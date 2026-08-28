import os
if 'TRITON_CACHE_DIR' not in os.environ:
    os.environ['TRITON_CACHE_DIR'] = os.path.abspath('.triton_cache')
# Generic Trainer: Configurable multi-stage training with live snapshot tournament evaluation.

import os
import sys
import subprocess
import time
import json
import argparse
from typing import List, Optional, Dict, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ts_engine as ts
from ai.models.coldwar_net import ColdWarNet, create_coldwar_net
from ai.models.coldwar_net_v2 import ColdWarNetV2, create_coldwar_net_v2
from ai.models.coldwar_net_v3 import ColdWarNetV3, create_coldwar_net_v3
from ai.models.coldwar_net_v4 import ColdWarNetV4, create_coldwar_net_v4
from ai.rewards.reward_calculator import ZeroSumTerminalReward, ShapedZeroSumReward, BlunderAwareRewardCalculator, UsefulActionsReward
from bindings.ts_env import TsVectorizedEnv
from ai.training.rollout_buffer import RolloutBuffer
from ai.training.nash_pg import NashPGTrainer, OracleGuidedNashPGTrainer
from ai.training.warmup_dataset_loader import WarmupDataset
from tools.lib.player_agent import PlayerAgent, NeuralAgent, load_agent, resolve_device
from tools.lib.tournament_evaluator import TournamentEvaluator


def run_behavioral_cloning_warmup(
    model: nn.Module,
    dataset_path: str,
    output_checkpoint_path: str,
    epochs: int = 5,
    batch_size: int = 1024,
    lr: float = 1e-3,
    max_games: Optional[int] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> None:
    dev = resolve_device(device)
    model.to(dev)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"=== Loading Warm-up Dataset (OOM-Safe Streaming Mode) from: {dataset_path} ===", flush=True)
    t0 = time.time()
    ds = WarmupDataset(dataset_path)

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        total_loss = 0.0
        correct_actions = 0
        samples_seen = 0

        for b_obs, b_mask, b_act, b_val, b_vp in ds.stream_batches(
            batch_size=batch_size, max_games=max_games, device=dev, shuffle_buffer_size=4096
        ):
            logits, v_win, v_vp = model(b_obs, b_mask)
            policy_loss = F.cross_entropy(logits, b_act)
            val_win_loss = F.mse_loss(v_win.squeeze(-1), b_val)
            val_vp_loss = F.mse_loss(v_vp.squeeze(-1), b_vp)

            loss = policy_loss + 0.5 * val_win_loss + 0.05 * val_vp_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds = torch.argmax(logits, dim=-1)
            correct_actions += (preds == b_act).sum().item()
            cur_b_size = b_obs.size(0)
            total_loss += loss.item() * cur_b_size
            samples_seen += cur_b_size

            if samples_seen % 100000 < batch_size:
                cur_l = total_loss / max(1, samples_seen)
                cur_acc = (correct_actions / max(1, samples_seen)) * 100.0
                dt = max(1e-2, time.time() - t_epoch)
                print(f"  Epoch {epoch:2d}/{epochs:2d} | {samples_seen:,} samples ({samples_seen/dt:.0f} samples/s) | Loss: {cur_l:.4f} | Acc: {cur_acc:.2f}%", flush=True)

        avg_loss = total_loss / max(1, samples_seen)
        acc = (correct_actions / max(1, samples_seen)) * 100.0
        dt = max(1e-2, time.time() - t_epoch)
        print(f"  Epoch {epoch:2d}/{epochs:2d} COMPLETED in {dt:.1f}s | Loss: {avg_loss:.4f} | Action Acc: {acc:.2f}% | Total Samples: {samples_seen:,}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_checkpoint_path)), exist_ok=True)
    torch.save(model.state_dict(), output_checkpoint_path)
    print(f"=== Warm-up Complete in {time.time() - t0:.1f}s. Saved to: {output_checkpoint_path} ===", flush=True)


def evaluate_and_log_snapshot(
    model: nn.Module,
    opponents: List[PlayerAgent],
    elapsed_seconds: int,
    out_dir: str,
    report_path: str,
    games_per_side: int = 25,
    device: Optional[Union[torch.device, str]] = None,
    add_to_opponents_after: bool = True,
    arch: str = "v2",
) -> None:
    dev = resolve_device(device)
    snap_name = f"snapshot_{elapsed_seconds}s"
    current_agent = NeuralAgent(model=model, device=dev, name=snap_name)

    print(f"\n--- Evaluating Newest Snapshot @ {elapsed_seconds}s against {len(opponents)} Opponents ({games_per_side*2} games each) ---", flush=True)
    report_entry = [f"### Snapshot @ {elapsed_seconds}s (Evaluated against {len(opponents)} baselines / past snapshots)\n\n"]
    report_entry.append("| Opponent | Overall Win Rate | As US Win Rate | As USSR Win Rate | Top Loss Causes (US) | Top Loss Causes (USSR) |\n")
    report_entry.append("|:---|:---:|:---:|:---:|:---|:---|\n")

    for opp in opponents:
        res = TournamentEvaluator.play_matchup(current_agent, opp, games_per_side=games_per_side)
        wr_tot = res["win_rate_a"] * 100.0
        wr_us = res["win_rate_a_as_us"] * 100.0
        wr_ussr = res["win_rate_a_as_ussr"] * 100.0

        top_us = ", ".join([f"{k} ({v})" for k, v in list(res["causes_loss_us"].items())[:3]]) or "None (0 losses)"
        top_ussr = ", ".join([f"{k} ({v})" for k, v in list(res["causes_loss_ussr"].items())[:3]]) or "None (0 losses)"

        print(f"  vs {opp.name:<25s} -> Overall: {wr_tot:5.1f}% ({res['a_wins']}W-{res['b_wins']}L) | US: {wr_us:5.1f}% ({res['a_wins_as_us']}W-{res['a_losses_as_us']}L) | USSR: {wr_ussr:5.1f}% ({res['a_wins_as_ussr']}W-{res['a_losses_as_ussr']}L)", flush=True)
        print(f"       Losses as US:   {top_us}", flush=True)
        print(f"       Losses as USSR: {top_ussr}", flush=True)

        report_entry.append(f"| **{opp.name}** | **{wr_tot:.1f}%** ({res['a_wins']}W-{res['b_wins']}L) | {wr_us:.1f}% ({res['a_wins_as_us']}W-{res['a_losses_as_us']}L) | {wr_ussr:.1f}% ({res['a_wins_as_ussr']}W-{res['a_losses_as_ussr']}L) | {top_us} | {top_ussr} |\n")

    report_entry.append("\n---\n\n")
    print("-" * 80 + "\n", flush=True)

    with open(report_path, "a", encoding="utf-8") as f:
        f.write("".join(report_entry))

    if add_to_opponents_after:
        if arch == "v4":
            frozen_net = create_coldwar_net_v4(dev)
        elif arch == "v3":
            frozen_net = create_coldwar_net_v3(dev)
        elif arch == "v2":
            frozen_net = create_coldwar_net_v2(dev)
        else:
            frozen_net = create_coldwar_net(dev)
        frozen_net.load_state_dict(model.state_dict())
        frozen_net.to(dev)
        frozen_net.eval()
        opponents.append(NeuralAgent(model=frozen_net, device=dev, name=f"Snapshot_{elapsed_seconds}s"))


def run_post_training_tournament(
    checkpoint_dir: str,
    additional_models: Optional[List[str]] = None,
    games_per_side: int = 500,
    device: Optional[Union[torch.device, str]] = None,
) -> None:
    from tools.tournament import run_massive_tournament
    from tools.lib.checkpoint_utils import discover_checkpoints

    dev = resolve_device(device)
    print("\n" + "=" * 80, flush=True)
    print(" LAUNCHING POST-TRAINING MASSIVE TOURNAMENT BENCHMARK", flush=True)
    print("=" * 80 + "\n", flush=True)

    ckpts = discover_checkpoints(checkpoint_dir)
    models_to_evaluate = [c["path"] for c in ckpts]

    baselines = additional_models or ["heuristic", "random"]
    for b in baselines:
        if b not in models_to_evaluate:
            models_to_evaluate.append(b)

    report_out = os.path.join(checkpoint_dir, "final_tournament_report.md")
    json_out = os.path.join(checkpoint_dir, "final_tournament_results.json")

    run_massive_tournament(
        model_specs=models_to_evaluate,
        games_per_side=games_per_side,
        device=str(dev),
        anchor_model="HeuristicBot",
        anchor_elo=1500.0,
        output_report=report_out,
        output_json=json_out,
    )


def train_pipeline(
    arch: str = "v2",
    warmup_checkpoint: Optional[str] = None,
    warmup_dataset: Optional[str] = None,
    bc_epochs: int = 5,
    duration_seconds: int = 3600,
    snapshot_interval_seconds: int = 600,
    eval_opponents: Optional[List[str]] = None,
    eval_games_per_side: int = 50,
    num_envs: int = 512,
    buffer_size: int = 128,
    batch_size: int = 4096,
    lr: float = 3e-4,
    eta: float = 0.1,
    entropy_coef: float = 0.01,
    reward_scheme: str = "blunder_aware",
    output_dir: Optional[str] = None,
    description: Optional[str] = None,
    device: Optional[Union[torch.device, str]] = None,
    post_tournament: bool = False,
    post_tournament_models: Optional[List[str]] = None,
    post_tournament_games: int = 500,
    curriculum_switch_seconds: Optional[int] = None,
    curriculum_switch_fraction: float = 0.5,
) -> None:
    dev = resolve_device(device)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = output_dir or os.path.join("data", "checkpoints", f"run_{arch}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "training_metrics.jsonl")
    report_path = os.path.join(out_dir, "tournament_report.md")

    # Write metadata.json recording git commit, training mode, and description
    git_commit = "unknown"
    git_message = "unknown"
    git_dirty = False
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        git_message = subprocess.check_output(["git", "log", "-1", "--pretty=%B"], text=True).strip()
        status_out = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        git_dirty = bool(status_out)
    except Exception:
        pass

    metadata_path = os.path.join(out_dir, "metadata.json")
    metadata_info = {
        "run_id": os.path.basename(out_dir),
        "arch": arch,
        "base_commit": git_commit,
        "commit_message": git_message,
        "git_dirty": git_dirty,
        "training_mode": reward_scheme,
        "reward_scheme": reward_scheme,
        "duration_seconds": duration_seconds,
        "snapshot_interval_seconds": snapshot_interval_seconds,
        "num_envs": num_envs,
        "description": description or f"Self-play RL training with arch={arch}, reward={reward_scheme}, duration={duration_seconds}s.",
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_info, f, indent=2)

    # 1. Initialize Model
    if arch == "v4":
        model = create_coldwar_net_v4(dev)
    elif arch == "v3":
        model = create_coldwar_net_v3(dev)
    elif arch == "v2":
        model = create_coldwar_net_v2(dev)
    else:
        model = create_coldwar_net(dev)

    # 2. Handle Warm-up
    if warmup_checkpoint and os.path.exists(warmup_checkpoint):
        print(f"Loading Warm-up Checkpoint from: {warmup_checkpoint}", flush=True)
        model.load_state_dict(torch.load(warmup_checkpoint, map_location=dev, weights_only=True))
        model.to(dev)
    elif warmup_dataset and os.path.exists(warmup_dataset):
        warmup_save_path = os.path.join(out_dir, f"coldwar_net_{arch}_warmup.pt")
        run_behavioral_cloning_warmup(
            model=model,
            dataset_path=warmup_dataset,
            output_checkpoint_path=warmup_save_path,
            epochs=bc_epochs,
            device=dev,
        )
    else:
        print("No warmup checkpoint or dataset specified. Starting from fresh weights.", flush=True)

    # 3. Setup Reward Calculator & Vectorized Env
    is_curriculum = (reward_scheme == "curriculum")
    if is_curriculum or reward_scheme == "useful_actions":
        reward_calc = UsefulActionsReward()
    elif reward_scheme == "blunder_aware":
        reward_calc = BlunderAwareRewardCalculator()
    elif reward_scheme == "shaped":
        reward_calc = ShapedZeroSumReward()
    else:
        reward_calc = ZeroSumTerminalReward()
    env = TsVectorizedEnv(num_envs=num_envs, base_seed=12345, reward_calculator=reward_calc)

    # Curriculum timing configuration
    if is_curriculum:
        if curriculum_switch_seconds is not None:
            curriculum_switch_at = float(curriculum_switch_seconds)
        else:
            curriculum_switch_at = float(duration_seconds) * float(curriculum_switch_fraction)
        curriculum_switched = False
        print(f"[CURRICULUM] Active: Stage 1 = UsefulActionsReward (first {curriculum_switch_at:.0f}s), Stage 2 = BlunderAwareRewardCalculator", flush=True)
    else:
        curriculum_switch_at = float("inf")
        curriculum_switched = False

    # 4. Instantiate Unified NashPG Trainer (OracleGuided for V4, standard for V1/V2/V3)
    TrainerCls = OracleGuidedNashPGTrainer if arch == "v4" else NashPGTrainer
    trainer = TrainerCls(
        active_net=model,
        env=env,
        num_envs=num_envs,
        buffer_size=buffer_size,
        batch_size=batch_size,
        lr=lr,
        eta=eta,
        ent_coef=entropy_coef,
        gamma=0.999,
        gae_lambda=0.98,
        num_epochs=4,
        ref_update_freq=200_000,
        max_grad_norm=1.0,
        slice_turn_boundaries=(reward_scheme == "blunder_aware"),
        temperature_schedule=True,
        device=dev,
    )

    # Opponent agents for evaluation (starts with baselines, dynamically appends past snapshots)
    opp_specs = eval_opponents or ["random", "heuristic"]
    opponents: List[PlayerAgent] = []
    for spec in opp_specs:
        try:
            opponents.append(load_agent(spec, device=dev))
        except Exception as e:
            print(f"Warning: Could not load opponent \"{spec}\": {e}", flush=True)

    t_start = time.time()
    next_eval_time = snapshot_interval_seconds
    it = 0

    print("=" * 80, flush=True)
    print(f"STARTING GENERIC TRAINING PIPELINE ({duration_seconds}s, Snapshots every {snapshot_interval_seconds}s)", flush=True)
    print(f"Arch: {arch} | Envs: {num_envs} | Opponents to evaluate: {[o.name for o in opponents]}", flush=True)
    print("=" * 80, flush=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Snapshot Tournament Evaluation Report ({arch.upper()})\n\n")

    # Initial Snapshot (0s / start)
    snap_0_path = os.path.join(out_dir, "snapshot_0s.pt")
    torch.save(model.state_dict(), snap_0_path)
    evaluate_and_log_snapshot(
        model=model,
        opponents=opponents,
        elapsed_seconds=0,
        out_dir=out_dir,
        report_path=report_path,
        games_per_side=eval_games_per_side,
        device=dev,
        add_to_opponents_after=True,
        arch=arch,
    )

    while True:
        elapsed = time.time() - t_start
        if elapsed >= duration_seconds:
            break

        # Curriculum stage switch from UsefulActionsReward to BlunderAwareRewardCalculator
        if is_curriculum and not curriculum_switched and elapsed >= curriculum_switch_at:
            curriculum_switched = True
            trainer.set_reward_calculator(BlunderAwareRewardCalculator())
            trainer.set_slice_turn_boundaries(True)
            print(f"\n{'=' * 80}", flush=True)
            print(f"[CURRICULUM] STAGE 2 SWITCH: Replaced UsefulActionsReward with BlunderAwareRewardCalculator at elapsed={elapsed:.1f}s / {duration_seconds}s", flush=True)
            print(f"{'=' * 80}\n", flush=True)

        it += 1
        iteration_metrics = trainer.train_iteration()
        total_env_steps = trainer.total_env_steps

        # Log training step metrics
        step_metrics = {
            "iteration": it,
            "elapsed_seconds": int(elapsed),
            "total_steps": total_env_steps,
            "steps_per_sec": int(total_env_steps / max(1.0, elapsed)),
            "loss": iteration_metrics["loss"],
            "policy_loss": iteration_metrics["policy_loss"],
            "value_loss": iteration_metrics["val_loss"],
            "kl_div": iteration_metrics["kl_div"],
            "entropy": iteration_metrics["entropy"],
            "clip_frac": iteration_metrics.get("clip_frac", 0.0),
            "belief_loss": iteration_metrics.get("belief_loss", 0.0),
            "oracle_loss": iteration_metrics.get("oracle_loss", 0.0),
            "distill_loss": iteration_metrics.get("distill_loss", 0.0),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(step_metrics) + "\n")

        if it % 10 == 0:
            print(
                f"[{int(elapsed)}s/{duration_seconds}s] It {it:4d} | Steps: {total_env_steps:,} ({step_metrics['steps_per_sec']:,} st/s) | "
                f"Loss: {step_metrics['loss']:.3f} | KL: {step_metrics['kl_div']:.4f} | Ent: {step_metrics['entropy']:.3f} | Clip: {step_metrics['clip_frac']*100:.1f}%",
                flush=True,
            )

        # Snapshot Evaluation
        if elapsed >= next_eval_time:
            snap_path = os.path.join(out_dir, f"snapshot_{int(elapsed)}s.pt")
            torch.save(model.state_dict(), snap_path)
            evaluate_and_log_snapshot(
                model=model,
                opponents=opponents,
                elapsed_seconds=int(elapsed),
                out_dir=out_dir,
                report_path=report_path,
                games_per_side=eval_games_per_side,
                device=dev,
                add_to_opponents_after=True,
                arch=arch,
            )
            next_eval_time += snapshot_interval_seconds

    # Final Snapshot
    final_snap_path = os.path.join(out_dir, "snapshot_final.pt")
    torch.save(model.state_dict(), final_snap_path)
    evaluate_and_log_snapshot(
        model=model,
        opponents=opponents,
        elapsed_seconds=int(time.time() - t_start),
        out_dir=out_dir,
        report_path=report_path,
        games_per_side=eval_games_per_side,
        device=dev,
        add_to_opponents_after=False,
        arch=arch,
    )

    print(f"\n=== Training Complete. Final Checkpoint: {final_snap_path} ===", flush=True)

    # Post-training massive tournament
    if post_tournament:
        run_post_training_tournament(
            checkpoint_dir=out_dir,
            additional_models=post_tournament_models or ["random", "heuristic"],
            games_per_side=post_tournament_games,
            device=dev,
        )
