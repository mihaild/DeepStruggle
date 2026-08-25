# Generic Trainer: Configurable multi-stage training with live snapshot tournament evaluation.

import os
import sys
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
from ai.env.reward_calculator import ZeroSumTerminalReward, ShapedZeroSumReward
from ai.env.ts_env import TsVectorizedEnv
from ai.training.rollout_buffer import RolloutBuffer
from ai.training.warmup_dataset_loader import WarmupDataset
from ai.eval.player_agent import PlayerAgent, NeuralAgent, load_agent, resolve_device
from ai.eval.tournament_evaluator import TournamentEvaluator


def run_behavioral_cloning_warmup(
    model: nn.Module,
    dataset_path: str,
    output_checkpoint_path: str,
    epochs: int = 5,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: Optional[Union[torch.device, str]] = None,
) -> None:
    dev = resolve_device(device)
    model.to(dev)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"=== Loading Warm-up Dataset from: {dataset_path} ===", flush=True)
    t0 = time.time()
    ds = WarmupDataset(dataset_path)
    tensors = ds.build_in_memory_tensors(device=dev)
    obs = tensors["observations"]
    masks = tensors["action_masks"]
    acts = tensors["actions"]
    val_wins = tensors["win_targets"]
    val_vps = tensors["vp_targets"]
    num_samples = obs.size(0)
    print(f"Loaded {num_samples:,} transition samples in {time.time() - t0:.1f}s. Training {epochs} epochs...", flush=True)

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(num_samples, device=dev)
        total_loss = 0.0
        correct_actions = 0

        for idx in range(0, num_samples, batch_size):
            batch_indices = perm[idx : idx + batch_size]
            b_obs = obs[batch_indices]
            b_mask = masks[batch_indices]
            b_act = acts[batch_indices]
            b_val = val_wins[batch_indices]
            b_vp = val_vps[batch_indices]

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
            total_loss += loss.item() * b_obs.size(0)

        avg_loss = total_loss / max(1, num_samples)
        acc = (correct_actions / max(1, num_samples)) * 100.0
        print(f"  Epoch {epoch:2d}/{epochs:2d} | Loss: {avg_loss:.4f} | Action Acc: {acc:.2f}% | Samples: {num_samples}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_checkpoint_path)), exist_ok=True)
    torch.save(model.state_dict(), output_checkpoint_path)
    print(f"=== Warm-up Complete in {time.time() - t0:.1f}s. Saved to: {output_checkpoint_path} ===\n", flush=True)


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
    reward_scheme: str = "terminal",
    output_dir: Optional[str] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> None:
    dev = resolve_device(device)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = output_dir or os.path.join("checkpoints", f"run_{arch}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "training_metrics.jsonl")
    report_path = os.path.join(out_dir, "tournament_report.md")

    # 1. Initialize Model
    if arch == "v2":
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
    reward_calc = ShapedZeroSumReward() if reward_scheme == "shaped" else ZeroSumTerminalReward()
    env = TsVectorizedEnv(num_envs=num_envs, base_seed=12345, reward_calculator=reward_calc)

    ref_model = create_coldwar_net_v2(dev) if arch == "v2" else create_coldwar_net(dev)
    ref_model.load_state_dict(model.state_dict())
    ref_model.to(dev)
    ref_model.eval()

    buffer = RolloutBuffer(
        buffer_size=buffer_size,
        num_envs=num_envs,
        obs_dim=4293,
        action_dim=212,
        device=dev,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Stratified exploration temperatures across environments
    env_temps = np.zeros(num_envs, dtype=np.float32)
    env_temps[0 : num_envs // 4] = 0.15
    env_temps[num_envs // 4 : num_envs // 2] = 0.50
    env_temps[num_envs // 2 : 3 * num_envs // 4] = 0.10
    env_temps[3 * num_envs // 4 :] = 0.35

    # Opponent agents for evaluation (starts with baselines, dynamically appends past snapshots)
    opp_specs = eval_opponents or ["random", "heuristic"]
    opponents: List[PlayerAgent] = []
    for spec in opp_specs:
        try:
            opponents.append(load_agent(spec, device=dev))
        except Exception as e:
            print(f"Warning: Could not load opponent \"{spec}\": {e}", flush=True)

    obs_np, masks_np, _ = env.reset_all()

    t_start = time.time()
    next_eval_time = snapshot_interval_seconds
    it = 0
    total_env_steps = 0
    steps_since_ref_update = 0
    ref_update_freq = 200_000

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

        it += 1
        model.eval()
        buffer.reset()

        # Rollout Collection
        for step in range(buffer_size):
            obs_t = torch.from_numpy(obs_np).float().to(dev)
            masks_t = torch.from_numpy(masks_np).to(dev)

            with torch.no_grad():
                logits, v_win, v_vp = model(obs_t, masks_t)
                scaled_logits = logits / torch.from_numpy(env_temps).unsqueeze(-1).to(dev)
                dist = torch.distributions.Categorical(logits=scaled_logits)
                actions_t = dist.sample()
                log_probs_t = dist.log_prob(actions_t)

            actions_np = actions_t.cpu().numpy()
            next_obs_np, next_masks_np, rewards_np, dones_np, info = env.step(actions_np)

            # Store exact acting players
            buffer.add(
                obs=obs_t,
                masks=masks_t,
                actions=actions_t,
                log_probs=log_probs_t,
                values_win=v_win.squeeze(-1),
                values_vp=v_vp.squeeze(-1),
                rewards=torch.from_numpy(rewards_np).float().to(dev),
                dones=torch.from_numpy(dones_np).float().to(dev),
                players=torch.from_numpy(info["acting_players"]).to(dev),
            )

            obs_np = next_obs_np
            masks_np = next_masks_np

        # GAE Bootstrap
        last_obs_t = torch.from_numpy(obs_np).float().to(dev)
        last_masks_t = torch.from_numpy(masks_np).to(dev)
        with torch.no_grad():
            _, last_v_win, last_v_vp = model(last_obs_t, last_masks_t)
            last_dones = torch.from_numpy(dones_np).to(dev)
            last_players = torch.from_numpy(info["acting_players"]).to(dev)

        buffer.compute_gae(
            last_v_win=last_v_win.squeeze(-1),
            last_v_vp=last_v_vp.squeeze(-1),
            last_dones=last_dones,
            last_players=last_players,
            gamma=0.999,
            gae_lambda=0.98,
        )

        steps_collected = buffer_size * num_envs
        total_env_steps += steps_collected
        steps_since_ref_update += steps_collected

        # Inner Loop SGD Updates
        model.train()
        total_loss_accum = 0.0
        pol_loss_accum = 0.0
        val_loss_accum = 0.0
        kl_accum = 0.0
        entropy_accum = 0.0
        clip_frac_accum = 0.0
        num_updates = 0

        for _ in range(4):
            for b_obs, b_mask, b_act, b_old_lp, b_adv, b_ret_win, b_ret_vp in buffer.get_batches(batch_size):
                cur_logits, cur_v_win, cur_v_vp = model(b_obs, b_mask)
                cur_v_win = cur_v_win.squeeze(-1)
                cur_v_vp = cur_v_vp.squeeze(-1)

                cur_dist = torch.distributions.Categorical(logits=cur_logits)
                cur_lp = cur_dist.log_prob(b_act)
                cur_entropy = cur_dist.entropy()

                ratio = torch.exp(cur_lp - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 0.8, 1.2) * b_adv
                ppo_loss = -torch.min(surr1, surr2).mean()

                clip_frac = ((ratio < 0.8) | (ratio > 1.2)).float().mean().item()

                with torch.no_grad():
                    ref_logits, _, _ = ref_model(b_obs, b_mask)
                    ref_log_p = F.log_softmax(ref_logits, dim=-1)

                cur_p = F.softmax(cur_logits, dim=-1)
                cur_log_p = F.log_softmax(cur_logits, dim=-1)
                kl_div = torch.sum(cur_p * (cur_log_p - ref_log_p), dim=-1).mean()

                policy_loss = ppo_loss + eta * kl_div - entropy_coef * cur_entropy.mean()
                val_loss = F.mse_loss(cur_v_win, b_ret_win) + 0.05 * F.mse_loss(cur_v_vp, b_ret_vp)
                loss = policy_loss + 0.5 * val_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss_accum += loss.item()
                pol_loss_accum += ppo_loss.item()
                val_loss_accum += val_loss.item()
                kl_accum += kl_div.item()
                entropy_accum += cur_entropy.mean().item()
                clip_frac_accum += clip_frac
                num_updates += 1

        if steps_since_ref_update >= ref_update_freq:
            ref_model.load_state_dict(model.state_dict())
            ref_model.to(dev)
            steps_since_ref_update = 0

        # Log training step metrics
        step_metrics = {
            "iteration": it,
            "elapsed_seconds": int(elapsed),
            "total_steps": total_env_steps,
            "steps_per_sec": int(total_env_steps / max(1.0, elapsed)),
            "loss": total_loss_accum / max(1, num_updates),
            "policy_loss": pol_loss_accum / max(1, num_updates),
            "value_loss": val_loss_accum / max(1, num_updates),
            "kl_div": kl_accum / max(1, num_updates),
            "entropy": entropy_accum / max(1, num_updates),
            "clip_frac": clip_frac_accum / max(1, num_updates),
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
    print(f"=== Training Complete. Final Checkpoint: {final_snap_path} ===", flush=True)


def format_loss_causes(causes: Dict[str, int]) -> str:
    """Formats loss causes breakdown string."""
    if not causes:
        return "None (0 losses)"
    sorted_items = sorted(causes.items(), key=lambda x: x[1], reverse=True)
    return ", ".join([f"{k} ({v})" for k, v in sorted_items])


def evaluate_and_log_snapshot(
    model: nn.Module,
    opponents: List[PlayerAgent],
    elapsed_seconds: int,
    out_dir: str,
    report_path: str,
    games_per_side: int,
    device: torch.device,
    add_to_opponents_after: bool = True,
    arch: str = "v2",
) -> None:
    """Runs tournament matches evaluating ONLY the newest model snapshot against all registered opponents.

    Tracks separate win rates and specific loss causes for US and USSR sides.
    """
    model.eval()
    model.to(device)
    active_agent = NeuralAgent(model=model, name=f"Checkpoint_{elapsed_seconds}s", device=device)
    print(f"\n--- Evaluating Newest Snapshot @ {elapsed_seconds}s against {len(opponents)} Opponents ({games_per_side*2} games each) ---", flush=True)

    header = f"### Snapshot @ {elapsed_seconds}s ({elapsed_seconds // 60}m {elapsed_seconds % 60}s)\n\n"
    table = "| Opponent | Overall Win Rate | Win Rate (US) | Win Rate (USSR) | Avg Turn / Steps | Avg VP Margin | Loss Causes as US | Loss Causes as USSR |\n|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|\n"

    for opp in opponents:
        res = TournamentEvaluator.play_matchup(
            agent_a=active_agent,
            agent_b=opp,
            games_per_side=games_per_side,
            base_seed=10000 + elapsed_seconds,
        )

        w = res["a_wins"]
        l = res["b_wins"]
        d = res["draws"]
        wr_all = res["win_rate_a"] * 100.0

        w_us = res["a_wins_as_us"]
        l_us = res["a_losses_as_us"]
        wr_us = res["win_rate_a_as_us"] * 100.0

        w_ussr = res["a_wins_as_ussr"]
        l_ussr = res["a_losses_as_ussr"]
        wr_ussr = res["win_rate_a_as_ussr"] * 100.0

        avg_turn = res["avg_turn"]
        avg_steps = res["avg_steps"]
        vp_m = res["avg_vp_margin_a"]

        causes_us_str = format_loss_causes(res["causes_loss_us"])
        causes_ussr_str = format_loss_causes(res["causes_loss_ussr"])

        row = (
            f"| **{opp.name}** | **{wr_all:.1f}%** ({w}W-{l}L-{d}D) | "
            f"**{wr_us:.1f}%** ({w_us}W-{l_us}L) | "
            f"**{wr_ussr:.1f}%** ({w_ussr}W-{l_ussr}L) | "
            f"T{avg_turn:.1f} / {avg_steps:.0f} | {vp_m:+.1f} | "
            f"{causes_us_str} | {causes_ussr_str} |\n"
        )
        table += row

        print(
            f"  vs {opp.name:25s} -> Overall: {wr_all:5.1f}% ({w}W-{l}L) | US: {wr_us:5.1f}% ({w_us}W-{l_us}L) | USSR: {wr_ussr:5.1f}% ({w_ussr}W-{l_ussr}L)\n"
            f"       Losses as US:   {causes_us_str}\n"
            f"       Losses as USSR: {causes_ussr_str}",
            flush=True,
        )

    with open(report_path, "a", encoding="utf-8") as f:
        f.write(header + table + "\n\n")
    print("--------------------------------------------------------------------------------\n", flush=True)

    # Register this snapshot into opponents list for future snapshots to test against
    if add_to_opponents_after:
        frozen_model = create_coldwar_net_v2(device) if arch == "v2" else create_coldwar_net(device)
        frozen_model.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
        frozen_model.to(device)
        frozen_model.eval()
        snapshot_agent = NeuralAgent(model=frozen_model, name=f"Snapshot_{elapsed_seconds}s", device=device)
        opponents.append(snapshot_agent)
