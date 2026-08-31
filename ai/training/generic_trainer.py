import os
if 'TRITON_CACHE_DIR' not in os.environ:
    os.environ['TRITON_CACHE_DIR'] = os.path.abspath('.triton_cache')
# Generic Trainer: Configurable multi-stage training with live snapshot tournament evaluation.

import os
import sys
import subprocess
import time
import collections
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
from ai.training.start_pool import DEFAULT_TURN_MIX, StartPositionPool
from ai.training.warmup_dataset_loader import WarmupDataset
from bindings.ts_env import ENDING_REASON_KEYS
from tools.lib.player_agent import PlayerAgent, NeuralAgent, load_agent, resolve_device
from tools.lib.batch_tournament import BatchMatchRunner
from tools.lib.tournament_evaluator import TournamentEvaluator

# TensorBoard is optional: a missing (or broken) install must never take down a
# multi-hour training run, so the writer degrades to a no-op and JSONL logging carries on.
_SUMMARY_WRITER_CLS: Optional[Any] = None
_SUMMARY_WRITER_IMPORT_ERROR: Optional[str] = None
try:
    from torch.utils.tensorboard import SummaryWriter as _ImportedSummaryWriter
    _SUMMARY_WRITER_CLS = _ImportedSummaryWriter
except Exception as _tb_err:  # pragma: no cover - depends on the local install
    _SUMMARY_WRITER_IMPORT_ERROR = str(_tb_err)


# Maps training_metrics.jsonl keys to TensorBoard tags. Every JSONL metric is mirrored;
# anything not listed here lands under "misc/" so new metrics cannot silently go missing.
TB_TAGS: Dict[str, str] = {
    "elapsed_seconds": "progress/elapsed_seconds",
    "total_steps": "progress/total_steps",
    "steps_per_sec": "progress/steps_per_sec",
    "loss": "train/loss",
    "policy_loss": "train/policy_loss",
    "value_loss": "train/value_loss",
    "kl_div": "train/kl_div",
    "entropy": "train/entropy",
    "clip_frac": "train/clip_frac",
    "defcon_risk_loss": "train/defcon_risk_loss",
    "belief_loss": "train/belief_loss",
    "oracle_loss": "train/oracle_loss",
    "distill_loss": "train/distill_loss",
    "explained_variance": "diagnostics/explained_variance",
    "adv_std": "diagnostics/adv_std",
    "adv_std_raw": "diagnostics/adv_std_raw",
    "adv_frac_near_zero": "diagnostics/adv_frac_near_zero",
    "entropy_fixed_probe": "diagnostics/entropy_fixed_probe",
    "diag/mean_final_turn": "positions/mean_final_turn",
    "diag/frac_reaching_turn9": "positions/frac_reaching_turn9",
    "diag/empty_battlegrounds_turn8": "positions/empty_battlegrounds_turn8",
    "diag/empty_battlegrounds_turn5": "positions/empty_battlegrounds_turn5",
    "diag/salvageable_frac_turn6": "positions/salvageable_frac_turn6",
    "diag/salvageable_given_reached_turn6": "positions/salvageable_given_reached_turn6",
    "decisive_win_take_rate": "decisive/win_take_rate",
    "decisive_loss_avoid_rate": "decisive/loss_avoid_rate",
    "decisive_win_available": "decisive/win_available",
    "decisive_loss_avoidable": "decisive/loss_avoidable",
    "episodes_completed": "game/episodes_completed",
    "mean_turn": "game/mean_turn",
    "median_turn": "game/median_turn",
    "ending_frac_20vp": "endings/20vp",
    "ending_frac_final_scoring": "endings/final_scoring",
    "ending_frac_defcon1_self": "endings/defcon1_self",
    "ending_frac_defcon1_provoked": "endings/defcon1_provoked",
    "ending_frac_held_scoring": "endings/held_scoring",
    "ending_frac_wargames": "endings/wargames",
}

# Per-start-turn variants. With mid-game start sampling on, a pooled game metric mixes
# real games with resumed ones; these keep the series separable in TensorBoard.
for _t in (1, 4, 6, 8, 10):
    TB_TAGS[f"episodes_completed_start{_t}"] = f"game_start{_t}/episodes_completed"
    TB_TAGS[f"mean_turn_start{_t}"] = f"game_start{_t}/mean_turn"
    TB_TAGS[f"median_turn_start{_t}"] = f"game_start{_t}/median_turn"
    for _k in ENDING_REASON_KEYS:
        TB_TAGS[f"ending_frac_{_k}_start{_t}"] = f"endings_start{_t}/{_k}"

# Metrics that describe completed episodes; meaningless (and misleading as zeros) on an
# iteration where no game finished, so they are held back from TensorBoard then.
EPISODE_DEPENDENT_KEYS = frozenset(
    ["mean_turn", "median_turn"] + [f"ending_frac_{k}" for k in ENDING_REASON_KEYS]
)


def episode_dependent_in(stats: Dict[str, float]) -> frozenset:
    """Which keys of `stats` are episode-dependent, including per-start-turn variants.

    A per-start-turn group is only emitted when that start turn actually completed an
    episode, so the suffixed keys cannot be enumerated up front -- publishing zeros for
    absent groups would be the misleading thing this hold-back exists to prevent.
    """
    return frozenset(
        k for k in stats
        if k in EPISODE_DEPENDENT_KEYS
        or any(k.startswith(f"{stem}_start") for stem in EPISODE_DEPENDENT_KEYS)
    )


class TensorBoardLogger:
    """Best-effort TensorBoard writer. Any failure disables it instead of raising."""

    def __init__(self, log_dir: str, enabled: bool = True) -> None:
        self.log_dir = log_dir
        self.writer: Optional[Any] = None
        if not enabled:
            return
        if _SUMMARY_WRITER_CLS is None:
            print(
                f"Warning: TensorBoard logging unavailable ({_SUMMARY_WRITER_IMPORT_ERROR}); "
                f"continuing with JSONL metrics only. Install with: pip install tensorboard",
                flush=True,
            )
            return
        try:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = _SUMMARY_WRITER_CLS(log_dir=log_dir)
            print(f"TensorBoard logging enabled -> {log_dir}  (tensorboard --logdir {log_dir})", flush=True)
        except Exception as e:
            self.writer = None
            print(f"Warning: Could not start TensorBoard writer at {log_dir}: {e}. Continuing without it.", flush=True)

    @property
    def active(self) -> bool:
        return self.writer is not None

    def log_metrics(self, metrics: Dict[str, Any], step: int, skip_keys: Optional[frozenset[str]] = None) -> None:
        if self.writer is None:
            return
        try:
            for key, value in metrics.items():
                if key == "iteration" or (skip_keys is not None and key in skip_keys):
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                self.writer.add_scalar(TB_TAGS.get(key, f"misc/{key}"), float(value), step)
        except Exception as e:
            print(f"Warning: TensorBoard logging failed ({e}); disabling TensorBoard for the rest of the run.", flush=True)
            self.writer = None

    def log_text(self, tag: str, text: str, step: int) -> None:
        if self.writer is None:
            return
        try:
            self.writer.add_text(tag, text, step)
        except Exception:
            pass

    def flush(self) -> None:
        if self.writer is None:
            return
        try:
            self.writer.flush()
        except Exception:
            pass

    def close(self) -> None:
        if self.writer is None:
            return
        try:
            self.writer.close()
        except Exception:
            pass
        self.writer = None


def _episode_group_stats(episodes: List[Dict[str, Any]], suffix: str) -> Dict[str, float]:
    stats: Dict[str, float] = {f"episodes_completed{suffix}": float(len(episodes))}
    turns = [float(ep["turn"]) for ep in episodes if "turn" in ep]
    stats[f"mean_turn{suffix}"] = float(np.mean(turns)) if turns else 0.0
    stats[f"median_turn{suffix}"] = float(np.median(turns)) if turns else 0.0

    reasons = [str(ep.get("ending_reason", "")) for ep in episodes]
    counted = [r for r in reasons if r]
    for key in ENDING_REASON_KEYS:
        stats[f"ending_frac_{key}{suffix}"] = (
            float(sum(1 for r in counted if r == key)) / float(len(counted)) if counted else 0.0
        )
    return stats


def summarize_completed_episodes(episodes: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregates the episodes that finished during one iteration into scalar metrics.

    Game length is reported at the game-turn granularity (mean and median terminal turn),
    and the ending-reason mix as a fraction of the episodes completed this iteration.

    Everything is also reported per *start* turn. With mid-game start sampling on, half the
    environments begin partway through a game, so a pooled mean turn or ending mix
    describes neither the real game nor the resumed one -- a run reads a mean turn of 8
    while its turn-1 games still end at 6. The unsuffixed keys keep the pooled figures for
    continuity; read the _start1 series when comparing against runs without a pool.
    """
    stats = _episode_group_stats(episodes, "")

    by_start: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
    for ep in episodes:
        by_start[int(ep.get("start_turn", 1))].append(ep)
    for start_turn, group in by_start.items():
        stats.update(_episode_group_stats(group, f"_start{start_turn}"))
    return stats


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
    decisive_games: int = 128,
    position_games: int = 128,
    num_baselines: int = 0,
    max_snapshot_opponents: int = 4,
) -> Dict[str, float]:
    dev = resolve_device(device)
    snap_name = f"snapshot_{elapsed_seconds}s"
    current_agent = NeuralAgent(model=model, device=dev, name=snap_name)

    # Decisive-decision rates. These move long before win rate does, because a forced win
    # or avoidable forced loss arises at well under 1% of decisions -- rare enough to be
    # invisible in aggregate results, decisive enough that each one is a whole game.
    decisive_metrics: Dict[str, float] = {}
    try:
        from ai.eval.decisive_probe import measure_decisive_batched
        # Batched: the single-state loop spends 96.9% of its time in the policy forward,
        # so handing the GPU one state at a time was the whole cost.
        stats = measure_decisive_batched(model, num_envs=decisive_games)
        decisive_metrics = stats.as_metrics()
        print(f"  decisive: takes {stats.win_take_rate * 100:.0f}% of {stats.win_available} forced wins | "
              f"avoids {stats.loss_avoid_rate * 100:.0f}% of {stats.loss_avoidable} avoidable losses",
              flush=True)
    except Exception as exc:
        print(f"  (decisive probe unavailable: {exc})", flush=True)

    # Position diagnostics. Win rate plateaus while the play underneath stays incoherent;
    # these read the board instead. empty_battlegrounds_turn8 is the sharpest -- roughly a
    # quarter of battlegrounds sit untouched from turn 8 on, always the same ones, and the
    # count stops falling rather than slowly improving.
    position_metrics: Dict[str, float] = {}
    try:
        from ai.eval.position_diagnostics import format_report, profile_self_play_batched
        # Batched: ~106k decisions/sec against ~890 for the one-state-at-a-time loop, so
        # this costs seconds rather than minutes of every snapshot evaluation.
        profile = profile_self_play_batched(model, num_envs=position_games)
        position_metrics = profile["scalars"]
        print(f"  positions: {profile['scalars']['diag/empty_battlegrounds_turn8']:.1f} empty "
              f"battlegrounds at turn 8 | "
              f"{100 * profile['scalars']['diag/frac_reaching_turn9']:.0f}% reach turn 9 | "
              f"{100 * profile['scalars']['diag/salvageable_frac_turn6']:.0f}% salvageable at turn 6",
              flush=True)
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(f"\n<details><summary>Positions @ {elapsed_seconds}s</summary>\n\n```\n"
                    f"{format_report(profile)}\n```\n</details>\n\n")
    except Exception as exc:
        print(f"  (position diagnostics unavailable: {exc})", flush=True)

    print(f"\n--- Evaluating Newest Snapshot @ {elapsed_seconds}s against {len(opponents)} Opponents ({games_per_side*2} games each) ---", flush=True)
    report_entry = [f"### Snapshot @ {elapsed_seconds}s (Evaluated against {len(opponents)} baselines / past snapshots)\n\n"]
    report_entry.append("| Opponent | Overall Win Rate | As US Win Rate | As USSR Win Rate | Top Loss Causes (US) | Top Loss Causes (USSR) |\n")
    report_entry.append("|:---|:---:|:---:|:---:|:---|:---|\n")

    for opp in opponents:
        res = BatchMatchRunner.play_parallel_matchup(
            current_agent, opp, games_per_side=games_per_side, device=dev, temperature=0.1)
        wr_tot = res["win_rate_a"] * 100.0
        wr_us = res["win_rate_a_as_us"] * 100.0
        wr_ussr = res["win_rate_a_as_ussr"] * 100.0

        top_us = ", ".join([f"{k} ({v})" for k, v in sorted(res["causes_loss_us"].items(), key=lambda x: x[1], reverse=True)]) or "None (0 losses)"
        top_ussr = ", ".join([f"{k} ({v})" for k, v in sorted(res["causes_loss_ussr"].items(), key=lambda x: x[1], reverse=True)]) or "None (0 losses)"

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
        # Evaluate against the baselines plus only the most recent snapshots. The opponent
        # list otherwise grows by one every interval, so eval cost is quadratic in run
        # length: the last evaluation of a 3-hour run faced 14 opponents and took 957s
        # against a 900s snapshot interval, which starved training entirely. Older
        # snapshots are also the least informative comparison for a policy that has moved
        # well past them.
        if max_snapshot_opponents > 0:
            excess = len(opponents) - num_baselines - max_snapshot_opponents
            if excess > 0:
                del opponents[num_baselines:num_baselines + excess]

    return {**decisive_metrics, **position_metrics}


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
    train_steps: int = 0,
    max_snapshot_opponents: int = 4,
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
    slice_turn_boundaries: Optional[bool] = None,
    blunder_window: bool = True,
    gamma: float = 1.0,
    priority_alpha: float = 0.0,
    defcon_coef: float = 0.0,
    start_pool_frac: float = 0.0,
    start_pool_capacity: int = 512,
    start_pool_episodes: int = 600,
    ref_update_freq: int = 200_000,
    tensorboard: bool = True,
) -> None:
    dev = resolve_device(device)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = output_dir or os.path.join("data", "checkpoints", f"run_{arch}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "training_metrics.jsonl")
    report_path = os.path.join(out_dir, "tournament_report.md")
    tb = TensorBoardLogger(os.path.join(out_dir, "tb"), enabled=tensorboard)

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
    tb.log_text("run/metadata", "```json\n" + json.dumps(metadata_info, indent=2) + "\n```", 0)

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
        from tools.lib.player_agent import load_checkpoint_into
        load_checkpoint_into(model, torch.load(warmup_checkpoint, map_location=dev, weights_only=True))
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
    # Mid-game start positions. Self-play from turn 1 reaches the late game rarely and
    # plays it badly, so a share of environments resume from saved turn-boundary positions
    # instead. The pool is rebuilt as the policy moves on; see ai/training/start_pool.py.
    start_pool: Optional[StartPositionPool] = None
    env_start_turns: List[Optional[int]] = [None] * num_envs
    if start_pool_frac > 0.0:
        mix = dict(DEFAULT_TURN_MIX)
        pool_turns = tuple(t for t in mix if t != 1)
        start_pool = StartPositionPool(turns=pool_turns,
                                       capacity_per_turn=start_pool_capacity)

        def _start_provider(env_idx: int) -> Optional[Any]:
            turn = env_start_turns[env_idx]
            if turn is None or start_pool is None:
                return None
            return start_pool.sample(turn)

        env = TsVectorizedEnv(num_envs=num_envs, base_seed=12345,
                              reward_calculator=reward_calc,
                              start_provider=_start_provider)
    else:
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
        gamma=gamma,
        gae_lambda=0.98,
        num_epochs=4,
        ref_update_freq=ref_update_freq,
        max_grad_norm=1.0,
        # "auto" (None) means: rely on per-episode blunder windowing, not the blunt
        # global truncation. Global slicing measurably degrades the policy -- it strips
        # the outcome signal from clean wins too -- so it is opt-in for ablations only.
        slice_turn_boundaries=(False if slice_turn_boundaries is None else slice_turn_boundaries),
        blunder_window=blunder_window,
        priority_alpha=priority_alpha,
        defcon_coef=defcon_coef,
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
    # Evaluation and pool refreshes are charged to their own clock, not to the training
    # budget. Previously they came out of the same wall clock as training: one 3-hour run
    # spent 61% of its budget evaluating and completed 473 iterations while its A/B partner
    # completed 1024, purely because the arm that plays longer games has costlier
    # evaluations. That made a wall-clock budget silently policy-dependent.
    overhead_seconds = 0.0
    next_eval_time = snapshot_interval_seconds
    # A step budget makes two arms of an experiment exactly comparable; a time budget
    # cannot, because steps/sec depends on the policy. duration_seconds still bounds
    # wall time when no step budget is given.
    step_budget = int(train_steps)
    next_eval_steps = 0
    eval_every_steps = 0
    if step_budget > 0:
        evals_planned = max(1, duration_seconds // max(1, snapshot_interval_seconds))
        eval_every_steps = max(1, step_budget // evals_planned)
        next_eval_steps = eval_every_steps
    it = 0

    print("=" * 80, flush=True)
    print(f"STARTING GENERIC TRAINING PIPELINE ({duration_seconds}s, Snapshots every {snapshot_interval_seconds}s)", flush=True)
    num_baselines = len(opponents)
    budget_desc = (f"{train_steps:,} steps" if train_steps > 0
                   else f"{duration_seconds}s of training (evaluation excluded)")
    print(f"Arch: {arch} | Envs: {num_envs} | Budget: {budget_desc} | "
          f"Opponents to evaluate: {[o.name for o in opponents]} "
          f"(+ up to {max_snapshot_opponents} recent snapshots)", flush=True)
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
        num_baselines=num_baselines,
        max_snapshot_opponents=max_snapshot_opponents,
    )

    def _refresh_start_pool(tag: str) -> Dict[str, float]:
        """Rebuild the pool from the current policy and re-draw which env starts where."""
        if start_pool is None:
            return {}
        stats = start_pool.harvest(model, num_envs=min(256, max(32, num_envs)),
                                   num_episodes=start_pool_episodes)
        for i in range(num_envs):
            env_start_turns[i] = None
        assignment = start_pool.assign_starts(num_envs, DEFAULT_TURN_MIX)
        for i, turn in enumerate(assignment):
            env_start_turns[i] = turn
        sizes = {t: start_pool.size(t) for t in sorted(start_pool.buckets)}
        resumed = sum(1 for t in assignment if t is not None)
        print(f"  start pool ({tag}): {sizes} | {resumed}/{num_envs} envs resume mid-game",
              flush=True)
        return stats.as_metrics()

    def _progress_label(train_elapsed: float, steps_done: int) -> str:
        """Show progress against the active budget, with a projected wall-clock finish.

        A step budget is what makes two arms comparable, but it says nothing about how long
        the run will take, so project the finish from the rate observed so far and include
        evaluation overhead measured to date.
        """
        if step_budget <= 0:
            return f"{int(train_elapsed)}s/{duration_seconds}s"
        rate = steps_done / max(train_elapsed, 1e-6)
        remaining = max(0, step_budget - steps_done) / max(rate, 1e-6)
        eta = int(train_elapsed + overhead_seconds + remaining)
        return f"{steps_done:,}/{step_budget:,} steps, ETA {eta}s"

    _refresh_start_pool("initial")

    while True:
        elapsed = time.time() - t_start - overhead_seconds
        if step_budget > 0:
            if trainer.total_env_steps >= step_budget:
                break
        elif elapsed >= duration_seconds:
            break

        # Curriculum stage switch from UsefulActionsReward to BlunderAwareRewardCalculator
        if is_curriculum and not curriculum_switched and elapsed >= curriculum_switch_at:
            curriculum_switched = True
            trainer.set_reward_calculator(BlunderAwareRewardCalculator())
            trainer.set_slice_turn_boundaries(False if slice_turn_boundaries is None else slice_turn_boundaries)
            print(f"\n{'=' * 80}", flush=True)
            print(f"[CURRICULUM] STAGE 2 SWITCH: Replaced UsefulActionsReward with BlunderAwareRewardCalculator at elapsed={elapsed:.1f}s / {duration_seconds}s", flush=True)
            print(f"{'=' * 80}\n", flush=True)

        it += 1
        iteration_metrics = trainer.train_iteration()
        total_env_steps = trainer.total_env_steps
        episode_stats = summarize_completed_episodes(iteration_metrics.get("completed_episodes", []))

        # Log training step metrics
        step_metrics: Dict[str, Any] = {
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
            "defcon_risk_loss": iteration_metrics.get("defcon_risk_loss", 0.0),
            "belief_loss": iteration_metrics.get("belief_loss", 0.0),
            "oracle_loss": iteration_metrics.get("oracle_loss", 0.0),
            "distill_loss": iteration_metrics.get("distill_loss", 0.0),
            "explained_variance": float(iteration_metrics.get("explained_variance", 0.0)),
            "adv_std": float(iteration_metrics.get("adv_std", 0.0)),
            "adv_std_raw": float(iteration_metrics.get("adv_std_raw", 0.0)),
            "adv_frac_near_zero": float(iteration_metrics.get("adv_frac_near_zero", 0.0)),
        }
        step_metrics.update(episode_stats)
        if "entropy_fixed_probe" in iteration_metrics:
            step_metrics["entropy_fixed_probe"] = float(iteration_metrics["entropy_fixed_probe"])

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(step_metrics) + "\n")

        tb.log_metrics(
            step_metrics,
            step=it,
            skip_keys=(episode_dependent_in(episode_stats)
                       if episode_stats["episodes_completed"] == 0.0 else None),
        )
        if it % 10 == 0:
            tb.flush()

        if it % 10 == 0:
            print(
                f"[{_progress_label(elapsed, total_env_steps)}] It {it:4d} | Steps: {total_env_steps:,} ({step_metrics['steps_per_sec']:,} st/s) | "
                f"Loss: {step_metrics['loss']:.3f} | KL: {step_metrics['kl_div']:.4f} | Ent: {step_metrics['entropy']:.3f} | Clip: {step_metrics['clip_frac']*100:.1f}% | "
                f"EV: {step_metrics['explained_variance']:+.3f} | Turn: {step_metrics['mean_turn']:.1f}",
                flush=True,
            )

        # Snapshot Evaluation
        due = (total_env_steps >= next_eval_steps) if step_budget > 0 else (elapsed >= next_eval_time)
        if due:
            t_eval0 = time.time()
            snap_path = os.path.join(out_dir, f"snapshot_{int(elapsed)}s.pt")
            torch.save(model.state_dict(), snap_path)
            decisive = evaluate_and_log_snapshot(
                model=model,
                opponents=opponents,
                elapsed_seconds=int(elapsed),
                out_dir=out_dir,
                report_path=report_path,
                games_per_side=eval_games_per_side,
                device=dev,
                add_to_opponents_after=True,
                arch=arch,
                num_baselines=num_baselines,
                max_snapshot_opponents=max_snapshot_opponents,
            )
            pool_metrics = _refresh_start_pool(f"@{int(elapsed)}s")
            if pool_metrics:
                decisive = {**(decisive or {}), **pool_metrics}
            if decisive:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"iteration": it, "elapsed_seconds": int(elapsed), **decisive}) + "\n")
                if tb is not None:
                    tb.log_metrics(decisive, it)
            if step_budget > 0:
                next_eval_steps += eval_every_steps
            else:
                next_eval_time += snapshot_interval_seconds
            overhead_seconds += time.time() - t_eval0

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
        num_baselines=num_baselines,
        max_snapshot_opponents=max_snapshot_opponents,
    )

    tb.flush()
    tb.close()

    print(f"\n=== Training Complete. Final Checkpoint: {final_snap_path} ===", flush=True)

    # Post-training massive tournament
    if post_tournament:
        run_post_training_tournament(
            checkpoint_dir=out_dir,
            additional_models=post_tournament_models or ["random", "heuristic"],
            games_per_side=post_tournament_games,
            device=dev,
        )
