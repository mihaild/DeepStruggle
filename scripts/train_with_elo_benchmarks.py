"""Automated Twilight Struggle NashPG Training Harness.

Features:
- Separate run directory per training execution under checkpoints/
- 10-Minute Snapshotting with batched tournament evaluation
- 50 Games as US + 50 Games as USSR against all past versions, HeuristicBot, and RandomBot
- Separate US / USSR win rate reporting
- Automatic full-game self-play .tslog.json replay generation per snapshot
- Live Bradley-Terry Elo leaderboard tracking
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch

# Ensure repository root is on sys.path
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import ts_engine as ts
from ai.models.coldwar_net import ColdWarNet, create_coldwar_net
from ai.env.action_encoder import ActionEncoder
from ai.training.behavioral_cloning import BehavioralCloningTrainer, HeuristicPolicy
from ai.training.nash_pg import NashPGTrainer
from server.replay import ReplayLogger, REPLAYS_DIR
from ai.eval.self_play import generate_self_play_replay
from server.replay_types import EloBenchmarkReportDict


class EloCalculator:
    """Computes calibrated Elo ratings via Bradley-Terry maximum likelihood estimation."""

    @staticmethod
    def compute_elo_ratings(
        match_results: List[Tuple[str, str, float]],
        anchor_agent: str = "HeuristicBot",
        anchor_rating: float = 1500.0,
        iterations: int = 250,
        lr: float = 0.5,
    ) -> Dict[str, float]:
        agents = sorted(list(set([a for m in match_results for a in m[:2]])))
        if not agents:
            return {}

        gamma = {a: 0.0 for a in agents}
        n_matches = max(1, len(match_results))

        for _ in range(iterations):
            grad = {a: 0.0 for a in agents}
            for a, b, score_a in match_results:
                diff = np.clip(gamma[b] - gamma[a], -20.0, 20.0)
                p_a = 1.0 / (1.0 + np.exp(diff))
                grad[a] += (score_a - p_a)
                grad[b] += ((1.0 - score_a) - (1.0 - p_a))

            for a in agents:
                gamma[a] += lr * grad[a] / n_matches

        scale = 400.0 / np.log(10.0)
        ratings = {a: gamma[a] * scale for a in agents}

        if anchor_agent in ratings:
            shift = anchor_rating - ratings[anchor_agent]
            for a in ratings:
                ratings[a] += shift

        return ratings


class BatchedEvaluator:
    """High-speed vectorized match evaluator playing 50 US and 50 USSR games simultaneously."""

    def __init__(self, device: torch.device):
        self.device = device

    def evaluate_pair(
        self,
        name_a: str,
        model_a: Optional[ColdWarNet],
        name_b: str,
        model_b: Optional[ColdWarNet],
        games_per_side: int = 50,
        base_seed: int = 42,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """Plays games_per_side with A as US and games_per_side with B as US."""
        total_games = games_per_side * 2
        runner = ts.VectorizedBatchRunner(total_games, base_seed)

        completed = [False] * total_games
        winners = ["NONE"] * total_games
        final_vps = [0] * total_games
        step_counts = [0] * total_games

        steps = 0
        max_steps = 4000

        while not all(completed) and steps < max_steps:
            steps += 1
            obs_all = np.array(runner.get_observations(), copy=False)
            masks_all = np.array(runner.get_action_masks(), copy=False)
            players = runner.get_decision_players()
            terminals = runner.get_terminals()
            term_utils = runner.get_terminal_utilities()
            vps = runner.get_victory_points()

            for i in range(total_games):
                if not completed[i] and terminals[i]:
                    completed[i] = True
                    step_counts[i] = steps
                    final_vps[i] = int(vps[i])
                    if term_utils[i] > 0:
                        winners[i] = "US"
                    elif term_utils[i] < 0:
                        winners[i] = "USSR"
                    else:
                        winners[i] = "DRAW"

            if all(completed):
                break

            active_indices = [i for i in range(total_games) if not completed[i]]
            if not active_indices:
                break

            actions = [0] * total_games

            indices_for_a = []
            indices_for_b = []
            indices_heuristic = []
            indices_random = []

            for i in active_indices:
                p = players[i]
                is_agent_a = (i < games_per_side and (p == 1 or p == int(ts.Player.US))) or (i >= games_per_side and (p == -1 or p == int(ts.Player.USSR)))
                agent_name = name_a if is_agent_a else name_b
                model = model_a if is_agent_a else model_b

                if agent_name == "HeuristicBot":
                    indices_heuristic.append(i)
                elif agent_name == "RandomBot":
                    indices_random.append(i)
                else:
                    if is_agent_a:
                        indices_for_a.append(i)
                    else:
                        indices_for_b.append(i)

            if indices_for_a and model_a is not None:
                obs_sub = torch.from_numpy(obs_all[indices_for_a]).float().to(self.device)
                mask_sub = torch.from_numpy(masks_all[indices_for_a]).to(self.device)
                with torch.no_grad():
                    act_t, _, _, _, _ = model_a.sample_action(obs_sub, mask_sub, temperature=temperature, deterministic=False)
                act_list = act_t.cpu().numpy().tolist()
                for idx_sub, env_i in enumerate(indices_for_a):
                    actions[env_i] = int(act_list[idx_sub])

            if indices_for_b and model_b is not None:
                obs_sub = torch.from_numpy(obs_all[indices_for_b]).float().to(self.device)
                mask_sub = torch.from_numpy(masks_all[indices_for_b]).to(self.device)
                with torch.no_grad():
                    act_t, _, _, _, _ = model_b.sample_action(obs_sub, mask_sub, temperature=temperature, deterministic=False)
                act_list = act_t.cpu().numpy().tolist()
                for idx_sub, env_i in enumerate(indices_for_b):
                    actions[env_i] = int(act_list[idx_sub])

            for env_i in indices_heuristic:
                state_ref = runner.get_state(env_i)
                actions[env_i] = HeuristicPolicy.select_action(state_ref)

            for env_i in indices_random:
                m = masks_all[env_i]
                legal_ids = [int(x) for x in np.where(m > 0)[0]]
                actions[env_i] = int(np.random.choice(legal_ids)) if legal_ids else ActionEncoder.CONFIRM_DONE_INDEX

            runner.step_flat_all(actions)

        us_wins_a = sum(1 for i in range(games_per_side) if winners[i] == "US")
        us_losses_a = sum(1 for i in range(games_per_side) if winners[i] == "USSR")
        us_draws_a = sum(1 for i in range(games_per_side) if winners[i] == "DRAW")

        ussr_wins_a = sum(1 for i in range(games_per_side, total_games) if winners[i] == "USSR")
        ussr_losses_a = sum(1 for i in range(games_per_side, total_games) if winners[i] == "US")
        ussr_draws_a = sum(1 for i in range(games_per_side, total_games) if winners[i] == "DRAW")

        total_wins_a = us_wins_a + ussr_wins_a
        total_losses_a = us_losses_a + ussr_losses_a
        total_draws_a = us_draws_a + ussr_draws_a
        total_points_a = total_wins_a + 0.5 * total_draws_a

        return {
            "agent_a": name_a,
            "agent_b": name_b,
            "games_per_side": games_per_side,
            "total_games": total_games,
            "us_wins_a": us_wins_a,
            "us_losses_a": us_losses_a,
            "us_draws_a": us_draws_a,
            "us_win_rate_a": us_wins_a / max(1, games_per_side),
            "ussr_wins_a": ussr_wins_a,
            "ussr_losses_a": ussr_losses_a,
            "ussr_draws_a": ussr_draws_a,
            "ussr_win_rate_a": ussr_wins_a / max(1, games_per_side),
            "total_wins_a": total_wins_a,
            "total_losses_a": total_losses_a,
            "total_draws_a": total_draws_a,
            "total_points_a": total_points_a,
            "total_win_rate_a": total_points_a / max(1, total_games),
        }


def dump_sample_self_play_game(
    model: ColdWarNet,
    model_name: str,
    output_paths: List[str],
    device: torch.device,
    seed: int = 2026,
    temperature: float = 0.3,
) -> str:
    """Simulates a sample self-play game to full completion and saves .tslog.json replay."""
    _, saved_path = generate_self_play_replay(
        model=model,
        model_name=model_name,
        output_path=output_paths,
        device=device,
        seed=seed,
        temperature=temperature,
        verbose=False,
    )
    return saved_path


def run_full_training_campaign(
    total_hours: float = 1.0,
    snapshot_interval_minutes: float = 10.0,
    games_per_side: int = 50,
    num_envs: int = 256,
    buffer_size: int = 128,
    run_dir: Optional[str] = None,
    device_str: str = "cuda",
):
    start_time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not run_dir:
        run_dir = os.path.join("checkpoints", f"run_{start_time_stamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(REPLAYS_DIR, exist_ok=True)

    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    print("\n" + "=" * 80)
    print(" Twilight Struggle NashPG Training Campaign (Canonical Representation)")
    print(f" Run Directory: {run_dir}")
    print(f" Device: {device} | Total Duration: {total_hours} hr | Snapshots every {snapshot_interval_minutes} min")
    print(f" Parallel Envs: {num_envs} (OpenMP Multi-Threaded) | Buffer Size: {buffer_size}")
    print(f" Evaluation per Pair: {games_per_side} as US + {games_per_side} as USSR (100 total)")
    print("=" * 80 + "\n")

    evaluator = BatchedEvaluator(device)

    # 1. Phase 0: Behavioral Cloning Pre-training
    bc_path = os.path.join(run_dir, "coldwar_net_bc.pt")
    snap0_path = os.path.join(run_dir, "snapshot_0m.pt")

    print(">>> [Phase 0] Training Behavioral Cloning Foundation Model...")
    base_model = create_coldwar_net(device)
    bc_trainer = BehavioralCloningTrainer(base_model, lr=1e-3, batch_size=256, device=device)
    dataset = bc_trainer.generate_demonstration_dataset(num_games=500)
    bc_trainer.train(dataset, epochs=10, batch_size=256, save_path=bc_path)

    torch.save(base_model.state_dict(), snap0_path)
    print(f"Saved initial baseline checkpoint to {snap0_path}\n")

    # Snapshot registry
    snapshots: Dict[str, ColdWarNet] = {}
    snapshots["Snapshot_0m (BC)"] = create_coldwar_net(device)
    snapshots["Snapshot_0m (BC)"].load_state_dict(torch.load(snap0_path, map_location=device))
    snapshots["Snapshot_0m (BC)"].eval()

    all_match_history: List[Tuple[str, str, float]] = []
    eval_records: List[Dict[str, Any]] = []

    # Baseline match between HeuristicBot and RandomBot
    res_hr = evaluator.evaluate_pair("HeuristicBot", None, "RandomBot", None, games_per_side=games_per_side, base_seed=200)
    eval_records.append(res_hr)
    for _ in range(res_hr["total_wins_a"]):
        all_match_history.append(("HeuristicBot", "RandomBot", 1.0))
    for _ in range(res_hr["total_losses_a"]):
        all_match_history.append(("HeuristicBot", "RandomBot", 0.0))
    for _ in range(res_hr["total_draws_a"]):
        all_match_history.append(("HeuristicBot", "RandomBot", 0.5))

    # Initial Baseline Tournament for Snapshot_0m vs Heuristic and Random
    print("--- Evaluating Snapshot_0m Baseline ---")
    for opp_name in ["HeuristicBot", "RandomBot"]:
        opp_m = None
        res = evaluator.evaluate_pair(
            "Snapshot_0m (BC)", snapshots["Snapshot_0m (BC)"],
            opp_name, opp_m,
            games_per_side=games_per_side, base_seed=100
        )
        eval_records.append(res)
        for _ in range(res["total_wins_a"]):
            all_match_history.append(("Snapshot_0m (BC)", opp_name, 1.0))
        for _ in range(res["total_losses_a"]):
            all_match_history.append(("Snapshot_0m (BC)", opp_name, 0.0))
        for _ in range(res["total_draws_a"]):
            all_match_history.append(("Snapshot_0m (BC)", opp_name, 0.5))

        print(f"  vs {opp_name:12s} | US Win: {res['us_win_rate_a']*100:5.1f}% | USSR Win: {res['ussr_win_rate_a']*100:5.1f}% | Total: {res['total_win_rate_a']*100:5.1f}% ({res['total_points_a']:.1f}/{res['total_games']})")

    initial_elos = EloCalculator.compute_elo_ratings(all_match_history, anchor_agent="HeuristicBot", anchor_rating=1500.0)
    print("\nInitial Elo Ratings:")
    for ag, r in sorted(initial_elos.items(), key=lambda x: -x[1]):
        print(f"  {ag:20s}: {r:6.1f}")
    print()

    # 2. Phase 1: NashPG RL Self-Play Training Loop
    active_net = create_coldwar_net(device)
    active_net.load_state_dict(torch.load(snap0_path, map_location=device))

    trainer = NashPGTrainer(
        active_net=active_net,
        num_envs=num_envs,
        buffer_size=buffer_size,
        lr=3e-4,
        gamma=0.999,
        gae_lambda=0.95,
        eta=0.1,
        device=device,
    )

    total_seconds = total_hours * 3600.0
    snapshot_interval_seconds = snapshot_interval_minutes * 60.0
    next_snapshot_time = snapshot_interval_seconds

    t_start = time.time()
    iter_idx = 0
    snapshot_count = 0

    print(f">>> [Phase 1] Launching NashPG Self-Play RL for {total_hours} Hours...")

    while True:
        elapsed = time.time() - t_start
        if elapsed >= total_seconds:
            break

        iter_idx += 1
        metrics = trainer.train_iteration(ref_update_freq=250_000, ppo_epochs=4, batch_size=512)

        # Print iteration status
        if iter_idx % 5 == 0 or iter_idx == 1:
            total_steps = iter_idx * num_envs * buffer_size
            speed = total_steps / max(elapsed, 0.001)
            print(
                f"[NashPG] Iter {iter_idx:4d} | Elapsed: {elapsed/60.0:4.1f}m | Total Steps: {total_steps:,} | "
                f"Speed: {speed:,.0f} step/s | Loss: {metrics['loss']:.4f} (Pol: {metrics['policy_loss']:.4f}, Val: {metrics.get('val_loss', metrics.get('value_loss', 0.0)):.4f}, KL: {metrics['kl_div']:.4f})"
            )

        # Check if snapshot milestone is reached
        if elapsed >= next_snapshot_time or (elapsed + 30 >= total_seconds and snapshot_count < int(total_seconds / snapshot_interval_seconds)):
            snapshot_count += 1
            snap_min = int(round(elapsed / 60.0))
            snap_name = f"Snapshot_{snap_min}m"
            snap_file = os.path.join(run_dir, f"snapshot_{snap_min}m.pt")
            torch.save(active_net.state_dict(), snap_file)

            # Register snapshot model
            snap_model = create_coldwar_net(device)
            snap_model.load_state_dict(torch.load(snap_file, map_location=device))
            snap_model.eval()
            snapshots[snap_name] = snap_model

            print(f"\n{'*'*80}")
            print(f" [SNAPSHOT MILESTONE] {snap_name} reached at {elapsed/60.0:.1f} minutes! Saved to {snap_file}")
            print(f" Running Batched Tournament ({games_per_side} as US + {games_per_side} as USSR per opponent)...")
            print(f"{'*'*80}")

            # Dump sample self-play replay for this snapshot
            replay_run_path = os.path.join(run_dir, f"{snap_name.lower()}_self_play.tslog.json")
            replay_web_path = os.path.join(REPLAYS_DIR, f"{os.path.basename(run_dir)}_{snap_name.lower()}_self_play.tslog.json")
            dump_sample_self_play_game(snap_model, snap_name, [replay_run_path, replay_web_path], device=device, seed=2026 + snapshot_count)
            print(f" Sample Self-Play Replay recorded to: {replay_web_path}")

            # Evaluate against all previous snapshots and baselines
            opponents_to_eval = [s for s in snapshots.keys() if s != snap_name] + ["HeuristicBot", "RandomBot"]

            for opp_name in opponents_to_eval:
                opp_m = snapshots.get(opp_name, None)
                res = evaluator.evaluate_pair(
                    snap_name, snap_model,
                    opp_name, opp_m,
                    games_per_side=games_per_side,
                    base_seed=1000 * snapshot_count + len(eval_records)
                )
                eval_records.append(res)
                for _ in range(res["total_wins_a"]):
                    all_match_history.append((snap_name, opp_name, 1.0))
                for _ in range(res["total_losses_a"]):
                    all_match_history.append((snap_name, opp_name, 0.0))
                for _ in range(res["total_draws_a"]):
                    all_match_history.append((snap_name, opp_name, 0.5))

                print(
                    f"  vs {opp_name:18s} | US Win: {res['us_win_rate_a']*100:5.1f}% ({res['us_wins_a']:2d}/{games_per_side}) | "
                    f"USSR Win: {res['ussr_win_rate_a']*100:5.1f}% ({res['ussr_wins_a']:2d}/{games_per_side}) | "
                    f"Total: {res['total_win_rate_a']*100:5.1f}% ({res['total_points_a']:.1f}/{res['total_games']})"
                )

            # Compute and display updated Elo ratings
            current_elos = EloCalculator.compute_elo_ratings(all_match_history, anchor_agent="HeuristicBot", anchor_rating=1500.0)
            print(f"\n ---------------- CURRENT ELO LEADERBOARD ({elapsed/60.0:.1f}m) ----------------")
            print(f" | Rank | Agent / Model Snapshot      | Elo Rating |")
            print(f" +------+-----------------------------+------------+")
            for rank, (ag, r) in enumerate(sorted(current_elos.items(), key=lambda x: -x[1]), 1):
                print(f" | {rank:4d} | {ag:27s} | {r:10.1f} |")
            print(f" ---------------------------------------------------------\n")

            # Write updated markdown report
            write_run_report(run_dir, current_elos, eval_records, iter_idx, elapsed)

            next_snapshot_time += snapshot_interval_seconds

    # Save final model
    latest_path = os.path.join(run_dir, "coldwar_net_latest.pt")
    torch.save(active_net.state_dict(), latest_path)
    print(f"\nTraining Complete! Final model saved to {latest_path}")


def write_run_report(run_dir: str, elo_ratings: Dict[str, float], eval_records: List[Dict[str, Any]], iters: int, elapsed_sec: float):
    report_file = os.path.join(run_dir, "training_report.md")
    json_file = os.path.join(run_dir, "elo_history.json")

    with open(json_file, "w") as f:
        elo_report: EloBenchmarkReportDict = {"elo_ratings": elo_ratings, "iterations": [iters], "elapsed_seconds": elapsed_sec}
        json.dump(elo_report, f, indent=2)

    with open(report_file, "w") as f:
        f.write(f"# Twilight Struggle RL Training Report\n\n")
        f.write(f"- **Run Directory**: `{run_dir}`\n")
        f.write(f"- **Elapsed Training Time**: {elapsed_sec/60.0:.1f} minutes ({iters} NashPG iterations)\n")
        f.write(f"- **Anchor**: `HeuristicBot` @ 1500.0 Elo\n\n")
        f.write(f"## Current Elo Leaderboard\n\n")
        f.write(f"| Rank | Agent / Model Snapshot | Elo Rating |\n")
        f.write(f"|:----:|:-----------------------|:----------:|\n")
        for rank, (ag, r) in enumerate(sorted(elo_ratings.items(), key=lambda x: -x[1]), 1):
            f.write(f"| **{rank}** | **{ag}** | **{r:.1f}** |\n")

        f.write(f"\n## Head-to-Head Matchup Breakdown (50 US / 50 USSR)\n\n")
        f.write(f"| Snapshot (Agent A) | Opponent (Agent B) | US Win Rate (as US) | USSR Win Rate (as USSR) | Total Win Rate | Total Points |\n")
        f.write(f"|:---|:---|:---:|:---:|:---:|:---:|\n")
        for r in eval_records:
            f.write(
                f"| **{r['agent_a']}** | {r['agent_b']} | "
                f"{r['us_win_rate_a']*100:.1f}% ({r['us_wins_a']}/{r['games_per_side']}) | "
                f"{r['ussr_win_rate_a']*100:.1f}% ({r['ussr_wins_a']}/{r['games_per_side']}) | "
                f"**{r['total_win_rate_a']*100:.1f}%** | {r['total_points_a']:.1f} / {r['total_games']} |\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Twilight Struggle NashPG Training Campaign")
    parser.add_argument("--hours", type=float, default=1.0, help="Total training hours")
    parser.add_argument("--interval", type=float, default=10.0, help="Snapshot interval in minutes")
    parser.add_argument("--games-per-side", type=int, default=50, help="Evaluation games as US and USSR (default: 50)")
    parser.add_argument("--num-envs", type=int, default=256, help="Number of parallel environments")
    parser.add_argument("--buffer-size", type=int, default=128, help="Rollout buffer size per environment")
    parser.add_argument("--run-dir", type=str, default=None, help="Custom run directory under checkpoints/")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device ('cuda' or 'cpu')")

    args = parser.parse_args()
    run_full_training_campaign(
        total_hours=args.hours,
        snapshot_interval_minutes=args.interval,
        games_per_side=args.games_per_side,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        run_dir=args.run_dir,
        device_str=args.device,
    )
