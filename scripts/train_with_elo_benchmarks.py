"""1-Hour Twilight Struggle NashPG Training Runner with 10-Minute Snapshots and Elo Evaluation."""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple, Any
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


class EloCalculator:
    """Computes Elo ratings via Bradley-Terry maximum likelihood or iterative updates."""

    @staticmethod
    def compute_elo_ratings(
        match_results: List[Tuple[str, str, float]],
        anchor_agent: str = "HeuristicBot",
        anchor_rating: float = 1500.0,
        k_factor: float = 24.0,
        iterations: int = 100,
    ) -> Dict[str, float]:
        """Calculates Elo ratings given a list of (agent_a, agent_b, score_a).

        score_a: 1.0 (win), 0.5 (draw), 0.0 (loss).
        """
        agents = sorted(list(set([a for m in match_results for a in m[:2]])))
        ratings = {a: 1500.0 for a in agents}
        if "RandomBot" in ratings:
            ratings["RandomBot"] = 1000.0

        for _ in range(iterations):
            for agent_a, agent_b, score_a in match_results:
                r_a = ratings[agent_a]
                r_b = ratings[agent_b]
                e_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
                e_b = 1.0 - e_a
                score_b = 1.0 - score_a

                ratings[agent_a] += k_factor * (score_a - e_a)
                ratings[agent_b] += k_factor * (score_b - e_b)

        # Re-center so anchor has anchor_rating
        if anchor_agent in ratings:
            shift = anchor_rating - ratings[anchor_agent]
            for a in ratings:
                ratings[a] += shift

        return {k: round(v, 1) for k, v in sorted(ratings.items(), key=lambda x: x[1], reverse=True)}


class CrossEvaluator:
    """Plays head-to-head matches between model snapshots, HeuristicBot, and RandomBot."""

    def __init__(self, device: torch.device):
        self.device = device
        self.models_cache: Dict[str, ColdWarNet] = {}

    def get_model(self, name: str, path: str) -> ColdWarNet:
        if name not in self.models_cache:
            m = create_coldwar_net(self.device)
            m.load_state_dict(torch.load(path, map_location=self.device))
            m.eval()
            self.models_cache[name] = m
        return self.models_cache[name]

    def play_match(self, agent_a_name: str, agent_b_name: str, games_per_pair: int = 16) -> List[Tuple[str, str, float]]:
        """Plays games between agent_a and agent_b, half as US and half as USSR."""
        results = []
        half = games_per_pair // 2

        matchups = [(agent_a_name, agent_b_name)] * half + [(agent_b_name, agent_a_name)] * (games_per_pair - half)

        for us_agent, ussr_agent in matchups:
            state = ts.GameState()
            seed = int(np.random.randint(1, 1_000_000_000))
            ts.Engine.init_game(state, seed)

            step = 0
            while not ts.Engine.is_terminal(state) and step < 450:
                p = state.ctx().decision_player if state.ctx().decision_player != ts.Player.NONE else state.phasing_player
                active_agent = us_agent if p == ts.Player.US else ussr_agent

                if active_agent.startswith("Snapshot") or active_agent.startswith("ColdWarNet"):
                    model = self.models_cache[active_agent]
                    obs = ts.extract_observation(state, p)
                    mask = ActionEncoder.get_legal_mask(state)
                    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                    mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        act_t, _, _, _, _ = model.sample_action(obs_t, mask_t, temperature=0.3, deterministic=False)
                    action_idx = int(act_t.item())
                elif active_agent == "HeuristicBot":
                    action_idx = HeuristicPolicy.select_action(state)
                else: # RandomBot
                    mask = ActionEncoder.get_legal_mask(state)
                    legal_indices = [int(i) for i in np.where(mask > 0)[0]]
                    action_idx = int(np.random.choice(legal_indices)) if legal_indices else ActionEncoder.CONFIRM_DONE_INDEX

                ts.Engine.step_flat(state, action_idx)
                step += 1

            term_util = ts.Engine.get_terminal_utility(state)
            if term_util > 0: # US won
                score_for_us = 1.0
            elif term_util < 0: # USSR won
                score_for_us = 0.0
            else: # Draw
                score_for_us = 0.5

            if us_agent == agent_a_name:
                results.append((agent_a_name, agent_b_name, score_for_us))
            else:
                score_for_a = 1.0 - score_for_us
                results.append((agent_a_name, agent_b_name, score_for_a))

        return results


def run_1hour_training_with_benchmarks(
    total_minutes: int = 60,
    snapshot_interval_minutes: int = 10,
    num_envs: int = 256,
    buffer_size: int = 128,
    eta: float = 0.1,
    lr: float = 3e-4,
    device_str: str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"\n{'='*70}")
    print(f" Twilight Struggle: 1-Hour NashPG Training with 10-Minute Elo Benchmarks")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f" Envs: {num_envs} (OpenMP 24-core) | Buffer: {buffer_size} | NashPG η: {eta} | LR: {lr}")
    print(f" Total Duration: {total_minutes} mins | Snapshot Interval: {snapshot_interval_minutes} mins")
    print(f"{'='*70}\n")

    os.makedirs("checkpoints", exist_ok=True)
    bc_checkpoint = "checkpoints/coldwar_net_bc.pt"

    # Step 1: Ensure Phase 0 BC checkpoint exists
    model = create_coldwar_net(device)
    if not os.path.exists(bc_checkpoint):
        print(f"[Phase 0] Running Initial Behavioral Cloning Pre-Training...")
        bc_trainer = BehavioralCloningTrainer(model, device=device, lr=1e-3)
        dataset = bc_trainer.generate_demonstration_dataset(num_games=500)
        bc_trainer.train(dataset, epochs=10, batch_size=256, save_path=bc_checkpoint)
    else:
        print(f"[Phase 0] Loading existing BC baseline weights from {bc_checkpoint}")
        model.load_state_dict(torch.load(bc_checkpoint, map_location=device))

    # Initialize NashPG Trainer with corrected zero-sum GAE
    trainer = NashPGTrainer(
        active_net=model,
        num_envs=num_envs,
        buffer_size=buffer_size,
        lr=lr,
        eta=eta,
        ref_update_freq=250_000,
        device=device,
    )

    evaluator = CrossEvaluator(device)

    # Save initial Snapshot_0m (BC baseline)
    init_snap_name = "Snapshot_0m (BC)"
    init_snap_path = "checkpoints/snapshot_0m.pt"
    torch.save(model.state_dict(), init_snap_path)
    evaluator.get_model(init_snap_name, init_snap_path)

    snapshots: List[Tuple[str, str]] = [(init_snap_name, init_snap_path)]
    all_match_history: List[Tuple[str, str, float]] = []

    # Run initial baseline games vs HeuristicBot and RandomBot
    print(f"\n[Baseline Benchmark] Running initial evaluation vs Heuristic & Random Bot...")
    h_matches = evaluator.play_match(init_snap_name, "HeuristicBot", games_per_pair=20)
    all_match_history.extend(h_matches)
    h_wins = sum(1 for m in h_matches if m[0] == init_snap_name and m[2] == 1.0)
    print(f"  Snapshot_0m (BC) vs HeuristicBot: {h_wins}/{len(h_matches)} wins ({h_wins/len(h_matches)*100:.1f}%)")

    r_matches = evaluator.play_match(init_snap_name, "RandomBot", games_per_pair=12)
    all_match_history.extend(r_matches)
    r_wins = sum(1 for m in r_matches if m[0] == init_snap_name and m[2] == 1.0)
    print(f"  Snapshot_0m (BC) vs RandomBot:    {r_wins}/{len(r_matches)} wins ({r_wins/len(r_matches)*100:.1f}%)")

    all_match_history.extend(evaluator.play_match("HeuristicBot", "RandomBot", games_per_pair=12))

    initial_elo = EloCalculator.compute_elo_ratings(all_match_history, anchor_agent="HeuristicBot", anchor_rating=1500.0)
    print(f"Initial Elo Leaderboard: {initial_elo}\n")

    start_time = time.time()
    total_seconds = total_minutes * 60
    interval_seconds = snapshot_interval_minutes * 60
    next_snapshot_time = start_time + interval_seconds

    iteration = 0
    snapshot_idx = 1

    while True:
        iteration += 1
        rollout_stats = trainer.collect_rollouts()
        train_stats = trainer.train_step()

        elapsed = time.time() - start_time
        remaining = max(0, total_seconds - elapsed)

        if iteration % 5 == 0:
            print(
                f"[{elapsed/60:4.1f}m / {total_minutes}m] Iter {iteration:4d} | Steps: {trainer.total_env_steps:9,d} "
                f"({rollout_stats['fps']:6.0f} step/s) | Loss: {train_stats['loss']:.4f} (Pol: {train_stats['policy_loss']:.4f}, "
                f"Val: {train_stats['val_loss']:.4f}, KL: {train_stats['kl_div']:.4f}) | Left: {remaining/60:.1f}m"
            )

        # Check if snapshot interval is reached or training is complete
        if time.time() >= next_snapshot_time or elapsed >= total_seconds:
            curr_mins = int(round(elapsed / 60))
            snap_name = f"Snapshot_{curr_mins}m"
            snap_path = f"checkpoints/snapshot_{curr_mins}m.pt"

            print(f"\n{'*'*70}")
            print(f" >>> [SNAPSHOT {snapshot_idx}] Saving {snap_name} at {trainer.total_env_steps:,} total steps ({elapsed/60:.1f} mins) <<<")
            torch.save(trainer.active_net.state_dict(), snap_path)
            torch.save(trainer.active_net.state_dict(), "checkpoints/coldwar_net_latest.pt")
            evaluator.get_model(snap_name, snap_path)
            snapshots.append((snap_name, snap_path))

            # Run Cross-Tournament: New snapshot vs All past snapshots, HeuristicBot, and RandomBot
            print(f" >>> Running Cross-Evaluation Tournament for {snap_name}...")
            # 1. Play vs HeuristicBot
            h_matches = evaluator.play_match(snap_name, "HeuristicBot", games_per_pair=20)
            all_match_history.extend(h_matches)
            h_wins = sum(1 for m in h_matches if m[0] == snap_name and m[2] == 1.0)
            print(f"     vs HeuristicBot: {h_wins}/{len(h_matches)} wins ({h_wins/len(h_matches)*100:.1f}%)")

            # 2. Play vs RandomBot
            r_matches = evaluator.play_match(snap_name, "RandomBot", games_per_pair=12)
            all_match_history.extend(r_matches)
            r_wins = sum(1 for m in r_matches if m[0] == snap_name and m[2] == 1.0)
            print(f"     vs RandomBot:    {r_wins}/{len(r_matches)} wins ({r_wins/len(r_matches)*100:.1f}%)")

            # 3. Play vs Prior Snapshots
            for prev_name, _ in snapshots[:-1]:
                p_matches = evaluator.play_match(snap_name, prev_name, games_per_pair=16)
                all_match_history.extend(p_matches)
                p_wins = sum(1 for m in p_matches if m[0] == snap_name and m[2] == 1.0)
                print(f"     vs {prev_name:18s}: {p_wins}/{len(p_matches)} wins ({p_wins/len(p_matches)*100:.1f}%)")

            # Compute current Elo ratings
            current_elo = EloCalculator.compute_elo_ratings(
                all_match_history, anchor_agent="HeuristicBot", anchor_rating=1500.0
            )

            print(f"\n ---------------- ELO LEADERBOARD ({curr_mins} mins) ----------------")
            for rank, (agent, elo) in enumerate(current_elo.items(), start=1):
                star = " <-- CURRENT" if agent == snap_name else ""
                print(f"  {rank:2d}. {agent:22s} : {elo:6.1f} Elo{star}")
            print(f" --------------------------------------------------------\n")

            # Save Elo history to JSON
            elo_log = {
                "elapsed_minutes": curr_mins,
                "total_env_steps": trainer.total_env_steps,
                "leaderboard": current_elo,
                "snapshots": [s[0] for s in snapshots],
            }
            with open("checkpoints/elo_history.json", "w") as f:
                json.dump(elo_log, f, indent=2)

            # Write formatted report artifact
            write_markdown_report(snapshots, current_elo, trainer.total_env_steps, elapsed)

            snapshot_idx += 1
            next_snapshot_time = time.time() + interval_seconds

        if elapsed >= total_seconds:
            print(f"\n=======================================================")
            print(f" 1-Hour NashPG Training Complete! Total Steps: {trainer.total_env_steps:,}")
            print(f" Checkpoints and Elo Leaderboard saved in checkpoints/")
            print(f"=======================================================\n")
            break


def write_markdown_report(snapshots: List[Tuple[str, str]], elo_ratings: Dict[str, float], total_steps: int, elapsed_sec: float):
    report = f"""# Twilight Struggle Neural AI Training Report

- **Elapsed Time**: {elapsed_sec / 60:.1f} minutes
- **Total Environment Steps**: {total_steps:,}
- **Algorithm**: NashPG (Nash Policy Gradient with Iteratively Refined Regularization)

## Current Elo Leaderboard (Anchored vs HeuristicBot @ 1500 Elo)

| Rank | Agent / Checkpoint | Elo Rating |
|:----:|:-------------------|:----------:|
"""
    for rank, (agent, elo) in enumerate(elo_ratings.items(), start=1):
        report += f"| {rank} | **{agent}** | **{elo:.1f}** |\n"

    report += f"\n*Report automatically updated at {time.strftime('%Y-%m-%d %H:%M:%S')}*\n"

    with open("checkpoints/training_report.md", "w") as f:
        f.write(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1-Hour NashPG Training with 10-Minute Elo Benchmarks")
    parser.add_argument("--minutes", type=int, default=60, help="Total training time in minutes (default 60)")
    parser.add_argument("--interval", type=int, default=10, help="Snapshot interval in minutes (default 10)")
    parser.add_argument("--num-envs", type=int, default=256, help="Number of parallel environments (default 256)")
    parser.add_argument("--buffer-size", type=int, default=128, help="Buffer size per environment (default 128)")
    parser.add_argument("--eta", type=float, default=0.1, help="NashPG KL regularization coefficient")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device ('cuda' or 'cpu')")

    args = parser.parse_args()
    run_1hour_training_with_benchmarks(
        total_minutes=args.minutes,
        snapshot_interval_minutes=args.interval,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        eta=args.eta,
        lr=args.lr,
        device_str=args.device,
    )
