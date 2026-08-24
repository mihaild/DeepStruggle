"""Large-Scale Vectorized Round-Robin Tournament (200 games per pair)."""

import argparse
import itertools
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
from ai.training.behavioral_cloning import HeuristicPolicy
from scripts.train_with_elo_benchmarks import EloCalculator


class BatchedTournamentRunner:
    """Runs large tournaments with batched GPU neural inference & OpenMP C++ simulation."""

    def __init__(self, device: torch.device):
        self.device = device
        self.models_cache: Dict[str, ColdWarNet] = {}

    def load_model(self, name: str, path: str) -> None:
        if os.path.exists(path):
            m = create_coldwar_net(self.device)
            m.load_state_dict(torch.load(path, map_location=self.device))
            m.eval()
            self.models_cache[name] = m

    def play_batched_match(
        self,
        us_agent: str,
        ussr_agent: str,
        num_games: int = 100,
        base_seed: int = 42,
        temperature: float = 0.3,
    ) -> List[float]:
        """Plays `num_games` simultaneously with us_agent as US and ussr_agent as USSR.

        Returns list of scores for US (1.0 for US win, 0.5 for draw, 0.0 for USSR win).
        """
        runner = ts.VectorizedBatchRunner(num_games, base_seed)
        us_scores = [0.5] * num_games
        max_steps = 450
        step_count = 0

        while step_count < max_steps:
            step_count += 1
            terminals = runner.get_terminals()
            if all(terminals):
                break

            players = runner.get_decision_players()
            obs_all = runner.get_observations() # [num_games, 4293]
            masks_all = runner.get_action_masks() # [num_games, 212]

            actions = [0] * num_games

            # Group active games by decision player
            us_indices = []
            ussr_indices = []

            for i in range(num_games):
                if terminals[i]:
                    continue
                p = players[i]
                if p == 1: # US
                    us_indices.append(i)
                else: # USSR
                    ussr_indices.append(i)

            if not us_indices and not ussr_indices:
                break

            # Handle US decisions
            if us_indices:
                if us_agent in self.models_cache:
                    m = self.models_cache[us_agent]
                    sub_obs = torch.from_numpy(obs_all[us_indices]).float().to(self.device)
                    sub_masks = torch.from_numpy(masks_all[us_indices]).to(self.device)
                    with torch.no_grad():
                        sub_acts, _, _, _, _ = m.sample_action(sub_obs, sub_masks, temperature=temperature, deterministic=False)
                    for k, idx in enumerate(us_indices):
                        actions[idx] = int(sub_acts[k].item())
                elif us_agent == "HeuristicBot":
                    for idx in us_indices:
                        actions[idx] = HeuristicPolicy.select_action(runner.get_state(idx))
                else: # RandomBot
                    for idx in us_indices:
                        mask = masks_all[idx]
                        legal = [int(x) for x in np.where(mask > 0)[0]]
                        actions[idx] = int(np.random.choice(legal)) if legal else ActionEncoder.CONFIRM_DONE_INDEX

            # Handle USSR decisions
            if ussr_indices:
                if ussr_agent in self.models_cache:
                    m = self.models_cache[ussr_agent]
                    sub_obs = torch.from_numpy(obs_all[ussr_indices]).float().to(self.device)
                    sub_masks = torch.from_numpy(masks_all[ussr_indices]).to(self.device)
                    with torch.no_grad():
                        sub_acts, _, _, _, _ = m.sample_action(sub_obs, sub_masks, temperature=temperature, deterministic=False)
                    for k, idx in enumerate(ussr_indices):
                        actions[idx] = int(sub_acts[k].item())
                elif ussr_agent == "HeuristicBot":
                    for idx in ussr_indices:
                        actions[idx] = HeuristicPolicy.select_action(runner.get_state(idx))
                else: # RandomBot
                    for idx in ussr_indices:
                        mask = masks_all[idx]
                        legal = [int(x) for x in np.where(mask > 0)[0]]
                        actions[idx] = int(np.random.choice(legal)) if legal else ActionEncoder.CONFIRM_DONE_INDEX

            runner.step_flat_all(actions)

        # Collect terminal utility for each game
        term_utils = runner.get_terminal_utilities()
        for i in range(num_games):
            u = term_utils[i]
            if u > 0: # US won
                us_scores[i] = 1.0
            elif u < 0: # USSR won
                us_scores[i] = 0.0
            else: # Draw or timeout
                us_scores[i] = 0.5

        return us_scores


def run_full_200game_tournament(
    games_per_side: int = 100,
    temperature: float = 0.3,
    device_str: str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"\n{'='*75}")
    print(f" Twilight Struggle Grand Tournament: {games_per_side*2} Games Per Pair ({games_per_side} US / {games_per_side} USSR)")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"{'='*75}\n")

    runner = BatchedTournamentRunner(device)

    # Register all models and baselines
    model_configs = [
        ("Snapshot_60m", "checkpoints/snapshot_60m.pt"),
        ("Snapshot_51m", "checkpoints/snapshot_51m.pt"),
        ("Snapshot_41m", "checkpoints/snapshot_41m.pt"),
        ("Snapshot_31m", "checkpoints/snapshot_31m.pt"),
        ("Snapshot_20m", "checkpoints/snapshot_20m.pt"),
        ("Snapshot_10m", "checkpoints/snapshot_10m.pt"),
        ("Snapshot_0m (BC)", "checkpoints/snapshot_0m.pt"),
    ]

    available_agents = []
    for name, path in model_configs:
        if os.path.exists(path):
            runner.load_model(name, path)
            available_agents.append(name)
            print(f" Loaded {name:18s} from {path}")

    available_agents.extend(["HeuristicBot", "RandomBot"])
    print(f" Baselines loaded: HeuristicBot, RandomBot\n")

    pairs = list(itertools.combinations(available_agents, 2))
    total_matches = len(pairs)
    total_games = total_matches * (games_per_side * 2)
    print(f"Total Matchups: {total_matches} pairs | Total Games to Play: {total_games:,} games\n")

    all_match_results: List[Tuple[str, str, float]] = []
    pair_summary: Dict[str, Dict[str, Any]] = {}

    start_tournament_time = time.time()

    for idx, (agent_a, agent_b) in enumerate(pairs, start=1):
        pair_key = f"{agent_a} vs {agent_b}"
        print(f"[{idx:2d}/{total_matches}] Playing {pair_key:35s} ({games_per_side*2} games)...", end="", flush=True)
        t0 = time.time()

        # Leg 1: Agent A is US, Agent B is USSR
        scores_leg1 = runner.play_batched_match(
            us_agent=agent_a,
            ussr_agent=agent_b,
            num_games=games_per_side,
            base_seed=idx * 1000 + 1,
            temperature=temperature,
        )
        for s in scores_leg1:
            all_match_results.append((agent_a, agent_b, s))

        # Leg 2: Agent B is US, Agent A is USSR
        scores_leg2 = runner.play_batched_match(
            us_agent=agent_b,
            ussr_agent=agent_a,
            num_games=games_per_side,
            base_seed=idx * 1000 + 500,
            temperature=temperature,
        )
        for s in scores_leg2:
            all_match_results.append((agent_a, agent_b, 1.0 - s))

        # Compute pair stats
        a_wins_as_us = sum(1 for s in scores_leg1 if s == 1.0)
        a_draws_as_us = sum(1 for s in scores_leg1 if s == 0.5)
        a_losses_as_us = sum(1 for s in scores_leg1 if s == 0.0)

        b_wins_as_us = sum(1 for s in scores_leg2 if s == 1.0)
        b_draws_as_us = sum(1 for s in scores_leg2 if s == 0.5)
        b_losses_as_us = sum(1 for s in scores_leg2 if s == 0.0)

        total_a_points = sum(scores_leg1) + (games_per_side - sum(scores_leg2))
        total_games_played = games_per_side * 2
        a_win_rate = total_a_points / total_games_played

        elapsed_pair = time.time() - t0
        print(f" Done in {elapsed_pair:.2f}s | {agent_a}: {total_a_points:.1f}/{total_games_played} ({a_win_rate*100:.1f}%)")

        pair_summary[pair_key] = {
            "agent_a": agent_a,
            "agent_b": agent_b,
            "total_games": total_games_played,
            "agent_a_points": total_a_points,
            "agent_a_win_rate": a_win_rate,
            "as_us": {
                "agent_a_record": f"{a_wins_as_us}W - {a_losses_as_us}L - {a_draws_as_us}D",
                "agent_b_record": f"{b_wins_as_us}W - {b_losses_as_us}L - {b_draws_as_us}D",
            },
        }

    total_elapsed = time.time() - start_tournament_time
    print(f"\n{'='*75}")
    print(f" Tournament Complete! {total_games:,} games in {total_elapsed:.1f}s ({total_games/total_elapsed:.1f} games/second)")
    print(f"{'='*75}\n")

    # Compute Final Elo Ratings
    final_elo = EloCalculator.compute_elo_ratings(
        all_match_results,
        anchor_agent="HeuristicBot",
        anchor_rating=1500.0,
        k_factor=16.0,
        iterations=300,
    )

    print(f" ---------------- FINAL GRAND TOURNAMENT ELO LEADERBOARD ----------------")
    print(f" | Rank | Agent / Model Snapshot      | Elo Rating | Total Points / Games |")
    print(f" +------+-----------------------------+------------+----------------------+")
    for rank, (agent, elo) in enumerate(final_elo.items(), start=1):
        # Calculate total tournament score for agent
        agent_pts = sum(m[2] for m in all_match_results if m[0] == agent) + sum((1.0 - m[2]) for m in all_match_results if m[1] == agent)
        agent_total = len([m for m in all_match_results if m[0] == agent or m[1] == agent])
        pct = (agent_pts / agent_total * 100) if agent_total > 0 else 0
        print(f" |  {rank:2d}  | {agent:27s} |   {elo:6.1f}   | {agent_pts:5.1f} / {agent_total:4d} ({pct:4.1f}%) |")
    print(f" ------------------------------------------------------------------------\n")

    # Save detailed JSON and Markdown report
    report_data = {
        "games_per_side": games_per_side,
        "total_games": total_games,
        "total_elapsed_seconds": total_elapsed,
        "leaderboard": final_elo,
        "pair_summary": pair_summary,
    }
    with open("checkpoints/grand_tournament_results.json", "w") as f:
        json.dump(report_data, f, indent=2)

    # Write Markdown table
    md_content = f"""# Twilight Struggle Grand Tournament Results (200 Games Per Pair)

- **Total Games Played**: {total_games:,} games ({games_per_side} as US / {games_per_side} as USSR per matchup)
- **Tournament Duration**: {total_elapsed:.1f} seconds ({total_games/total_elapsed:.1f} games/second)
- **Anchor**: `HeuristicBot` @ 1500.0 Elo

## Final Elo Leaderboard

| Rank | Agent / Checkpoint | Elo Rating | Total Match Score | Win Rate (%) |
|:----:|:-------------------|:----------:|:-----------------:|:------------:|
"""
    for rank, (agent, elo) in enumerate(final_elo.items(), start=1):
        agent_pts = sum(m[2] for m in all_match_results if m[0] == agent) + sum((1.0 - m[2]) for m in all_match_results if m[1] == agent)
        agent_total = len([m for m in all_match_results if m[0] == agent or m[1] == agent])
        pct = (agent_pts / agent_total * 100) if agent_total > 0 else 0
        md_content += f"| **{rank}** | **{agent}** | **{elo:.1f}** | {agent_pts:.1f} / {agent_total} | **{pct:.1f}%** |\n"

    md_content += f"\n## Detailed Head-to-Head Cross-Table\n\n"
    md_content += f"| Agent | " + " | ".join([f"**{a[:7]}**" for a in final_elo.keys()]) + " |\n"
    md_content += "|:---|" + "|:---" * len(final_elo) + "|\n"

    for row_agent in final_elo.keys():
        row_str = f"| **{row_agent}** |"
        for col_agent in final_elo.keys():
            if row_agent == col_agent:
                row_str += " - |"
            else:
                pair_key1 = f"{row_agent} vs {col_agent}"
                pair_key2 = f"{col_agent} vs {row_agent}"
                if pair_key1 in pair_summary:
                    pts = pair_summary[pair_key1]["agent_a_points"]
                    tot = pair_summary[pair_key1]["total_games"]
                    row_str += f" {pts:.0f}/{tot} ({pts/tot*100:.0f}%) |"
                elif pair_key2 in pair_summary:
                    pts = pair_summary[pair_key2]["total_games"] - pair_summary[pair_key2]["agent_a_points"]
                    tot = pair_summary[pair_key2]["total_games"]
                    row_str += f" {pts:.0f}/{tot} ({pts/tot*100:.0f}%) |"
                else:
                    row_str += " ? |"
        md_content += row_str + "\n"

    with open("checkpoints/grand_tournament_report.md", "w") as f:
        f.write(md_content)
    print(" Grand tournament report written to checkpoints/grand_tournament_report.md\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Large-Scale Round Robin Tournament")
    parser.add_argument("--games-per-side", type=int, default=100, help="Games per side (default 100 -> 200 total per pair)")
    parser.add_argument("--temperature", type=float, default=0.3, help="Evaluation temperature")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device ('cuda' or 'cpu')")

    args = parser.parse_args()
    run_full_200game_tournament(
        games_per_side=args.games_per_side,
        temperature=args.temperature,
        device_str=args.device,
    )
