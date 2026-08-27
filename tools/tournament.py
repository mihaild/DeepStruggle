#!/usr/bin/env python3
"""Massive Parallel Tournament & Elo Matrix Evaluator for Twilight Struggle AI."""

import os
import sys
import time
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch

from tools.eval.player_agent import PlayerAgent, load_agent, resolve_device
from tools.eval.batch_tournament import BatchMatchRunner, compute_mle_elo


def format_loss_causes(causes: Dict[str, int], total_losses: int) -> str:
    if not causes or total_losses == 0:
        return "None (0 losses)"
    grouped: Dict[str, int] = {}
    for k, v in causes.items():
        label = "Held scoring" if k.startswith("Held scoring") else k
        grouped[label] = grouped.get(label, 0) + v
    actual_tot = sum(grouped.values())
    tot = max(actual_tot, total_losses, 1)
    sorted_items = sorted(grouped.items(), key=lambda x: x[1], reverse=True)
    return ", ".join([f"{k}: {v} ({v/tot*100:.1f}%)" for k, v in sorted_items])


def run_massive_tournament(
    model_specs: List[str],
    games_per_side: int = 1000,
    batch_chunk_size: int = 1000,
    anchor_model: str = "HeuristicBot",
    anchor_elo: float = 1500.0,
    output_report: Optional[str] = None,
    output_json: Optional[str] = None,
    device: Optional[str] = "cuda",
) -> Dict[str, Any]:
    dev = resolve_device(device)
    print("=" * 90)
    print(f"STARTING MASSIVE PARALLEL TOURNAMENT: {len(model_specs)} Models, {games_per_side * 2:,} games per matchup")
    print(f"Compute Device: {dev} | Batch Chunk Size: {batch_chunk_size}")
    print("=" * 90)

    # 1. Load All Agents
    agents: List[PlayerAgent] = []
    for spec in model_specs:
        try:
            agents.append(load_agent(spec, device=dev))
        except Exception as e:
            print(f"Error loading model '{spec}': {e}")
            sys.exit(1)

    M = len(agents)
    model_names = [a.name for a in agents]
    print(f"Loaded {M} Models: {model_names}\n")

    # Matrices: M x M
    win_matrix = np.zeros((M, M), dtype=np.int32)
    loss_matrix = np.zeros((M, M), dtype=np.int32)
    draw_matrix = np.zeros((M, M), dtype=np.int32)
    total_matrix = np.zeros((M, M), dtype=np.int32)

    ussr_win_matrix = np.zeros((M, M), dtype=np.int32)
    us_win_matrix = np.zeros((M, M), dtype=np.int32)

    matchup_details: Dict[str, Dict[str, Any]] = {}
    pair_count = (M * (M - 1)) // 2
    cur_pair = 0
    t0_all = time.time()

    # 2. Run All Pairwise Matches
    for i in range(M):
        for j in range(i + 1, M):
            cur_pair += 1
            agent_i = agents[i]
            agent_j = agents[j]
            pair_key = f"{agent_i.name}_vs_{agent_j.name}"

            print(f"[{cur_pair:2d}/{pair_count:2d}] Running Matchup: {agent_i.name} vs {agent_j.name} ({games_per_side*2:,} games)...", flush=True)
            res = BatchMatchRunner.play_parallel_matchup(
                agent_a=agent_i,
                agent_b=agent_j,
                games_per_side=games_per_side,
                batch_chunk_size=batch_chunk_size,
                base_seed=10000 + (cur_pair * 10000),
                device=dev,
            )

            # Record Results
            w_i = res["a_wins"]
            w_j = res["b_wins"]
            d = res["draws"]
            tot = res["total_games"]

            win_matrix[i, j] = w_i
            loss_matrix[i, j] = w_j
            draw_matrix[i, j] = d
            total_matrix[i, j] = tot

            win_matrix[j, i] = w_j
            loss_matrix[j, i] = w_i
            draw_matrix[j, i] = d
            total_matrix[j, i] = tot

            # Side specific
            ussr_win_matrix[i, j] = res["a_wins_as_ussr"]
            us_win_matrix[i, j] = res["a_wins_as_us"]
            ussr_win_matrix[j, i] = res["a_losses_as_us"]   # j won as USSR when i was US
            us_win_matrix[j, i] = res["a_losses_as_ussr"]   # j won as US when i was USSR

            matchup_details[pair_key] = res

            wr_i = (w_i / tot) * 100.0
            wr_j = (w_j / tot) * 100.0
            print(f"      -> {agent_i.name}: {wr_i:.1f}% ({w_i}W) | {agent_j.name}: {wr_j:.1f}% ({w_j}W) | Draws: {d} in {res['elapsed_seconds']:.1f}s ({tot/res['elapsed_seconds']:.1f} g/s)", flush=True)

    total_tournament_time = time.time() - t0_all
    print("\n" + "=" * 90)
    print(f"ALL MATCHUPS FINISHED in {total_tournament_time:.1f}s ({(pair_count * games_per_side * 2) / total_tournament_time:.1f} total games/sec)")
    print("=" * 90 + "\n")

    # 3. Compute Bradley-Terry Elo Ratings
    elo_ratings = compute_mle_elo(
        model_names=model_names,
        win_matrix=win_matrix,
        total_matrix=total_matrix,
        anchor_model=anchor_model if anchor_model in model_names else model_names[0],
        anchor_elo=anchor_elo,
    )

    # Sort models by Elo
    sorted_indices = sorted(range(M), key=lambda idx: elo_ratings[model_names[idx]], reverse=True)
    sorted_names = [model_names[idx] for idx in sorted_indices]

    # 4. Format Winning Matrices (Total, USSR, US)
    total_wr_matrix = np.zeros((M, M), dtype=np.float32)
    ussr_wr_matrix = np.zeros((M, M), dtype=np.float32)
    us_wr_matrix = np.zeros((M, M), dtype=np.float32)

    for i in range(M):
        for j in range(M):
            if i != j and total_matrix[i, j] > 0:
                total_wr_matrix[i, j] = (win_matrix[i, j] / total_matrix[i, j]) * 100.0
                ussr_wr_matrix[i, j] = (ussr_win_matrix[i, j] / games_per_side) * 100.0
                us_wr_matrix[i, j] = (us_win_matrix[i, j] / games_per_side) * 100.0

    # 5. Build Markdown Report
    report_lines = []
    report_lines.append("# Massive Parallel Tournament & Elo Rating Report\n\n")
    report_lines.append(f"- **Total Models**: {M}\n")
    report_lines.append(f"- **Games per Matchup**: {games_per_side * 2:,} ({games_per_side:,} as USSR, {games_per_side:,} as US)\n")
    report_lines.append(f"- **Total Games Played**: {pair_count * games_per_side * 2:,}\n")
    report_lines.append(f"- **Total Computation Time**: {total_tournament_time:.1f}s ({(pair_count * games_per_side * 2) / total_tournament_time:.1f} games/sec)\n")
    report_lines.append(f"- **Anchor Reference**: `{anchor_model}` = {anchor_elo:.0f} Elo\n\n")

    # Leaderboard Table
    report_lines.append("## 1. Overall Leaderboard & Elo Ratings\n\n")
    report_lines.append("| Rank | Model | Elo Rating | Total Matches | Total Record (W-L-D) | Overall Win Rate |\n")
    report_lines.append("|:---:|:---|:---:|:---:|:---:|:---:|\n")

    for rank, idx in enumerate(sorted_indices, start=1):
        name = model_names[idx]
        elo = elo_ratings[name]
        tot_w = int(np.sum(win_matrix[idx, :]))
        tot_l = int(np.sum(loss_matrix[idx, :]))
        tot_d = int(np.sum(draw_matrix[idx, :]))
        tot_g = tot_w + tot_l + tot_d
        wr = (tot_w / max(1, tot_g)) * 100.0
        report_lines.append(f"| **{rank}** | **{name}** | **{elo:.1f}** | {tot_g:,} | {tot_w:,}W - {tot_l:,}L - {tot_d:,}D | **{wr:.1f}%** |\n")
    report_lines.append("\n---\n\n")

    # Total Win Rate Matrix
    report_lines.append("## 2. Head-to-Head Total Win Rate Matrix (% Win for Row vs Column)\n\n")
    header_cols = " | ".join([f"**{name}**" for name in sorted_names])
    report_lines.append(f"| Model | {header_cols} |\n")
    report_lines.append(f"|:---|{'---:|' * M}\n")
    for i in sorted_indices:
        row_vals = []
        for j in sorted_indices:
            if i == j:
                row_vals.append("—")
            else:
                row_vals.append(f"{total_wr_matrix[i, j]:.1f}%")
        report_lines.append(f"| **{model_names[i]}** | {' | '.join(row_vals)} |\n")
    report_lines.append("\n---\n\n")

    # USSR Win Rate Matrix
    report_lines.append("## 3. USSR Win Rate Matrix (% Win when Row Model is USSR)\n\n")
    report_lines.append(f"| Model (as USSR) | {header_cols} |\n")
    report_lines.append(f"|:---|{'---:|' * M}\n")
    for i in sorted_indices:
        row_vals = []
        for j in sorted_indices:
            if i == j:
                row_vals.append("—")
            else:
                row_vals.append(f"{ussr_wr_matrix[i, j]:.1f}%")
        report_lines.append(f"| **{model_names[i]}** | {' | '.join(row_vals)} |\n")
    report_lines.append("\n---\n\n")

    # US Win Rate Matrix
    report_lines.append("## 4. US Win Rate Matrix (% Win when Row Model is US)\n\n")
    report_lines.append(f"| Model (as US) | {header_cols} |\n")
    report_lines.append(f"|:---|{'---:|' * M}\n")
    for i in sorted_indices:
        row_vals = []
        for j in sorted_indices:
            if i == j:
                row_vals.append("—")
            else:
                row_vals.append(f"{us_wr_matrix[i, j]:.1f}%")
        report_lines.append(f"| **{model_names[i]}** | {' | '.join(row_vals)} |\n")
    report_lines.append("\n---\n\n")

    # 6. Detailed Loss Cause Breakdown for the Final Model
    final_model_idx = M - 1  # Last model in input list (or snapshot_final)
    final_name = model_names[final_model_idx]
    report_lines.append(f"## 5. Granular Loss Causes Breakdown for Final Model (`{final_name}`)\n\n")
    report_lines.append("| Opponent | Overall Win Rate | Record (W-L-D) | Loss Reasons when `{final_name}` played as USSR | Loss Reasons when `{final_name}` played as US |\n")
    report_lines.append("|:---|:---:|:---:|:---|:---|\n")

    for j in range(M):
        if j == final_model_idx:
            continue
        opp_name = model_names[j]
        # Retrieve matchup result
        key_f_vs_j = f"{final_name}_vs_{opp_name}"
        key_j_vs_f = f"{opp_name}_vs_{final_name}"

        if key_f_vs_j in matchup_details:
            m_res = matchup_details[key_f_vs_j]
            w = m_res["a_wins"]
            l = m_res["b_wins"]
            d = m_res["draws"]
            tot = m_res["total_games"]
            wr = (w / tot) * 100.0
            causes_loss_ussr = format_loss_causes(m_res["causes_loss_ussr"], m_res["a_losses_as_ussr"])
            causes_loss_us = format_loss_causes(m_res["causes_loss_us"], m_res["a_losses_as_us"])
        else:
            m_res = matchup_details[key_j_vs_f]
            w = m_res["b_wins"]
            l = m_res["a_wins"]
            d = m_res["draws"]
            tot = m_res["total_games"]
            wr = (w / tot) * 100.0
            causes_loss_ussr = format_loss_causes(m_res["causes_loss_us"], m_res["a_wins_as_us"]) # j won as US -> final lost as USSR
            causes_loss_us = format_loss_causes(m_res["causes_loss_ussr"], m_res["a_wins_as_ussr"]) # j won as USSR -> final lost as US

        report_lines.append(f"| **{opp_name}** | **{wr:.1f}%** | {w}W - {l}L - {d}D | {causes_loss_ussr} | {causes_loss_us} |\n")

    report_text = "".join(report_lines)

    # Save outputs
    if output_report:
        os.makedirs(os.path.dirname(os.path.abspath(output_report)), exist_ok=True)
        with open(output_report, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"Saved Markdown Tournament Report to: {output_report}")

    summary_data = {
        "models": model_names,
        "elo_ratings": elo_ratings,
        "win_matrix": win_matrix.tolist(),
        "total_matrix": total_matrix.tolist(),
        "ussr_win_matrix": ussr_win_matrix.tolist(),
        "us_win_matrix": us_win_matrix.tolist(),
        "games_per_side": games_per_side,
        "total_time_seconds": total_tournament_time,
    }

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(summary_data, indent=2))
        print(f"Saved JSON Tournament Data to: {output_json}")

    print("\n" + report_text)
    return summary_data


def main():
    parser = argparse.ArgumentParser(description="Massive Parallel Tournament & Elo Rating Benchmark")
    parser.add_argument("--models", nargs="+", default=None, help="List of model checkpoints or bot names ('random', 'heuristic')")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to auto-discover all snapshot checkpoints")
    parser.add_argument("--include-baselines", action="store_true", default=True, help="Include RandomBot and HeuristicBot")
    parser.add_argument("--games-per-side", type=int, default=1000, help="Games per side per matchup (total 2x games per pair)")
    parser.add_argument("--batch-chunk-size", type=int, default=1000, help="Max parallel games executed in a single vectorized batch")
    parser.add_argument("--anchor-model", type=str, default="HeuristicBot", help="Model name to anchor Elo ratings")
    parser.add_argument("--anchor-elo", type=float, default=1500.0, help="Anchor Elo rating value")
    parser.add_argument("--output-report", type=str, default=None, help="Path to save Markdown report")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save JSON results")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device (cuda or cpu)")

    args = parser.parse_args()

    models = args.models or []
    if args.checkpoint_dir and os.path.exists(args.checkpoint_dir):
        discovered = [os.path.join(args.checkpoint_dir, f) for f in sorted(os.listdir(args.checkpoint_dir)) if f.endswith(".pt")]
        models.extend(discovered)

    if args.include_baselines:
        if "random" not in models and "RandomBot" not in models:
            models.insert(0, "random")
        if "heuristic" not in models and "HeuristicBot" not in models:
            models.insert(1, "heuristic")

    if len(models) < 2:
        print("Error: Need at least 2 models for a tournament.")
        sys.exit(1)

    out_rep = args.output_report or os.path.join(args.checkpoint_dir or "checkpoints", "massive_tournament_report.md")
    out_json = args.output_json or os.path.join(args.checkpoint_dir or "checkpoints", "massive_tournament_results.json")

    run_massive_tournament(
        model_specs=models,
        games_per_side=args.games_per_side,
        batch_chunk_size=args.batch_chunk_size,
        anchor_model=args.anchor_model,
        anchor_elo=args.anchor_elo,
        output_report=out_rep,
        output_json=out_json,
        device=args.device,
    )


if __name__ == "__main__":
    main()
