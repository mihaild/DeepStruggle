#!/usr/bin/env python3
"""Unified Tournament & Matchup Evaluator for Twilight Struggle AI.

Supports both 2-model head-to-head benchmarking with granular loss cause diagnostics
and massive round-robin tournaments with Bradley-Terry MLE Elo rating matrices.
"""

import os
import sys
import time
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from tools.lib.player_agent import PlayerAgent, load_agent, resolve_device
from tools.lib.batch_tournament import BatchMatchRunner, compute_mle_elo


def format_loss_causes(causes: Dict[str, int], total_losses: int) -> str:
    if not causes or total_losses == 0:
        return "None (0 losses)"
    tot = max(sum(causes.values()), total_losses, 1)
    sorted_items = sorted(causes.items(), key=lambda x: x[1], reverse=True)
    return ", ".join([f"{k}: {v} ({v/tot*100:.1f}%)" for k, v in sorted_items])


def run_head_to_head_report(
    agent_a: PlayerAgent,
    agent_b: PlayerAgent,
    matchup: Dict[str, Any],
    elo_ratings: Dict[str, float],
    elapsed_time: float,
) -> str:
    """Formats a detailed head-to-head match report for exactly two agents."""
    name_a = agent_a.name
    name_b = agent_b.name
    tot = matchup["total_games"]
    w_a = matchup["a_wins"]
    w_b = matchup["b_wins"]
    d = matchup["draws"]
    wr_a = (w_a / tot) * 100.0
    wr_b = (w_b / tot) * 100.0

    lines: List[str] = []
    lines.append("\n" + "=" * 85 + "\n")
    lines.append(f"TWILIGHT STRUGGLE HEAD-TO-HEAD BENCHMARK: {name_a} vs {name_b}\n")
    lines.append("=" * 85 + "\n\n")

    lines.append(f"• Total Games Played: {tot:,} ({tot // 2:,} per side) in {elapsed_time:.1f}s ({tot / max(0.1, elapsed_time):.1f} games/s)\n")
    lines.append(f"• Overall Result: {name_a} {w_a}W - {w_b}L - {d}D ({wr_a:.1f}% win rate)\n")
    lines.append(f"• Elo Ratings: {name_a} = {elo_ratings[name_a]:.1f} | {name_b} = {elo_ratings[name_b]:.1f} (Δ {elo_ratings[name_a] - elo_ratings[name_b]:+.1f})\n\n")

    lines.append("### Side-Specific Breakdown:\n")
    a_us_w = matchup["a_wins_as_us"]
    a_us_l = matchup["a_losses_as_us"]
    a_us_d = matchup.get("a_draws_as_us", 0)
    a_us_tot = a_us_w + a_us_l + a_us_d
    lines.append(f"  1. {name_a} (US) vs {name_b} (USSR):\n")
    lines.append(f"     - Record: {a_us_w}W - {a_us_l}L - {a_us_d}D ({a_us_w / max(1, a_us_tot) * 100:.1f}% US win rate)\n")
    lines.append(f"     - Average Steps: {matchup.get('avg_steps', 0.0):.1f}\n")
    lines.append(f"     - Losses by {name_a} (US): {format_loss_causes(matchup['causes_loss_us'], a_us_l)}\n\n")

    a_ussr_w = matchup["a_wins_as_ussr"]
    a_ussr_l = matchup["a_losses_as_ussr"]
    a_ussr_d = matchup.get("a_draws_as_ussr", 0)
    a_ussr_tot = a_ussr_w + a_ussr_l + a_ussr_d
    lines.append(f"  2. {name_a} (USSR) vs {name_b} (US):\n")
    lines.append(f"     - Record: {a_ussr_w}W - {a_ussr_l}L - {a_ussr_d}D ({a_ussr_w / max(1, a_ussr_tot) * 100:.1f}% USSR win rate)\n")
    lines.append(f"     - Average Steps: {matchup.get('avg_steps', 0.0):.1f}\n")
    lines.append(f"     - Losses by {name_a} (USSR): {format_loss_causes(matchup['causes_loss_ussr'], a_ussr_l)}\n\n")

    if "choice_stats" in matchup:
        cs = matchup["choice_stats"]
        lines.append("=" * 85 + "\n")
        lines.append("MICRO-ACTIONS & DETERMINISTIC SINGLE-CHOICE ANALYSIS (Grouped by Side)\n")
        lines.append("=" * 85 + "\n\n")
        lines.append(f"• USSR Decisions: {cs['ussr_single_choice_micro_actions']:,} single-choice / {cs['ussr_total_micro_actions']:,} total micro-actions ({cs['ussr_single_choice_pct']:.2f}% deterministic)\n")
        lines.append(f"• US Decisions:   {cs['us_single_choice_micro_actions']:,} single-choice / {cs['us_total_micro_actions']:,} total micro-actions ({cs['us_single_choice_pct']:.2f}% deterministic)\n")
        lines.append(f"• Combined Total: {cs['overall_single_choice_micro_actions']:,} single-choice / {cs['overall_total_micro_actions']:,} total micro-actions ({cs['overall_single_choice_pct']:.2f}% deterministic)\n\n")

        avg = cs["avg_per_game"]
        lines.append("• Per-Game Averages:\n")
        lines.append(f"  - USSR: {avg['ussr_total']:.1f} total micro-actions, {avg['ussr_single']:.1f} single-choice ({avg['ussr_single']/max(0.1, avg['ussr_total'])*100:.2f}%)\n")
        lines.append(f"  - US:   {avg['us_total']:.1f} total micro-actions, {avg['us_single']:.1f} single-choice ({avg['us_single']/max(0.1, avg['us_total'])*100:.2f}%)\n")
        lines.append(f"  - Both: {avg['combined_total']:.1f} total micro-actions, {avg['combined_single']:.1f} single-choice ({avg['combined_single']/max(0.1, avg['combined_total'])*100:.2f}%)\n\n")

        lines.append("### Breakdown of Deterministic (1 Valid Choice) Actions by Category:\n\n")
        lines.append("| Action Category | USSR Count (% of USSR 1-choice) | US Count (% of US 1-choice) | Total (% of All 1-choice) |\n")
        lines.append("|:---|:---:|:---:|:---:|\n")
        all_cats = sorted(set(list(cs["category_counts_ussr"].keys()) + list(cs["category_counts_us"].keys())))
        for cat in all_cats:
            cnt_ussr = cs["category_counts_ussr"].get(cat, 0)
            cnt_us = cs["category_counts_us"].get(cat, 0)
            cnt_tot = cnt_ussr + cnt_us
            pct_ussr = (cnt_ussr / max(1, cs["ussr_single_choice_micro_actions"])) * 100.0
            pct_us = (cnt_us / max(1, cs["us_single_choice_micro_actions"])) * 100.0
            pct_tot = (cnt_tot / max(1, cs["overall_single_choice_micro_actions"])) * 100.0
            lines.append(f"| **{cat}** | {cnt_ussr:,} ({pct_ussr:.1f}%) | {cnt_us:,} ({pct_us:.1f}%) | {cnt_tot:,} ({pct_tot:.1f}%) |\n")
        lines.append("\n")

    lines.append("=" * 85 + "\n")
    return "".join(lines)


def run_massive_tournament(
    model_specs: List[str],
    games_per_side: int = 500,
    temperature: float = 0.1,
    batch_chunk_size: int = 1000,
    anchor_model: str = "HeuristicBot",
    anchor_elo: float = 1500.0,
    output_report: Optional[str] = None,
    output_json: Optional[str] = None,
    device: str = "cuda",
    track_choices: bool = False,
    log_games: Optional[str] = None,
    auto_advance: bool = True,
) -> Dict[str, Any]:
    """Runs high-throughput round-robin tournament across all specified models."""
    dev = resolve_device(device)

    print("\n" + "=" * 85)
    print(f" INITIALIZING TOURNAMENT EVALUATOR: {len(model_specs)} Models, {games_per_side * 2} Games/Pair")
    print(f" Device: {dev} | Chunk Size: {batch_chunk_size}")
    print("=" * 85)

    agents: List[PlayerAgent] = []
    for spec in model_specs:
        agent = load_agent(spec, device=dev)
        agents.append(agent)
        print(f" Loaded Agent: {agent.name:<35s} (from {spec})")

    # Names must be distinct AND attributable. They used to be disambiguated by appending #1 and
    # #2, which kept the tournament running but made its report unciteable: two runs snapshot at
    # identical step counts, so a field could report "snapshot_150011904steps#1 beat
    # snapshot_150011904steps#2" with no way to tell which run either was. Agents are now named
    # <run>@<steps> by checkpoint_id, so a genuine clash is a mistake and is refused.
    name_counts: Dict[str, int] = {}
    for a in agents:
        name_counts[a.name] = name_counts.get(a.name, 0) + 1
    clashes = sorted(n for n, c in name_counts.items() if c > 1)
    if clashes:
        raise ValueError(
            "tournament entrants must have distinct names; these are repeated: "
            + ", ".join(clashes)
            + ". Each is <run>@<steps>, so a repeat means the same checkpoint was entered twice "
              "or two runs share a short name.")

    M = len(agents)
    model_names = [a.name for a in agents]

    win_matrix = np.zeros((M, M), dtype=np.int32)
    loss_matrix = np.zeros((M, M), dtype=np.int32)
    draw_matrix = np.zeros((M, M), dtype=np.int32)
    ussr_win_matrix = np.zeros((M, M), dtype=np.int32)
    us_win_matrix = np.zeros((M, M), dtype=np.int32)
    total_matrix = np.zeros((M, M), dtype=np.int32)

    total_pairs = (M * (M - 1)) // 2
    pair_idx = 0
    t_start = time.time()
    matchup_details: Dict[str, Any] = {}

    for i in range(M):
        for j in range(i + 1, M):
            pair_idx += 1
            agent_a = agents[i]
            agent_b = agents[j]
            pair_start = time.time()

            m_res = BatchMatchRunner.play_parallel_matchup(
                agent_a,
                agent_b,
                games_per_side=games_per_side,
                device=dev,
                temperature=temperature,
                batch_chunk_size=batch_chunk_size,
                track_choices=track_choices,
                log_games_file=log_games,
                auto_advance=auto_advance,
            )
            pair_time = time.time() - pair_start
            matchup_key = f"{agent_a.name}_vs_{agent_b.name}"
            matchup_details[matchup_key] = m_res

            w_a = m_res["a_wins"]
            w_b = m_res["b_wins"]
            d = m_res["draws"]
            tot = m_res["total_games"]

            win_matrix[i, j] = w_a
            win_matrix[j, i] = w_b
            loss_matrix[i, j] = w_b
            loss_matrix[j, i] = w_a
            draw_matrix[i, j] = d
            draw_matrix[j, i] = d
            total_matrix[i, j] = tot
            total_matrix[j, i] = tot

            ussr_win_matrix[i, j] = m_res["a_wins_as_ussr"]
            ussr_win_matrix[j, i] = m_res["a_losses_as_us"]
            us_win_matrix[i, j] = m_res["a_wins_as_us"]
            us_win_matrix[j, i] = m_res["a_losses_as_ussr"]

            wr_a = (w_a / tot) * 100.0
            print(
                f"[{pair_idx:2d}/{total_pairs:2d}] {agent_a.name:<25s} vs {agent_b.name:<25s} -> "
                f"{w_a:4d}W - {w_b:4d}L - {d:2d}D ({wr_a:5.1f}% win) in {pair_time:5.1f}s"
            )

    total_tournament_time = time.time() - t_start
    total_games_played = np.sum(total_matrix) // 2

    # Compute Bradley-Terry MLE Elo ratings
    elo_ratings = compute_mle_elo(model_names, win_matrix, total_matrix, anchor_model=anchor_model, anchor_elo=anchor_elo)

    # If exactly 2 models, print detailed head-to-head report
    if M == 2:
        m_key = f"{agents[0].name}_vs_{agents[1].name}"
        report_text = run_head_to_head_report(agents[0], agents[1], matchup_details[m_key], elo_ratings, total_tournament_time)
        print(report_text)
        if output_report:
            os.makedirs(os.path.dirname(os.path.abspath(output_report)), exist_ok=True)
            with open(output_report, "w", encoding="utf-8") as f:
                f.write(report_text)
        return {
            "models": model_names,
            "elo_ratings": elo_ratings,
            "matchup": matchup_details[m_key],
            "total_time_seconds": total_tournament_time,
        }

    # Full Round-Robin Tournament Report
    sorted_indices = sorted(range(M), key=lambda idx: elo_ratings[model_names[idx]], reverse=True)
    sorted_names = [model_names[idx] for idx in sorted_indices]

    total_wr_matrix = np.zeros((M, M))
    ussr_wr_matrix = np.zeros((M, M))
    us_wr_matrix = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            if i != j:
                total_wr_matrix[i, j] = (win_matrix[i, j] / max(1, total_matrix[i, j])) * 100.0
                ussr_wr_matrix[i, j] = (ussr_win_matrix[i, j] / max(1, games_per_side)) * 100.0
                us_wr_matrix[i, j] = (us_win_matrix[i, j] / max(1, games_per_side)) * 100.0

    report_lines: List[str] = []
    report_lines.append("# Twilight Struggle AI: Massive Tournament Evaluation Report\n\n")
    report_lines.append(f"- **Total Models**: {M}\n")
    report_lines.append(f"- **Total Games Played**: {total_games_played:,}\n")
    report_lines.append(f"- **Games Per Matchup Pair**: {games_per_side * 2:,} ({games_per_side} per side)\n")
    report_lines.append(f"- **Sampling Temperature**: {temperature} ({'deterministic' if temperature <= 0.05 else 'sampled'})\n")
    report_lines.append(f"- **Total Evaluation Time**: {total_tournament_time:.1f} seconds ({total_games_played / max(0.1, total_tournament_time):.1f} games/sec)\n\n")

    report_lines.append("## 1. Bradley-Terry MLE Elo Leaderboard\n\n")
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

    # Matrices
    report_lines.append("## 2. Head-to-Head Total Win Rate Matrix (% Win for Row vs Column)\n\n")
    header_cols = " | ".join([f"**{name}**" for name in sorted_names])
    report_lines.append(f"| Model | {header_cols} |\n")
    report_lines.append(f"|:---|{'---:|' * M}\n")
    for i in sorted_indices:
        row_vals = []
        for j in sorted_indices:
            row_vals.append("—" if i == j else f"{total_wr_matrix[i, j]:.1f}%")
        report_lines.append(f"| **{model_names[i]}** | {' | '.join(row_vals)} |\n")
    report_lines.append("\n---\n\n")

    # Per-side matrix. The pooled matrix above averages the two sides, which hides exactly the
    # thing worth seeing in an asymmetric game: an arm can be at 50% overall while winning
    # almost every game on one side and almost none on the other.
    report_lines.append("## 3. Per-Side Win Rate Matrix (Row vs Column: as USSR / as US)\n\n")
    report_lines.append(
        "Each cell is the **row** model's win rate against the **column** model, "
        f"playing USSR and playing US, over {games_per_side} games per side.\n\n")
    report_lines.append(f"| Model | {header_cols} |\n")
    report_lines.append(f"|:---|{'---:|' * M}\n")
    for i in sorted_indices:
        row_vals = []
        for j in sorted_indices:
            if i == j:
                row_vals.append("—")
            else:
                row_vals.append(f"{ussr_wr_matrix[i, j]:.1f}% / {us_wr_matrix[i, j]:.1f}%")
        report_lines.append(f"| **{model_names[i]}** | {' | '.join(row_vals)} |\n")
    report_lines.append("\n")

    # The side split each arm shows across the whole field, and the gap between the two.
    report_lines.append("### Side balance per arm (across all opponents)\n\n")
    report_lines.append("| Model | as USSR | as US | USSR - US |\n")
    report_lines.append("|:---|---:|---:|---:|\n")
    for i in sorted_indices:
        opps = [j for j in range(M) if j != i]
        if not opps:
            continue
        as_ussr = float(np.mean([ussr_wr_matrix[i, j] for j in opps]))
        as_us = float(np.mean([us_wr_matrix[i, j] for j in opps]))
        report_lines.append(
            f"| **{model_names[i]}** | {as_ussr:.1f}% | {as_us:.1f}% | "
            f"{as_ussr - as_us:+.1f} pp |\n")
    report_lines.append("\n---\n\n")

    report_text = "".join(report_lines)

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
        "games_per_side": games_per_side,
        "temperature": temperature,
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
    parser.add_argument("--include-baselines", action="store_true", default=False, help="Explicitly include RandomBot and HeuristicBot")
    parser.add_argument("--games-per-side", type=int, default=500, help="Games per side per matchup (total 2x games per pair)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature for every neural agent. <= 0.05 plays the "
                             "argmax. The default 0.1 samples, which is fine for comparing "
                             "models of similar sharpness but confounds a comparison between "
                             "models whose policy entropy differs -- the sharper one plays "
                             "closer to its own argmax and wins on temperature rather than on "
                             "strength. Re-rate at 0.0 when the arms differ in entropy.")
    parser.add_argument("--batch-chunk-size", type=int, default=1000, help="Max parallel games executed in a single vectorized batch")
    parser.add_argument("--anchor-model", type=str, default="HeuristicBot", help="Model name to anchor Elo ratings")
    parser.add_argument("--anchor-elo", type=float, default=1500.0, help="Anchor Elo rating value")
    parser.add_argument("--output-report", type=str, default=None, help="Path to save Markdown report")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save JSON results")
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Compute device; falls back to cpu when there is no GPU (resolve_device). "
             "The default was cpu, which contradicted this module's own function "
             "signature and made every tournament far slower than it needed to be -- "
             "200 games took 118s on cpu against 6.4s on cuda, an 18x difference.")
    parser.add_argument("--track-choices", action="store_true", default=False, help="Track and report micro-actions with exactly 1 valid choice")
    parser.add_argument("--log-games", type=str, default=None, help="Path to save per-game JSONL execution logs")
    parser.add_argument("--self-play", action="store_true", default=False, help="Evaluate model against itself")
    # Default ON. The function signature defaulted to True while this flag defaulted to
    # False and line 413 passes it through unconditionally, so every CLI run settled
    # nothing and the signature change was invisible -- "the training setup you think you
    # configured is not the one that ran". The opt-out is explicit instead.
    parser.add_argument("--no-auto-advance", dest="auto_advance", action="store_false",
                        default=True,
                        help="Present decisions with a single legal action instead of "
                             "settling them in the engine. Changes the decision stream, so "
                             "a run with it is not comparable with a run without.")

    args = parser.parse_args()

    models = list(args.models) if args.models else []
    if args.checkpoint_dir and os.path.exists(args.checkpoint_dir):
        discovered = [os.path.join(args.checkpoint_dir, f) for f in sorted(os.listdir(args.checkpoint_dir)) if f.endswith(".pt")]
        models.extend(discovered)
        # When evaluating a checkpoint directory, default to including baselines unless explicitly specified
        if not args.models and not args.include_baselines:
            args.include_baselines = True

    if args.include_baselines:
        if "random" not in models and "RandomBot" not in models:
            models.insert(0, "random")
        if "heuristic" not in models and "HeuristicBot" not in models:
            models.insert(1, "heuristic")

    if (args.self_play or len(models) == 1) and len(models) == 1:
        models = [models[0], models[0]]

    if len(models) < 2:
        print("Error: Need at least 2 models for an evaluation or tournament (or pass --self-play).")
        sys.exit(1)

    out_rep = args.output_report
    out_json = args.output_json

    run_massive_tournament(
        model_specs=models,
        games_per_side=args.games_per_side,
        temperature=args.temperature,
        batch_chunk_size=args.batch_chunk_size,
        anchor_model=args.anchor_model,
        anchor_elo=args.anchor_elo,
        output_report=out_rep,
        output_json=out_json,
        device=args.device,
        track_choices=args.track_choices,
        log_games=args.log_games,
        auto_advance=args.auto_advance,
    )


if __name__ == "__main__":
    main()
