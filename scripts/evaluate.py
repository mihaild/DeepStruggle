#!/usr/bin/env python3
"""Unified Entry Point for Twilight Struggle Tournament Evaluation."""

import argparse
import sys
from ai.eval.player_agent import load_agent
from ai.eval.tournament_evaluator import TournamentEvaluator


def format_loss_causes(causes) -> str:
    if not causes:
        return "None (0 losses)"
    grouped = {}
    for k, v in causes.items():
        label = "Held scoring" if k.startswith("Held scoring") else k
        grouped[label] = grouped.get(label, 0) + v
    sorted_items = sorted(grouped.items(), key=lambda x: x[1], reverse=True)
    return ", ".join([f"{k} ({v})" for k, v in sorted_items])


def main():
    parser = argparse.ArgumentParser(description="Twilight Struggle Head-to-Head Tournament Evaluator")
    parser.add_argument("--agent-a", type=str, required=True, help="Agent A specifier (random, heuristic, or checkpoint path)")
    parser.add_argument("--agent-b", type=str, required=True, help="Agent B specifier (random, heuristic, or checkpoint path)")
    parser.add_argument("--games-per-side", type=int, default=50, help="Games played per side (total 2x games)")
    parser.add_argument("--base-seed", type=int, default=10000, help="Base seed for matches")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device (cuda or cpu)")

    args = parser.parse_args()

    agent_a = load_agent(args.agent_a, device=args.device)
    agent_b = load_agent(args.agent_b, device=args.device)

    print(f"=== Starting Matchup: {agent_a.name} vs {agent_b.name} ({args.games_per_side * 2} games) ===")
    res = TournamentEvaluator.play_matchup(agent_a, agent_b, games_per_side=args.games_per_side, base_seed=args.base_seed)

    w_all = res["a_wins"]
    l_all = res["b_wins"]
    d_all = res["draws"]
    wr_all = res["win_rate_a"] * 100.0

    w_us = res["a_wins_as_us"]
    l_us = res["a_losses_as_us"]
    wr_us = res["win_rate_a_as_us"] * 100.0

    w_ussr = res["a_wins_as_ussr"]
    l_ussr = res["a_losses_as_ussr"]
    wr_ussr = res["win_rate_a_as_ussr"] * 100.0

    print("\n--- Results ---")
    print(f"Total Games: {res['total_games']}")
    print(f"{agent_a.name} Overall Win Rate: {wr_all:.1f}% ({w_all}W - {l_all}L - {d_all}D)")
    print(f"{agent_a.name} Win Rate as US:   {wr_us:.1f}% ({w_us}W - {l_us}L - {res['a_draws_as_us']}D)")
    print(f"{agent_a.name} Win Rate as USSR: {wr_ussr:.1f}% ({w_ussr}W - {l_ussr}L - {res['a_draws_as_ussr']}D)")
    print(f"Average Game Length: {res['avg_steps']:.1f} steps (Turn {res['avg_turn']:.1f})")
    print(f"Average VP Margin for {agent_a.name}: {res['avg_vp_margin_a']:+.1f}")

    print(f"\nLoss Causes for {agent_a.name} as US:")
    print(f"  {format_loss_causes(res['causes_loss_us'])}")

    print(f"\nLoss Causes for {agent_a.name} as USSR:")
    print(f"  {format_loss_causes(res['causes_loss_ussr'])}")


if __name__ == "__main__":
    main()
