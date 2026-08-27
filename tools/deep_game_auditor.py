#!/usr/bin/env python3
"""
Deep Game Auditor: Runs 10 full exploratory games across diverse random seeds
with exploratory agents exploring edge cases, coups, space race, realignments, and complex events.
"""

import sys
import os
import json
from typing import List, Dict, Any

import ts_engine
from bot.play_and_audit_full_game import run_full_game_simulation, generate_readable_game_summary

def run_deep_audit(seeds: List[int], bot_type: str = "exploratory"):
    os.makedirs("replays", exist_ok=True)
    total_steps_all = 0
    total_violations_all = 0
    all_summaries = []

    print("=" * 85)
    print(f"STARTING 10-GAME EXPLORATORY ENGINE AUDITOR (Mode: {bot_type.upper()})")
    print("=" * 85)

    for idx, s in enumerate(seeds, 1):
        print(f"\n--- [Game {idx:02d}/10] Running Simulation for Seed {s} ({bot_type}) ---")
        logs, violations = run_full_game_simulation(seed=s, bot_type=bot_type)
        total_steps_all += len(logs)
        total_violations_all += len(violations)

        last_entry = logs[-1]
        winner = last_entry.get("winner", "IN_PROGRESS")
        final_vp = last_entry.get("final_vp", 0)
        turns = last_entry.get("turn", 1)

        summary_line = f"Game {idx:02d} (Seed {s:5d}): {len(logs):4d} steps | Turn {turns:2d} | Winner: {winner:4s} | Final VP: {final_vp:+3d} | Violations: {len(violations)}"
        print(f"Summary: {summary_line}")
        all_summaries.append(summary_line)

        # Save individual trace
        txt_path = f"replays/exploratory_game_{idx:02d}_seed_{s}.txt"
        generate_readable_game_summary(logs, txt_path)

    print("\n" + "=" * 85)
    print("EXPLORATORY DEEP AUDITOR OVERALL RESULTS:")
    print("=" * 85)
    for line in all_summaries:
        print(f"  • {line}")
    print("-" * 85)
    print(f"Total Games: {len(seeds)} | Total Micro-Steps Audited: {total_steps_all} | Total Invariant Violations: {total_violations_all}")
    if total_violations_all == 0:
        print("🎉 ALL 10 EXPLORATORY GAMES PASSED 100% INVARIANT AND RULE AUDIT WITH ZERO ERRORS!")
    else:
        print(f"⚠️ DETECTED {total_violations_all} ERRORS ACROSS SIMULATIONS!")
    print("=" * 85)

if __name__ == "__main__":
    test_seeds = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
    run_deep_audit(test_seeds, bot_type="exploratory")
