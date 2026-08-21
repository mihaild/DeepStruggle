#!/usr/bin/env python3
"""
Deep Game Auditor: Runs multiple complete games across diverse random seeds
and performs a comprehensive, independent post-game and step-by-step invariant audit.
"""

import sys
import os
import json
from typing import List, Dict, Any

import ts_engine
from bot.play_and_audit_full_game import run_full_game_simulation, generate_readable_game_summary

def run_deep_audit(seeds: List[int]):
    os.makedirs("replays", exist_ok=True)
    total_steps_all = 0
    total_violations_all = 0
    all_summaries = []

    print("=" * 80)
    print("STARTING MULTI-GAME DEEP ENGINE AUDITOR ACROSS DIVERSE SEEDS")
    print("=" * 80)

    for s in seeds:
        print(f"\n--- Running Full Simulation for Seed {s} ---")
        logs, violations = run_full_game_simulation(seed=s)
        total_steps_all += len(logs)
        total_violations_all += len(violations)

        last_entry = logs[-1]
        winner = last_entry.get("winner", "IN_PROGRESS")
        final_vp = last_entry.get("final_vp", 0)
        turns = last_entry.get("turn", 1)

        summary_line = f"Seed {s:5d}: {len(logs):3d} steps | End Turn: {turns:2d} | Winner: {winner:4s} | Final VP: {final_vp:+3d} | Violations: {len(violations)}"
        print(f"Summary: {summary_line}")
        all_summaries.append(summary_line)

        # Save individual trace
        txt_path = f"replays/audit_game_seed_{s}.txt"
        generate_readable_game_summary(logs, txt_path)

    print("\n" + "=" * 80)
    print("DEEP AUDITOR OVERALL RESULTS:")
    print("=" * 80)
    for line in all_summaries:
        print(f"  • {line}")
    print("-" * 80)
    print(f"Total Games: {len(seeds)} | Total Micro-Steps Audited: {total_steps_all} | Total Invariant Violations: {total_violations_all}")
    if total_violations_all == 0:
        print("🎉 ALL SIMULATIONS PASSED 100% INVARIANT AND RULE AUDIT WITH ZERO ERRORS!")
    else:
        print(f"⚠️ DETECTED {total_violations_all} ERRORS ACROSS SIMULATIONS!")
    print("=" * 80)

if __name__ == "__main__":
    seeds_to_test = [42, 100, 123, 777, 2026]
    run_deep_audit(seeds_to_test)
