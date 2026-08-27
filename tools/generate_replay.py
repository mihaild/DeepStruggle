#!/usr/bin/env python3
"""Unified Replay Generator for Self-Play or Head-to-Head Matches (.tslog.json)."""

import os
import sys
import argparse
from typing import Optional

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from tools.eval.self_play import generate_self_play_replay
from tools.eval.player_agent import load_agent


def main():
    parser = argparse.ArgumentParser(description="Generate Twilight Struggle Game Replay (.tslog.json)")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path (or bot name)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for game initialization")
    parser.add_argument("--temperature", "--temp", type=float, default=0.2, help="Policy sampling temperature")
    parser.add_argument("--game-id", type=str, default=None, help="Custom game identifier")
    parser.add_argument("--output", type=str, default=None, help="Output .tslog.json path")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device ('cuda' or 'cpu')")

    args = parser.parse_args()

    # Determine default game ID and path if not specified
    model_base = os.path.splitext(os.path.basename(args.model))[0]
    gid = args.game_id or f"{model_base}_selfplay_s{args.seed}"
    out_path = args.output or os.path.join("replays", f"{gid}.tslog.json")

    log_dict, saved_path = generate_self_play_replay(
        model_path=args.model,
        model_name=model_base,
        seed=args.seed,
        temperature=args.temperature,
        game_id=gid,
        output_path=out_path,
        device=args.device,
        verbose=True,
    )

    replay_filename = os.path.basename(saved_path)
    print("=" * 80)
    print("REPLAY READY FOR WEB WORKBENCH INSPECTION")
    print(f"File Path: {saved_path}")
    print(f"Workbench URL: http://localhost:8000/?replay={replay_filename}")
    print("=" * 80)


if __name__ == "__main__":
    main()
