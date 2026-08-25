"""Generates a full self-play game replay (.tslog.json) using trained ColdWarNet checkpoints."""

import argparse
import os
import sys
from typing import Optional

# Ensure repository root is on sys.path
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from ai.eval.self_play import generate_self_play_replay


def run_self_play_cli(
    model_path: str = "checkpoints/snapshot_20m.pt",
    seed: int = 42,
    temperature: float = 0.3,
    game_id: str = "snapshot_20m_self_play",
    output_filename: Optional[str] = None,
    device_str: str = "cuda",
) -> str:
    _, saved_path = generate_self_play_replay(
        model=None,
        model_path=model_path,
        seed=seed,
        temperature=temperature,
        game_id=game_id,
        output_path=output_filename,
        device=device_str,
        verbose=True,
    )
    return saved_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Self-Play Game Replay")
    parser.add_argument("--model", type=str, default="checkpoints/snapshot_20m.pt", help="Path to model checkpoint")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for game initialization")
    parser.add_argument("--temp", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--game-id", type=str, default="snapshot_20m_self_play", help="Replay Game ID")
    parser.add_argument("--output", type=str, default=None, help="Custom output path")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device ('cuda' or 'cpu')")

    args = parser.parse_args()
    run_self_play_cli(
        model_path=args.model,
        seed=args.seed,
        temperature=args.temp,
        game_id=args.game_id,
        output_filename=args.output,
        device_str=args.device,
    )
