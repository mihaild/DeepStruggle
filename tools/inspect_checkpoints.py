#!/usr/bin/env python3
"""Checkpoint Inspector: lists checkpoints with their size, date and architecture.

    PYTHONPATH=.:build/release python tools/inspect_checkpoints.py [directory]

The directory defaults to the shared data tree's `checkpoints/` (tools/lib/data_root.py). The
architecture is the one the evaluation loader builds from the weights
(`tools.lib.player_agent.NeuralAgent`), so this listing cannot disagree with what a tournament
would run. Each checkpoint is loaded in full, on the CPU. `resume_*.pt` files are training state,
not policies, and are listed without one. A file that fails to load is listed with its error and
makes the command exit 1.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence

from tools.lib.checkpoint_utils import discover_checkpoints
from tools.lib.data_root import data_path
from tools.lib.player_agent import NeuralAgent


def checkpoint_architecture(path: str) -> str:
    """The class of the network the evaluation loader builds from this checkpoint's weights."""
    agent = NeuralAgent.from_checkpoint(path, device="cpu")
    split = ("side heads" if getattr(agent.model, "side_heads", False)
             else "side nets" if getattr(agent.model, "ussr_net", None) is not None else "")
    return type(agent.model).__name__ + (f" ({split})" if split else "")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", nargs="?", default=None,
                    help="where to look (default: the shared data tree's checkpoints/)")
    a = ap.parse_args(argv)
    directory = a.directory or data_path("checkpoints")
    checkpoints = discover_checkpoints(directory)

    print("=" * 95)
    print("TWILIGHT STRUGGLE AI: CHECKPOINT REGISTRY")
    print("=" * 95)
    print(f"{'Filename':35s} | {'Size':8s} | {'Modified':19s} | {'Architecture':30s}")
    print("-" * 95)

    failed: List[str] = []
    for c in checkpoints:
        if c["filename"].startswith("resume_"):
            arch = "(training state, not a policy)"
        else:
            try:
                arch = checkpoint_architecture(c["path"])
            except (RuntimeError, ValueError, KeyError, OSError) as e:
                arch = f"LOAD FAILED: {type(e).__name__}: {e}"
                failed.append(c["path"])
        rel_path = os.path.relpath(c["path"], directory)
        print(f"{rel_path:35s} | {c['size_mb']:6.1f} MB | {c['mtime']:19s} | {arch}")

    print("=" * 95)
    print(f"Total Checkpoints Found: {len(checkpoints)}\n")
    if failed:
        print(f"{len(failed)} checkpoint(s) failed to load:", file=sys.stderr)
        for p in failed:
            print(f"  {p}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
