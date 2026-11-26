#!/usr/bin/env python3
"""A --ladder-side-nets (A39) checkpoint from one side's network and a base's other side (owner, 2026-11-26).

    PYTHONPATH=. .venv/bin/python tools/scripts/recombine_side_nets.py --side ussr --side-net <side .pt> \\
        --base <a39 .pt> --output <out .pt>

The inverse of extract_side_net.py for plain checkpoints: `--side`'s network from `--side-net` (a single-side
exploiter's snapshot or SWA), the other side's from `--base` bit for bit, so tournaments, reads and the leaderboard
play it on both seats with the untouched side exactly the base's. Write it inside the exploiter's run directory, so
its view and features are read from that run (tools/lib/side_nets.py)."""

from __future__ import annotations

import argparse
import os

import torch

from tools.lib.side_nets import SIDES, combine


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--side", required=True, choices=SIDES)
    ap.add_argument("--side-net", required=True, help="the side's network alone (a plain LadderNet checkpoint)")
    ap.add_argument("--base", required=True, help="the A39 checkpoint whose other side is kept")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"{args.output} exists; refusing to overwrite it")
    side = torch.load(args.side_net, map_location="cpu", weights_only=False)
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    torch.save(combine(side, base, args.side), args.output)
    print(f"wrote {args.output}: {args.side} from {args.side_net}, the other side from {args.base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
