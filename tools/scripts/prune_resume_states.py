#!/usr/bin/env python3
"""Thin out step-tagged resume files, keeping one roughly every N steps.

A resume file carries the optimiser state as well as the weights -- 48MB against a snapshot's
13MB -- so at a fine snapshot interval they dominate a run directory. They are worth keeping at
*some* spacing, because they are what lets an experiment branch from a partial result, which is
how the current P10/P11 arms were started from E3-17-22's 80M checkpoint. They are not worth
keeping at every snapshot.

Selection is by spacing, not by modulo. Real step counts are 5046272, 40042496, 80019456 --
never exact multiples of anything -- so `steps % stride == 0` keeps nothing at all.

`resume_state.pt` (the newest, untagged) is never touched: it is what an interrupted run restarts
from, and a run in progress rewrites it continuously.

Dry run by default. Pass --apply to delete.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Tuple

TAGGED = re.compile(r"^resume_(\d+)steps\.pt$")


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def plan(run_dir: str, stride: int) -> Tuple[List[str], List[str]]:
    """(kept, removed) tagged resume filenames for one run directory."""
    found: Dict[int, str] = {}
    for name in os.listdir(run_dir):
        m = TAGGED.match(name)
        if m:
            found[int(m.group(1))] = name
    keep: List[str] = []
    last = -1
    for steps in sorted(found):
        if last < 0 or steps - last >= stride:
            keep.append(found[steps])
            last = steps
    keep_set = set(keep)
    removed = [found[s] for s in sorted(found) if found[s] not in keep_set]
    return keep, removed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="/workspace/data/checkpoints",
                    help="Directory of run directories, or a single run directory")
    ap.add_argument("--stride", type=int, default=40_000_000,
                    help="Keep a tagged resume roughly every this many steps")
    ap.add_argument("--apply", action="store_true", help="Actually delete (default: dry run)")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    runs = ([root] if any(TAGGED.match(f) for f in os.listdir(root))
            else [os.path.join(root, d) for d in sorted(os.listdir(root))
                  if os.path.isdir(os.path.join(root, d))])

    total_freed = 0
    total_removed = 0
    for run in runs:
        try:
            keep, removed = plan(run, args.stride)
        except OSError:
            continue
        if not keep and not removed:
            continue
        freed = sum(os.path.getsize(os.path.join(run, f)) for f in removed
                    if os.path.exists(os.path.join(run, f)))
        total_freed += freed
        total_removed += len(removed)
        print(f"{os.path.basename(run)}: keep {len(keep)}, remove {len(removed)} "
              f"({human(freed)})")
        for f in keep:
            print(f"    keep   {f}")
        if args.apply:
            for f in removed:
                path = os.path.join(run, f)
                try:
                    os.remove(path)
                except OSError as e:
                    print(f"    FAILED {f}: {e}")

    verb = "freed" if args.apply else "would free"
    print(f"\n{verb} {human(total_freed)} across {total_removed} files")
    if not args.apply:
        print("dry run -- pass --apply to delete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
