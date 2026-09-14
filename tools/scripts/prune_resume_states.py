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

**Nothing is deleted.** Files are *moved* to a quarantine directory, preserving the run directory
name, so what was thinned can be reviewed and removed by hand. An earlier version of this script
deleted outright and destroyed 34.8GB across 678 files that were not recoverable; moving costs
nothing on the same filesystem and makes the operation reversible.

Dry run by default. Pass --apply to move.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
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
    ap.add_argument("--apply", action="store_true",
                    help="Actually move the files (default: dry run)")
    ap.add_argument("--quarantine", default="/workspace/data/checkpoints_cleanup",
                    help="Where thinned files are moved to, under <run-name>/<file>")
    ap.add_argument("--delete", action="store_true",
                    help="Delete instead of moving. Not the default, and not reversible.")
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
            dest_dir = os.path.join(args.quarantine, os.path.basename(run))
            if not args.delete:
                os.makedirs(dest_dir, exist_ok=True)
            for f in removed:
                path = os.path.join(run, f)
                try:
                    if args.delete:
                        os.remove(path)
                    else:
                        dest = os.path.join(dest_dir, f)
                        # shutil.move rather than os.rename: the quarantine may be on a
                        # different filesystem, where rename fails with EXDEV.
                        shutil.move(path, dest)
                except OSError as e:
                    print(f"    FAILED {f}: {e}")

    if not args.apply:
        verb = "would move"
    elif args.delete:
        verb = "DELETED"
    else:
        verb = "moved"
    print(f"\n{verb} {human(total_freed)} across {total_removed} files")
    if args.apply and not args.delete and total_removed:
        print(f"quarantined under {args.quarantine}/<run-name>/ -- review and remove by hand")
    if not args.apply:
        print("dry run -- pass --apply to move (or --apply --delete to remove outright)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
