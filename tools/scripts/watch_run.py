#!/usr/bin/env python3
"""Emit one line per event for a training run, so a launch cannot be lost track of.

Written after losing time twice to launches whose failure looked like success:

* a resumed run exceeded its cumulative `--train-steps` budget before the first loop
  iteration. It printed "Training Complete", wrote `snapshot_final.pt` and a tournament
  report, and logged **no metrics file at all**. Nothing in the log said anything was wrong.
* a sweep launched with `nohup ... &` died with its parent session and left no trace.

So watching for a completion banner is not enough, and neither is watching for a traceback.
**Silence and success look identical**, and so do "finished" and "exited without doing anything".
This checks the thing that actually matters -- whether the step counter is advancing -- and
emits on every terminal state:

    START    the run was seen alive with a step count
    PROGRESS periodic, carrying steps and the collapse instruments
    STALL    the process is alive but the step count has not moved
    NOSTART  the process is gone and the run never logged an iteration  <- the silent one
    CRASH    the process is gone before reaching the target
    DONE     the target step count was reached

Every line is an event. Exits when the run reaches a terminal state.

    tools/scripts/watch_run.py <run-dir> --target-steps 240000000 [--interval 120]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Tuple

WATCHED = ("adv_std_raw", "critic_auc", "critic_brier_skill", "critic_base_rate")


def emit(line: str) -> None:
    print(line, flush=True)


def read_metrics(run_dir: str) -> Tuple[Optional[int], Optional[Dict[str, Any]], int]:
    """(latest step count, latest row, number of iterations logged)."""
    path = os.path.join(run_dir, "training_metrics.jsonl")
    if not os.path.exists(path):
        return None, None, 0
    last: Optional[Dict[str, Any]] = None
    n = 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line)
                    n += 1
                except json.JSONDecodeError:
                    continue  # a partially written final line is normal
    except OSError:
        return None, None, 0
    if last is None:
        return None, None, 0
    return int(last.get("total_steps", 0)), last, n


def process_alive(pattern: str) -> bool:
    try:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        return out.returncode == 0 and bool(out.stdout.strip())
    except OSError:
        return False


def summary(row: Optional[Dict[str, Any]]) -> str:
    if not row:
        return ""
    bits = [f"{k}={row[k]:.4f}" for k in WATCHED if isinstance(row.get(k), (int, float))]
    return "  " + " ".join(bits) if bits else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--target-steps", type=int, required=True,
                    help="Cumulative step count that means the run is finished. For a RESUMED "
                         "run this includes the steps it inherited, not just the increment.")
    ap.add_argument("--interval", type=int, default=120, help="Seconds between checks")
    ap.add_argument("--pattern", default="tools/train.py",
                    help="pgrep pattern identifying the training process")
    ap.add_argument("--grace", type=int, default=600,
                    help="Seconds to allow before a run with no iterations counts as NOSTART")
    ap.add_argument("--stall-after", type=int, default=1800,
                    help="Seconds without the step count advancing before reporting STALL")
    args = ap.parse_args()

    run = os.path.basename(args.run_dir.rstrip("/"))
    t0 = time.time()
    last_steps: Optional[int] = None
    last_move = time.time()
    started = False
    stall_reported = False

    while True:
        steps, row, iters = read_metrics(args.run_dir)
        alive = process_alive(args.pattern)
        now = time.time()

        if steps is not None and not started:
            started = True
            emit(f"START {run}: {steps:,} steps, {iters} iterations logged")

        if steps is not None and steps != last_steps:
            last_steps, last_move, stall_reported = steps, now, False

        # The silent failure: the process is gone and nothing was ever logged.
        if not alive and iters == 0:
            if now - t0 > args.grace:
                emit(f"NOSTART {run}: process gone and no iteration was ever logged. "
                     f"A resumed run whose --train-steps budget is below its inherited step "
                     f"count exits here, after writing a final checkpoint that looks complete.")
                return 1
        elif not alive:
            if steps is not None and steps >= args.target_steps:
                emit(f"DONE {run}: reached {steps:,} steps ({iters} iterations){summary(row)}")
                return 0
            emit(f"CRASH {run}: process gone at {steps:,}/{args.target_steps:,} steps "
                 f"({iters} iterations){summary(row)}")
            return 1

        if steps is not None and steps >= args.target_steps:
            emit(f"DONE {run}: reached {steps:,} steps ({iters} iterations){summary(row)}")
            return 0

        if started and not stall_reported and now - last_move > args.stall_after:
            stall_reported = True
            emit(f"STALL {run}: alive but stuck at {last_steps:,} steps for "
                 f"{int(now - last_move)}s")

        if started and steps is not None:
            pct = 100.0 * steps / max(args.target_steps, 1)
            emit(f"PROGRESS {run}: {steps:,}/{args.target_steps:,} ({pct:.1f}%)"
                 f"{summary(row)}")

        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
