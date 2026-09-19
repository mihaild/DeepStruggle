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

**This is the progress channel, not the completion channel.** Prefer to launch the run itself in
the background so that its own exit is the notification -- one event, for success and crash
alike, with nothing to arm and nothing to remember. Use this script when the *intermediate*
states matter: a stall, or a run that exits without logging an iteration.

If it is watched by something with a deadline, the deadline must outlast the run. A 160M-step arm
is about 3.5 hours; a watch capped at one hour is killed first, and the completion it existed to
report then arrives as silence -- which is indistinguishable from the run still going. That is a
real failure that happened, not a hypothetical.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Tuple

#: What a PROGRESS line carries, ordered by how much it is worth.
#:
#: `opp_pool_size` and `opp_win_rate_mean` first, because they are the ONLY things that have
#: separated a collapsed run from a healthy control twice -- E4-01-01 (no pool at all) and X4b
#: (pool starved to 1, beaten 96% of the time). They measure the cause rather than a symptom and
#: are readable at iteration 1.
#:
#: The rest are printed as a *description*, not as a trigger. Measured across both collapses
#: (`research/method/detecting_collapse.md`), no dynamics metric generalises: entropy separated
#: E4's pair enormously and was noise on E3's, `clip_frac` and `adv_std_raw` reversed direction
#: between them, and `kl_div` was 115x on E3 and flat on E4. They are worth seeing because when
#: something goes wrong they say what KIND of wrong -- E4's policy barely moved, E3's moved
#: violently -- but a threshold on any of them describes one collapse only.
#:
#: Deliberately absent: the three critic metrics this used to print. Critic quality never
#: separated in either pair, and by some measures the collapsing run scored better, because a
#: degenerate policy is an easy prediction problem. `critic_base_rate` is `max(p, 1-p)` and
#: carries no direction at all.
WATCHED = ("opp_pool_size", "opp_win_rate_mean", "entropy", "clip_frac",
           "adv_std_raw", "kl_div", "us_episode_frac")


def us_episode_frac(row: Dict[str, Any]) -> Optional[float]:
    """Fraction of this iteration's finished episodes the US won, from self-play alone.

    The two known collapse modes are mutually blind (`research/method/detecting_collapse.md`):
    E3-24-28 ran totally one-sided for its whole 40M with `kl_div` at a healthy 0.06, and
    E3-31-28 reached `kl_div` 193.7 with side balance never flagging. `opp_pool_size` catches
    starvation only on a *pooled* run -- E3-24-28 had no pool, so nothing here would have caught
    it. This closes that hole, and needs no external opponent.

    Not sufficient alone: one side genuinely improving faster looks the same, so a per-side win
    rate against a frozen reference stays the arbiter. A value pinned at 0.0 or 1.0 is the alarm.
    """
    done = row.get("episodes_completed")
    won = row.get("episodes_completed_won_us")
    if not isinstance(done, (int, float)) or not isinstance(won, (int, float)) or done <= 0:
        return None
    return won / done


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
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a partially written final line is normal
                n += 1
                # Only rows that actually carry a step count. The run writes a summary row at
                # shutdown (blunder statistics) with no `total_steps`; taking the last row
                # unconditionally read it as 0 steps and reported CRASH on a run that had just
                # finished successfully.
                if "total_steps" in row:
                    last = row
    except OSError:
        return None, None, 0
    if last is None:
        return None, None, 0
    return int(last.get("total_steps", 0)), last, n


def _cmdline(pid: str) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def pid_alive(pid: int) -> bool:
    """Is this exact process still running? Signal 0 tests existence without touching it."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def run_pid(run_dir: str) -> Optional[int]:
    """The PID the run recorded for itself, if it recorded one.

    Preferred over any pattern match, because a pattern cannot distinguish two runs. Runs started
    before the trainer wrote this file have none, and the caller falls back.
    """
    try:
        with open(os.path.join(run_dir, "run.pid"), encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def process_alive(pattern: str) -> bool:
    """Is *a* training process still running, by command-line pattern?

    The fallback for runs with no `run.pid`. It cannot tell one run from another: the default
    pattern is `tools/train.py`, which matches ANY run on the box, so a dead run reports STALL
    rather than CRASH while a sibling run is alive. That happened on 2026-09-16 when E3-22-28 was
    relaunched and the old watcher kept reporting on the run that had been killed. Prefer
    `run_pid`; this exists only for runs that predate it.

    **Must exclude this watcher and its shell.** `--pattern` is normally the run name, and the
    watcher's own command line contains the run name too (it is an argument), so a bare
    `pgrep -f <run-name>` matches the watcher itself. That makes `process_alive` permanently
    true, and CRASH can then never fire -- the watcher reports PROGRESS forever while the run it
    is watching is dead. That is the exact failure this script exists to prevent, so the
    self-match is filtered rather than assumed away.
    """
    try:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    except OSError:
        return False
    if out.returncode != 0:
        return False
    me = str(os.getpid())
    for pid in out.stdout.split():
        if pid == me:
            continue
        cmd = _cmdline(pid)
        if not cmd:
            continue
        # The watcher, its shell wrapper, and any sibling watcher are not the training run.
        if "watch_run.py" in cmd:
            continue
        if "shell-snapshots" in cmd or cmd.strip().startswith("/usr/bin/zsh -c"):
            continue
        return True
    return False


def summary(row: Optional[Dict[str, Any]]) -> str:
    if not row:
        return ""
    derived = {"us_episode_frac": us_episode_frac(row)}
    bits = []
    for k in WATCHED:
        v = derived[k] if k in derived else row.get(k)
        if isinstance(v, (int, float)):
            bits.append(f"{k}={v:.4f}")
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
                    help="pgrep -f pattern identifying the training process. The run name works; "
                         "this watcher and its shell are excluded from the match, since their own "
                         "command lines contain it too.")
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
        pid = run_pid(args.run_dir)
        alive = pid_alive(pid) if pid is not None else process_alive(args.pattern)
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
