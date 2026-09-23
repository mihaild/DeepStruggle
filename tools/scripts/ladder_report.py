#!/usr/bin/env python3
"""The standing table for the P21 architecture ladder.

One table per new arm, so twelve modifications produce twelve comparable reports rather than
twelve hand-written summaries each emphasising something different. Rows are every arm in the
tournament -- the ladder rungs so far, plus `E4-03-01@80M`, `E4-04-01@80M` and `HeuristicBot`.
Columns are:

    Elo              from the tournament's Bradley-Terry fit
    steps/s          the arm's own measured throughput, so the compute price is never implicit
    USSR / US vs previous   per side against the rung below -- the matched comparison
    USSR / US vs anchor     per side against E4-03-01@80M -- absolute placement

**Per side, never pooled.** A pooled number hides the failure this ladder surfaced on its first
rung: M0 beats the E4 defaults 80% as USSR and 49% as US, and the pooled 64.5% shows neither.

**One tournament per table.** Bradley-Terry ratings are field-relative -- the anchor rated 2107.2
in one tournament and 2179.5 in another, unchanged, because the entrants differed. Only deltas
inside a single tournament mean anything, so this reads exactly one JSON.

    tools/scripts/ladder_report.py <tournament.json> --previous <label> [--anchor <label>]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Any, Dict, Optional, Tuple

#: The P21 anchor: best of the first four E4 arms, at 80M and at any budget alike.
ANCHOR_DEFAULT = "E4-03-01@80M"
CHECKPOINT_ROOT = "/workspace/data/checkpoints"


def per_side(data: Dict[str, Any], a: str, b: str) -> Optional[Tuple[float, float]]:
    """(a's win rate as USSR, as US) against b, or None if the pair was not played."""
    if a == b:
        return None
    ps = data.get("per_side", {})
    e = ps.get(f"{a}_vs_{b}")
    if e is not None:
        return float(e["win_rate_a_as_ussr"]), float(e["win_rate_a_as_us"])
    e = ps.get(f"{b}_vs_{a}")
    if e is not None:
        # Stored from b's perspective: a played US in b's USSR games and vice versa.
        return 1.0 - float(e["win_rate_a_as_us"]), 1.0 - float(e["win_rate_a_as_ussr"])
    return None


def side_balance(data: Dict[str, Any], arm: str) -> Optional[Tuple[float, float]]:
    """(arm's win rate as USSR, as US) across EVERY opponent in this tournament.

    The per-arm columns only ever show a rate *against something*, so the anchor's own row is
    blank there -- it cannot play itself. This is the row that says how an arm is doing from each
    seat overall, which is how the ladder noticed that M0 is a competent USSR and a poor US while
    the anchor sits within a point of even.
    """
    ussr_w = ussr_n = us_w = us_n = 0
    for key, e in data.get("per_side", {}).items():
        a, _, b = key.partition("_vs_")
        n = int(e["games_per_side"])
        if a == arm:
            ussr_w += int(e["a_wins_as_ussr"]); ussr_n += n
            us_w += int(e["a_wins_as_us"]); us_n += n
        elif b == arm:
            # a played USSR in those games, so `arm` played US, and vice versa.
            us_w += n - int(e["a_wins_as_ussr"]); us_n += n
            ussr_w += n - int(e["a_wins_as_us"]); ussr_n += n
    if not ussr_n or not us_n:
        return None
    return ussr_w / ussr_n, us_w / us_n


def steps_per_sec(entrant: str) -> Optional[float]:
    """Median post-warmup throughput for the run behind an entrant label, if it is a checkpoint.

    Reported because this ladder's rungs differ by 4-5x in speed -- M0 runs 60,814 steps/s
    against the anchor's 11,732 -- so an Elo quoted without its compute price is half a result.
    """
    m = re.match(r"^(E\d+-\d+-\d+)@", entrant)
    if not m:
        return None
    dirs = [d for d in glob.glob(os.path.join(CHECKPOINT_ROOT, m.group(1) + "_*"))
            if "_VOID_" not in d]
    if not dirs:
        return None
    vals = []
    for d in sorted(dirs, key=os.path.getmtime, reverse=True):
        path = os.path.join(d, "training_metrics.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                v = r.get("steps_per_sec_avg") or r.get("steps_per_sec")
                # Skip the first 50 iterations: startup, compile and the snapshot@0 evaluation
                # all distort them.
                if isinstance(v, (int, float)) and v > 0 and int(r.get("iteration", 0)) > 50:
                    vals.append(float(v))
        if vals:
            break
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def _cell(got: Optional[Tuple[float, float]], which: int) -> str:
    if got is None:
        return "—"
    return f"{100 * got[which]:.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tournament_json")
    ap.add_argument("--previous", default=None,
                    help="entrant name of the rung below. Omit for M0, which is the floor.")
    ap.add_argument("--anchor", default=ANCHOR_DEFAULT)
    args = ap.parse_args()

    with open(args.tournament_json, encoding="utf-8") as f:
        data = json.load(f)
    elo = data.get("elo_ratings", {})

    prev_label = args.previous or "—"
    print(f"games per side {data.get('games_per_side')}, tau={data.get('temperature')}   "
          f"previous rung: {prev_label}   anchor: {args.anchor}\n")
    head = (f"| {'arm':26s} | {'Elo':>7s} | {'steps/s':>8s} "
            f"| {'USSR all':>8s} | {'US all':>7s} | {'gap':>6s} "
            f"| {'USSR v prev':>11s} | {'US v prev':>9s} "
            f"| {'USSR v anch':>11s} | {'US v anch':>9s} |")
    print(head)
    print("|" + "|".join(["-" * (len(c) + 2) for c in
                          ["x" * 26, "x" * 7, "x" * 8, "x" * 8, "x" * 7, "x" * 6,
                           "x" * 11, "x" * 9, "x" * 11, "x" * 9]]) + "|")

    for name, rating in sorted(elo.items(), key=lambda kv: -kv[1]):
        sps = steps_per_sec(name)
        vp = per_side(data, name, args.previous) if args.previous else None
        va = per_side(data, name, args.anchor)
        sb = side_balance(data, name)
        gap = f"{100 * (sb[0] - sb[1]):+.1f}" if sb else "—"
        print(f"| {name:26s} | {rating:7.1f} | "
              f"{(f'{sps:,.0f}' if sps else '—'):>8s} | "
              f"{_cell(sb, 0):>8s} | {_cell(sb, 1):>7s} | {gap:>6s} | "
              f"{_cell(vp, 0):>11s} | {_cell(vp, 1):>9s} | "
              f"{_cell(va, 0):>11s} | {_cell(va, 1):>9s} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
