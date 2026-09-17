#!/usr/bin/env python3
"""What changes in the gameplay as a run collapses?

The X4b arms go from 1674 Elo to 1162 inside 5M steps, and every account of why has come from the
training loss -- which has now ruled out the CE coefficient and the reference schedule, and left a
collapse whose entropy RISES and whose CE gradient share FALLS. This reads the other side: what
the policy actually does, snapshot by snapshot.

Per checkpoint, from self-play with full traces:

* **chosen-probability and entropy, per seat** — the split that identified E3-30-28's US seat as
  having no policy rather than a bad one.
* **per decision type** — so a collapse concentrated in placements is distinguishable from one
  spread across every choice.
* **game length and outcome** — a policy that has stopped playing loses differently from one that
  plays badly.
Blunder rates are deliberately NOT re-measured here: the run's own training_metrics.jsonl
already carries them per snapshot, and a second estimate from six games would be noisier than
the one that exists.

    tools/scripts/collapse_trace.py --run <run-dir> --every 4 --games 6
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import re
import statistics
import subprocess
import sys
import tempfile
from typing import Counter as CounterT, Dict, List, NamedTuple

NAMES = {1: "CARD", 2: "PLAY_MODE", 3: "TIMING", 4: "OP_MODE", 5: "POINT", 6: "BRANCH"}


def snapshots(run_dir: str) -> List[tuple]:
    out = []
    for f in glob.glob(os.path.join(run_dir, "snapshot_*steps.pt")):
        m = re.search(r"snapshot_(\d+)steps\.pt$", f)
        if m:
            out.append((int(m.group(1)), f))
    return sorted(out)


def play(ckpt: str, games: int, outdir: str, seed0: int = 4100) -> List[str]:
    """Self-play through the sanctioned CLI, so replays carry the standard trace schema."""
    paths = []
    for g in range(games):
        p = os.path.join(outdir, "g%d.tslog.json" % g)
        cmd = [sys.executable, "tools/play_match.py", "--agent", ckpt,
               "--seed", str(seed0 + g), "--temperature", "0.0",
               "--commentary", "--trace", "--trace-critic-every", "decision",
               "--game-id", "collapse_%d" % g, "--output", p, "--device", "cuda"]
        env = dict(os.environ, PYTHONPATH=".:build/release")
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if os.path.exists(p):
            paths.append(p)
        elif r.returncode != 0:
            print("    (game %d failed: %s)" % (g, r.stderr.strip().splitlines()[-1:]), flush=True)
    return paths


class Summary(NamedTuple):
    seat_p: Dict[str, List[float]]
    seat_ent: Dict[str, List[float]]
    per_type: Dict[str, List[float]]
    lengths: List[int]
    winners: CounterT[str]
    reasons: CounterT[str]
    turns: List[int]


def summarise(paths: List[str]) -> Summary:
    import json

    per_seat: Dict[str, List[float]] = {"US": [], "USSR": []}
    ent_seat: Dict[str, List[float]] = {"US": [], "USSR": []}
    per_type: Dict[str, List[float]] = collections.defaultdict(list)
    lengths: List[int] = []
    winners: collections.Counter = collections.Counter()
    reasons: collections.Counter = collections.Counter()
    turns: List[int] = []

    for p in paths:
        b = json.load(open(p, encoding="utf-8"))
        steps = b.get("steps", [])
        lengths.append(len(steps))
        # metadata.result carries winner, margin, end_turn and reason -- the reason matters
        # here, because a run that stops playing loses differently from one that plays badly.
        res = ((b.get("metadata") or {}).get("result")) or {}
        winners[str(res.get("winner", "?"))] += 1
        reasons[str(res.get("reason", "?"))] += 1
        turns.append(int(res.get("end_turn", 0) or 0))
        for st in steps:
            po = st.get("policy") or {}
            pl = st.get("player")
            pc = po.get("p_chosen")
            if pl in per_seat and isinstance(pc, (int, float)):
                per_seat[pl].append(float(pc))
                if isinstance(po.get("entropy"), (int, float)):
                    ent_seat[pl].append(float(po["entropy"]))
            dt = (st.get("action") or {}).get("decision_type")
            if isinstance(pc, (int, float)) and dt in NAMES:
                per_type[NAMES[dt]].append(float(pc))
    return Summary(seat_p=per_seat, seat_ent=ent_seat, per_type=dict(per_type),
                   lengths=lengths, winners=winners, reasons=reasons, turns=turns)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--run", required=True)
    ap.add_argument("--every", type=int, default=4, help="use every Nth snapshot")
    ap.add_argument("--games", type=int, default=6)
    a = ap.parse_args()

    snaps = snapshots(a.run)[:: a.every]
    if not snaps:
        print("no snapshots in", a.run)
        return 1

    print("%-10s %7s %7s %7s %7s %6s %6s  %s" % (
        "steps", "US p", "USSR p", "US ent", "SU ent", "turn", "len", "winners / reason"))
    print("-" * 104)
    by_type: List[tuple] = []
    for steps, path in snaps:
        with tempfile.TemporaryDirectory() as d:
            paths = play(path, a.games, d)
            if not paths:
                print("%-10d  (no games)" % steps)
                continue
            s = summarise(paths)
        def f(v: List[float]) -> float:
            return statistics.mean(v) if v else float("nan")

        top_reason = s.reasons.most_common(1)[0][0] if s.reasons else "?"
        print("%-10d %7.3f %7.3f %7.3f %7.3f %6.1f %6.0f  %s / %s" % (
            steps, f(s.seat_p["US"]), f(s.seat_p["USSR"]),
            f(s.seat_ent["US"]), f(s.seat_ent["USSR"]),
            statistics.mean(s.turns) if s.turns else 0.0,
            statistics.mean(s.lengths), dict(s.winners), top_reason[:34]))
        by_type.append((steps, {k: f(v) for k, v in s.per_type.items()}))

    # Second table: mean chosen-probability per decision type. This is the column that
    # distinguishes a collapse concentrated in one kind of choice -- placements, say -- from one
    # spread evenly across every decision the policy makes. Summarising it and then dropping it,
    # as this script did, left the tool unable to answer the question it was written for.
    kinds = [k for k in NAMES.values() if any(k in d for _, d in by_type)]
    if kinds:
        print()
        print("mean p_chosen by decision type")
        print("%-10s %s" % ("steps", " ".join("%9s" % k for k in kinds)))
        print("-" * (11 + 10 * len(kinds)))
        for steps, d in by_type:
            cells = " ".join(
                ("%9.3f" % d[k]) if k in d and d[k] == d[k] else "%9s" % "-" for k in kinds)
            print("%-10d %s" % (steps, cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
