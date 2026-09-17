#!/usr/bin/env python3
"""Where does a checkpoint put its opening influence, and how did that change over training?

The setup phase is the cheapest possible probe of a policy: a handful of `POINT_NODE` decisions
before any card is drawn, identical in structure every game. It is also where a degenerate policy
shows most plainly — piling every point into one country is legal, scores nothing, and is the kind
of choice a collapsed seat makes.

Runs only the setup phase, so it costs nothing next to a game, and reports the placement
distribution per side along with how concentrated it is.

    tools/scripts/setup_placement.py --checkpoints a.pt b.pt --games 40
"""

from __future__ import annotations

import argparse
import collections
import math
import sys
from typing import Dict, List, NamedTuple, Tuple

import json
import os

import ts_engine as ts

from tools.lib.player_agent import load_agent

_MAP = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "rules", "map.json")
with open(_MAP, encoding="utf-8") as _f:
    _NAMES = {int(c["id"]): c["name"] for c in json.load(_f)["countries"]}


class SideStats(NamedTuple):
    placements: int
    distinct: int
    entropy: float
    top_share: float
    ref_overlap: float
    top: List[Tuple[str, int]]
    top_full: List[Tuple[str, int]]


#: The reference openings named as correct: US takes 4 West Germany / 3 Italy unless holding
#: Marshall Plan; USSR takes 4 East Germany / 4 Poland / 1 Yugoslavia or Austria.
REFERENCE = {
    "US": {"West Germany": 4, "Italy": 3},
    "USSR": {"East Germany": 4, "Poland": 4, "Yugoslavia": 1, "Austria": 1},
}

#: The countries that actually decide the opening, as against the ones that merely appear in a
#: reasonable line. Poland is the one USSR must have; West Germany and Italy are the two the US
#: must have. Somewhere like Bulgaria is a defensible point but not a substitute for Poland, and a
#: single overlap percentage hides that difference -- which is why the per-country table below
#: reports placements a game against the target rather than only the aggregate.
CORE = {"US": ("West Germany", "Italy"), "USSR": ("Poland",)}


def reference_overlap(counts: Dict[str, int], side: str, games: int) -> float:
    """Share of a side's placements landing in a reference country, capped at its target.

    Capped, because seven points into West Germany is not seven points of correct opening -- four
    is what the opening calls for and the rest is the same waste this probe exists to catch.
    """
    ref = REFERENCE[side]
    tot = sum(counts.values())
    if tot <= 0:
        return 0.0
    good = 0
    for name, target in ref.items():
        good += min(counts.get(name, 0), target * games)
    return good / tot


def _entropy(counts: Dict[str, int]) -> float:
    tot = sum(counts.values())
    if tot <= 0:
        return 0.0
    ps = [c / tot for c in counts.values() if c > 0]
    return -sum(p * math.log(p) for p in ps)


def probe(spec: str, games: int, device: str) -> Dict[str, SideStats]:
    agent = load_agent(spec, device=device)
    per_side: Dict[str, collections.Counter] = {
        "US": collections.Counter(), "USSR": collections.Counter()}
    decisions = {"US": 0, "USSR": 0}

    for g in range(games):
        st = ts.GameState()
        ts.Engine.init_game(st, 7000 + g)
        guard = 0
        while st.current_phase == ts.Phase.SETUP and not ts.Engine.is_terminal(st):
            guard += 1
            if guard > 400:
                break
            ctx = st.ctx()
            pl = ctx.decision_player
            if pl == ts.Player.NONE:
                # chance node: drain it. `step` raises on a refused action and returns None
                # otherwise -- it is the checked half of the try_step/step split, so the result
                # must not be truth-tested. Doing that silently ended this probe after one
                # placement per game.
                if not ts.Engine.try_step(st, ts.MicroAction(ctx.decision_type, 0, 0, 0)):
                    break
                continue
            side = "US" if pl == ts.Player.US else "USSR"
            idx = agent.select_action(st, pl, temperature=0.0)
            ma = ts.decode_flat_action(st, int(idx))
            if ctx.decision_type == ts.DecisionType.POINT_NODE:
                cid = int(ma.primary_id)
                per_side[side][_NAMES.get(cid, "#%d" % cid)] += 1
                decisions[side] += 1
            if not ts.Engine.try_step(st, ma):
                break

    out: Dict[str, SideStats] = {}
    for side in ("USSR", "US"):
        c = per_side[side]
        tot = sum(c.values())
        top = c.most_common(4)
        out[side] = SideStats(
            placements=tot,
            distinct=len(c),
            entropy=_entropy(c),
            top_share=(top[0][1] / tot) if tot else 0.0,
            ref_overlap=reference_overlap(dict(c), side, games),
            top=top,
            top_full=list(c.items()),
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    labels = a.labels if a.labels and len(a.labels) == len(a.checkpoints) else a.checkpoints

    print("%-16s %-5s %7s %9s %9s %9s %8s  %s" % (
        "checkpoint", "side", "places", "distinct", "entropy", "top%", "ref%",
        "top countries"))
    print("-" * 120)
    for spec, lab in zip(a.checkpoints, labels):
        r = probe(spec, a.games, a.device)
        for side in ("USSR", "US"):
            d = r[side]
            tops = ", ".join("%s x%d" % (n, k) for n, k in d.top)
            print("%-16s %-5s %7d %9d %9.3f %8.1f%% %7.1f%%  %s" % (
                lab[:16], side, d.placements, d.distinct,
                d.entropy, 100 * d.top_share, 100 * d.ref_overlap, tops))
            counts = dict(d.top_full)
            detail = []
            for name, target in REFERENCE[side].items():
                per_game = counts.get(name, 0) / max(1, a.games)
                mark = "*" if name in CORE[side] else " "
                detail.append("%s%s %.1f/%d" % (mark, name, per_game, target))
            print("%-16s %-5s   per game vs target (* = decisive): %s" % ("", "", "  ".join(detail)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
