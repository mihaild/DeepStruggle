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
    top: List[Tuple[str, int]]


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
            top=top,
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

    print("%-16s %-5s %7s %9s %9s %9s  %s" % (
        "checkpoint", "side", "places", "distinct", "entropy", "top%", "top countries"))
    print("-" * 112)
    for spec, lab in zip(a.checkpoints, labels):
        r = probe(spec, a.games, a.device)
        for side in ("USSR", "US"):
            d = r[side]
            tops = ", ".join("%s x%d" % (n, k) for n, k in d.top)
            print("%-16s %-5s %7d %9d %9.3f %8.1f%%  %s" % (
                lab[:16], side, d.placements, d.distinct,
                d.entropy, 100 * d.top_share, tops))
    return 0


if __name__ == "__main__":
    sys.exit(main())
