#!/usr/bin/env python3
"""The standing report for one rung of the P21 architecture ladder.

Every rung is reported the same way, so twelve modifications produce twelve comparable blocks
rather than twelve hand-written summaries that each emphasise something different:

  1. **Elo**, from the tournament's Bradley-Terry fit.
  2. **Per-side win rate against the previous rung** — the matched comparison, and the only one
     that supports a causal claim about the mechanism this rung adds.
  3. **Per-side win rate against `E4-03-01@80M`** — the anchor, for absolute placement.

Per side, not pooled, because a pooled number hides the failure mode this ladder has already
turned up: M0 beats HeuristicBot 90% as USSR and 59% as US, and the pooled 74.5% shows neither.

    tools/scripts/ladder_report.py <tournament.json> --rung <label> [--previous <label>]

Labels are the tournament's own entrant names, as `checkpoint_id.checkpoint_label` produces them.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional, Tuple

#: The P21 anchor: best of the first four E4 arms, at 80M and at any budget alike.
ANCHOR_DEFAULT = "E4-03-01@final"


def _per_side(data: Dict[str, Any], a: str, b: str) -> Optional[Tuple[float, float, bool]]:
    """(a's win rate as USSR, as US, whether the stored pair was reversed)."""
    ps = data.get("per_side", {})
    key = f"{a}_vs_{b}"
    if key in ps:
        e = ps[key]
        return float(e["win_rate_a_as_ussr"]), float(e["win_rate_a_as_us"]), False
    key = f"{b}_vs_{a}"
    if key in ps:
        e = ps[key]
        # Stored from b's perspective: a played US in b's USSR games and vice versa.
        return 1.0 - float(e["win_rate_a_as_us"]), 1.0 - float(e["win_rate_a_as_ussr"]), True
    return None


def _line(label: str, got: Optional[Tuple[float, float, bool]]) -> str:
    if got is None:
        return f"  vs {label:26s} — not in this tournament"
    ussr, us, _ = got
    pooled = 0.5 * (ussr + us)
    gap = 100 * (ussr - us)
    return (f"  vs {label:26s} {100*ussr:5.1f}% as USSR | {100*us:5.1f}% as US "
            f"| {100*pooled:5.1f}% pooled | side gap {gap:+5.1f} pp")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tournament_json")
    ap.add_argument("--rung", required=True, help="entrant name of the rung being reported")
    ap.add_argument("--previous", default=None,
                    help="entrant name of the rung below. Omit for M0, which is the floor.")
    ap.add_argument("--anchor", default=ANCHOR_DEFAULT)
    args = ap.parse_args()

    with open(args.tournament_json, encoding="utf-8") as f:
        data = json.load(f)

    elo = data.get("elo_ratings", {})
    if args.rung not in elo:
        print(f"'{args.rung}' is not in this tournament. Entrants: {sorted(elo)}",
              file=sys.stderr)
        return 1

    print(f"# {args.rung}")
    print(f"\n## Elo  ({data.get('games_per_side')} games per side, "
          f"tau={data.get('temperature')})\n")
    for i, (name, rating) in enumerate(sorted(elo.items(), key=lambda kv: -kv[1]), start=1):
        mark = "  <-- this rung" if name == args.rung else ""
        print(f"  {i}. {name:24s} {rating:7.1f}{mark}")

    print("\n## Per side\n")
    if args.previous:
        print(_line(f"{args.previous} (previous rung)", _per_side(data, args.rung,
                                                                  args.previous)))
    else:
        print("  vs previous rung             — none; this is the floor of the ladder")
    print(_line(f"{args.anchor} (anchor)", _per_side(data, args.rung, args.anchor)))

    if args.previous:
        prev = elo.get(args.previous)
        if prev is not None:
            print(f"\n  delta vs previous rung: {elo[args.rung] - prev:+.1f} Elo")
    anc = elo.get(args.anchor)
    if anc is not None:
        print(f"  delta vs anchor:        {elo[args.rung] - anc:+.1f} Elo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
