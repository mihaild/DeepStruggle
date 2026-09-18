"""Named opening setups that can be forced on an agent in place of its own placements.

A trained policy fights where it was placed and rarely opens a new front, so its own setup
decides much of what the rest of the game can look like. Forcing a known opening is how that gets
separated: same agent, same engine, different starting board.

One definition, used by both the probe that measures the effect (`ai/eval/forced_setup.py`) and
the replay generators that produce games to watch (`tools/lib/self_play.py`,
`tools/play_match.py`). If these drifted, a replay labelled "the human opening" would not be the
opening the numbers were measured on.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts
from bindings.action_encoder import ActionEncoder

#: Flat action index of country `cid` at a POINT_NODE (`bindings/action_encoder.py`).
NODE_OFFSET = ActionEncoder.NODE_OFFSET

#: 6 USSR placements, then 7 US in Western Europe, then 2 US bonus.
SETUP_DECISIONS = 15

WEST_GERMANY, ITALY, EAST_GERMANY, POLAND, YUGOSLAVIA, IRAN = 7, 10, 14, 15, 18, 25

#: The standard human opening, as (country, points) in placement order. The USSR overcontrols
#: Poland and East Germany by a point each; the US takes West Germany and Italy with a buffer and
#: spends its two bonus points on Iran.
USSR_OPENING: Sequence[Tuple[int, int]] = ((EAST_GERMANY, 1), (POLAND, 4), (YUGOSLAVIA, 1))
US_OPENING: Sequence[Tuple[int, int]] = ((WEST_GERMANY, 4), (ITALY, 3), (IRAN, 2))


def expand(opening: Sequence[Tuple[int, int]]) -> List[int]:
    """(country, points) pairs to one country id per influence point, in placement order."""
    return [cid for cid, n in opening for _ in range(n)]


OPENINGS: Dict[str, Dict[str, List[int]]] = {
    "human": {"US": expand(US_OPENING), "USSR": expand(USSR_OPENING)},
}


def scripted_setup_index(state: ts.GameState, side: str, opening: str,
                         cursor: Dict[str, int]) -> Optional[int]:
    """The flat action index of `side`'s next scripted placement, or None if there is none.

    None means "let the agent decide": either setup is over, or this side's script is spent.
    An *illegal* scripted placement raises instead, because falling through to the agent would
    produce a partly-forced opening -- which is neither opening, and would be reported as one.

    `cursor` is per-side and mutated in place; pass `{"US": 0, "USSR": 0}` at the start of a game.
    """
    if state.current_phase != ts.Phase.SETUP:
        return None
    script = OPENINGS[opening][side]
    k = cursor[side]
    if k >= len(script):
        return None
    idx = NODE_OFFSET + script[k]
    mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
    if not mask[idx]:
        raise RuntimeError(
            f"{side} placement {k} of the '{opening}' opening (country {script[k]}, flat action "
            f"{idx}) is not legal at this setup decision -- refusing to fall back to the agent.")
    cursor[side] = k + 1
    return idx


def acting_side(state: ts.GameState) -> str:
    """Which side is making this decision.

    The decision player, never `phasing_player`: the USSR is the phasing player for the whole of
    setup, so keying off it sends every US placement down the USSR script.
    """
    return "US" if state.ctx().decision_player == ts.Player.US else "USSR"
