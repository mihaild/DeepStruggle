"""Playing an opponent's card for Ops resolves its event exactly once.

An opponent-associated card played for Operations fires the owner's event. With EVENT_FIRST
timing the engine pushes a frame for the Ops half, runs the event, then returns. A pushed
frame starts zeroed, and zero is TimingBranch::OPS_FIRST, so advance_after_ops saw
"OPS_FIRST on an opponent card" once the event's own Ops finished and fired the event a
second time.

Observed in review_91004: the USSR played Grain Sales to Soviets for Ops, and the US took
the "return the card, take 2 Ops" branch and couped Cuba -- twice -- driving DEFCON from 3
to 1 and ending the game. The USSR also received none of the Ops it had played the card
for.
"""

from typing import List, Tuple

import numpy as np
import pytest
import ts_engine as ts

from ai.eval.positions import PLAY_MODE_ACTION, PositionBuilder
from bindings.action_encoder import ActionEncoder

GRAIN_SALES, COMECON, WARSAW_PACT = 67, 14, 16
# P17: the timing is carried by the resolution itself. On an opponent's card EVENT (110) is
# event-first and any OPS_* is ops-first, so these name resolutions rather than a separate
# timing node.
TIMING_OPS_FIRST, TIMING_EVENT_FIRST = 112, 110


def _play_for_ops(timing: int) -> Tuple[int, int]:
    """USSR plays Grain Sales for Ops; returns (event offers, US op-mode nodes)."""
    st = PositionBuilder(
        hand=[GRAIN_SALES, COMECON, WARSAW_PACT], side=ts.Player.USSR,
        turn=8, defcon=4, opponent_hand=[4, 23, 34],
    ).build()
    ts.Engine.step_flat(st, GRAIN_SALES - 1)

    legal = np.flatnonzero(ActionEncoder.get_legal_mask(st))
    assert timing in legal.tolist(), f"resolution {timing} not offered"
    ts.Engine.step_flat(st, timing)

    offers = us_ops = 0
    for _ in range(30):
        if ts.Engine.is_terminal(st):
            break
        ctx = st.ctx()
        legal = np.flatnonzero(ActionEncoder.get_legal_mask(st))
        if not len(legal):
            break
        if (ctx.decision_type == ts.DecisionType.CHOOSE_BRANCH
                and int(ctx.resolving_card) == GRAIN_SALES):
            offers += 1
            pick = int(legal[1]) if len(legal) > 1 else int(legal[0])
        else:
            pick = int(legal[0])
            if (ctx.decision_type == ts.DecisionType.SELECT_OP_MODE
                    and ctx.decision_player == ts.Player.US):
                us_ops += 1
        ts.Engine.step_flat(st, pick)
        while (not ts.Engine.is_terminal(st)
               and st.ctx().decision_player == ts.Player.NONE
               and st.ctx().decision_type == ts.DecisionType.ROLL_DIE):
            ts.Engine.step(st, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
    return offers, us_ops


@pytest.mark.parametrize("timing,label", [
    (TIMING_EVENT_FIRST, "EVENT_FIRST"),
    (TIMING_OPS_FIRST, "OPS_FIRST"),
])
def test_opponent_event_resolves_once(timing: int, label: str) -> None:
    offers, us_ops = _play_for_ops(timing)
    assert offers == 1, f"{label}: Grain Sales offered its branch {offers} times, expected 1"
    assert us_ops <= 1, f"{label}: the US got {us_ops} Ops actions from one event"
