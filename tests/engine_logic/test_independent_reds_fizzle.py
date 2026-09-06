"""Independent Reds must not open a decision it has no target for.

Two conditions decide the same thing for this card and they disagree:

* `early_war.cpp:227` (`trigger_independent_reds`) finishes the event early only when **no**
  target country has any USSR Influence at all -- `ussr_influence > 0`.
* `card_dispatcher.cpp:1576` (the target mask) offers a country only where the USSR is **ahead**
  -- `ussr_influence > us_influence`.

The mask is the correct test. The card reads "Add US Influence to ... to match the USSR Influence
in that country", so a country where the US already equals or exceeds the USSR has nothing to add.

In the gap between the two the engine opens a POINT_NODE with an empty mask. The safety net in
`generate_flat_mask_212` then inserts CONFIRM_DONE so the game is not stuck, and prints
"POINT_NODE with no legal target and no early stop" -- which is where that message in dataset
generation and replay conversion comes from. The cost is a decision that is not a decision: the
policy is asked to choose from one no-op action, and it lands in the training stream as though a
choice had been made.

The fix is to give the trigger the mask's condition, so the event fizzles before opening anything.
Per AGENTS.md invariant 11 that is the user's call, so this is marked xfail rather than applied.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

import ts_engine as ts

INDEPENDENT_REDS = 22
TARGETS = ("Yugoslavia", "Romania", "Bulgaria", "Hungary", "Czechoslovakia")
CONFIRM_DONE = 211


def _cid(name: str) -> int:
    for cid in range(84):
        if ts.MapData.get_country_info(cid)["name"].lower() == name.lower():
            return cid
    raise AssertionError(f"country {name!r} not found")


def _state_with(us: int, ussr: int) -> ts.GameState:
    """A live position where every Independent Reds target holds `us`/`ussr` Influence."""
    state = ts.GameState()
    ts.Engine.init_game(state, 5)
    state.current_phase = ts.Phase.ACTION_ROUND
    for name in TARGETS:
        state.set_country(_cid(name), us, ussr)
    return state


def _legal(state: ts.GameState) -> List[int]:
    return [int(a) for a in np.flatnonzero(np.array(ts.get_flat_action_mask(state)))]


@pytest.mark.xfail(strict=True,
                   reason="trigger_independent_reds guards on ussr>0, the mask requires ussr>us")
def test_no_decision_is_opened_when_the_us_already_matches_every_target() -> None:
    """USSR present everywhere, but never ahead: there is nothing to add, so nothing to decide."""
    state = _state_with(us=2, ussr=1)
    done = ts.CardHandlers.trigger_event(state, INDEPENDENT_REDS, ts.Player.US)
    assert done, (
        "the event should finish immediately: no target has more USSR Influence than US, so "
        f"there is nothing to match; instead it opened {state.ctx().decision_type} with "
        f"legal actions {_legal(state)}")


def test_the_gap_is_real_and_this_is_what_it_looks_like() -> None:
    """Pins the current behaviour, so the xfail above cannot pass for an unrelated reason."""
    state = _state_with(us=2, ussr=1)
    ts.CardHandlers.trigger_event(state, INDEPENDENT_REDS, ts.Player.US)
    ctx = state.ctx()
    assert ctx.decision_type == ts.DecisionType.POINT_NODE
    assert ctx.allow_early_stop == 0
    # Only the safety net's CONFIRM_DONE, which is not a choice.
    assert _legal(state) == [CONFIRM_DONE]


def test_an_empty_board_still_fizzles_cleanly() -> None:
    """The case the trigger's own guard already covers: no USSR Influence anywhere."""
    state = _state_with(us=0, ussr=0)
    assert ts.CardHandlers.trigger_event(state, INDEPENDENT_REDS, ts.Player.US)


def test_a_real_target_still_opens_a_real_choice() -> None:
    """Where the USSR is ahead the event must still work."""
    state = _state_with(us=0, ussr=0)
    state.set_country(_cid("Yugoslavia"), 1, 3)
    done = ts.CardHandlers.trigger_event(state, INDEPENDENT_REDS, ts.Player.US)
    assert not done
    legal = _legal(state)
    assert legal and legal != [CONFIRM_DONE], f"expected a country to choose, got {legal}"
