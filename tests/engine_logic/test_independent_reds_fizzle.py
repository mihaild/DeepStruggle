"""Independent Reds must not open a decision it has no target for.

Two conditions used to decide the same thing for this card, and they disagreed:

* `trigger_independent_reds` (`early_war.cpp`) finished the event early only when **no** target
  country had any USSR Influence at all -- `ussr_influence > 0`.
* the target mask (`card_dispatcher.cpp`) offers a country only where the USSR is **ahead** --
  `ussr_influence > us_influence`.

The mask is the correct test. The card reads "Add US Influence to ... to match the USSR Influence
in that country", so a country where the US already equals or exceeds the USSR has nothing to add.

In the gap between the two the engine opened a POINT_NODE with an empty mask. The safety net in
`generate_flat_mask_212` then inserted CONFIRM_DONE so the game was not stuck, and printed
"POINT_NODE with no legal target and no early stop" -- which is where that message in dataset
generation and replay conversion came from. What it cost was a decision that is not a decision:
the policy was asked to choose from one no-op action, and it landed in the training stream as
though a choice had been made. Four such nodes appeared in 5,000 generated games.

The trigger now uses the mask's condition, so the event fizzles before opening anything. These
are the regression.
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


def test_no_decision_is_opened_when_the_us_already_matches_every_target() -> None:
    """USSR present everywhere, but never ahead: there is nothing to add, so nothing to decide."""
    state = _state_with(us=2, ussr=1)
    done = ts.CardHandlers.trigger_event(state, INDEPENDENT_REDS, ts.Player.US)
    assert done, (
        "the event should finish immediately: no target has more USSR Influence than US, so "
        f"there is nothing to match; instead it opened {state.ctx().decision_type} with "
        f"legal actions {_legal(state)}")


def test_the_fizzle_leaves_no_independent_reds_node_behind() -> None:
    """Finishing must mean no node at all, not a node whose only action is CONFIRM_DONE.

    Before the fix the mask came back as exactly `[CONFIRM_DONE]` here -- the safety net's
    no-op -- and that is what made it into the training stream as a decision.
    """
    state = _state_with(us=2, ussr=1)
    assert ts.CardHandlers.trigger_event(state, INDEPENDENT_REDS, ts.Player.US)
    assert state.ctx().resolving_card != INDEPENDENT_REDS
    assert _legal(state) != [CONFIRM_DONE]


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
