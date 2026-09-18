"""Aldrich Ames Remix: the USSR must discard a card from the US hand, not may.

`trigger_aldrich_ames` opened a SELECT_CARD decision without setting `allow_early_stop`, so it
inherited whatever the previous decision left there. An inherited 1 makes `generate_flat_mask`
offer CONFIRM_DONE alongside the cards, turning a required discard into an optional one -- a
self-play game was seen being offered both US cards plus a pass, and taking the pass.

The one case where declining is correct is an empty US hand: there is nothing to discard, the
event mask finds no legal card, and the mask falls back to pass on its own.
"""

import numpy as np
import pytest
import ts_engine
from bindings.action_encoder import ActionEncoder

ALDRICH_AMES = 98
PASS_INDEX = ActionEncoder.CONFIRM_DONE_INDEX
def _state_with_us_hand(cards: list[int]) -> ts_engine.GameState:
    state = ts_engine.GameState()
    ts_engine.Engine.init_game(state, 4242)
    # Clear every hand, then deal the US exactly what the test asks for.
    for cid in range(1, 111):
        state.set_card_location(cid, ts_engine.CardLocation.DRAW_DECK)
    for cid in cards:
        state.set_card_location(cid, ts_engine.CardLocation.HAND_US_KNOWN)
    return state


def _trigger(state: ts_engine.GameState) -> np.ndarray:
    # Set early stop first: the bug was that the handler inherited it rather than clearing it,
    # so a test that starts from 0 would pass against the unfixed engine.
    state.ctx().allow_early_stop = 1
    ts_engine.CardHandlers.trigger_event(state, ALDRICH_AMES, ts_engine.Player.USSR)
    return np.asarray(ts_engine.ActionMask.generate_flat_mask(state))


def test_the_discard_cannot_be_declined_when_the_us_holds_cards() -> None:
    state = _state_with_us_hand([55, 75])
    mask = _trigger(state)

    assert state.ctx().decision_player == ts_engine.Player.USSR
    assert state.ctx().resolving_card == ALDRICH_AMES
    assert mask[PASS_INDEX] == 0, (
        "CONFIRM_DONE is offered, so the USSR can decline a mandatory discard")
    assert mask[55 - 1] == 1 and mask[75 - 1] == 1, "both US cards must be discardable"
    assert int(mask.sum()) == 2, f"only the US hand should be legal, got {np.flatnonzero(mask)}"


def test_the_handler_clears_an_inherited_early_stop() -> None:
    """The specific defect: the flag was inherited from the previous decision."""
    state = _state_with_us_hand([55])
    state.ctx().allow_early_stop = 1
    ts_engine.CardHandlers.trigger_event(state, ALDRICH_AMES, ts_engine.Player.USSR)
    assert state.ctx().allow_early_stop == 0


def test_an_empty_us_hand_falls_back_to_pass() -> None:
    # Nothing to discard. The mask must still offer something legal rather than dead-end.
    state = _state_with_us_hand([])
    mask = _trigger(state)
    assert mask[PASS_INDEX] == 1, "with no US cards the decision must be passable"
    assert int(mask.sum()) == 1


@pytest.mark.parametrize("hand", [[55], [55, 75], [4, 55, 75, 83]])
def test_every_card_in_the_us_hand_is_offered(hand: list[int]) -> None:
    mask = _trigger(_state_with_us_hand(hand))
    assert sorted(int(i) + 1 for i in np.flatnonzero(mask)) == sorted(hand)
