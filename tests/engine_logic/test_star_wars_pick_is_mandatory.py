"""Star Wars (#85): the pick is mandatory, and fizzles only on an empty discard pile.

Same defect as Aldrich Ames -- `trigger_star_wars` opened a SELECT_CARD without setting
`allow_early_stop`, so it inherited the previous decision's value, and an inherited 1 let the US
decline a pick the rules require.

It differs in the fizzle: the handler returns before opening any decision when the discard pile
holds no non-scoring card, so unlike Aldrich Ames there is never a legal-but-empty decision to
fall back from. Once the decision exists there is always at least one card, and pass must never
be offered.
"""

import numpy as np
import pytest
import ts_engine
from bindings.action_encoder import ActionEncoder

STAR_WARS = 85
PASS_INDEX = ActionEncoder.CONFIRM_DONE_INDEX
def _state(discard: list[int], us_ahead: bool = True) -> ts_engine.GameState:
    state = ts_engine.GameState()
    ts_engine.Engine.init_game(state, 8888)
    for cid in range(1, 111):
        state.set_card_location(cid, ts_engine.CardLocation.DRAW_DECK)
    for cid in discard:
        state.set_card_location(cid, ts_engine.CardLocation.DISCARD_PILE)
    state.us_space_track = 5 if us_ahead else 0
    state.ussr_space_track = 0 if us_ahead else 5
    # The bug was an inherited flag, so every case starts with it set.
    state.ctx().allow_early_stop = 1
    return state


def test_the_pick_cannot_be_declined() -> None:
    state = _state(discard=[55, 75])
    fizzled = ts_engine.CardHandlers.trigger_event(state, STAR_WARS, ts_engine.Player.US)
    assert not fizzled, "with cards in the discard pile the event must open a decision"

    mask = np.asarray(ts_engine.ActionMask.generate_flat_mask(state))
    assert state.ctx().decision_player == ts_engine.Player.US
    assert state.ctx().resolving_card == STAR_WARS
    assert mask[PASS_INDEX] == 0, "CONFIRM_DONE is offered, so a mandatory pick can be declined"
    assert int(mask.sum()) >= 1


def test_the_handler_clears_an_inherited_early_stop() -> None:
    state = _state(discard=[55])
    ts_engine.CardHandlers.trigger_event(state, STAR_WARS, ts_engine.Player.US)
    assert state.ctx().allow_early_stop == 0


def test_it_fizzles_on_an_empty_discard_pile() -> None:
    state = _state(discard=[])
    fizzled = ts_engine.CardHandlers.trigger_event(state, STAR_WARS, ts_engine.Player.US)
    assert fizzled, "no non-scoring card in the discard pile means the event does nothing"
    assert state.ctx().resolving_card != STAR_WARS


def test_scoring_cards_in_the_discard_pile_do_not_count() -> None:
    # A discard pile of nothing but scoring cards is an empty one for this event.
    scoring = [c for c in range(1, 111) if ts_engine.CardData.get_card_info(c).get("is_scoring")]
    assert scoring, "expected the deck to contain scoring cards"
    state = _state(discard=scoring[:3])
    assert ts_engine.CardHandlers.trigger_event(state, STAR_WARS, ts_engine.Player.US)


def test_it_does_nothing_when_the_us_is_not_ahead_on_the_space_race() -> None:
    state = _state(discard=[55, 75], us_ahead=False)
    assert ts_engine.CardHandlers.trigger_event(state, STAR_WARS, ts_engine.Player.US)
    assert state.ctx().resolving_card != STAR_WARS
