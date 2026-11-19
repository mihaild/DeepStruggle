"""A card Grain Sales draws fires its Event whichever way the US spends its Ops.

The USSR plays Grain Sales for its Event, the US takes a USSR card from its hand and plays it for
Operations: the card's Event is still owed to the USSR. It was owed correctly when the Ops were
Influence and dropped when they were a Coup or a Realignment. Those finish from the die roll, where
the decision is nobody's, and `settle_card_played_on_pushed_frame` took the spender from it: an
opponent's card is judged against a player, and NONE has no opponent. Quagmire went to the discard
pile unplayed, NORAD stayed, and the US was never made to discard (T8 AR1 of a game played on
twilightstruggle.app).
"""
from typing import List

import pytest

import ts_engine as ts
from bindings.action_encoder import ActionEncoder
from tests.engine_logic.test_grain_sales_through_an_event import GRAIN_SALES, PLAY_EVENT, _at_action_round

QUAGMIRE = 42
NASSER = 15           # a 1 Op card, which Quagmire's discard may not use
DUCK_AND_COVER = 4    # a 3 Op card
EB = ts.EffectBits
EVENT_FIRST = ActionEncoder.PLAY_MODE_OFFSET + int(ts.Resolution.EVENT)
SPACE = ActionEncoder.PLAY_MODE_OFFSET + int(ts.Resolution.SPACE)
INFLUENCE = ActionEncoder.PLAY_MODE_OFFSET + int(ts.Resolution.OPS_INFLUENCE)
COUP = ActionEncoder.PLAY_MODE_OFFSET + int(ts.Resolution.OPS_COUP)
REALIGN = ActionEncoder.PLAY_MODE_OFFSET + int(ts.Resolution.OPS_REALIGN)


def _legal(state: ts.GameState) -> List[int]:
    return [i for i, v in enumerate(ts.ActionMask.generate_flat_mask(state)) if v]


def _us_plays_the_drawn_quagmire(mode: int) -> ts.GameState:
    """USSR: Grain Sales (Event) draws Quagmire; US plays it as `mode`; play on to the US's next round."""
    state = _at_action_round()
    state.phasing_player = ts.Player.USSR
    state.ctx().decision_player = ts.Player.USSR
    state.set_card_location(GRAIN_SALES, ts.hand_of(ts.Player.USSR))
    state.set_card_location(QUAGMIRE, ts.hand_of(ts.Player.USSR))
    state.set_flag(EB.NORAD_ACTIVE)
    for card in (DUCK_AND_COVER, NASSER):
        state.set_card_location(card, ts.hand_of(ts.Player.US))
    ts.Engine.step_flat(state, GRAIN_SALES - 1)
    ts.Engine.step_flat(state, PLAY_EVENT)
    assert ts.in_hand_of(state.get_card_location(QUAGMIRE), ts.Player.US)
    ts.Engine.step_flat(state, mode)
    for _ in range(60):
        ctx = state.ctx()
        if ctx.decision_player == ts.Player.NONE and ctx.decision_type == ts.DecisionType.ROLL_DIE:
            ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
        elif state.phasing_player == ts.Player.US and ctx.decision_type == ts.DecisionType.SELECT_CARD:
            return state
        else:
            ts.Engine.step_flat(state, _legal(state)[0])
    raise AssertionError("the US's next action round was never reached")


@pytest.mark.parametrize("mode", [EVENT_FIRST, INFLUENCE, COUP, REALIGN],
                         ids=["event", "influence", "coup", "realign"])
def test_the_drawn_cards_event_fires_however_its_ops_are_spent(mode: int) -> None:
    state = _us_plays_the_drawn_quagmire(mode)
    assert state.has_flag(EB.QUAGMIRE_ACTIVE)
    assert not state.has_flag(EB.NORAD_ACTIVE)
    # A starred Event that occurred leaves the game.
    assert state.get_card_location(QUAGMIRE) == ts.CardLocation.REMOVED_FROM_GAME


def test_the_us_then_must_discard_a_card_of_two_ops_or_more() -> None:
    state = _us_plays_the_drawn_quagmire(COUP)
    assert _legal(state) == [DUCK_AND_COVER - 1]


def test_space_race_with_the_drawn_card_still_does_not_fire_its_event() -> None:
    """The Event is owed to Ops. A Space Race attempt has none, so Quagmire is only discarded."""
    state = _us_plays_the_drawn_quagmire(SPACE)
    assert not state.has_flag(EB.QUAGMIRE_ACTIVE)
    assert state.has_flag(EB.NORAD_ACTIVE)
    assert state.get_card_location(QUAGMIRE) == ts.CardLocation.DISCARD_PILE
