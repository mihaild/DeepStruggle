"""Missile Envy's forced play constrains action rounds, not headlines.

The card says the opponent must use it for Operations "during their next action round", so a
headline is untouched: any card may be headlined and resolves as its Event. The engine applied
the restriction in every phase, so at turn 6's headline of ts-replayer game 114 the USSR --
holding Missile Envy from a turn 5 exchange -- could headline nothing else, though the human
headlined Portuguese Empire Crumbles.
"""
from typing import List

import numpy as np
import ts_engine as ts

MISSILE_ENVY = 49
PORTUGUESE_EMPIRE_CRUMBLES = 52


def _state(phase: ts.Phase) -> ts.GameState:
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    while (not ts.Engine.is_terminal(state)
           and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
        mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
        ts.Engine.step_flat(state, int(np.flatnonzero(mask)[0]))
    state.current_phase = phase
    state.action_round = 1
    state.phasing_player = ts.Player.USSR
    for c in range(1, 111):
        if ts.in_hand_of(state.get_card_location(c), ts.Player.USSR):
            state.set_card_location(c, ts.CardLocation.DISCARD_PILE)
    for c in (MISSILE_ENVY, PORTUGUESE_EMPIRE_CRUMBLES):
        state.set_card_location(c, ts.hand_of(ts.Player.USSR))
    state.forced_card_player = ts.Player.USSR
    state.forced_card_id = MISSILE_ENVY
    state.ctx().decision_player = ts.Player.USSR
    state.ctx().decision_type = ts.DecisionType.SELECT_CARD
    return state


def _legal_cards(state: ts.GameState) -> List[int]:
    mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
    return [int(a) + 1 for a in np.flatnonzero(mask) if int(a) < 110]


def test_any_card_may_be_headlined_while_holding_a_forced_card() -> None:
    legal = _legal_cards(_state(ts.Phase.HEADLINE))
    assert PORTUGUESE_EMPIRE_CRUMBLES in legal, (
        "a headline is not the action round Missile Envy reserves")
    assert MISSILE_ENVY in legal, "and the forced card itself stays available"


def test_the_action_round_is_still_forced() -> None:
    legal = _legal_cards(_state(ts.Phase.ACTION_ROUND))
    assert legal == [MISSILE_ENVY], (
        f"the action round must offer only the forced card, got {legal}")


def test_a_headlined_card_may_still_be_played_as_its_event() -> None:
    """The Ops-only restriction on the forced card must not leak into the headline either."""
    state = _state(ts.Phase.HEADLINE)
    ts.Engine.step(state, ts.MicroAction(
        ts.DecisionType.SELECT_CARD, PORTUGUESE_EMPIRE_CRUMBLES, 0, 0))
    if state.ctx().decision_type == ts.DecisionType.SELECT_PLAY_MODE:
        mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
        assert mask[110 + int(ts.Resolution.EVENT)] == 1, (
            "a headlined card resolves as its Event")
