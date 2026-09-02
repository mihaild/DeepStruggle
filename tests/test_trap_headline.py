"""Quagmire and Bear Trap constrain action rounds, not headlines.

The trap says what a player may do on their turn: play a 2+ Ops card and roll to escape. It
says nothing about the headline, where any card may be played and its event resolves normally.
The engine applied the restriction in both phases, so at turn 9's headline of ts-replayer game
105 the USSR could not headline Europe Scoring -- a scoring card is not a 2 Ops card -- though
the human did exactly that while trapped.
"""
from typing import List

import numpy as np
import pytest
import ts_engine as ts

EUROPE_SCORING = 2
DUCK_AND_COVER = 4          # 3 Ops, a legal trap escape


def _hand(state: ts.GameState, player: ts.Player, cards: List[int]) -> None:
    loc = ts.CardLocation.HAND_US if player == ts.Player.US else ts.CardLocation.HAND_USSR
    for c in range(1, 111):
        if state.get_card_location(c) == loc:
            state.set_card_location(c, ts.CardLocation.DISCARD_PILE)
    for c in cards:
        state.set_card_location(c, loc)


def _trapped_ussr(phase: ts.Phase) -> ts.GameState:
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    while (not ts.Engine.is_terminal(state)
           and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
        mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
        ts.Engine.step_flat(state, int(np.flatnonzero(mask)[0]))
    state.set_flag(ts.EffectBits.BEAR_TRAP_ACTIVE)
    state.current_phase = phase
    state.action_round = 1
    state.phasing_player = ts.Player.USSR
    _hand(state, ts.Player.USSR, [EUROPE_SCORING, DUCK_AND_COVER])
    state.ctx().decision_player = ts.Player.USSR
    state.ctx().decision_type = ts.DecisionType.SELECT_CARD
    return state


def _legal_cards(state: ts.GameState) -> List[int]:
    mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
    return [int(a) + 1 for a in np.flatnonzero(mask) if int(a) < 110]


def test_any_card_may_be_headlined_while_trapped() -> None:
    state = _trapped_ussr(ts.Phase.HEADLINE)
    legal = _legal_cards(state)
    assert EUROPE_SCORING in legal, "a scoring card may be headlined even under Bear Trap"
    assert DUCK_AND_COVER in legal


def test_the_headlined_event_resolves_rather_than_rolling_to_escape() -> None:
    state = _trapped_ussr(ts.Phase.HEADLINE)
    ts.Engine.step(state, ts.MicroAction(ts.DecisionType.SELECT_CARD, EUROPE_SCORING, 0, 0))
    assert state.ctx().decision_type != ts.DecisionType.ROLL_DIE, (
        "a headline is not a trap escape attempt")
    assert state.has_flag(ts.EffectBits.BEAR_TRAP_ACTIVE), (
        "and it does not clear the trap either")


def test_the_action_round_is_still_constrained() -> None:
    state = _trapped_ussr(ts.Phase.ACTION_ROUND)
    legal = _legal_cards(state)
    assert EUROPE_SCORING not in legal, (
        "in an action round the trap still forbids anything under 2 Ops")
    assert DUCK_AND_COVER in legal


@pytest.mark.parametrize("flag,player", [
    (ts.EffectBits.QUAGMIRE_ACTIVE, ts.Player.US),
    (ts.EffectBits.BEAR_TRAP_ACTIVE, ts.Player.USSR),
])
def test_both_traps_behave_the_same(flag: int, player: ts.Player) -> None:
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    while (not ts.Engine.is_terminal(state)
           and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
        mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
        ts.Engine.step_flat(state, int(np.flatnonzero(mask)[0]))
    state.set_flag(flag)
    state.current_phase = ts.Phase.HEADLINE
    state.phasing_player = player
    _hand(state, player, [EUROPE_SCORING, DUCK_AND_COVER])
    state.ctx().decision_player = player
    state.ctx().decision_type = ts.DecisionType.SELECT_CARD
    assert EUROPE_SCORING in _legal_cards(state)
