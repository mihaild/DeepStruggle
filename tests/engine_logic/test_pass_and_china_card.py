"""When a player may pass an action round, and when they may not.

A player holding cards must play one; passing is only for a player who has run out. The China
Card is the exception: it is not part of the hand a player is required to spend, so holding it
and nothing else leaves a real choice between playing it and passing.

The engine offered no pass to a player whose only card was the China Card, which made a legal
skip impossible to reconstruct and, in ordinary play, forced the card out of a hand that meant
to keep it.
"""
from typing import List, Tuple

import pytest
import ts_engine as ts

CHINA_CARD = 6
DUCK_AND_COVER = 4
FIVE_YEAR_PLAN = 5
PASS = 211


def _at_an_action_round() -> ts.GameState:
    """A game played forward far enough to be asked for an action round card."""
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    guard = 0
    while state.current_phase != ts.Phase.ACTION_ROUND and guard < 500:
        guard += 1
        ctx = state.ctx()
        if (ctx.decision_player == ts.Player.NONE
                and ctx.decision_type == ts.DecisionType.ROLL_DIE):
            ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
            continue
        legal = [i for i, v in enumerate(ts.ActionMask.generate_flat_mask(state)) if v]
        assert legal, "the game stalled before an action round"
        ts.Engine.step_flat(state, legal[0])
    assert state.current_phase == ts.Phase.ACTION_ROUND
    assert state.ctx().decision_type == ts.DecisionType.SELECT_CARD
    return state


def _offered(hand: List[int], china: bool) -> Tuple[List[int], bool]:
    """The cards offered and whether passing is, for a mover holding `hand`."""
    state = _at_an_action_round()
    mover = state.ctx().decision_player
    loc = ts.CardLocation.HAND_US if mover == ts.Player.US else ts.CardLocation.HAND_USSR
    for c in range(1, 111):
        if state.get_card_location(c) == loc:
            state.set_card_location(c, ts.CardLocation.DISCARD_PILE)
    for c in hand:
        state.set_card_location(c, loc)
    if china:
        state.china_card_holder = mover
        state.china_card_playable = True
    else:
        state.china_card_holder = (
            ts.Player.US if mover == ts.Player.USSR else ts.Player.USSR)
    mask = ts.ActionMask.generate_flat_mask(state)
    return [i + 1 for i in range(110) if mask[i]], bool(mask[PASS])


def test_a_player_who_has_run_out_may_pass() -> None:
    """At turn 10 AR7 of replay 105 the US has no cards left and skips the round."""
    cards, can_pass = _offered([], china=False)
    assert cards == []
    assert can_pass


def test_a_player_holding_only_the_china_card_may_pass_or_play_it() -> None:
    cards, can_pass = _offered([], china=True)
    assert cards == [CHINA_CARD]
    assert can_pass, "the China Card is a choice, not an obligation"


def test_a_player_holding_a_card_may_not_pass() -> None:
    cards, can_pass = _offered([DUCK_AND_COVER], china=False)
    assert cards == [DUCK_AND_COVER]
    assert not can_pass


def test_the_china_card_does_not_excuse_a_player_who_holds_other_cards() -> None:
    cards, can_pass = _offered([DUCK_AND_COVER], china=True)
    assert sorted(cards) == sorted([DUCK_AND_COVER, CHINA_CARD])
    assert not can_pass


def test_several_cards_still_leave_no_pass() -> None:
    cards, can_pass = _offered([DUCK_AND_COVER, FIVE_YEAR_PLAN], china=True)
    assert sorted(cards) == sorted([DUCK_AND_COVER, FIVE_YEAR_PLAN, CHINA_CARD])
    assert not can_pass


def test_passing_gives_the_round_to_the_other_player() -> None:
    """The pass has to be a move the engine accepts, not merely a mask bit."""
    state = _at_an_action_round()
    mover = state.ctx().decision_player
    loc = ts.CardLocation.HAND_US if mover == ts.Player.US else ts.CardLocation.HAND_USSR
    for c in range(1, 111):
        if state.get_card_location(c) == loc:
            state.set_card_location(c, ts.CardLocation.DISCARD_PILE)
    state.china_card_holder = ts.Player.US if mover == ts.Player.USSR else ts.Player.USSR
    assert ts.ActionMask.generate_flat_mask(state)[PASS]
    ts.Engine.step_flat(state, PASS)
    assert state.ctx().decision_player != mover or state.current_phase != ts.Phase.ACTION_ROUND
