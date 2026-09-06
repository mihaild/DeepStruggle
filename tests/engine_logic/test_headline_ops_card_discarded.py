"""A card played through a headline event is spent, and leaves the hand it was played from.

The headline machinery relocates the two headline cards themselves. A card played *through* one
of them was nobody's to clear away: Grain Sales To Soviets draws a card at random from the
opponent's hand, moves it into the US hand and has the US play it, and when the Ops it granted
ran out the engine simply moved on to the second headline.

At turn 4's headline of ts-replayer game 14 the US headlines Grain Sales To Soviets, draws
Marshall Plan out of the USSR hand and coups Brazil with it. Marshall Plan was still in the US
hand afterwards: in the running when the USSR's Missile Envy asked for the highest Ops card --
the log says the US gave Muslim Revolution -- and playable a second time.
"""
from typing import List

import pytest
import ts_engine as ts

GRAIN_SALES = 67
MARSHALL_PLAN = 23
DUCK_AND_COVER = 4
NUCLEAR_TEST_BAN = 34
# 2 Ops and USSR-associated, so Grain Sales (also 2, and the US wins ties) resolves first, and
# with no scoring card in the US hand it finds nothing and returns at once.
CAMBRIDGE_FIVE = 104


def _at_headline() -> ts.GameState:
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    guard = 0
    while state.current_phase == ts.Phase.SETUP and guard < 500:
        guard += 1
        legal = [i for i, v in enumerate(ts.ActionMask.generate_flat_mask(state)) if v]
        assert legal
        ts.Engine.step_flat(state, legal[0])
    assert state.current_phase == ts.Phase.HEADLINE
    return state


def _set_hand(state: ts.GameState, player: ts.Player, cards: List[int]) -> None:
    loc = ts.CardLocation.HAND_US if player == ts.Player.US else ts.CardLocation.HAND_USSR
    for c in range(1, 111):
        if state.get_card_location(c) == loc:
            state.set_card_location(c, ts.CardLocation.DISCARD_PILE)
    for c in cards:
        state.set_card_location(c, loc)


def _grain_sales_headline() -> ts.GameState:
    """US headlines Grain Sales; the USSR holds exactly Marshall Plan to be drawn."""
    state = _at_headline()
    _set_hand(state, ts.Player.US, [GRAIN_SALES, DUCK_AND_COVER])
    _set_hand(state, ts.Player.USSR, [MARSHALL_PLAN, CAMBRIDGE_FIVE])
    for _ in range(2):
        p = state.ctx().decision_player
        card = GRAIN_SALES if p == ts.Player.US else CAMBRIDGE_FIVE
        ts.Engine.step_flat(state, card - 1)
    return state


def _play_until(state: ts.GameState, limit: int = 300) -> None:
    """Drive the headline out with whatever the engine offers first."""
    guard = 0
    while state.current_phase == ts.Phase.HEADLINE and guard < limit:
        guard += 1
        ctx = state.ctx()
        if (ctx.decision_player == ts.Player.NONE
                and ctx.decision_type == ts.DecisionType.ROLL_DIE):
            ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
            continue
        legal = [i for i, v in enumerate(ts.ActionMask.generate_flat_mask(state)) if v]
        if not legal:
            break
        ts.Engine.step_flat(state, legal[0])


def test_grain_sales_hands_the_drawn_card_to_the_us_to_play() -> None:
    """The setup this rests on: the card really does move into the US hand to be played."""
    state = _grain_sales_headline()
    assert int(state.ctx().resolving_card) == GRAIN_SALES
    assert state.get_card_location(MARSHALL_PLAN) in (
        ts.CardLocation.HAND_USSR, ts.CardLocation.HAND_US,
        ts.CardLocation.PEEKED_TEMP), "the drawn card is Grain Sales' to move"


def test_the_played_card_does_not_stay_in_hand() -> None:
    state = _grain_sales_headline()
    _play_until(state)
    assert state.get_card_location(MARSHALL_PLAN) not in (
        ts.CardLocation.HAND_US, ts.CardLocation.HAND_USSR), (
        "Marshall Plan was played, and a played card is not still held")


def test_the_headline_cards_themselves_are_still_cleared() -> None:
    """The new relocation must not stand in for the one that was already right."""
    state = _grain_sales_headline()
    _play_until(state)
    for card in (GRAIN_SALES, CAMBRIDGE_FIVE):
        assert state.get_card_location(card) not in (
            ts.CardLocation.HAND_US, ts.CardLocation.HAND_USSR,
            ts.CardLocation.HEADLINE_COMMITTED), f"card {card} was left committed"


def test_a_card_nobody_played_is_left_alone() -> None:
    """Only the card the Ops were spent on moves; the rest of the hand is untouched."""
    state = _grain_sales_headline()
    _play_until(state)
    assert state.get_card_location(DUCK_AND_COVER) == ts.CardLocation.HAND_US, (
        "the US still holds what it never played")
