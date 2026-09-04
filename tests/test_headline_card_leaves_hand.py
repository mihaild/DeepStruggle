"""A headlined card is out of hand before either headline resolves.

Both headlines are played at once, face down, and only then resolved in Ops order. Neither
card is in a hand while the other plays out, so a card that reads a hand must not find it
there.

The engine moved a headline card's location only after its own event had run, which left the
*second* headline sitting in its owner's hand for the whole of the first one's resolution. At
turn 2's headline of replay 144 the US headlines Middle East Scoring and the USSR headlines
The Cambridge Five, which resolves first at 2 Ops against a scoring card's 0 -- and it found
the scoring card still in the US hand and asked the USSR where to place an Influence. The log
reads "US has no cards to reveal".

574 headlines across 236 of the 287 recorded games pair a hand-reading card with another
headline. Missile Envy is the sharp case: it takes the opponent's highest Ops card, so the
opponent's own headline was in the running to be handed over.
"""
from typing import List, Tuple

import pytest
import ts_engine as ts

CAMBRIDGE_FIVE = 104
MIDDLE_EAST_SCORING = 3
MISSILE_ENVY = 49
DUCK_AND_COVER = 4
NUCLEAR_TEST_BAN = 34


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


def _headline(us_card: int, ussr_card: int,
              us_hand: List[int], ussr_hand: List[int]) -> ts.GameState:
    state = _at_headline()
    _set_hand(state, ts.Player.US, us_hand + [us_card])
    _set_hand(state, ts.Player.USSR, ussr_hand + [ussr_card])
    for _ in range(2):
        p = state.ctx().decision_player
        ts.Engine.step_flat(state, (us_card if p == ts.Player.US else ussr_card) - 1)
    return state


def test_the_location_exists_and_is_neither_hand_nor_a_pile() -> None:
    """It cannot be the discard pile: Star Wars and SALT Negotiations read that."""
    assert ts.CardLocation.HEADLINE_COMMITTED not in (
        ts.CardLocation.HAND_US, ts.CardLocation.HAND_USSR,
        ts.CardLocation.DISCARD_PILE, ts.CardLocation.REMOVED_FROM_GAME,
        ts.CardLocation.DRAW_DECK, ts.CardLocation.ONGOING_EVENT,
        ts.CardLocation.PEEKED_TEMP, ts.CardLocation.UNAVAILABLE)


def test_the_cambridge_five_does_not_see_the_headlined_scoring_card() -> None:
    """Replay 144 turn 2: "US has no cards to reveal"."""
    state = _headline(us_card=MIDDLE_EAST_SCORING, ussr_card=CAMBRIDGE_FIVE,
                      us_hand=[DUCK_AND_COVER], ussr_hand=[NUCLEAR_TEST_BAN])
    ctx = state.ctx()
    assert int(ctx.resolving_card) != CAMBRIDGE_FIVE, (
        "it found a scoring card that had already been played")
    assert ctx.decision_type != ts.DecisionType.POINT_NODE


def test_the_cambridge_five_still_sees_a_scoring_card_that_is_really_held() -> None:
    """The fix must not make the event inert -- only blind to the headline."""
    state = _headline(us_card=DUCK_AND_COVER, ussr_card=CAMBRIDGE_FIVE,
                      us_hand=[MIDDLE_EAST_SCORING], ussr_hand=[NUCLEAR_TEST_BAN])
    ctx = state.ctx()
    assert int(ctx.resolving_card) == CAMBRIDGE_FIVE
    assert ctx.decision_type == ts.DecisionType.POINT_NODE
    assert ctx.decision_player == ts.Player.USSR


def test_a_headlined_card_reaches_its_ordinary_destination_afterwards() -> None:
    """The committed location is a waypoint, not where the card ends up."""
    state = _headline(us_card=MIDDLE_EAST_SCORING, ussr_card=CAMBRIDGE_FIVE,
                      us_hand=[DUCK_AND_COVER], ussr_hand=[NUCLEAR_TEST_BAN])
    guard = 0
    while state.current_phase == ts.Phase.HEADLINE and guard < 200:
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
    for card in (MIDDLE_EAST_SCORING, CAMBRIDGE_FIVE):
        assert state.get_card_location(card) != ts.CardLocation.HEADLINE_COMMITTED, (
            f"card {card} was left committed after the headline resolved")


def test_missile_envy_does_not_take_the_opponents_headline() -> None:
    """It takes the opponent's highest Ops card, and a headline is not theirs to give.

    Nuclear Test Ban is 4 Ops and Duck and Cover 1. Headline Nuclear Test Ban and the card
    Missile Envy can reach is Duck and Cover.
    """
    state = _headline(us_card=MISSILE_ENVY, ussr_card=NUCLEAR_TEST_BAN,
                      us_hand=[], ussr_hand=[DUCK_AND_COVER])
    assert state.get_card_location(NUCLEAR_TEST_BAN) != ts.CardLocation.HAND_USSR, (
        "the USSR headlined it, so it is no longer theirs to hand over")
