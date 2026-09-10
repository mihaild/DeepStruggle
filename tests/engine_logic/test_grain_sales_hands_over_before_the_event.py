"""A card Grain Sales hands over leaves the peek before its own Event can look at one.

Grain Sales draws a card out of the USSR hand and shows it to the US, who keeps it or returns
it. "Being looked at" is a location -- CardLocation::PEEKED_TEMP -- which is what lets the player
choosing see the card they are choosing about, without a special case in the observation.

That makes one ordering load-bearing. If the card is Our Man in Tehran, an Event that peeks five
cards of its own, and the US keeps it, then the card must be out of PEEKED_TEMP *before* its
Event fires. Otherwise it is sitting inside its own peek and can discard itself -- the card
disappearing in the middle of resolving itself.

The handler does move it, on the line before it switches to SELECT_PLAY_MODE, and Our Man in
Tehran peeks from the draw deck rather than from PEEKED_TEMP, so there are two independent
reasons this is safe. Neither is obvious from reading either handler on its own, which is why
it is pinned here.
"""

from __future__ import annotations

from typing import List

import pytest

import ts_engine as ts

GRAIN_SALES = 67
OUR_MAN_IN_TEHRAN = 108
ISRAEL = 23

PLAY_THE_CARD = 0
RETURN_THE_CARD = 1


def _peeked(state: ts.GameState) -> List[int]:
    return [c for c in range(1, 111)
            if state.get_card_location(c) == ts.CardLocation.PEEKED_TEMP]


def _grain_sales_offering(card: int) -> ts.GameState:
    """Grain Sales resolved so that `card` is the one drawn and offered to the US."""
    state = ts.GameState()
    ts.Engine.init_game(state, 4242)
    state.current_phase = ts.Phase.ACTION_ROUND
    state.turn = 8
    state.action_round = 1
    state.phasing_player = ts.Player.US
    # The US controls a Middle East country, so Our Man in Tehran's Event can occur.
    state.set_country(ISRAEL, 5, 0)

    # Exactly one card in the USSR hand, so the random draw has only one answer.
    for c in range(1, 111):
        if ts.in_hand_of(state.get_card_location(c), ts.Player.USSR):
            state.set_card_location(c, ts.CardLocation.DRAW_DECK)
    state.set_card_location(card, ts.hand_of(ts.Player.USSR))

    assert not ts.CardHandlers.trigger_event(state, GRAIN_SALES, ts.Player.US), (
        "Grain Sales should open a choice rather than finish"
    )
    assert _peeked(state) == [card], (
        f"Grain Sales should be offering exactly {card}; peek is {_peeked(state)}"
    )
    return state


def test_the_offered_card_is_the_one_being_looked_at() -> None:
    state = _grain_sales_offering(OUR_MAN_IN_TEHRAN)
    assert state.get_card_location(OUR_MAN_IN_TEHRAN) == ts.CardLocation.PEEKED_TEMP
    assert not ts.in_hand_of(state.get_card_location(OUR_MAN_IN_TEHRAN), ts.Player.USSR), (
        "the card has been drawn out of the USSR hand, so it is no longer in it"
    )


def test_keeping_our_man_in_tehran_does_not_put_it_inside_its_own_peek() -> None:
    state = _grain_sales_offering(OUR_MAN_IN_TEHRAN)
    ts.CardHandlers.handle_event_step(
        state, ts.MicroAction(ts.DecisionType.CHOOSE_BRANCH, PLAY_THE_CARD, 0, 0))

    # It is in the US hand now, and the peek it opens must not contain it.
    assert ts.in_hand_of(state.get_card_location(OUR_MAN_IN_TEHRAN), ts.Player.US)

    assert ts.Engine.step(
        state, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, 0, 0, 0)), (
        "playing the kept card as its Event should be accepted"
    )
    peek = _peeked(state)
    assert OUR_MAN_IN_TEHRAN not in peek, (
        f"Our Man in Tehran is inside its own peek {peek} and could discard itself"
    )
    assert ts.in_hand_of(state.get_card_location(OUR_MAN_IN_TEHRAN), ts.Player.US) \
        or state.get_card_location(OUR_MAN_IN_TEHRAN) in (
            ts.CardLocation.DISCARD_PILE, ts.CardLocation.REMOVED_FROM_GAME), (
        "the card should be in a hand or placed by its own resolution, never peeked"
    )


def test_returning_the_card_gives_it_back_and_remembers_it_was_seen() -> None:
    """Returned to the hand it came from, marked known: the US has seen it."""
    state = _grain_sales_offering(OUR_MAN_IN_TEHRAN)
    ts.CardHandlers.handle_event_step(
        state, ts.MicroAction(ts.DecisionType.CHOOSE_BRANCH, RETURN_THE_CARD, 0, 0))

    where = state.get_card_location(OUR_MAN_IN_TEHRAN)
    assert ts.in_hand_of(where, ts.Player.USSR), f"the card should go back; it is at {where}"
    assert ts.known_to_opponent(where), (
        "the US saw the card, and that is what the location records"
    )
    assert _peeked(state) == [], "nothing should be left being looked at"


def test_the_peek_is_empty_either_way() -> None:
    """Whichever branch the US takes, the card leaves PEEKED_TEMP in that same step."""
    for branch in (PLAY_THE_CARD, RETURN_THE_CARD):
        state = _grain_sales_offering(OUR_MAN_IN_TEHRAN)
        ts.CardHandlers.handle_event_step(
            state, ts.MicroAction(ts.DecisionType.CHOOSE_BRANCH, branch, 0, 0))
        assert _peeked(state) == [], (
            f"branch {branch} left {_peeked(state)} still being looked at"
        )
