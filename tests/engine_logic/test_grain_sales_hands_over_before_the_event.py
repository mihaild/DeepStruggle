"""A card Grain Sales hands over is in the US hand before its own Event can look at a peek.

Grain Sales draws a card out of the USSR hand and offers it to the US, who resolves it or hands
it back. P17 section 5 made that offer the card's own resolution node -- one decision, on Grain
Sales' frame -- and the drawn card moves into the US hand when it is drawn, not when it is kept.

The location is load-bearing and it is NOT a presentation choice. The action mask is generated
against a position and `step` validates against it, so every predicate that asks where the card
is must get the same answer from both. With the card staged at PEEKED_TEMP they disagreed:
ActionMask's Missile Envy forced-play test read `in_hand_of` as false and offered the Event,
and the identical test inside `step` read it as true and refused. The old two-decision shape hid
this, because the card moved BETWEEN the branch and the play mode, so each mask was generated
against the location its own step would see.

One ordering stays load-bearing either way. If the card is Our Man in Tehran, an Event that peeks
five cards of its own, it must not be sitting in PEEKED_TEMP when its Event fires, or it is inside
its own peek and can discard itself -- the card disappearing in the middle of resolving itself.
Moving it into the hand at draw time settles that earlier than before, and Our Man in Tehran peeks
from the draw deck rather than from PEEKED_TEMP, so there are two independent reasons it is safe.
Neither is obvious from reading either handler alone, which is why it is pinned here.
"""

from __future__ import annotations

from typing import List

import ts_engine as ts

GRAIN_SALES = 67
OUR_MAN_IN_TEHRAN = 108
ISRAEL = 23

# P17 section 5: the decline is CONFIRM_DONE at the resolution node, not a branch index.
DECLINE = ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, 255, 0, 0x80)  # 0x80 = CONFIRM_DONE


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
    assert int(state.ctx().pending_op_card) == card, (
        f"Grain Sales should be offering exactly {card}; "
        f"it names {int(state.ctx().pending_op_card)}"
    )
    return state


def test_the_offered_card_is_named_by_the_decision() -> None:
    """The US can see what it is being offered, and the offer is one decision."""
    state = _grain_sales_offering(OUR_MAN_IN_TEHRAN)
    assert state.ctx().decision_type == ts.DecisionType.SELECT_PLAY_MODE
    assert int(state.ctx().resolving_card) == GRAIN_SALES
    assert int(state.ctx().pending_op_card) == OUR_MAN_IN_TEHRAN
    # Already the US's, marked known, so the mask and step agree about where it is.
    where = state.get_card_location(OUR_MAN_IN_TEHRAN)
    assert ts.in_hand_of(where, ts.Player.US), f"the card should be in the US hand; it is at {where}"
    assert ts.known_to_opponent(where), "both players know which card left the USSR hand"
    assert _peeked(state) == [], "nothing is left being looked at"


def test_keeping_our_man_in_tehran_does_not_put_it_inside_its_own_peek() -> None:
    state = _grain_sales_offering(OUR_MAN_IN_TEHRAN)

    # One decision: play the drawn card as its Event. No branch first.
    assert ts.Engine.try_step(
        state, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE,
                              int(ts.Resolution.EVENT), 0, 0)), (
        "playing the offered card as its Event should be accepted"
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


def test_declining_gives_it_back_and_remembers_it_was_seen() -> None:
    """Handed back to the hand it came from, marked known: the US has seen it."""
    state = _grain_sales_offering(OUR_MAN_IN_TEHRAN)
    ts.CardHandlers.handle_event_step(state, DECLINE)

    where = state.get_card_location(OUR_MAN_IN_TEHRAN)
    assert ts.in_hand_of(where, ts.Player.USSR), f"the card should go back; it is at {where}"
    assert ts.known_to_opponent(where), (
        "the US saw the card, and that is what the location records"
    )
    assert _peeked(state) == [], "nothing should be left being looked at"
    # ...and the US takes Grain Sales' own 2 Ops instead.
    assert state.ctx().decision_type == ts.DecisionType.SELECT_OP_MODE
    assert int(state.ctx().pending_ops_value) == 2


def test_the_drawn_card_is_never_the_one_being_looked_at() -> None:
    """Whichever answer is given, the drawn card itself is never left in a peek.

    Not "the peek is empty": playing Our Man in Tehran opens a peek of five cards, which is the
    card working. What must never happen is the drawn card appearing in it.
    """
    state = _grain_sales_offering(OUR_MAN_IN_TEHRAN)
    assert ts.Engine.try_step(
        state, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE,
                              int(ts.Resolution.EVENT), 0, 0))
    assert OUR_MAN_IN_TEHRAN not in _peeked(state), (
        f"playing left the card inside its own peek {_peeked(state)}"
    )

    state = _grain_sales_offering(OUR_MAN_IN_TEHRAN)
    ts.CardHandlers.handle_event_step(state, DECLINE)
    assert _peeked(state) == [], f"declining left {_peeked(state)} still being looked at"
