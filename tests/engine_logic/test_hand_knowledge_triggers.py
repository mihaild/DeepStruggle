"""What makes a card in an opponent's hand public, and what quietly makes it private again.

Five things reveal a card in Twilight Struggle, and the engine now records all of them:

* **CIA Created** -- "The USSR reveals their hand of cards for this turn."
* **"Lone Gunman"** -- "The US reveals their hand of cards."
* **Aldrich Ames Remix** -- "The US reveals their hand of cards, face-up, for the remainder of
  the turn."
* **SALT Negotiations** -- the retrieved card is revealed to the opponent before it is taken.
* **the draw deck running out** -- at that moment every card is in a hand, the discard, removed,
  in play, or never in the game, and all of those but the hands are public, so each player can
  name the other's hand as the complement of what they can see.

Missile Envy and Grain Sales also move known cards, and are covered in the card suites.

The second half of each test is the part that is easy to get wrong: knowledge attaches to the
*cards*, not to the player. A card drawn after a reveal is hidden, and a card that leaves the hand
takes its knownness with it. Storing this as a location rather than a flag is what makes that
automatic, and these tests are what say it actually is.
"""

from __future__ import annotations

from typing import List, Set

import numpy as np
import pytest

import ts_engine as ts

CIA_CREATED = 26
LONE_GUNMAN = 62
ALDRICH_AMES = 98


def _fresh(seed: int = 31337) -> ts.GameState:
    state = ts.GameState()
    ts.Engine.init_game(state, seed)
    return state


def _hand(state: ts.GameState, p: ts.Player) -> List[int]:
    return [c for c in range(1, 111) if ts.in_hand_of(state.get_card_location(c), p)]


def _known(state: ts.GameState, p: ts.Player) -> Set[int]:
    """Cards in `p`'s hand that the opponent can name."""
    return {c for c in _hand(state, p) if ts.known_to_opponent(state.get_card_location(c))}


def _stock_hand(state: ts.GameState, p: ts.Player, cards: List[int]) -> None:
    for c in range(1, 111):
        if ts.in_hand_of(state.get_card_location(c), p):
            state.set_card_location(c, ts.CardLocation.DRAW_DECK)
    for c in cards:
        state.set_card_location(c, ts.hand_of(p))


# --------------------------------------------------------------------------------------------
# The helper itself
# --------------------------------------------------------------------------------------------

def test_revealing_a_hand_marks_every_card_in_it_and_nothing_else() -> None:
    state = _fresh()
    _stock_hand(state, ts.Player.USSR, [10, 11, 12])
    _stock_hand(state, ts.Player.US, [20, 21])

    before_us = set(_hand(state, ts.Player.US))
    assert _known(state, ts.Player.USSR) == set()

    ts.reveal_hand(state, ts.Player.USSR)
    assert _known(state, ts.Player.USSR) == {10, 11, 12}
    assert _known(state, ts.Player.US) == set(), "the other hand must be untouched"
    assert set(_hand(state, ts.Player.US)) == before_us, "no card changed hands"
    assert set(_hand(state, ts.Player.USSR)) == {10, 11, 12}, "no card left the revealed hand"


def test_revealing_is_idempotent() -> None:
    state = _fresh()
    _stock_hand(state, ts.Player.USSR, [10, 11])
    ts.reveal_hand(state, ts.Player.USSR)
    once = [state.get_card_location(c) for c in (10, 11)]
    ts.reveal_hand(state, ts.Player.USSR)
    assert [state.get_card_location(c) for c in (10, 11)] == once


def test_a_card_drawn_after_a_reveal_is_hidden_again() -> None:
    """Knowledge is about cards that were seen, not about the player who was caught."""
    state = _fresh()
    _stock_hand(state, ts.Player.USSR, [10, 11])
    ts.reveal_hand(state, ts.Player.USSR)
    state.set_card_location(12, ts.hand_of(ts.Player.USSR))
    assert _known(state, ts.Player.USSR) == {10, 11}, (
        "card 12 arrived after the reveal and nobody has seen it")


def test_knowledge_leaves_with_the_card() -> None:
    """A card played or discarded loses its knownness because its location is overwritten."""
    state = _fresh()
    _stock_hand(state, ts.Player.USSR, [10, 11])
    ts.reveal_hand(state, ts.Player.USSR)
    state.set_card_location(10, ts.CardLocation.DISCARD_PILE)
    assert not ts.known_to_opponent(state.get_card_location(10))
    assert ts.hand_holder(state.get_card_location(10)) == ts.Player.NONE


# --------------------------------------------------------------------------------------------
# The cards
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("card,revealed_side,other_side", [
    (CIA_CREATED, ts.Player.USSR, ts.Player.US),
    (LONE_GUNMAN, ts.Player.US, ts.Player.USSR),
    (ALDRICH_AMES, ts.Player.US, ts.Player.USSR),
])
def test_the_reveal_cards_reveal_the_right_hand(card: int, revealed_side: ts.Player,
                                                other_side: ts.Player) -> None:
    """Each card names whose hand goes face up, and it is not always the opponent's."""
    state = _fresh()
    _stock_hand(state, ts.Player.US, [20, 21, 22])
    _stock_hand(state, ts.Player.USSR, [30, 31, 32])
    # The card being played is not in either hand, so it cannot be caught up in its own reveal.
    state.set_card_location(card, ts.CardLocation.DRAW_DECK)

    expected = set(_hand(state, revealed_side))
    ts.CardHandlers.trigger_event(state, card, other_side)

    assert _known(state, revealed_side) == expected, (
        f"card {card} should have revealed the whole {revealed_side} hand")
    assert _known(state, other_side) == set(), (
        f"card {card} revealed the wrong hand as well as the right one")


def test_aldrich_ames_reveal_survives_into_the_observation() -> None:
    """End to end: play the card, then read the observation the USSR is given."""
    state = _fresh()
    _stock_hand(state, ts.Player.US, [20, 21, 22])
    state.set_card_location(ALDRICH_AMES, ts.CardLocation.DRAW_DECK)

    board, features, known_slot = 84 * 26, 14, 2
    before = np.asarray(ts.extract_observation(state, ts.Player.USSR))
    assert all(before[board + (c - 1) * features + known_slot] == 0.0 for c in (20, 21, 22))

    ts.CardHandlers.trigger_event(state, ALDRICH_AMES, ts.Player.USSR)
    after = np.asarray(ts.extract_observation(state, ts.Player.USSR))
    for c in (20, 21, 22):
        assert after[board + (c - 1) * features + known_slot] == 1.0, (
            f"the USSR should be able to name US card {c} after Aldrich Ames")



# --------------------------------------------------------------------------------------------
# The deck running out
# --------------------------------------------------------------------------------------------

def test_exhausting_the_deck_makes_both_hands_public() -> None:
    """Deal with an almost-empty deck: the complement is deducible, so both hands go face up."""
    state = _fresh()
    _stock_hand(state, ts.Player.US, [20, 21])
    _stock_hand(state, ts.Player.USSR, [30, 31])
    # Everything not in a hand goes to the discard *after* the hands are set -- _stock_hand
    # returns the previous hand to the deck, so doing this first would leave those cards in it
    # and the deck would never run out, which is how this test failed the first time.
    for c in range(1, 111):
        if ts.hand_holder(state.get_card_location(c)) == ts.Player.NONE:
            state.set_card_location(c, ts.CardLocation.DISCARD_PILE)
    # Two cards left in the deck against two hands that want eight each: it must run out.
    for c in (40, 41):
        state.set_card_location(c, ts.CardLocation.DRAW_DECK)
    assert sum(1 for c in range(1, 111)
               if state.get_card_location(c) == ts.CardLocation.DRAW_DECK) == 2

    ts.StateMachine.deal_cards_to_hands(state)

    for side, originals in ((ts.Player.US, {20, 21}), (ts.Player.USSR, {30, 31})):
        known = _known(state, side)
        assert originals <= known, (
            f"{side} held {sorted(originals)} when the deck ran out; those are deducible")


def test_the_scheduled_era_reshuffles_do_not_reveal_anything() -> None:
    """Turn 4 and turn 8 shuffle the discard back in, but the deck is not empty and nothing is
    deducible. A reveal there would be a rule the game does not have."""
    state = _fresh()
    _stock_hand(state, ts.Player.US, [20, 21])
    _stock_hand(state, ts.Player.USSR, [30, 31])
    assert _known(state, ts.Player.US) == set()

    ts.StateMachine.reshuffle_discard_into_draw(state)

    assert _known(state, ts.Player.US) == set(), "a plain reshuffle revealed a hand"
    assert _known(state, ts.Player.USSR) == set(), "a plain reshuffle revealed a hand"


def test_a_whole_game_never_marks_a_card_known_outside_a_hand() -> None:
    """The invariant that makes one field enough: knownness only ever lives on a held card."""
    for seed in (5, 6, 7):
        state = _fresh(seed)
        for _ in range(300):
            if ts.Engine.is_terminal(state):
                break
            for c in range(1, 111):
                loc = state.get_card_location(c)
                if ts.known_to_opponent(loc):
                    assert ts.hand_holder(loc) != ts.Player.NONE, (
                        f"card {c} is marked known while in {loc}")
            mask = np.asarray(ts.get_flat_action_mask(state))
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                break
            ts.Engine.step_flat(state, int(legal[0]))
