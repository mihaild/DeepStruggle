"""A deal can be driven from a recording, so a replay survives a changed shuffle.

A die roll has had an override for a long time -- a ROLL_DIE action carries the value. A deal had
none, because a deal is not a decision and there is no action to hang one on. The consequence
(BUGS.md TODO) is that a replay cannot be made self-contained: re-drive it on an engine whose
shuffle or draw order has changed and it diverges at the first deal, silently, producing a
different game rather than an error.

The test that matters is the last one: same recording, deliberately different RNG, same hands.
That is the property the affordance exists for, and nothing else here implies it.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pytest
import ts_engine as ts


def _hand(state: ts.GameState, player: ts.Player) -> list[int]:
    loc = ts.hand_of(player)
    return [c for c in range(1, 111) if state.get_card_location(c) == loc]


def _fresh(seed: int) -> ts.GameState:
    s = ts.GameState()
    ts.Engine.init_game(s, seed)
    return s


def _empty_hand(state: ts.GameState, player: ts.Player) -> None:
    """Put a player's hand back in the draw deck, so the next deal actually deals.

    init_game already deals, so calling deal_cards_to_hands on a fresh state finds the hand at
    its target size and the loop never runs -- which is how the first version of these tests
    passed while proving nothing.
    """
    for c in range(1, 111):
        if state.get_card_location(c) == ts.hand_of(player):
            state.set_card_location(c, ts.CardLocation.DRAW_DECK)


def test_it_is_inert_when_nobody_names_anything() -> None:
    """Normal play must be untouched -- the queues are empty in every real game."""
    a, b = _fresh(4242), _fresh(4242)
    assert _hand(a, ts.Player.US) == _hand(b, ts.Player.US)
    assert a.get_forced_deal_remaining(ts.Player.US) == 0
    assert a.get_forced_deal_remaining(ts.Player.USSR) == 0


def test_a_named_card_is_the_one_dealt() -> None:
    s = _fresh(7)
    _empty_hand(s, ts.Player.USSR)
    deck = [c for c in range(1, 111)
            if s.get_card_location(c) == ts.CardLocation.DRAW_DECK]
    assert len(deck) >= 3
    want = deck[:3]
    s.set_forced_deal(ts.Player.USSR, want)
    assert s.get_forced_deal_remaining(ts.Player.USSR) == 3

    ts.StateMachine.deal_cards_to_hands(s)
    got = _hand(s, ts.Player.USSR)
    for card in want:
        assert card in got, f"card {card} was named for the deal and did not arrive"


def test_the_queue_does_not_leak_into_the_next_deal() -> None:
    """One deal, one queue. A leftover would be applied to a deal it was never recorded for."""
    s = _fresh(11)
    _empty_hand(s, ts.Player.US)
    deck = [c for c in range(1, 111)
            if s.get_card_location(c) == ts.CardLocation.DRAW_DECK]
    s.set_forced_deal(ts.Player.US, deck[:2])
    ts.StateMachine.deal_cards_to_hands(s)
    assert s.get_forced_deal_remaining(ts.Player.US) == 0
    assert s.get_forced_deal_remaining(ts.Player.USSR) == 0


@pytest.mark.parametrize("cards", [list(range(1, 11)), [0], [111]])
def test_a_malformed_queue_is_refused_rather_than_truncated(cards: list[int]) -> None:
    """Too many cards, or an id that is not a card. Refusing beats silently dealing something."""
    s = _fresh(3)
    with pytest.raises(Exception):
        s.set_forced_deal(ts.Player.US, cards)


def test_the_same_recording_deals_the_same_hands_under_a_different_rng() -> None:
    """The property the whole affordance exists for.

    Record what a deal gave each side, then replay it into a game whose RNG stream is
    deliberately different. Without the override the draw is
    `Prng::random_index(state.rng_state, ...)` and the hands diverge; with it, the recording
    decides. The control below is what makes this meaningful -- it shows the two RNGs really do
    deal differently when nothing is forced.
    """
    def _dealt(seed: int, rng_xor: int,
               force: Optional[Tuple[List[int], List[int]]]) -> Tuple[List[int], List[int]]:
        s = _fresh(seed)
        _empty_hand(s, ts.Player.US)
        _empty_hand(s, ts.Player.USSR)
        if rng_xor:
            s.rng_state = s.rng_state ^ rng_xor
        if force:
            s.set_forced_deal(ts.Player.US, force[0])
            s.set_forced_deal(ts.Player.USSR, force[1])
        ts.StateMachine.deal_cards_to_hands(s)
        return _hand(s, ts.Player.US), _hand(s, ts.Player.USSR)

    rec_us, rec_ussr = _dealt(2024, 0, None)
    assert len(rec_us) >= 8 and len(rec_ussr) >= 8, "the recording deal did not happen"

    # Control: a different RNG, nothing forced -> different hands. Without this the test below
    # would pass even if the override did nothing at all.
    free_us, free_ussr = _dealt(2024, 0xDEADBEEF, None)
    assert (free_us, free_ussr) != (rec_us, rec_ussr), (
        "the two RNG streams dealt identically, so this proves nothing -- pick another xor")

    # Same different RNG, now driven by the recording -> the recording's hands.
    got_us, got_ussr = _dealt(2024, 0xDEADBEEF, (rec_us[:9], rec_ussr[:9]))
    assert got_us == rec_us, "the recording did not drive the US deal"
    assert got_ussr == rec_ussr, "the recording did not drive the USSR deal"
