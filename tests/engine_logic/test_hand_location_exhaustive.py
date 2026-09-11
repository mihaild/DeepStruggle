"""Every hand is one of four locations, and the observation folds them the way it should.

The `CardLocation` split replaced two hand values with four. The compiler proved that no *read*
site still names the old ones, because deleting them made every such site fail to build. It cannot
prove the two things this file checks:

* that the four variants reach the slots they should -- my own hand in slot 1 whether or not the
  opponent has seen the card, the opponent's in slot 2 when I have seen it and slot 0, folded in
  with the draw deck, when I have not;
* that no engine path leaves a card somewhere that disagrees with the hand the rest of the code
  thinks it is in, which is checked here against real play rather than by inspection.

The hand-size question is checked alongside, because it is the same invariant seen from the other
end: the count the network is given has to be the number of cards actually held, counting the ones
the opponent has seen.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pytest

import ts_engine as ts

BOARD = 84 * 26
CARD_FEATURES = 14
GLOBAL_BASE = BOARD + 110 * CARD_FEATURES

#: observation.cpp:240-241 -- the opponent's hand size, then mine.
OPP_HAND_COUNT, MY_HAND_COUNT = 70, 71
#: The divisor those two are stored with.
HAND_COUNT_SCALE = 10.0

DECK_OR_HIDDEN, MY_HAND, KNOWN_OPPONENT_HAND = 0, 1, 2

#: Every hand variant, and who holds it.
HAND_LOCATIONS: List[Tuple[ts.CardLocation, ts.Player, bool]] = [
    (ts.CardLocation.HAND_US_UNKNOWN, ts.Player.US, False),
    (ts.CardLocation.HAND_US_KNOWN, ts.Player.US, True),
    (ts.CardLocation.HAND_USSR_UNKNOWN, ts.Player.USSR, False),
    (ts.CardLocation.HAND_USSR_KNOWN, ts.Player.USSR, True),
]

#: The whole enum, so a value added later cannot quietly go unconsidered.
ALL_LOCATIONS = [
    ts.CardLocation.UNAVAILABLE, ts.CardLocation.DRAW_DECK,
    ts.CardLocation.HAND_US_UNKNOWN, ts.CardLocation.HAND_US_KNOWN,
    ts.CardLocation.HAND_USSR_UNKNOWN, ts.CardLocation.HAND_USSR_KNOWN,
    ts.CardLocation.DISCARD_PILE, ts.CardLocation.REMOVED_FROM_GAME,
    ts.CardLocation.ONGOING_EVENT, ts.CardLocation.PEEKED_TEMP,
    ts.CardLocation.HEADLINE_COMMITTED,
]


def _fresh(seed: int = 909) -> ts.GameState:
    state = ts.GameState()
    ts.Engine.init_game(state, seed)
    return state


def _slots(obs: np.ndarray, card: int, features: int) -> List[int]:
    base = BOARD + (card - 1) * features
    n_slots = 8
    return [k for k in range(n_slots) if obs[base + k] == 1.0]


def test_the_enum_is_exactly_these_eleven_values() -> None:
    """If someone adds a location, this file has to consider it."""
    named = {n for n in dir(ts.CardLocation) if n.isupper()}
    assert named == {loc.name for loc in ALL_LOCATIONS}, (
        "CardLocation changed; the mappings below need a decision for the new value")


def test_helpers_agree_with_the_four_hand_variants() -> None:
    for loc, holder, known in HAND_LOCATIONS:
        other = ts.Player.USSR if holder == ts.Player.US else ts.Player.US
        assert ts.in_hand_of(loc, holder) is True
        assert ts.in_hand_of(loc, other) is False
        assert ts.known_to_opponent(loc) is known
        assert ts.hand_holder(loc) == holder
        assert ts.hand_of(holder, known) == loc
    for loc in ALL_LOCATIONS:
        if loc not in {l for l, _h, _k in HAND_LOCATIONS}:
            assert ts.hand_holder(loc) == ts.Player.NONE
            assert not ts.in_hand_of(loc, ts.Player.US)
            assert not ts.in_hand_of(loc, ts.Player.USSR)


@pytest.mark.parametrize("loc,holder,known", HAND_LOCATIONS)
def test_each_hand_variant_reaches_the_right_slot(loc: ts.CardLocation, holder: ts.Player,
                                                  known: bool) -> None:
    """My own hand is my own hand; what the opponent has seen is what the opponent has seen."""
    state = _fresh()
    card = 42
    state.set_card_location(card, loc)
    other = ts.Player.USSR if holder == ts.Player.US else ts.Player.US

    holder_view = np.asarray(ts.extract_observation(state, holder))
    other_view = np.asarray(ts.extract_observation(state, other))

    assert _slots(holder_view, card, CARD_FEATURES) == [MY_HAND], (
        "the holder must see it in slot 1 whether or not the opponent has seen it")
    assert _slots(other_view, card, CARD_FEATURES) == (
        [KNOWN_OPPONENT_HAND] if known else [DECK_OR_HIDDEN]), (
        "a card the opponent has seen is public and gets its own slot; one they have not is "
        "folded in with the draw deck, because those two are what an observer cannot separate")


@pytest.mark.parametrize("loc", ALL_LOCATIONS)
def test_every_location_lights_exactly_one_slot(loc: ts.CardLocation) -> None:
    """Every location maps to exactly one slot, for both viewers.

    HEADLINE_COMMITTED is the case worth naming: a card committed to the headline is given its
    owner's hand slot, because whose it is and that it is in play is what the slot is read for.
    """
    state = _fresh()
    card = 42
    state.set_card_location(card, loc)
    for side in (ts.Player.US, ts.Player.USSR):
        slots = _slots(np.asarray(ts.extract_observation(state, side)), card, CARD_FEATURES)
        assert len(slots) == 1, f"got {slots} for {loc.name} viewed by {side}"


def test_v2_separates_known_from_unknown_only_for_the_opponent() -> None:
    state = _fresh()
    card = 42
    for holder in (ts.Player.US, ts.Player.USSR):
        other = ts.Player.USSR if holder == ts.Player.US else ts.Player.US
        for known, expected in ((False, DECK_OR_HIDDEN), (True, KNOWN_OPPONENT_HAND)):
            probe = state.clone()
            probe.set_card_location(card, ts.hand_of(holder, known))
            opp_view = np.asarray(ts.extract_observation(probe, other))
            own_view = np.asarray(ts.extract_observation(probe, holder))
            assert _slots(opp_view, card, CARD_FEATURES) == [expected]
            assert _slots(own_view, card, CARD_FEATURES) == [MY_HAND], (
                "my own hand is my own hand; the split is only about what the opponent has seen")


# --------------------------------------------------------------------------------------------
# Hand sizes
# --------------------------------------------------------------------------------------------

def _reported_counts(state: ts.GameState, side: ts.Player) -> Tuple[float, float]:
    obs = np.asarray(ts.extract_observation(state, side))
    return (obs[GLOBAL_BASE + OPP_HAND_COUNT] * HAND_COUNT_SCALE,
            obs[GLOBAL_BASE + MY_HAND_COUNT] * HAND_COUNT_SCALE)


def _actual_counts(state: ts.GameState, side: ts.Player) -> Tuple[int, int]:
    other = ts.Player.USSR if side == ts.Player.US else ts.Player.US
    mine = sum(1 for c in range(1, 111) if ts.in_hand_of(state.get_card_location(c), side))
    theirs = sum(1 for c in range(1, 111) if ts.in_hand_of(state.get_card_location(c), other))
    return theirs, mine


def test_the_model_is_told_how_many_cards_the_opponent_holds() -> None:
    """A constructed hand of a known size, counted from both sides."""
    state = _fresh()
    for c in range(1, 111):
        if ts.hand_holder(state.get_card_location(c)) != ts.Player.NONE:
            state.set_card_location(c, ts.CardLocation.DRAW_DECK)

    # Seven for the USSR, four of them already seen by the US; three for the US.
    for i, c in enumerate(range(20, 27)):
        state.set_card_location(c, ts.hand_of(ts.Player.USSR, known=i < 4))
    for c in range(60, 63):
        state.set_card_location(c, ts.hand_of(ts.Player.US))

    opp_from_us, mine_from_us = _reported_counts(state, ts.Player.US)
    assert opp_from_us == pytest.approx(7.0), (
        "the US must be told the USSR holds seven, counting the three it has not seen")
    assert mine_from_us == pytest.approx(3.0)

    opp_from_ussr, mine_from_ussr = _reported_counts(state, ts.Player.USSR)
    assert opp_from_ussr == pytest.approx(3.0)
    assert mine_from_ussr == pytest.approx(7.0)


def test_hand_counts_include_cards_the_opponent_has_seen() -> None:
    """Revealing a card must not change how many the opponent is reported to hold."""
    state = _fresh()
    card = next(c for c in range(1, 111)
                if ts.in_hand_of(state.get_card_location(c), ts.Player.USSR))
    before = _reported_counts(state, ts.Player.US)
    state.set_card_location(card, ts.revealed(state.get_card_location(card)))
    after = _reported_counts(state, ts.Player.US)
    assert before == after, "a card becoming public did not change whose hand it is in"


def test_reported_hand_sizes_match_reality_through_a_whole_game() -> None:
    """The count the network reads has to be the number of cards actually held, all game.

    Played out rather than constructed, because the thing this catches is an engine path that
    puts a card somewhere the observation does not count -- which is exactly the failure the
    four-way split could have introduced and inspection would not find.
    """
    checked = 0
    for seed in (11, 12, 13):
        state = _fresh(seed)
        for _ in range(200):
            if ts.Engine.is_terminal(state):
                break
            mask = np.asarray(ts.get_flat_action_mask(state))
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                break
            for side in (ts.Player.US, ts.Player.USSR):
                expected_opp, expected_mine = _actual_counts(state, side)
                got_opp, got_mine = _reported_counts(state, side)
                assert got_opp == pytest.approx(float(expected_opp)), (
                    f"seed {seed}, turn {state.turn}: {side} told the opponent holds {got_opp}, "
                    f"actually {expected_opp}")
                assert got_mine == pytest.approx(float(expected_mine)), (
                    f"seed {seed}, turn {state.turn}: {side} told it holds {got_mine}, "
                    f"actually {expected_mine}")
                checked += 1
            ts.Engine.step_flat(state, int(legal[0]))
    assert checked > 200, "the walk ended too early to be evidence"


def test_no_card_ever_sits_in_a_hand_the_helpers_disagree_about() -> None:
    """Across real play, every location is either in nobody's hand or in exactly one player's."""
    for seed in (21, 22):
        state = _fresh(seed)
        for _ in range(200):
            if ts.Engine.is_terminal(state):
                break
            for c in range(1, 111):
                loc = state.get_card_location(c)
                in_us = ts.in_hand_of(loc, ts.Player.US)
                in_ussr = ts.in_hand_of(loc, ts.Player.USSR)
                assert not (in_us and in_ussr), f"card {c} is in both hands at once ({loc})"
                holder = ts.hand_holder(loc)
                assert (holder != ts.Player.NONE) == (in_us or in_ussr), (
                    f"card {c}: hand_holder says {holder} but in_hand_of says "
                    f"US={in_us} USSR={in_ussr}")
                if ts.known_to_opponent(loc):
                    assert holder != ts.Player.NONE, (
                        f"card {c} is marked known to an opponent while in no hand ({loc})")
            mask = np.asarray(ts.get_flat_action_mask(state))
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                break
            ts.Engine.step_flat(state, int(legal[0]))
