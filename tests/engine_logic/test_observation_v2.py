"""The v2 observation layout: what it splits, and what it must keep merged.

v2 widens the card block from 12 features to 13 to separate two things the legacy slot 0 could
not: a card the opponent is *known* to hold, and a card that is not in the game yet. The second
is public information the network was simply never given. The first is the point of the whole
`CardLocation` split.

The test that matters most here is the negative one. Slot 0 must go on merging the draw deck
with the *unknown* part of the opponent's hand, because that is exactly the pair an observer
cannot distinguish -- separating them would hand the network the hidden information the game is
played to discover, and every result measured against it would be worthless.
"""

from __future__ import annotations

import numpy as np
import pytest

import ts_engine as ts

BOARD = 84 * 28
V2_FEATURES = 13
LEGACY_FEATURES = 12

DECK_OR_HIDDEN, MY_HAND, KNOWN_OPPONENT_HAND = 0, 1, 2
DISCARD, REMOVED, ONGOING, PEEKED, NOT_IN_GAME = 3, 4, 5, 6, 7
V2_PROPERTY_BASE = 8
LEGACY_PROPERTY_BASE = 7


def _fresh() -> ts.GameState:
    state = ts.GameState()
    ts.Engine.init_game(state, 4242)
    return state


def _v2(state: ts.GameState, side: ts.Player) -> np.ndarray:
    return np.asarray(ts.extract_observation(state, side, legacy=False), dtype=np.float32)


def _legacy(state: ts.GameState, side: ts.Player) -> np.ndarray:
    return np.asarray(ts.extract_observation(state, side, legacy=True), dtype=np.float32)


def _card_slot(obs: np.ndarray, card: int, slot: int, features: int = V2_FEATURES) -> float:
    return float(obs[BOARD + (card - 1) * features + slot])


def test_widths() -> None:
    state = _fresh()
    assert _legacy(state, ts.Player.US).shape == (ts.OBS_SIZE_LEGACY,) == (4293,)
    assert _v2(state, ts.Player.US).shape == (ts.OBS_SIZE_V2,) == (4403,)


def test_legacy_is_the_default() -> None:
    """Existing callers pass no flag and must keep the layout their checkpoints were trained on."""
    state = _fresh()
    assert np.array_equal(np.asarray(ts.extract_observation(state, ts.Player.US)),
                          _legacy(state, ts.Player.US))


def test_v2_differs_from_legacy_only_in_the_card_block() -> None:
    """Board, globals, history, turn aggregates and active player must be untouched."""
    state = _fresh()
    for side in (ts.Player.US, ts.Player.USSR):
        legacy, v2 = _legacy(state, side), _v2(state, side)
        assert np.array_equal(legacy[:BOARD], v2[:BOARD]), "board features drifted"
        tail_legacy = legacy[BOARD + 110 * LEGACY_FEATURES:]
        tail_v2 = v2[BOARD + 110 * V2_FEATURES:]
        assert np.array_equal(tail_legacy, tail_v2), (
            "everything after the card block must be identical between layouts")


def test_card_properties_survive_the_shift() -> None:
    """The five property features move from 7..11 to 8..12 and must carry the same values."""
    state = _fresh()
    legacy, v2 = _legacy(state, ts.Player.US), _v2(state, ts.Player.US)
    for card in (1, 30, 33, 110):
        for k in range(5):
            assert _card_slot(v2, card, V2_PROPERTY_BASE + k) == pytest.approx(
                _card_slot(legacy, card, LEGACY_PROPERTY_BASE + k, LEGACY_FEATURES)), (
                f"property {k} of card {card} changed value across layouts")


def test_unavailable_is_split_out_of_the_deck() -> None:
    """A card not yet in the game gets its own slot instead of looking like a deck card."""
    state = _fresh()
    unavailable = [c for c in range(1, 111)
                   if state.get_card_location(c) == ts.CardLocation.UNAVAILABLE]
    deck = [c for c in range(1, 111)
            if state.get_card_location(c) == ts.CardLocation.DRAW_DECK]
    assert unavailable, "a fresh game should have later-era cards not yet in the deck"
    assert deck, "a fresh game should have a draw deck"

    obs = _v2(state, ts.Player.US)
    for card in unavailable:
        assert _card_slot(obs, card, NOT_IN_GAME) == 1.0
        assert _card_slot(obs, card, DECK_OR_HIDDEN) == 0.0
    for card in deck:
        assert _card_slot(obs, card, DECK_OR_HIDDEN) == 1.0
        assert _card_slot(obs, card, NOT_IN_GAME) == 0.0

    # In the legacy layout both of those were the same slot, which is what v2 is fixing.
    legacy = _legacy(state, ts.Player.US)
    assert _card_slot(legacy, unavailable[0], DECK_OR_HIDDEN, LEGACY_FEATURES) == 1.0
    assert _card_slot(legacy, deck[0], DECK_OR_HIDDEN, LEGACY_FEATURES) == 1.0


def test_a_known_opponent_card_becomes_visible_and_an_unknown_one_does_not() -> None:
    """The split, and the thing it must not overreach into."""
    state = _fresh()
    hidden, known = 40, 41
    state.set_card_location(hidden, ts.hand_of(ts.Player.USSR))
    state.set_card_location(known, ts.hand_of(ts.Player.USSR, True))

    us = _v2(state, ts.Player.US)
    # The card the US has seen is named.
    assert _card_slot(us, known, KNOWN_OPPONENT_HAND) == 1.0
    assert _card_slot(us, known, DECK_OR_HIDDEN) == 0.0
    # The card it has not seen is indistinguishable from a card in the deck. This is the
    # information-hiding property; if it ever fails the observation is cheating.
    assert _card_slot(us, hidden, KNOWN_OPPONENT_HAND) == 0.0
    assert _card_slot(us, hidden, DECK_OR_HIDDEN) == 1.0

    # From the USSR's own side both are simply its hand, knowledge or not.
    ussr = _v2(state, ts.Player.USSR)
    for card in (hidden, known):
        assert _card_slot(ussr, card, MY_HAND) == 1.0
        assert _card_slot(ussr, card, KNOWN_OPPONENT_HAND) == 0.0


#: global_features indices for the two public counts, from observation.cpp:230 and :240.
GLOBAL_BASE = BOARD + 110 * V2_FEATURES
DRAW_PILE_COUNT = GLOBAL_BASE + 62
OPPONENT_HAND_COUNT = GLOBAL_BASE + 70


def test_hidden_opponent_cards_are_indistinguishable_from_deck_cards() -> None:
    """Moving a card into the opponent's unseen hand must not say *which* card moved.

    Two things do legitimately change, and the test names them rather than allowing any
    difference: the draw-pile count and the opponent's hand size. Both are public at the table --
    you can see how many cards someone is holding -- so hiding them would model the game wrongly
    in the other direction. What must not change is anything card-specific.
    """
    state = _fresh()
    deck = next(c for c in range(1, 111)
                if state.get_card_location(c) == ts.CardLocation.DRAW_DECK)
    probe = state.clone()
    probe.set_card_location(deck, ts.hand_of(ts.Player.USSR))

    before = _v2(state, ts.Player.US)
    after = _v2(probe, ts.Player.US)
    moved = set(np.flatnonzero(before != after).tolist())

    card_block = set(range(BOARD, BOARD + 110 * V2_FEATURES))
    leaked = moved & card_block
    assert not leaked, (
        f"the card block changed at {sorted(leaked)[:8]} when a card moved into the opponent's "
        f"unseen hand -- that identifies the card and is a hidden-information leak")
    assert moved <= {DRAW_PILE_COUNT, OPPONENT_HAND_COUNT}, (
        f"unexpected features changed: {sorted(moved - {DRAW_PILE_COUNT, OPPONENT_HAND_COUNT})}")
    assert after[OPPONENT_HAND_COUNT] > before[OPPONENT_HAND_COUNT], (
        "the opponent's hand grew and the public count should say so")


def test_revealing_a_card_already_in_the_opponents_hand_changes_only_that_card() -> None:
    """Knowledge arriving must move exactly one card's features and no count."""
    state = _fresh()
    held = next(c for c in range(1, 111)
                if state.get_card_location(c) == ts.CardLocation.DRAW_DECK)
    hidden = state.clone()
    hidden.set_card_location(held, ts.hand_of(ts.Player.USSR))
    known = hidden.clone()
    known.set_card_location(held, ts.revealed(hidden.get_card_location(held)))

    before, after = _v2(hidden, ts.Player.US), _v2(known, ts.Player.US)
    moved = set(np.flatnonzero(before != after).tolist())
    base = BOARD + (held - 1) * V2_FEATURES
    assert moved == {base + DECK_OR_HIDDEN, base + KNOWN_OPPONENT_HAND}, (
        f"revealing one card should move exactly its two location slots, got {sorted(moved)}")
    assert after[base + KNOWN_OPPONENT_HAND] == 1.0
    assert after[base + DECK_OR_HIDDEN] == 0.0


def test_the_batch_runner_reports_its_own_width() -> None:
    legacy_runner = ts.VectorizedBatchRunner(4, 7, True)
    v2_runner = ts.VectorizedBatchRunner(4, 7, False)
    assert legacy_runner.obs_width == 4293 and legacy_runner.legacy_obs
    assert v2_runner.obs_width == 4403 and not v2_runner.legacy_obs
    assert np.asarray(legacy_runner.get_observations()).shape == (4, 4293)
    assert np.asarray(v2_runner.get_observations()).shape == (4, 4403)


def test_the_batch_runner_defaults_to_legacy() -> None:
    """Every existing call site constructs a runner without the flag."""
    runner = ts.VectorizedBatchRunner(2, 99)
    assert runner.legacy_obs and runner.obs_width == 4293


def test_runner_rows_match_the_single_state_extractor() -> None:
    """The batched path and the one-off path must agree, in both layouts."""
    for legacy in (True, False):
        runner = ts.VectorizedBatchRunner(3, 555, legacy)
        rows = np.asarray(runner.get_observations())
        for i in range(3):
            state = runner.get_state(i)
            ctx = state.ctx()
            side = ctx.decision_player if ctx.decision_player != ts.Player.NONE \
                else state.phasing_player
            direct = np.asarray(ts.extract_observation(state, side, legacy=legacy))
            assert np.array_equal(rows[i], direct), (
                f"batched and direct observations disagree at env {i}, legacy={legacy}")
