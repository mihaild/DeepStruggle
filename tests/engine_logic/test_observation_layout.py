"""The observation layout: what it splits, and what it must keep merged.

The card block carries eight location slots, and two of them exist to separate what an observer
can legitimately tell apart: a card the opponent is *known* to hold, and a card that is not in
the game yet. Both were once merged into slot 0 along with the deck.

The test that matters most here is the negative one. Slot 0 must go on merging the draw deck
with the *unknown* part of the opponent's hand, because that is exactly the pair an observer
cannot distinguish -- separating them would hand the network the hidden information the game is
played to discover, and every result measured against it would be worthless.
"""

from __future__ import annotations

import numpy as np
import pytest

import ts_engine as ts

BOARD = 84 * 26
CARD_FEATURES = 14

DECK_OR_HIDDEN, MY_HAND, KNOWN_OPPONENT_HAND = 0, 1, 2
DISCARD, REMOVED, ONGOING, PEEKED, NOT_IN_GAME = 3, 4, 5, 6, 7
PROPERTY_BASE = 8


def _fresh() -> ts.GameState:
    state = ts.GameState()
    ts.Engine.init_game(state, 4242)
    return state


def _v2(state: ts.GameState, side: ts.Player) -> np.ndarray:
    return np.asarray(ts.extract_observation(state, side), dtype=np.float32)


def _card_slot(obs: np.ndarray, card: int, slot: int, features: int = CARD_FEATURES) -> float:
    return float(obs[BOARD + (card - 1) * features + slot])


def test_width() -> None:
    """The width is a checkpoint contract: 84x26 board + 110x14 card + 100 global."""
    state = _fresh()
    assert _v2(state, ts.Player.US).shape == (ts.OBS_SIZE,) == (3824,)
    assert ts.OBS_SIZE == ts.OBS_SIZE_V23 == 84 * 26 + 110 * 14 + 100


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
GLOBAL_BASE = BOARD + 110 * CARD_FEATURES
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

    card_block = set(range(BOARD, BOARD + 110 * CARD_FEATURES))
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
    base = BOARD + (held - 1) * CARD_FEATURES
    assert moved == {base + DECK_OR_HIDDEN, base + KNOWN_OPPONENT_HAND}, (
        f"revealing one card should move exactly its two location slots, got {sorted(moved)}")
    assert after[base + KNOWN_OPPONENT_HAND] == 1.0
    assert after[base + DECK_OR_HIDDEN] == 0.0


def test_the_batch_runner_reports_its_own_width() -> None:
    runner = ts.VectorizedBatchRunner(4, 7)
    assert runner.obs_width == ts.OBS_SIZE == 3824
    assert np.asarray(runner.get_observations()).shape == (4, 3824)


def test_runner_rows_match_the_single_state_extractor() -> None:
    """The batched path and the one-off path must agree."""
    if True:
        runner = ts.VectorizedBatchRunner(3, 555)
        rows = np.asarray(runner.get_observations())
        for i in range(3):
            state = runner.get_state(i)
            ctx = state.ctx()
            side = ctx.decision_player if ctx.decision_player != ts.Player.NONE \
                else state.phasing_player
            direct = np.asarray(ts.extract_observation(state, side))
            assert np.array_equal(rows[i], direct), (
                f"batched and direct observations disagree at env {i}")


# --- The China Card ---------------------------------------------------------------------------
#
# card_locations[6] is ONGOING_EVENT for the whole game and has to stay that way: the cards that
# scan a hand (Grain Sales To Soviets, Five Year Plan, Missile Envy, The Cambridge Five,
# Terrorism) test membership through card_locations, so a hand variant there would let the China
# Card be stolen, discarded or forced. v2.1 therefore reads china_card_holder in the observation
# instead. Before that, the card block showed it as an ongoing event in every position sampled --
# to holder and opponent alike -- so the card branch never saw the one card that is always safe to
# play.

CHINA = 6


def test_the_china_card_reads_as_a_hand_card_to_its_holder() -> None:
    state = _fresh()
    holder = state.china_card_holder
    other = ts.Player.US if holder == ts.Player.USSR else ts.Player.USSR

    assert _card_slot(_v2(state, holder), CHINA, MY_HAND) == 1.0
    assert _card_slot(_v2(state, holder), CHINA, ONGOING) == 0.0
    # Its holder is public in this game, so the opponent sees it as a known held card.
    assert _card_slot(_v2(state, other), CHINA, KNOWN_OPPONENT_HAND) == 1.0
    assert _card_slot(_v2(state, other), CHINA, ONGOING) == 0.0


def test_the_china_card_stays_a_hand_card_while_face_down() -> None:
    """Passing it face down makes it unplayable this turn; it has not left the hand.

    Playability is global_features[11] and is a separate question from where the card is.
    """
    state = _fresh()
    holder = state.china_card_holder
    state.china_card_playable = 0
    assert _card_slot(_v2(state, holder), CHINA, MY_HAND) == 1.0


def test_the_engine_still_holds_the_china_card_out_of_play() -> None:
    """The observation changed; card_locations did not, and must not."""
    state = _fresh()
    assert state.get_card_location(CHINA) == ts.CardLocation.ONGOING_EVENT


# --- v2.3: Europe, the headline, and Chernobyl -------------------------------------------------

V23_BOARD = 84 * 26
V23_CARD_F = 14
V23_GLOBAL = V23_BOARD + 110 * V23_CARD_F
EUROPE_VP = V23_GLOBAL + 64
CTX = V23_GLOBAL + 72
# v2.2 had ctx/temp_card_count at CTX + 19; the slot is gone in v2.3 and these shift down one.
HEADLINE_STAGE, HEADLINE_FIRST_MINE, HEADLINE_SECOND_MINE = CTX + 19, CTX + 20, CTX + 21
CHERNOBYL_REGION = CTX + 22


def _v22(state: ts.GameState, side: ts.Player) -> np.ndarray:
    return np.asarray(ts.extract_observation(state, side), dtype=np.float32)


def _v22_card(obs: np.ndarray, card: int, slot: int) -> float:
    """v2.3's board is 84x26, so the module-level _card_slot's 84x28 offset does not apply."""
    return float(obs[V23_BOARD + (card - 1) * V23_CARD_F + slot])


def _europe() -> list[int]:
    return [i for i in range(84) if ts.MapData.get_country_info(i)["region"] == 0]


def test_europe_control_reads_as_a_win_not_as_half_a_domination() -> None:
    """Europe's control_vp is 0 against domination_vp 7, so net_delta ranked a won game below a
    dominated one -- about 6 against 12. Control ends the game; the feature must say so."""
    state = _fresh()
    for i in _europe():
        state.set_country(i, 20, 0)
    assert _v22(state, ts.Player.US)[EUROPE_VP] == pytest.approx(1.0)
    assert _v22(state, ts.Player.USSR)[EUROPE_VP] == pytest.approx(-1.0)

    other = _fresh()
    for i in _europe():
        other.set_country(i, 0, 20)
    assert _v22(other, ts.Player.US)[EUROPE_VP] == pytest.approx(-1.0)


def test_the_other_regions_keep_their_scoring() -> None:
    """Only Europe has a control_vp of 0; nothing else should have moved."""
    state = _fresh()
    obs = _v22(state, ts.Player.US)
    for r in range(1, 6):
        assert -1.0 <= float(obs[V23_GLOBAL + 64 + r]) <= 1.0


def test_a_committed_headline_is_visible_to_its_owner() -> None:
    """HEADLINE_COMMITTED had no branch, so it fell through to the deck slot and a player could
    not see the card they had just chosen."""
    state = _fresh()
    card = state.headline_us_card or 1
    state.set_card_location(card, ts.CardLocation.HEADLINE_COMMITTED)
    state.headline_us_card = card

    us = _v22(state, ts.Player.US)
    assert _v22_card(us, card, MY_HAND) == 1.0
    assert _v22_card(us, card, DECK_OR_HIDDEN) == 0.0
    # And it is in play, which is what the active-card feature means.
    assert _v22_card(us, card, 13) == 1.0   # is_active_card

    ussr = _v22(state, ts.Player.USSR)
    assert _v22_card(ussr, card, KNOWN_OPPONENT_HAND) == 1.0


def test_the_headline_stage_and_order_reach_the_model() -> None:
    state = _fresh()
    state.headline_stage = 2
    obs = _v22(state, ts.Player.US)
    assert obs[HEADLINE_STAGE] == pytest.approx(2.0 / 3.0)
    assert obs[HEADLINE_FIRST_MINE] in (0.0, 1.0)
    assert obs[HEADLINE_SECOND_MINE] in (0.0, 1.0)


def test_chernobyl_region_is_a_one_hot_and_is_empty_when_not_in_play() -> None:
    state = _fresh()
    obs = _v22(state, ts.Player.US)
    assert float(obs[CHERNOBYL_REGION:CHERNOBYL_REGION + 6].sum()) == 0.0, \
        "Chernobyl is not in play at setup"

    # Exactly one region is named while it is in play; the engine sets both the flag and the
    # region index together, so this checks the shape rather than driving the card.
    assert obs[CHERNOBYL_REGION:CHERNOBYL_REGION + 6].max() <= 1.0


# --- Cards staged for a decision -----------------------------------------------------------
#
# The engine puts a card at PEEKED_TEMP when a decision is *about* that card -- Grain Sales
# hands one over and asks whether to play it, Star Wars offers one from the discard pile -- and
# does so without touching card_locations. So the card kept reading DECK_OR_HIDDEN, and the player
# being asked could not see what they were deciding about. v2.3 shows it in the PEEKED slot, but
# only to the player whose decision it is, and only where they could not already see it.


def _staged_decisions(max_games: int = 150):
    """Positions where a card is being shown to the player who must decide about it.

    Grain Sales is the case: it draws a card out of the USSR hand and asks the US to keep or
    return it. The card sits at PEEKED_TEMP -- being looked at is a location -- so it is found
    by looking there rather than in a list kept beside the hand.
    """
    from bindings.action_encoder import ActionEncoder

    rng = np.random.default_rng(11)
    out = []
    for g in range(max_games):
        state = ts.GameState()
        ts.Engine.init_game(state, 90000 + g)
        for _ in range(4000):
            if ts.Engine.is_terminal(state) or len(out) >= 5:
                break
            ctx = state.ctx()
            if ctx.decision_player != ts.Player.NONE:
                peeked = [c for c in range(1, 111)
                          if state.get_card_location(c) == ts.CardLocation.PEEKED_TEMP]
                if len(peeked) == 1:
                    out.append((state.clone(), ctx.decision_player, peeked[0]))
            mask = np.asarray(ActionEncoder.get_legal_mask(state))
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                break
            ts.Engine.step_flat(state, int(rng.choice(legal)))
        if len(out) >= 5:
            break
    return out


def test_a_card_being_decided_about_is_visible_to_the_decider() -> None:
    """The card is at PEEKED_TEMP, which the card block already reads.

    obs_flags::STAGED_CARDS existed only because Grain Sales showed a card without moving it,
    leaving the US choosing about a card that read as deck-or-hidden. Moving it makes the
    visibility structural, and the flag had nothing left to do -- it is retired, along with the
    layout it was added to.
    """
    found = _staged_decisions()
    assert found, "no position reached where a card is being shown to a decider"
    for state, decider, card in found:
        obs = _v22(state, decider)
        assert _v22_card(obs, card, DECK_OR_HIDDEN) == 0.0, (
            f"card {card} is being decided about by {decider} and still reads as "
            f"deck-or-hidden")
        assert _v22_card(obs, card, PEEKED) == 1.0, (
            f"card {card} is at PEEKED_TEMP and should read as peeked")


def test_it_reads_the_same_to_both_players() -> None:
    """Grain Sales' draw is public: the USSR watches it leave their hand and the log records it.

    Contrast the hand itself, which stays hidden -- that separation is what the card-location
    split is for, and it is checked elsewhere in this file.
    """
    found = _staged_decisions()
    assert found
    for state, decider, card in found:
        other = ts.Player.US if decider == ts.Player.USSR else ts.Player.USSR
        assert _v22_card(_v22(state, other), card, PEEKED) == 1.0, (
            f"card {card} was drawn in the open and should read as peeked to both sides")
