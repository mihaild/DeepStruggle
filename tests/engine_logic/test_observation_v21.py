"""The v2 observation layout: what it splits, and what it must keep merged.

v2.1 widens the card block from 12 features to 13 to separate two things the legacy slot 0 could
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
V21_FEATURES = 13
LEGACY_FEATURES = 12

DECK_OR_HIDDEN, MY_HAND, KNOWN_OPPONENT_HAND = 0, 1, 2
DISCARD, REMOVED, ONGOING, PEEKED, NOT_IN_GAME = 3, 4, 5, 6, 7
V21_PROPERTY_BASE = 8
LEGACY_PROPERTY_BASE = 7


def _fresh() -> ts.GameState:
    state = ts.GameState()
    ts.Engine.init_game(state, 4242)
    return state


def _v2(state: ts.GameState, side: ts.Player) -> np.ndarray:
    return np.asarray(ts.extract_observation(state, side, layout="v2.1"), dtype=np.float32)


def _legacy(state: ts.GameState, side: ts.Player) -> np.ndarray:
    return np.asarray(ts.extract_observation(state, side, layout="legacy"), dtype=np.float32)


def _card_slot(obs: np.ndarray, card: int, slot: int, features: int = V21_FEATURES) -> float:
    return float(obs[BOARD + (card - 1) * features + slot])


def test_widths() -> None:
    state = _fresh()
    assert _legacy(state, ts.Player.US).shape == (ts.OBS_SIZE_LEGACY,) == (4293,)
    assert _v2(state, ts.Player.US).shape == (ts.OBS_SIZE_V21,) == (3891,)


def test_legacy_is_the_default() -> None:
    """Existing callers pass no flag and must keep the layout their checkpoints were trained on."""
    state = _fresh()
    assert np.array_equal(np.asarray(ts.extract_observation(state, ts.Player.US)),
                          _legacy(state, ts.Player.US))


def test_v21_changes_only_the_card_block_and_drops_history() -> None:
    """Two differences from legacy, and nothing else.

    The card block widens by one feature, and the 512-float history block is gone -- it was never
    written by anything, so it carried no information to lose. Every other section has to survive
    untouched, and the tails cannot simply be compared end to end any more because they are now
    different lengths.
    """
    state = _fresh()
    for side in (ts.Player.US, ts.Player.USSR):
        legacy, v21 = _legacy(state, side), _v2(state, side)
        assert np.array_equal(legacy[:BOARD], v21[:BOARD]), "board features drifted"

        l_glob = BOARD + 110 * LEGACY_FEATURES
        v_glob = BOARD + 110 * V21_FEATURES
        assert np.array_equal(legacy[l_glob:l_glob + 76], v21[v_glob:v_glob + 76]), (
            "global features drifted")

        # Legacy: globals, then 512 of history, then turn aggregates and active player.
        # v2.1: globals, then straight to turn aggregates and active player.
        assert np.array_equal(legacy[l_glob + 76 + 512:], v21[v_glob + 76:]), (
            "turn aggregates or active player drifted")

        # And the block that was dropped really was all zeros, so nothing was thrown away.
        assert not legacy[l_glob + 76: l_glob + 76 + 512].any(), (
            "the history block was non-zero, so dropping it discarded real information")


def test_card_properties_survive_the_shift() -> None:
    """The five property features move from 7..11 to 8..12 and must carry the same values."""
    state = _fresh()
    legacy, v2 = _legacy(state, ts.Player.US), _v2(state, ts.Player.US)
    for card in (1, 30, 33, 110):
        for k in range(5):
            assert _card_slot(v2, card, V21_PROPERTY_BASE + k) == pytest.approx(
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
GLOBAL_BASE = BOARD + 110 * V21_FEATURES
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

    card_block = set(range(BOARD, BOARD + 110 * V21_FEATURES))
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
    base = BOARD + (held - 1) * V21_FEATURES
    assert moved == {base + DECK_OR_HIDDEN, base + KNOWN_OPPONENT_HAND}, (
        f"revealing one card should move exactly its two location slots, got {sorted(moved)}")
    assert after[base + KNOWN_OPPONENT_HAND] == 1.0
    assert after[base + DECK_OR_HIDDEN] == 0.0


def test_the_batch_runner_reports_its_own_width() -> None:
    legacy_runner = ts.VectorizedBatchRunner(4, 7, "legacy")
    v2_runner = ts.VectorizedBatchRunner(4, 7, "v2.1")
    assert legacy_runner.obs_width == 4293 and legacy_runner.layout == "legacy"
    assert v2_runner.obs_width == 3891 and v2_runner.layout == "v2.1"
    assert np.asarray(legacy_runner.get_observations()).shape == (4, 4293)
    assert np.asarray(v2_runner.get_observations()).shape == (4, 3891)


def test_the_batch_runner_defaults_to_legacy() -> None:
    """Every existing call site constructs a runner without the flag."""
    runner = ts.VectorizedBatchRunner(2, 99)
    assert runner.layout == "legacy" and runner.obs_width == 4293


def test_runner_rows_match_the_single_state_extractor() -> None:
    """The batched path and the one-off path must agree, in every layout."""
    for layout in ("legacy", "v2.1", "v2.2"):
        runner = ts.VectorizedBatchRunner(3, 555, layout)
        rows = np.asarray(runner.get_observations())
        for i in range(3):
            state = runner.get_state(i)
            ctx = state.ctx()
            side = ctx.decision_player if ctx.decision_player != ts.Player.NONE \
                else state.phasing_player
            direct = np.asarray(ts.extract_observation(state, side, layout=layout))
            assert np.array_equal(rows[i], direct), (
                f"batched and direct observations disagree at env {i}, layout={layout}")


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


def test_the_legacy_layout_is_untouched_by_the_china_fix() -> None:
    """Legacy has to keep reproducing the observation its checkpoints were trained against."""
    state = _fresh()
    holder = state.china_card_holder
    obs = _legacy(state, holder)
    assert _card_slot(obs, CHINA, ONGOING, LEGACY_FEATURES) == 1.0
    assert _card_slot(obs, CHINA, MY_HAND, LEGACY_FEATURES) == 0.0


# --- v2.2: Europe, the headline, and Chernobyl -------------------------------------------------

V22_BOARD = 84 * 26
V22_CARD_F = 14
V22_GLOBAL = V22_BOARD + 110 * V22_CARD_F
EUROPE_VP = V22_GLOBAL + 64
CTX = V22_GLOBAL + 72
HEADLINE_STAGE, HEADLINE_FIRST_MINE, HEADLINE_SECOND_MINE = CTX + 20, CTX + 21, CTX + 22
CHERNOBYL_REGION = CTX + 23


def _v22(state: ts.GameState, side: ts.Player) -> np.ndarray:
    return np.asarray(ts.extract_observation(state, side, layout="v2.2"), dtype=np.float32)


def _v22_card(obs: np.ndarray, card: int, slot: int) -> float:
    """v2.2's board is 84x26, so the module-level _card_slot's 84x28 offset does not apply."""
    return float(obs[V22_BOARD + (card - 1) * V22_CARD_F + slot])


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
        assert -1.0 <= float(obs[V22_GLOBAL + 64 + r]) <= 1.0


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
# The engine stages a card in ctx.temp_cards when a decision is *about* that card -- Grain Sales
# hands one over and asks whether to play it, Star Wars offers one from the discard pile -- and
# does so without touching card_locations. So the card kept reading DECK_OR_HIDDEN, and the player
# being asked could not see what they were deciding about. v2.2 shows it in the PEEKED slot, but
# only to the player whose decision it is, and only where they could not already see it.


def _staged_decisions(max_games: int = 150):
    """Real positions where a card is staged for the player to move. Reached by play, because
    ctx.temp_cards is exposed to Python as a copy and cannot be set."""
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
            if int(ctx.temp_card_cnt) >= 1 and ctx.decision_player != ts.Player.NONE:
                card = int(ctx.temp_cards[0])
                if 1 <= card <= 110:
                    out.append((state.clone(), ctx.decision_player, card))
            mask = np.asarray(ActionEncoder.get_legal_mask(state))
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                break
            ts.Engine.step_flat(state, int(rng.choice(legal)))
        if len(out) >= 5:
            break
    return out


V23_CARD_F = 15
STAGED_FOR_ME = 14


def _v23(state: ts.GameState, side: ts.Player) -> np.ndarray:
    return np.asarray(ts.extract_observation(state, side, layout="v2.3"), dtype=np.float32)


def _v23_card(obs: np.ndarray, card: int, slot: int) -> float:
    return float(obs[V22_BOARD + (card - 1) * V23_CARD_F + slot])


def test_v22_is_unchanged_by_the_staged_card_work() -> None:
    """Arms F and F2 trained on v2.2 before any of this. Its width is identical to what the
    staged-card change would have produced, so a silent alteration would never have been caught --
    the checkpoints would simply have been evaluated against an observation they never saw."""
    assert int(ts.OBS_SIZE_V22) == 84 * 26 + 110 * 14 + 101
    found = _staged_decisions()
    assert found
    for state, decider, card in found:
        if state.get_card_location(card) == ts.CardLocation.PEEKED_TEMP:
            continue
        obs = np.asarray(ts.extract_observation(state, decider, layout="v2.2"), dtype=np.float32)
        # v2.2 does not reveal it; that is the behaviour F and F2 learned against.
        assert _v22_card(obs, card, DECK_OR_HIDDEN) in (0.0, 1.0)


def test_v23_marks_a_staged_card_for_the_player_deciding() -> None:
    found = _staged_decisions()
    assert found, "no staged-card decision reached; widen the search before trusting this"
    for state, decider, card in found:
        assert _v23_card(_v23(state, decider), card, STAGED_FOR_ME) == 1.0, (
            f"card {card} is staged for {decider} and not marked")


def test_v23_does_not_mark_it_for_the_other_player() -> None:
    found = _staged_decisions()
    assert found
    for state, decider, card in found:
        other = ts.Player.US if decider == ts.Player.USSR else ts.Player.USSR
        assert _v23_card(_v23(state, other), card, STAGED_FOR_ME) == 0.0


def test_v23_keeps_the_cards_real_location() -> None:
    """The mark is a feature, not an overwritten slot -- which is what stops Ask Not, whose
    staged cards are the player's own, from reading as something other than theirs."""
    found = _staged_decisions()
    assert found
    for state, decider, card in found:
        loc = state.get_card_location(card)
        if ts.in_hand_of(loc, decider):
            obs = _v23(state, decider)
            assert _v23_card(obs, card, MY_HAND) == 1.0
            assert _v23_card(obs, card, STAGED_FOR_ME) == 1.0


def test_v23_hides_peeked_temp_from_a_player_who_is_not_deciding() -> None:
    """PEEKED_TEMP mapped to the peeked slot for both sides. Measured over 1,500 games the
    opponent never moves during either card that uses it, so this closes a latent hole rather
    than a live leak -- but the next card to stage across a change of mover would have one."""
    found = _staged_decisions()
    assert found
    for state, decider, card in found:
        if state.get_card_location(card) != ts.CardLocation.PEEKED_TEMP:
            continue
        other = ts.Player.US if decider == ts.Player.USSR else ts.Player.USSR
        obs = _v23(state, other)
        assert _v23_card(obs, card, PEEKED) == 0.0
        assert _v23_card(obs, card, DECK_OR_HIDDEN) == 1.0
