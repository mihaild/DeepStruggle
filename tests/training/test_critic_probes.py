"""Guards for the card-sensitivity and critic-usefulness probes."""
from __future__ import annotations

import numpy as np
import pytest
import ts_engine as ts

from ai.eval.card_sensitivity import (BOARD_FLOATS, CARD_FLOATS, HANDS, SCORING_CARDS,
                                      YARDSTICK_COUNTRIES, _deck, _hand)


def test_card_block_offsets_match_the_layout() -> None:
    """The probe slices the card block by hand; if the layout moved this must fail loudly."""
    assert BOARD_FLOATS == 84 * 26
    assert CARD_FLOATS == 110 * 14
    assert BOARD_FLOATS + CARD_FLOATS + 100 == int(ts.OBS_SIZE)


def test_hand_and_deck_partition_the_cards() -> None:
    state = ts.GameState()
    ts.Engine.init_game(state, 91)
    us, su, deck = _hand(state, ts.Player.US), _hand(state, ts.Player.USSR), _deck(state)
    assert us and su and deck
    assert not (set(us) & set(su)), "a card cannot be in both hands"
    assert not (set(us) & set(deck)) and not (set(su) & set(deck))


def test_set_card_location_moves_a_card_and_changes_the_observation() -> None:
    """The perturbation must actually alter the card block, or the probe measures nothing."""
    state = ts.GameState()
    ts.Engine.init_game(state, 91)
    hand = _hand(state, ts.Player.US)
    deck = _deck(state)
    assert hand and deck

    def card_block(st):
        o = np.asarray(ts.extract_observation(st, ts.Player.US), dtype=np.float32)
        return o[BOARD_FLOATS:BOARD_FLOATS + CARD_FLOATS].copy()

    before = card_block(state)
    moved = state.clone()
    loc = moved.get_card_location(hand[0])
    moved.set_card_location(hand[0], ts.CardLocation.DRAW_DECK)
    moved.set_card_location(deck[0], loc)
    assert moved.get_card_location(deck[0]) == loc
    assert not np.array_equal(card_block(moved), before)


def test_clone_isolates_the_perturbation() -> None:
    """Every perturbation is applied to a clone; the base state must be untouched."""
    state = ts.GameState()
    ts.Engine.init_game(state, 91)
    hand = _hand(state, ts.Player.US)
    original = state.get_card_location(hand[0])
    scratch = state.clone()
    scratch.set_card_location(hand[0], ts.CardLocation.REMOVED_FROM_GAME)
    assert state.get_card_location(hand[0]) == original


def test_yardstick_countries_are_europe_battlegrounds() -> None:
    for cid in YARDSTICK_COUNTRIES:
        info = ts.MapData.get_country_info(cid)
        assert bool(info["battleground"])
        assert str(info["region"]) == str(ts.Region.EUROPE)


def test_set_country_changes_influence() -> None:
    state = ts.GameState()
    ts.Engine.init_game(state, 91)
    cid = YARDSTICK_COUNTRIES[0]
    c = state.get_country(cid)
    us_i, su_i = int(c.us_influence), int(c.ussr_influence)
    scratch = state.clone()
    scratch.set_country(cid, us_i + 4, su_i)
    assert int(scratch.get_country(cid).us_influence) == us_i + 4
    assert int(state.get_country(cid).us_influence) == us_i


@pytest.mark.parametrize("side", [ts.Player.US, ts.Player.USSR])
def test_hands_map_covers_known_and_unknown(side) -> None:
    assert len(HANDS[side]) == 2
    assert len(SCORING_CARDS) == 7


def test_explained_variance_is_circular_by_construction() -> None:
    """G = A + V, so EV = 1 - Var(A)/Var(G); collapsing advantages inflate it toward 1.

    This is why critic_usefulness exists, so pin the identity the argument rests on.
    """
    import torch

    from ai.training.rollout_buffer import explained_variance

    torch.manual_seed(3)
    values = torch.randn(4096)
    for adv_scale, expect_above in ((1.0, None), (0.05, 0.99)):
        adv = torch.randn(4096) * adv_scale
        returns = adv + values
        ev = explained_variance(returns, values)
        # The identity: G - V is exactly A.
        assert torch.allclose(returns - values, adv, atol=1e-6)
        if expect_above is not None:
            assert ev > expect_above, f"small advantages should inflate EV, got {ev}"
