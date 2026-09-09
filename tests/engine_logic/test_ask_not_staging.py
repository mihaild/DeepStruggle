"""Ask Not stages a player's cards without moving them out of their hand.

It used to park each selected card in CardLocation::PEEKED_TEMP, which is what stopped the mask
offering it a second time. That worked, but it meant a player's own cards read as PEEKED to
themselves -- and to the opponent, since PEEKED_TEMP was shown to both sides. "Already selected"
is now tracked in ctx.temp_cards, which is where the handler was recording it anyway.

Reached by play: ctx.temp_cards is exposed to Python as a copy, so the position cannot be built
directly.
"""

from typing import List, Tuple

import numpy as np
import pytest

import ts_engine as ts
from bindings.action_encoder import ActionEncoder

ASK_NOT = 77
#: get_legal_mask returns the 212-dim flat space, not the engine's 112-wide card mask: a card is
#: at index (id - 1) and the pass/confirm is at 211. Indexing it by card id silently checks the
#: next card along, which made one of these tests pass for the wrong reason.
PASS_ACTION = 211


def _card_idx(card: int) -> int:
    return card - 1


def _staging_positions(limit: int = 4) -> List[Tuple[ts.GameState, List[int]]]:
    rng = np.random.default_rng(23)
    out: List[Tuple[ts.GameState, List[int]]] = []
    for g in range(1500):
        if len(out) >= limit:
            break
        state = ts.GameState()
        ts.Engine.init_game(state, 120000 + g)
        for _ in range(4000):
            if ts.Engine.is_terminal(state) or len(out) >= limit:
                break
            ctx = state.ctx()
            if int(ctx.resolving_card) == ASK_NOT and int(ctx.temp_card_cnt) >= 1:
                staged = [int(c) for c in ctx.temp_cards[:int(ctx.temp_card_cnt)]]
                out.append((state.clone(), staged))
            mask = np.asarray(ActionEncoder.get_legal_mask(state))
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                break
            ts.Engine.step_flat(state, int(rng.choice(legal)))
    return out


@pytest.fixture(scope="module")
def positions():
    found = _staging_positions()
    assert found, "no Ask Not staging position reached; widen the search before trusting this"
    return found


def test_staged_cards_stay_in_their_owners_hand(positions) -> None:
    for state, staged in positions:
        for card in staged:
            assert ts.in_hand_of(state.get_card_location(card), ts.Player.US), (
                f"card {card} is staged for discard and has already left the US hand; "
                f"it reads as {state.get_card_location(card)}")


def test_a_staged_card_is_not_offered_again(positions) -> None:
    """The property that moving the card to PEEKED_TEMP used to provide."""
    for state, staged in positions:
        mask = np.asarray(ActionEncoder.get_legal_mask(state))
        again = [c for c in staged if mask[_card_idx(c)]]
        assert not again, f"the mask offers already-staged cards {again} a second time"


def test_unstaged_hand_cards_are_still_offered(positions) -> None:
    for state, staged in positions:
        mask = np.asarray(ActionEncoder.get_legal_mask(state))
        rest = [c for c in range(1, 111)
                if ts.in_hand_of(state.get_card_location(c), ts.Player.US) and c not in staged]
        if rest:
            assert any(mask[_card_idx(c)] for c in rest), (
                "no unstaged hand card is selectable, so the player cannot discard more")


def test_ask_not_no_longer_uses_peeked_temp(positions) -> None:
    """Our Man In Tehran is now its only user, where the cards really are off the deck."""
    for state, _ in positions:
        peeked = [c for c in range(1, 111)
                  if state.get_card_location(c) == ts.CardLocation.PEEKED_TEMP]
        assert not peeked, f"Ask Not still parks cards in PEEKED_TEMP: {peeked}"


def test_confirming_discards_exactly_the_staged_cards(positions) -> None:
    """The cards leave the hand on confirmation, not before."""
    state, staged = positions[0]
    work = state.clone()
    mask = np.asarray(ActionEncoder.get_legal_mask(work))
    assert mask[PASS_ACTION], "confirm-done must be available; Ask Not sets allow_early_stop"
    ts.Engine.step_flat(work, PASS_ACTION)
    for card in staged:
        assert not ts.in_hand_of(work.get_card_location(card), ts.Player.US), (
            f"card {card} was staged and confirmed but is still in hand")
