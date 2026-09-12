"""Ask Not discards each card as it is chosen, so the hand is the record of what is left.

Two designs preceded this one and both kept the chosen cards somewhere other than where the
rules put them. The first parked each selection in CardLocation::PEEKED_TEMP, which stopped the
mask offering it twice but made a player's own cards read as PEEKED -- to themselves and, since
PEEKED_TEMP was shown to both sides, to the opponent. The second left them in hand and tracked
"already chosen" in ctx.temp_cards, a list beside the hand it described.

Neither is needed. A discarded card is in the discard pile, so "already chosen" is "no longer in
hand", which the mask reads directly. How many replacements to draw is what the discard
allowance has been spent down by, so no tally is kept either.

Reached by play rather than built directly: the position needs the engine to have offered Ask
Not's selection, and the interesting property is what the mask does across two consecutive
choices.
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
MAX_DISCARDS = 9


def _card_idx(card: int) -> int:
    return card - 1


def _us_hand(state: ts.GameState) -> List[int]:
    want = ts.hand_of(ts.Player.US)
    known = ts.hand_of(ts.Player.US, True)
    return [c for c in range(1, 111) if state.get_card_location(c) in (want, known)]


def _staging_positions(limit: int = 4) -> List[ts.GameState]:
    """Positions where Ask Not is asking the US which card to discard next."""
    rng = np.random.default_rng(23)
    out: List[ts.GameState] = []
    for g in range(1500):
        if len(out) >= limit:
            break
        state = ts.GameState()
        ts.Engine.init_game(state, 120000 + g)
        for _ in range(4000):
            if ts.Engine.is_terminal(state) or len(out) >= limit:
                break
            ctx = state.ctx()
            if (int(ctx.resolving_card) == ASK_NOT
                    and ctx.decision_type == ts.DecisionType.SELECT_CARD
                    and len(_us_hand(state)) >= 2):
                out.append(state.clone())
                break
            mask = np.asarray(ActionEncoder.get_legal_mask(state))
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                break
            ts.Engine.step_flat(state, int(rng.choice(legal)))
            while (not ts.Engine.is_terminal(state)
                   and state.ctx().decision_player == ts.Player.NONE
                   and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
                ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
    return out


@pytest.fixture(scope="module")
def positions() -> List[ts.GameState]:
    found = _staging_positions()
    assert found, "no Ask Not selection position reached; widen the search before judging"
    return found


def test_the_search_actually_found_the_case(positions) -> None:
    assert len(positions) >= 1


def test_a_chosen_card_leaves_the_hand_and_is_not_offered_again(positions) -> None:
    for state in positions:
        probe = state.clone()
        before = _us_hand(probe)
        chosen = before[0]

        ts.Engine.step(probe, ts.MicroAction(ts.DecisionType.SELECT_CARD, chosen, 0, 0))

        assert probe.get_card_location(chosen) == ts.CardLocation.DISCARD_PILE, (
            f"card {chosen} was chosen for discard and should be in the discard pile, "
            f"not {probe.get_card_location(chosen)}"
        )
        mask = np.asarray(ActionEncoder.get_legal_mask(probe))
        assert mask[_card_idx(chosen)] == 0, (
            f"card {chosen} is already discarded and must not be offered again"
        )


def test_the_rest_of_the_hand_is_still_offered(positions) -> None:
    for state in positions:
        probe = state.clone()
        before = _us_hand(probe)
        chosen = before[0]
        ts.Engine.step(probe, ts.MicroAction(ts.DecisionType.SELECT_CARD, chosen, 0, 0))

        mask = np.asarray(ActionEncoder.get_legal_mask(probe))
        for card in before[1:]:
            assert mask[_card_idx(card)] == 1, (
                f"card {card} is still in the US hand and must still be offered"
            )


def test_no_card_is_parked_in_peeked_temp(positions) -> None:
    """The design this replaced made a player's own cards read as PEEKED to both sides."""
    for state in positions:
        probe = state.clone()
        chosen = _us_hand(probe)[0]
        ts.Engine.step(probe, ts.MicroAction(ts.DecisionType.SELECT_CARD, chosen, 0, 0))
        parked = [c for c in range(1, 111)
                  if probe.get_card_location(c) == ts.CardLocation.PEEKED_TEMP]
        assert parked == [], f"Ask Not left {parked} in PEEKED_TEMP"


def test_confirming_draws_one_replacement_per_discard(positions) -> None:
    """The allowance is the tally: what it has been spent down by is what is drawn back."""
    for state in positions:
        probe = state.clone()
        hand_before = _us_hand(probe)
        discards = hand_before[:2] if len(hand_before) >= 2 else hand_before[:1]
        for card in discards:
            ts.Engine.step(probe, ts.MicroAction(ts.DecisionType.SELECT_CARD, card, 0, 0))

        spent = MAX_DISCARDS - int(probe.ctx().remaining_steps)
        assert spent == len(discards), (
            f"discarded {len(discards)} cards but the allowance moved by {spent}"
        )

        ts.Engine.step(probe, ts.MicroAction(ts.DecisionType.SELECT_CARD, 0, 0, 0x80))

        hand_after = _us_hand(probe)
        assert len(hand_after) == len(hand_before), (
            f"the US discarded {len(discards)} and should have drawn {len(discards)} back: "
            f"hand went from {len(hand_before)} to {len(hand_after)}"
        )
        for card in discards:
            assert card not in hand_after or probe.get_card_location(card) != ts.CardLocation.DISCARD_PILE
