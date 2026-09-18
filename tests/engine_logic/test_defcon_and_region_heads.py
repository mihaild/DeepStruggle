"""P17 section 4: two heads that name concepts the game already has.

A head earns its place by naming something first-class. DEFCON and Region are -- scoring cards
and DEFCON restrictions are stated in terms of them -- so Summit, How I Learned and Chernobyl get
value heads rather than branch indices that mean nothing to a network.

The test that matters is `test_summit_can_leave_defcon_alone`. Summit reads "may degrade or
improve the DEFCON level by 1", so leaving it is a legal outcome, and as a pair of +1/-1 branches
there was no index that meant it. As "set DEFCON to V" over {current-1, current, current+1} it is
a positive choice like any other.
"""

from __future__ import annotations

import numpy as np
import pytest
import ts_engine as ts

from bindings.action_encoder import ActionEncoder

SUMMIT = ts.CardData.get_card_by_name("Summit")
HOW_I_LEARNED = ts.CardData.get_card_by_name("How I Learned to Stop Worrying")
CHERNOBYL = ts.CardData.get_card_by_name("Chernobyl")


def _at_branch(card: int, defcon: int = 5, seed: int = 5) -> ts.GameState:
    """Drive `card`'s event to its CHOOSE_BRANCH, draining the dice Summit rolls first."""
    s = ts.GameState()
    ts.Engine.init_game(s, seed)
    s.current_phase = ts.Phase.ACTION_ROUND
    s.defcon = defcon
    ts.CardHandlers.trigger_event(s, card, ts.Player.US)
    for _ in range(8):
        if s.ctx().decision_type != ts.DecisionType.ROLL_DIE:
            break
        ts.Engine.step(s, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
    assert s.ctx().decision_type == ts.DecisionType.CHOOSE_BRANCH, (
        f"card {card} did not reach a branch: {s.ctx().decision_type}")
    return s


def _legal(state: ts.GameState) -> list[int]:
    return [int(i) for i in np.flatnonzero(
        np.asarray(ts.Engine.get_flat_action_mask(state), dtype=np.uint8))]


@pytest.mark.parametrize("defcon,expected", [
    (5, [4, 5]),          # +1 clipped away
    (3, [2, 3, 4]),
    (1, [1, 2]),          # -1 clipped away
])
def test_summit_offers_exactly_one_step_either_way(defcon: int, expected: list[int]) -> None:
    s = _at_branch(SUMMIT, defcon=defcon)
    levels = sorted(i - ActionEncoder.DEFCON_VALUE_OFFSET + 1 for i in _legal(s))
    assert levels == expected


def test_summit_can_leave_defcon_alone() -> None:
    """The option the +1/-1 encoding could not express, which is why this is a head."""
    s = _at_branch(SUMMIT, defcon=3)
    same = ActionEncoder.DEFCON_VALUE_OFFSET + 3 - 1
    assert same in _legal(s), "the current level must be offered -- the card says 'may'"
    assert ts.Engine.try_step_flat(s, same)
    assert int(s.defcon) == 3, "choosing the current level changed DEFCON"


@pytest.mark.parametrize("target", [1, 2, 3, 4, 5])
def test_how_i_learned_offers_every_level(target: int) -> None:
    """"Set the DEFCON level to any level desired (1-5)" -- the same head, a wider mask."""
    s = _at_branch(HOW_I_LEARNED, defcon=4)
    idx = ActionEncoder.DEFCON_VALUE_OFFSET + target - 1
    assert idx in _legal(s)
    assert ts.Engine.try_step_flat(s, idx)
    # DEFCON 1 ends the game, so the level only survives on the board above it.
    if target > 1:
        assert int(s.defcon) == target


@pytest.mark.parametrize("region", [0, 1, 2, 3, 4, 5])
def test_chernobyl_offers_every_region(region: int) -> None:
    s = _at_branch(CHERNOBYL)
    idx = ActionEncoder.REGION_OFFSET + region
    assert idx in _legal(s)
    assert ts.Engine.try_step_flat(s, idx)
    assert s.has_flag(ts.EffectBits.CHERNOBYL_ACTIVE)


@pytest.mark.parametrize("card", [SUMMIT, HOW_I_LEARNED, CHERNOBYL])
def test_the_head_round_trips(card: int) -> None:
    """Every offered index must decode to an action that re-encodes to the same index.

    Encoding routes on the RESOLVING CARD, not on a flag: a caller building a MicroAction from
    the per-decision mask has only a type and an index, so a flag it cannot know about would make
    every one of these choices illegal. The fuzzer found exactly that.
    """
    s = _at_branch(card, defcon=3)
    for idx in _legal(s):
        action = ts.decode_flat_action(s, idx)
        assert int(ts.encode_micro_action(s, action)) == idx
        assert ts.Engine.try_step_flat(s.clone(), idx)


@pytest.mark.parametrize("card", [SUMMIT, HOW_I_LEARNED, CHERNOBYL])
def test_no_generic_branch_slot_is_offered(card: int) -> None:
    """The point of a head: these cards no longer occupy anonymous branch indices."""
    s = _at_branch(card, defcon=3)
    branch_block = range(ActionEncoder.BRANCH_OFFSET,
                         ActionEncoder.BRANCH_OFFSET + 8)
    assert not (set(_legal(s)) & set(branch_block))
