"""The engine and the Python codec must agree about the flat layout, slot for slot.

The repack that added section 4's heads widened the space 212 -> 220 and briefly left the two
sides disagreeing: `ActionEncoder.FLAT_ACTION_SIZE` said 220 while the engine still returned a
212-wide mask, because the binding stated the array shape as a literal separately from the
allocation. Nothing raised. Every index past the resolution node would have been read as the
wrong thing.

Widths are the cheap half. The offsets are checked by round-tripping a real action through each
block, which is what catches a block that moved without its counterpart.
"""

from __future__ import annotations

import numpy as np
import pytest
import ts_engine as ts

from bindings.action_encoder import ActionEncoder


def _fresh() -> ts.GameState:
    s = ts.GameState()
    ts.Engine.init_game(s, 17)
    return s


def test_the_two_sides_agree_on_the_width() -> None:
    engine_width = len(ts.Engine.get_flat_action_mask(_fresh()))
    assert engine_width == ActionEncoder.FLAT_ACTION_SIZE == 220


def test_the_vectorized_runner_agrees_too() -> None:
    """The training path has its own mask shape, stated separately in the bindings."""
    runner = ts.VectorizedBatchRunner(4, 1)
    masks = np.asarray(runner.get_action_masks())
    assert masks.shape == (4, ActionEncoder.FLAT_ACTION_SIZE)


def test_the_blocks_are_contiguous_and_ordered() -> None:
    """No gaps and no overlaps: the repack closed the merge's 116..118 hole deliberately."""
    A = ActionEncoder
    assert A.CARD_OFFSET == 0
    assert A.PLAY_MODE_OFFSET == A.CARD_OFFSET + 110
    assert A.OP_MODE_OFFSET == A.PLAY_MODE_OFFSET + 2      # shares the OPS_* slots
    assert A.ROLL_DIE_INDEX == A.PLAY_MODE_OFFSET + 5
    assert A.NODE_OFFSET == A.ROLL_DIE_INDEX + 1
    assert A.BRANCH_OFFSET == A.NODE_OFFSET + 84
    assert A.CONFIRM_DONE_INDEX == A.BRANCH_OFFSET + 8
    assert A.DEFCON_VALUE_OFFSET == A.CONFIRM_DONE_INDEX + 1
    assert A.REGION_OFFSET == A.DEFCON_VALUE_OFFSET + 5
    assert A.FLAT_ACTION_SIZE == A.REGION_OFFSET + 6


@pytest.mark.parametrize("country", [0, 1, 42, 83])
def test_a_country_round_trips_through_the_node_block(country: int) -> None:
    """The block most likely to be read at the wrong offset, because it is the widest."""
    state = _fresh()
    idx = ActionEncoder.NODE_OFFSET + country
    action = ActionEncoder.decode(state, idx)
    assert action.decision_type == ts.DecisionType.POINT_NODE
    assert action.primary_id == country


@pytest.mark.parametrize("branch", [0, 3, 7])
def test_a_branch_round_trips(branch: int) -> None:
    state = _fresh()
    action = ActionEncoder.decode(state, ActionEncoder.BRANCH_OFFSET + branch)
    assert action.decision_type == ts.DecisionType.CHOOSE_BRANCH
    assert action.primary_id == branch


def test_confirm_done_is_one_index() -> None:
    state = _fresh()
    action = ActionEncoder.decode(state, ActionEncoder.CONFIRM_DONE_INDEX)
    assert action.is_confirm_done()
