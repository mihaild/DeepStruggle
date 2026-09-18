"""Indices 116..118 are unassigned after the P17 merge, and must stay refused rather than decoded.

The merge freed three slots: the old CHOOSE_TIMING_BRANCH block and the old SELECT_OP_MODE block
collapsed onto the OPS_* resolutions, leaving 116, 117 and 118 with no meaning. `decode_flat_
action_212` refuses them on purpose -- "nothing should arrive here; refuse rather than decode to a
neighbouring meaning" -- because the failure it prevents is silent: 116 sits one past ROLL_DIE and
three before the country block, so a decoder that guessed would hand back a plausible wrong action.

They are NOT reclaimed. Shrinking the space to 209 would save three logits out of 212 and require
moving NODE_OFFSET and BRANCH_OFFSET, which were written out in ~16 places across two languages
when this was measured. That consolidation is done; the shrink is a two-line change now if it is
ever wanted, and these tests are what would catch it going wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
import ts_engine as ts

from bindings.action_encoder import ActionEncoder

RESERVED = list(range(*ActionEncoder.UNASSIGNED_RANGE))


def test_the_range_is_what_the_layout_says() -> None:
    assert RESERVED == [116, 117, 118]
    # Bounded by the two live neighbours, so a layout change cannot leave this stale.
    assert RESERVED[0] == ActionEncoder.ROLL_DIE_INDEX + 1
    assert RESERVED[-1] + 1 == ActionEncoder.NODE_OFFSET


@pytest.mark.parametrize("idx", RESERVED)
def test_decoding_a_reserved_index_is_refused(idx: int) -> None:
    state = ts.GameState()
    ts.Engine.init_game(state, 11)
    action = ActionEncoder.decode(state, idx)
    assert action.decision_type == ts.DecisionType.NONE
    assert action.primary_id == 255


def test_no_mask_ever_offers_a_reserved_index() -> None:
    """The other half: refusing to decode them is only safe if nothing ever proposes one."""
    import random

    rng = random.Random(5)
    seen = 0
    for seed in range(60):
        s = ts.GameState()
        ts.Engine.init_game(s, 90_000 + seed)
        for _ in range(900):
            if ts.Engine.is_terminal(s):
                break
            mask = np.asarray(ts.Engine.get_flat_action_mask(s), dtype=np.uint8)
            for idx in RESERVED:
                assert mask[idx] == 0, f"mask offered reserved index {idx}"
            legal = [int(i) for i in np.flatnonzero(mask)]
            if not legal:
                break
            seen += 1
            ts.Engine.step_flat(s, rng.choice(legal))
    assert seen > 5_000, f"only {seen} positions; too thin to conclude anything"
