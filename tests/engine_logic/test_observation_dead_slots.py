"""Which decision-context slots carry signal, pinned.

Two of the eight DECISION_TYPE one-hot slots are permanently zero after P17 and are documented
as reserved in `ctx_slots` (`engine/include/ts/game_state.hpp`). A comment saying a slot is empty
decays the moment something starts writing to it, so the claim is measured here instead.

The other direction matters just as much. A first-legal walk reported OP_MODE[REALIGN] dead too,
and it is not: the lowest legal index at a resolution node is never OPS_REALIGN (114), so that
walk simply never realigns. Acting on it would have deleted a live feature. Random play is what
separates "never happens" from "this sample never went there" -- see
`research/method/measurement_pitfalls.md`.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import ts_engine as ts

GLOBAL_BASE = int(ts.OBS_SIZE) - 100
CTX_BASE = GLOBAL_BASE + 72
DECISION_TYPE = CTX_BASE + 0
OP_MODE = CTX_BASE + 8

#: Slot -> why it can never fire. Both are structural, not empirical: NONE is not a decision a
#: player is asked for, and P17 retired the CHOOSE_TIMING_BRANCH node outright.
RESERVED = {
    DECISION_TYPE + int(ts.DecisionType.NONE): "NONE is never asked of a player",
    DECISION_TYPE + 3: "CHOOSE_TIMING_BRANCH is retired; EVENT/OPS_* carry the timing",
}


@pytest.fixture(scope="module")
def span():
    """Per-slot (min, max) over random play, which reaches op modes a first-legal walk cannot."""
    rng = random.Random(7)
    lo = np.full(int(ts.OBS_SIZE), np.inf)
    hi = np.full(int(ts.OBS_SIZE), -np.inf)
    seen = 0
    for seed in range(120):
        s = ts.GameState()
        ts.Engine.init_game(s, 30_000 + seed)
        for _ in range(900):
            if ts.Engine.is_terminal(s):
                break
            mask = ts.Engine.get_flat_action_mask(s)
            legal = [i for i, v in enumerate(mask) if v]
            if not legal:
                break
            o = np.asarray(ts.extract_observation(s, s.ctx().decision_player), dtype=np.float64)
            np.minimum(lo, o, out=lo)
            np.maximum(hi, o, out=hi)
            seen += 1
            ts.Engine.step_flat(s, int(rng.choice(legal)))
    assert seen > 10_000, f"only {seen} positions; too thin to call a slot dead"
    return lo, hi


def test_the_reserved_slots_never_fire(span) -> None:
    lo, hi = span
    for slot, why in RESERVED.items():
        assert lo[slot] == 0.0 and hi[slot] == 0.0, (
            f"slot {slot} is documented as reserved ({why}) but reached {hi[slot]}; "
            f"either the engine changed or the comment in ctx_slots is now wrong")


def test_every_other_context_slot_carries_signal(span) -> None:
    """The complement, so "reserved" cannot quietly grow to cover a feature that broke."""
    lo, hi = span
    dead = [i for i in range(DECISION_TYPE, CTX_BASE + 19)
            if lo[i] == hi[i] and i not in RESERVED]
    assert not dead, (
        f"context slots {dead} never varied and are not documented as reserved -- either a "
        f"feature stopped being written, or this sample never reached what sets it")
