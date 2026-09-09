"""Auto-advance must not change the outcome of a game on the vectorized path.

``test_auto_advance_integration.py`` establishes equivalence for ``Engine::step_flat`` on single
states. This file covers the path evaluation actually runs on: ``VectorizedBatchRunner``, which is
what ``tools/lib/batch_tournament.py`` drives and therefore what every tournament number and Elo
anchor is measured through. Enabling ``--auto-advance`` there must be a pure speed change, so any
result taken with the flag on stays comparable to one taken with it off.

The policy has to be a pure function of the *position*, not of a call counter. With auto-advance on,
the engine resolves some decisions itself, so the two runs are asked for a different number of
actions; a policy that consumed a shared RNG stream would then diverge for a reason that has
nothing to do with the flag, and the test would fail while proving nothing. Hashing the mask keeps
the choice identical at identical positions while still spreading play across the action space
rather than pinning it to the lowest legal index.
"""

from __future__ import annotations

import hashlib
from typing import List, Tuple

import numpy as np

import ts_engine as ts

NUM_ENVS = 128
BASE_SEED = 20260906
MAX_STEPS = 4000


def _position_policy(mask_row: np.ndarray) -> int:
    """Pick a legal action as a deterministic function of the mask alone.

    Same position, same choice, regardless of how many decisions preceded it in this run.
    """
    legal = np.flatnonzero(mask_row)
    if legal.size == 0:
        return 0
    digest = hashlib.sha256(mask_row.tobytes()).digest()
    return int(legal[int.from_bytes(digest[:8], "big") % legal.size])


def _play_out(auto_advance: bool) -> Tuple[List[float], List[int], List[int]]:
    """Run NUM_ENVS games to completion and return their outcomes."""
    runner = ts.VectorizedBatchRunner(NUM_ENVS, BASE_SEED)

    for _ in range(MAX_STEPS):
        terminals = runner.get_terminals()
        if all(terminals):
            break
        masks = runner.get_action_masks()
        actions = [
            0 if terminals[i] else _position_policy(np.asarray(masks[i]))
            for i in range(NUM_ENVS)
        ]
        runner.step_flat_all(actions, auto_advance=auto_advance)

    return (
        list(runner.get_terminal_utilities()),
        list(runner.get_victory_points()),
        list(runner.get_turns()),
    )


def test_auto_advance_does_not_change_vectorized_outcomes() -> None:
    """Every game must end with the same result, score and turn either way."""
    util_off, vp_off, turn_off = _play_out(auto_advance=False)
    util_on, vp_on, turn_on = _play_out(auto_advance=True)

    assert vp_off == vp_on, "victory points diverged between auto_advance off and on"
    assert turn_off == turn_on, "final turn diverged between auto_advance off and on"
    assert util_off == util_on, "terminal utility diverged between auto_advance off and on"


def test_the_equivalence_check_is_not_vacuous() -> None:
    """Guard the test above: the games must actually be played and actually be decided.

    An equivalence assertion over games that never start, or that all end in the same trivial
    way, would pass no matter what auto-advance did. This repo has already been bitten by
    diagnostics that reported confident numbers while measuring nothing (research/metrics.md
    section 1), so the check needs its own check.
    """
    util, vp, turns = _play_out(auto_advance=True)

    assert len(util) == NUM_ENVS
    assert max(turns) >= 4, f"games ended implausibly early (max turn {max(turns)})"
    assert len({round(u) for u in util}) > 1, "every game produced the same result"
    assert len(set(vp)) > 1, "every game produced the same score"
