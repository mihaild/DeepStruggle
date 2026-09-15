"""The batched paths must not discard the engine's verdict.

`step_flat_all` reports 0 for refused, 1 for accepted, 2 for terminal, per game, and both callers
threw the vector away -- the same unchecked-return defect that put 1,355 phantom steps into a
replay, here in the two hottest paths. A refusal in a rollout means the game did not advance while
the trainer credits the transition anyway.

Measured at the time of writing: 0 refusals in 5,052 batched steps with actions sampled from the
legal mask, so these are guards rather than fixes. A guard nothing exercises is decoration, so the
first test forces a refusal and requires it to be raised.
"""

from __future__ import annotations

import numpy as np
import pytest

import ts_engine as ts
from bindings.ts_env import TsVectorizedEnv
from tools.lib.game_step import IllegalActionError


def _illegal_action_for(state: ts.GameState) -> int:
    """A flat index whose decode answers a different question than the one being asked.

    The engine rejects an action whose decision_type is not the context's, so this is refused
    wherever the game happens to be.
    """
    want = int(state.ctx().decision_type)
    # Flat 0..109 decode to SELECT_CARD; 110..113 to SELECT_PLAY_MODE.
    return 110 if want == int(ts.DecisionType.SELECT_CARD) else 0


def test_a_refused_action_in_a_batch_is_raised_not_swallowed() -> None:
    env = TsVectorizedEnv(num_envs=8, base_seed=4242)
    env.reset_all()

    masks = np.array(env.runner.get_action_masks(), copy=False)
    actions = [int(np.flatnonzero(masks[i])[0]) for i in range(8)]
    # One game in the batch gets an action for a decision nobody asked about.
    actions[3] = _illegal_action_for(env.runner.get_state(3))

    with pytest.raises(IllegalActionError) as exc:
        env.step(np.asarray(actions, dtype=np.int64))
    assert "env 3" in str(exc.value), f"the failing game was not named: {exc.value}"


def test_a_legal_batch_steps_cleanly() -> None:
    """The guard must not fire on ordinary play, or it would halt every run."""
    env = TsVectorizedEnv(num_envs=16, base_seed=99)
    env.reset_all()
    rng = np.random.default_rng(0)

    for _ in range(200):
        masks = np.array(env.runner.get_action_masks(), copy=False)
        actions = []
        for i in range(16):
            legal = np.flatnonzero(masks[i])
            actions.append(int(rng.choice(legal)) if len(legal) else 211)
        env.step(np.asarray(actions, dtype=np.int64))   # must not raise
