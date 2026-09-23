"""TsVectorizedEnv rebuilds the runner's cached observations and masks exactly when needed.

`reset_game` and `step_flat_all` rebuild every env they touch. `set_state` does not, so a start
position injected through it needs a refresh. The env used to refresh all envs whenever any game
ended, which cost ~4.4 ms of single-threaded observation building per step, and it injected start
positions in `reset_all` and `reset_env` without refreshing afterwards. These tests pin both halves.
"""

from __future__ import annotations

import numpy as np
import ts_engine as ts

from bindings.ts_env import TsVectorizedEnv


def _cached(env: TsVectorizedEnv) -> tuple[np.ndarray, np.ndarray]:
    return (np.array(env.runner.get_observations(), copy=True),
            np.array(env.runner.get_action_masks(), copy=True))


def _random_legal(masks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.array([int(rng.choice(np.flatnonzero(m))) if m.any() else 0 for m in masks])


def test_step_leaves_nothing_for_a_full_refresh_to_change() -> None:
    """After every step, including steps where games end and reset, the buffers are current."""
    env = TsVectorizedEnv(num_envs=16, base_seed=5)
    _, masks, _ = env.reset_all()
    rng = np.random.default_rng(0)
    ended = 0
    for _ in range(600):
        _, masks, _, dones, _ = env.step(_random_legal(masks, rng))
        ended += int(dones.sum())
        obs_before, masks_before = _cached(env)
        env.runner.refresh_all()
        obs_after, masks_after = _cached(env)
        assert np.array_equal(obs_before, obs_after)
        assert np.array_equal(masks_before, masks_after)
        masks = masks_after
    assert ended > 0, "no game ended, so the auto-reset path was not exercised"


def _state_after(steps: int, seed: int) -> ts.GameState:
    r = ts.VectorizedBatchRunner(1, seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        m = np.array(r.get_action_masks(), copy=False)[0]
        r.step_flat_all([int(rng.choice(np.flatnonzero(m)))], False)
    return r.get_state(0)


def _expected(state: ts.GameState) -> tuple[np.ndarray, np.ndarray]:
    ctx = state.ctx()
    p = ctx.decision_player if ctx.decision_player != ts.Player.NONE else state.phasing_player
    return (np.asarray(ts.extract_observation(state, p), dtype=np.float32),
            np.asarray(ts.ActionMask.generate_flat_mask(state), dtype=np.uint8))


def test_injected_start_positions_are_visible_in_the_buffers() -> None:
    start = _state_after(40, seed=11)
    exp_obs, exp_mask = _expected(start)
    env = TsVectorizedEnv(num_envs=4, base_seed=3, start_provider=lambda i: start if i == 2 else None)

    env.reset_all()
    obs, masks = _cached(env)
    assert np.array_equal(obs[2], exp_obs) and np.array_equal(masks[2], exp_mask), "reset_all"

    env.reset_env(2)
    obs, masks = _cached(env)
    assert np.array_equal(obs[2], exp_obs) and np.array_equal(masks[2], exp_mask), "reset_env"


def test_an_injection_during_step_auto_reset_is_refreshed() -> None:
    start = _state_after(40, seed=13)
    exp_obs, exp_mask = _expected(start)
    env = TsVectorizedEnv(num_envs=8, base_seed=9, start_provider=lambda i: start)
    _, masks, _ = env.reset_all()
    rng = np.random.default_rng(1)
    for _ in range(3000):
        _, masks, _, dones, _ = env.step(_random_legal(masks, rng))
        if dones.any():
            i = int(np.flatnonzero(dones)[0])
            obs, cached_masks = _cached(env)
            assert np.array_equal(obs[i], exp_obs) and np.array_equal(cached_masks[i], exp_mask)
            return
    raise AssertionError("no game ended within 3000 steps")
