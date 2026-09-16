"""The value target must be bootstrapped from the decider's own view of the result.

`compute_gae` normally bootstraps across a change of mover by negating the next step's value,
which assumes `V(s, me) = -V(s, opponent)`. That is exact under perfect information and false
here: `v_win` is computed from a perspective-filtered observation, so the two evaluate different
information sets. Measured over 227 real positions, `v_US + v_USSR` has mean +0.051 and mean
absolute 0.144 where the identity requires 0.

The error lands hardest on the last decision before the side switches -- a card played for its
event, resolving with no further decisions -- because that advantage is then scored entirely by
the opponent's opinion of the result.

See `research/log/search_cost_and_coverage.md` §8.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import ts_engine as ts
from ai.training.rollout_buffer import RolloutBuffer

OBS, ACT, N, T = 8, 4, 2, 6


def _buffer(next_own: np.ndarray | None) -> RolloutBuffer:
    """A buffer with alternating movers and known values, so the target is hand-checkable."""
    buf = RolloutBuffer(buffer_size=T, num_envs=N, obs_dim=OBS, action_dim=ACT, device="cpu")
    for t in range(T):
        players = torch.full((N,), 1 if t % 2 == 0 else -1, dtype=torch.int8)
        buf.add(
            obs=torch.zeros(N, OBS), masks=torch.ones(N, ACT, dtype=torch.uint8),
            actions=torch.zeros(N, dtype=torch.long), log_probs=torch.zeros(N),
            rewards=np.zeros(N, dtype=np.float32), dones=torch.zeros(N),
            values_win=torch.full((N,), 0.10 * (t + 1)), values_vp=torch.zeros(N),
            players=players,
            next_values_own=(torch.from_numpy(next_own[t]).float()
                             if next_own is not None else None),
        )
    return buf


def _gae(buf: RolloutBuffer, same: bool) -> torch.Tensor:
    buf.compute_gae(
        last_v_win=torch.zeros(N), last_v_vp=torch.zeros(N),
        last_dones=torch.zeros(N, dtype=torch.bool),
        last_players=torch.full((N,), 1, dtype=torch.int8),
        gamma=1.0, gae_lambda=0.98, same_perspective_bootstrap=same,
    )
    return buf.advantages.clone()


def test_it_changes_the_advantage() -> None:
    """A flag that changed nothing would make the arm measure nothing."""
    own = np.full((T, N), -0.5, dtype=np.float32)
    a_default = _gae(_buffer(own), same=False)
    a_same = _gae(_buffer(own), same=True)
    assert not torch.allclose(a_default, a_same), (
        "the same-perspective bootstrap produced identical advantages")


def test_it_uses_the_stored_value_and_does_not_negate_it() -> None:
    """The stored value is already in the actor's frame, so it must be used with sign +1.

    Constructed so the two readings differ in sign: if the implementation negated the stored
    value, the first advantage would come out with the opposite sign.
    """
    own = np.zeros((T, N), dtype=np.float32)
    own[0, :] = 0.8                       # the mover thinks the result is good FOR ITSELF
    buf = _buffer(own)
    adv = _gae(buf, same=True)
    # delta_0 = r + gamma * next_own[0] - v_0 = 0 + 0.8 - 0.1 = +0.7, before the lambda carry.
    # Negating the stored value would give -0.8 - 0.1 = -0.9. Only the sign is asserted, because
    # the GAE carry from later steps shifts the magnitude.
    assert adv[0, 0] > 0, (
        "a result the mover values at +0.8 produced a negative advantage; the stored value is "
        "being negated, but it is already in the actor's frame")


def test_it_refuses_to_run_without_the_stored_values() -> None:
    """Falling back silently would produce an arm that measures the default path."""
    buf = _buffer(None)
    with pytest.raises(ValueError, match="next_values_own"):
        _gae(buf, same=False) if False else buf.compute_gae(
            last_v_win=torch.zeros(N), last_v_vp=torch.zeros(N),
            last_dones=torch.zeros(N, dtype=torch.bool),
            last_players=torch.full((N,), 1, dtype=torch.int8),
            same_perspective_bootstrap=True)


def test_the_default_path_is_untouched() -> None:
    """The flag is off by default, so existing runs must be bit-identical."""
    a_no_field = _gae(_buffer(None), same=False)
    a_with_field = _gae(_buffer(np.full((T, N), 0.42, dtype=np.float32)), same=False)
    assert torch.allclose(a_no_field, a_with_field), (
        "storing next_values_own changed the default path")


def test_the_env_serves_observations_from_either_side() -> None:
    """The machinery the fix needs: a state viewed from a named perspective, per environment."""
    from bindings.ts_env import TsVectorizedEnv

    env = TsVectorizedEnv(num_envs=4, base_seed=77)
    env.reset_all()
    us = env.observations_for(np.array([1, 1, 1, 1], dtype=np.int8))
    ussr = env.observations_for(np.array([-1, -1, -1, -1], dtype=np.int8))
    assert us.shape == (4, int(ts.OBS_SIZE))
    assert not np.array_equal(us, ussr), "both perspectives returned the same observation"

    # And it must agree with the single-state call it stands in for.
    direct = np.asarray(ts.extract_observation(env.runner.get_state(2), ts.Player.USSR),
                        dtype=np.float32)
    assert np.array_equal(ussr[2], direct)
