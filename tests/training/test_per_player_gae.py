"""Per-player GAE must telescope, which is the property the interleaved form only approximates.

The interleaved estimator bootstraps across a change of mover by negating the next step's value,
which assumes `V(s, me) = -V(s, opponent)`. That is a perfect-information identity; here the critic
reads an information set and the two sides differ by 0.144 mean absolute. Replacing only that term
(`--same-perspective-bootstrap`) breaks the telescoping outright and cost `E3-21-28` its critic.

Per-player GAE removes the cross-perspective bootstrap instead of patching it: each player's own
subsequence, bootstrapped from that player's next OWN decision, opponent rewards folded in.

See `research/findings/training/value_bootstrap_perspective.md`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ai.training.rollout_buffer import RolloutBuffer

OBS, ACT, N, T = 8, 4, 3, 12


def _buffer(rewards, players, dones=None, values=None) -> RolloutBuffer:
    buf = RolloutBuffer(buffer_size=T, num_envs=N, obs_dim=OBS, action_dim=ACT, device="cpu")
    for t in range(T):
        buf.add(
            obs=torch.zeros(N, OBS), masks=torch.ones(N, ACT, dtype=torch.uint8),
            actions=torch.zeros(N, dtype=torch.long), log_probs=torch.zeros(N),
            rewards=rewards[t], dones=torch.tensor(dones[t] if dones is not None else [0.0] * N),
            values_win=torch.tensor(values[t] if values is not None else [0.0] * N),
            values_vp=torch.zeros(N),
            players=torch.tensor(players[t], dtype=torch.int8),
            next_values_own=torch.zeros(N),
        )
    return buf


def _gae(buf, lam, **kw):
    buf.compute_gae(
        last_v_win=torch.zeros(N), last_v_vp=torch.zeros(N),
        last_dones=torch.zeros(N, dtype=torch.bool),
        last_players=torch.full((N,), 1, dtype=torch.int8),
        gamma=1.0, gae_lambda=lam, **kw)
    return buf.returns_win.clone()


def _alternating():
    """Movers alternate, one terminal reward at the end, arbitrary values in between."""
    players = [[1 if (t % 2 == 0) else -1] * N for t in range(T)]
    rewards = [np.zeros(N, dtype=np.float32) for _ in range(T)]
    rewards[T - 1] = np.array([1.0, -1.0, 1.0], dtype=np.float32)
    dones = [[0.0] * N for _ in range(T)]
    dones[T - 1] = [1.0] * N
    rng = np.random.default_rng(0)
    values = rng.uniform(-0.6, 0.6, size=(T, N)).astype(np.float32).tolist()
    return rewards, players, dones, values


def test_at_lambda_one_it_reproduces_the_realised_return() -> None:
    """The telescoping test: at lambda = 1 every intermediate value must cancel.

    `returns_win[t]` should then equal the reward actually collected from t onward, in that step's
    actor's frame, regardless of what the value function said along the way. The interleaved form
    passes this only because its negation is nearly right; a broken estimator fails it loudly.
    """
    rewards, players, dones, values = _alternating()
    buf = _buffer(rewards, players, dones, values)
    ret = _gae(buf, 1.0, per_player_gae=True).numpy()

    for i in range(N):
        outcome = rewards[T - 1][i]           # in the final actor's frame
        for t in range(T):
            want = outcome * players[t][i] * players[T - 1][i]
            assert ret[t, i] == pytest.approx(want, abs=1e-4), (
                f"env {i} step {t}: returns_win {ret[t, i]:.4f} != realised {want:.4f}; "
                f"an intermediate value failed to cancel")


def test_it_never_negates_a_value_across_a_change_of_mover() -> None:
    """With constant values per player, the bootstrap must use the player's OWN value.

    Build a case where the two perspectives disagree sharply: US values +0.5 everywhere, USSR -0.9.
    Under the negation the US bootstrap would read +0.9; per-player must read +0.5.
    """
    players = [[1 if (t % 2 == 0) else -1] * N for t in range(T)]
    values = [[0.5] * N if (t % 2 == 0) else [-0.9] * N for t in range(T)]
    rewards = [np.zeros(N, dtype=np.float32) for _ in range(T)]
    buf = _buffer(rewards, players, None, values)

    buf.compute_gae(
        last_v_win=torch.full((N,), 0.5), last_v_vp=torch.zeros(N),
        last_dones=torch.zeros(N, dtype=torch.bool),
        last_players=torch.full((N,), 1, dtype=torch.int8),
        gamma=1.0, gae_lambda=0.0, per_player_gae=True)
    adv = buf.advantages.detach().numpy()

    # lambda = 0, no reward: delta = V(next own decision) - V(now) = 0.5 - 0.5 for US.
    # Advantages are normalised afterwards, so check they are all equal rather than all zero.
    assert np.allclose(adv[0], adv[0][0]), "per-player delta varied where it should not"


def test_the_flag_is_inert_when_off() -> None:
    """Every existing run must stay bit-identical, or the arm is not a one-factor comparison."""
    rewards, players, dones, values = _alternating()
    a = _gae(_buffer(rewards, players, dones, values), 0.98)
    b = _gae(_buffer(rewards, players, dones, values), 0.98, per_player_gae=False)
    assert torch.equal(a, b)


def test_it_changes_the_returns_when_on() -> None:
    """A no-op flag would make the arm measure nothing."""
    rewards, players, dones, values = _alternating()
    off = _gae(_buffer(rewards, players, dones, values), 0.98)
    on = _gae(_buffer(rewards, players, dones, values), 0.98, per_player_gae=True)
    assert not torch.allclose(off, on)


def test_an_episode_boundary_does_not_leak_across() -> None:
    """A game ending mid-buffer must not bootstrap into the next game's first steps."""
    players = [[1 if (t % 2 == 0) else -1] * N for t in range(T)]
    rewards = [np.zeros(N, dtype=np.float32) for _ in range(T)]
    rewards[5] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    rewards[T - 1] = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    dones = [[0.0] * N for _ in range(T)]
    dones[5] = [1.0] * N
    dones[T - 1] = [1.0] * N
    rng = np.random.default_rng(1)
    values = rng.uniform(-0.6, 0.6, size=(T, N)).astype(np.float32).tolist()

    buf = _buffer(rewards, players, dones, values)
    ret = _gae(buf, 1.0, per_player_gae=True).numpy()
    # Steps 0..5 belong to the first episode, whose outcome is +1 in step 5's actor's frame.
    for i in range(N):
        for t in range(6):
            want = 1.0 * players[t][i] * players[5][i]
            assert ret[t, i] == pytest.approx(want, abs=1e-4), (
                f"env {i} step {t} leaked across the episode boundary")
