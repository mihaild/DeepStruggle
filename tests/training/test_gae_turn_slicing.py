"""Audit T-4: --slice-turn-boundaries reaches the per-player GAE pass.

The interleaved pass cuts a transition whose game turn changes by zeroing its non_terminal, which for the
advantage and the return is a done at that row. The per-player pass (ts.gae_per_player) reads dones only as that
cut, so the same rows are marked done for it.
"""

from __future__ import annotations

import numpy as np
import torch

from ai.training.rollout_buffer import RolloutBuffer

T, N, OBS, ACT = 24, 3, 4, 6


def _buffer(seed: int, extra_dones: bool = False) -> RolloutBuffer:
    rng = np.random.default_rng(seed)
    turns = np.repeat(np.arange(1, T // 6 + 1), 6)[:T]
    turns = np.stack([turns, np.roll(turns, 2), np.roll(turns, 4)], axis=1).astype(np.int8)
    players = rng.choice([1, -1], size=(T, N)).astype(np.int8)
    dones = np.zeros((T, N), np.float32)
    dones[T - 5, 1] = 1.0
    if extra_dones:
        dones[:-1] = np.maximum(dones[:-1], (turns[:-1] != turns[1:]).astype(np.float32))
    rewards = np.where(rng.random((T, N)) < 0.2, rng.uniform(-1, 1, (T, N)), 0).astype(np.float32)
    values = rng.uniform(-0.7, 0.7, (T, N)).astype(np.float32)
    buf = RolloutBuffer(buffer_size=T, num_envs=N, obs_dim=OBS, action_dim=ACT, device="cpu")
    for t in range(T):
        buf.add(obs=torch.zeros(N, OBS), masks=torch.ones(N, ACT, dtype=torch.uint8),
                actions=torch.zeros(N, dtype=torch.long), log_probs=torch.zeros(N), rewards=rewards[t],
                dones=dones[t], values_win=torch.from_numpy(values[t]), values_vp=torch.zeros(N),
                players=players[t], turns=turns[t])
    return buf


def _gae(buf: RolloutBuffer, per_player: bool, sliced: bool) -> tuple:
    buf.compute_gae(torch.full((N,), 0.3), torch.zeros(N), torch.ones(N, dtype=torch.int8), gamma=1.0,
                    gae_lambda=0.9, slice_turn_boundaries=sliced, per_player_gae=per_player,
                    last_side_vp=torch.zeros(2, N))
    # normalise_advantages rescales; the returns are the unnormalised advantage + value.
    return buf.returns_win.clone(), buf.advantages.clone()


def test_the_interleaved_cut_is_a_done_at_the_turns_last_row() -> None:
    for seed in range(4):
        sliced = _gae(_buffer(seed), per_player=False, sliced=True)
        marked = _gae(_buffer(seed, extra_dones=True), per_player=False, sliced=False)
        for a, b in zip(sliced, marked):
            assert torch.allclose(a, b, atol=1e-6)


def test_the_per_player_pass_cuts_at_the_same_rows() -> None:
    for seed in range(4):
        sliced = _gae(_buffer(seed), per_player=True, sliced=True)
        marked = _gae(_buffer(seed, extra_dones=True), per_player=True, sliced=False)
        plain = _gae(_buffer(seed), per_player=True, sliced=False)
        for a, b in zip(sliced, marked):
            assert torch.allclose(a, b, atol=1e-6)
        assert not torch.allclose(sliced[0], plain[0])
