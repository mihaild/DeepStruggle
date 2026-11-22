"""Differential test: vectorized compute_gae must match the original elementwise form.

compute_gae was rewritten from a per-element Python loop (128 x 512 = 65,536 individual
GPU index operations per call, 22s at production size) into masked tensor ops over the env
dimension. The timestep loop stays sequential because GAE is a backward recursion.

Equality is pinned directly against the implementation it replaced, not just against
hand-picked cases. (The blunder window the oracle also covered was removed on 2026-11-22.)
"""

import pytest
import torch

from ai.training.rollout_buffer import RolloutBuffer


def reference(buf, last_v_win, last_players, gamma, gae_lambda, slice_tb):
    """The original per-element implementation, kept here only as an oracle."""
    N = buf.num_envs
    dev = buf.device
    last_gae = torch.zeros(N, dtype=torch.float32, device=dev)
    adv = torch.zeros_like(buf.advantages)
    ret = torch.zeros_like(buf.returns_win)

    for t in reversed(range(buf.buffer_size)):
        curr_p = buf.players[t]
        non_terminal = 1.0 - buf.dones[t].float()
        if t == buf.buffer_size - 1:
            sign = (curr_p * last_players).float()
            next_val = sign * last_v_win
        else:
            sign = (curr_p * buf.players[t + 1]).float()
            next_val = sign * buf.values_win[t + 1]
            if slice_tb:
                non_terminal = non_terminal * (buf.turns[t] == buf.turns[t + 1]).float()

        for e in range(N):
            d = buf.rewards[t, e] + gamma * next_val[e] * non_terminal[e] - buf.values_win[t, e]
            last_gae[e] = d + gamma * gae_lambda * sign[e] * non_terminal[e] * last_gae[e]
            adv[t, e] = last_gae[e]
            ret[t, e] = adv[t, e] + buf.values_win[t, e]
    return adv, ret




@pytest.mark.parametrize("slice_tb", [False, True])
def test_vectorized_matches_elementwise(slice_tb: bool) -> None:
    torch.manual_seed(7)
    B, N = 24, 16
    buf = RolloutBuffer(buffer_size=B, num_envs=N, device="cpu")
    buf.rewards.normal_()
    buf.values_win.normal_()
    buf.players.copy_(torch.randint(0, 2, (B, N)).mul(2).sub(1).to(torch.int8))
    buf.turns.copy_(torch.randint(1, 5, (B, N)).to(torch.int8))
    buf.dones.copy_(torch.rand(B, N) < 0.10)
    buf.defcon_blunder.copy_(torch.randint(-1, 2, (B, N)).to(torch.int8))

    lv = torch.randn(N)
    lp = torch.randint(0, 2, (N,)).mul(2).sub(1).to(torch.int8)

    ref_adv, ref_ret = reference(buf, lv, lp, 1.0, 0.95, slice_tb)
    buf.compute_gae(last_v_win=lv, last_v_vp=torch.zeros(N),
                    last_players=lp, gamma=1.0, gae_lambda=0.95,
                    slice_turn_boundaries=slice_tb)

    ref_norm = (ref_adv - ref_adv.mean()) / (ref_adv.std() + 1e-8)
    assert torch.allclose(buf.returns_win, ref_ret, atol=1e-5)
    assert torch.allclose(buf.advantages, ref_norm, atol=1e-5)
