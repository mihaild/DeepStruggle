"""--adv-norm-learner-only: the advantage statistics come from the learner's own transitions.

A frozen opponent's transitions carry roughly the negative of the learner's advantages. Averaged
in, they keep the batch mean near zero whatever the learner does, so a learner slightly behind its
pool sees every action it took pushed down -- the loop that collapsed the P24 exploiter.
"""

from __future__ import annotations

import torch

from ai.training.rollout_buffer import RolloutBuffer


def _buffer(own_only: bool) -> RolloutBuffer:
    b = RolloutBuffer(buffer_size=4, num_envs=2, obs_dim=8, device="cpu")
    # Env 0 is the learner's; env 1 the frozen opponent's, with mirrored advantages.
    b.advantages = torch.tensor([[-0.2, 0.2], [-0.4, 0.4], [-0.1, 0.1], [-0.3, 0.3]])
    b.learner = torch.tensor([[1.0, 0.0]] * 4)
    b.players = torch.tensor([[1, -1]] * 4, dtype=b.players.dtype)
    b.adv_norm_learner_only = own_only
    return b


def test_the_learners_own_advantages_are_centred() -> None:
    b = _buffer(True)
    b.normalise_advantages()
    own = b.advantages[:, 0]
    assert abs(float(own.mean())) < 1e-6
    assert abs(float(own.std()) - 1.0) < 1e-5


def test_without_the_flag_the_opponent_holds_the_mean_at_zero() -> None:
    b = _buffer(False)
    b.normalise_advantages()
    assert float(b.advantages[:, 0].mean()) < -0.5   # the learner's samples all pushed down


def test_self_play_is_unchanged_by_the_flag() -> None:
    a, c = _buffer(False), _buffer(True)
    for b in (a, c):
        b.learner = torch.ones(4, 2)
    a.normalise_advantages()
    c.normalise_advantages()
    assert torch.equal(a.advantages, c.advantages)


def test_a_seat_with_no_learner_rows_is_normalised_by_the_batch_statistics() -> None:
    """Audit T-3: under --per-seat-adv-norm with a locked learner seat, the frozen opponent's seat has no learner
    rows to take statistics from. It was left as torch.empty_like memory (NaN possible); it now takes the
    batch-wide (learner) statistics, and the learner's seat keeps its own."""
    b = _buffer(True)
    b.per_seat_adv_norm = True
    raw = b.advantages.clone()
    b.normalise_advantages()
    own = raw[:, 0]
    mean, std = own.mean(), own.std() + 1e-8
    assert torch.isfinite(b.advantages).all()
    assert torch.allclose(b.advantages[:, 0], (own - mean) / std, atol=1e-6)
    assert torch.allclose(b.advantages[:, 1], (raw[:, 1] - mean) / std, atol=1e-6)


def test_a_seat_with_one_learner_row_is_normalised_by_the_batch_statistics() -> None:
    b = _buffer(True)
    b.per_seat_adv_norm = True
    b.learner = torch.tensor([[1.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])   # one USSR learner row
    raw = b.advantages.clone()
    b.normalise_advantages()
    stat = raw[b.learner > 0.5]
    mean, std = stat.mean(), stat.std() + 1e-8
    us = raw[:, 0]
    assert torch.allclose(b.advantages[:, 0], (us - us.mean()) / (us.std() + 1e-8), atol=1e-6)
    assert torch.allclose(b.advantages[:, 1], (raw[:, 1] - mean) / std, atol=1e-6)
