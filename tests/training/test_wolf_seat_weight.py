"""--wolf-seat-weight: per-seat surrogate weights from the USSR's self-play win share (WoLF)."""

from __future__ import annotations

import pytest
import torch

from ai.training.nash_pg import wolf_sample_weights, wolf_seat_weights
from ai.training.rollout_buffer import RolloutBuffer


def test_an_even_split_weights_both_seats_one() -> None:
    assert wolf_seat_weights(0.5, 1.0) == pytest.approx((1.0, 1.0))
    assert wolf_seat_weights(0.5, 0.5) == pytest.approx((1.0, 1.0))


def test_power_one_is_the_owners_rule() -> None:
    """w_us = 2x and w_ussr = 2(1 - x): the winning USSR learns slowly, the losing US fast."""
    for x in (0.1, 0.3, 0.7, 0.9):
        w_us, w_ussr = wolf_seat_weights(x, 1.0)
        assert w_us == pytest.approx(2 * x)
        assert w_ussr == pytest.approx(2 * (1 - x))


def test_the_ratio_is_x_over_one_minus_x_to_the_power_and_the_mean_is_one() -> None:
    for x in (0.2, 0.8, 0.95):
        for p in (0.5, 1.0, 2.0):
            w_us, w_ussr = wolf_seat_weights(x, p)
            assert w_us / w_ussr == pytest.approx((x / (1 - x)) ** p)
            assert (w_us + w_ussr) / 2 == pytest.approx(1.0)


def test_neither_weight_reaches_zero() -> None:
    w_us, w_ussr = wolf_seat_weights(1.0, 1.0)
    assert w_ussr > 0.0 and w_us < 2.0


def test_sample_weights_follow_the_acting_seat() -> None:
    players = torch.tensor([1, -1, 1, 0, -1], dtype=torch.int8)
    w = wolf_sample_weights(players, 1.8, 0.2)
    assert w.tolist() == pytest.approx([1.8, 0.2, 1.8, 1.0, 0.2])


def test_get_batches_carries_the_acting_seat_last() -> None:
    buf = RolloutBuffer(buffer_size=2, num_envs=3, obs_dim=4, device="cpu")
    buf.players = torch.tensor([[1, -1, 1], [-1, 1, -1]], dtype=torch.int8)
    batch = next(buf.get_batches(batch_size=6))
    assert len(batch) == 12
    assert sorted(batch[11].tolist()) == [-1, -1, -1, 1, 1, 1]
