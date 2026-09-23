"""Seat balancing and per-seat advantage normalisation: the anti-collapse levers.

Both are off by default, and the first thing tested is that off means untouched: a run launched
without the flags must make exactly the random draws it made before they existed, or every
baseline launched after this change would silently differ from the ones before it.
"""

from __future__ import annotations

import random

import pytest
import torch
import torch.nn as nn

from ai.training.opponent_pool import OpponentPool
from ai.training.rollout_buffer import RolloutBuffer


def _pool(seat_balance: bool, seed: int = 7, members: int = 3) -> OpponentPool:
    nets = [nn.Linear(2, 2) for _ in range(members)]
    return OpponentPool(nets, num_envs=64, frac=0.3, seed=seed, seat_balance=seat_balance)


def test_off_makes_the_same_draws_as_before() -> None:
    """With seat balancing off, sides and opponents come from the same RNG calls as ever."""
    a, b = _pool(False), _pool(False)
    b.observe_selfplay(1.0, 1000.0)          # would matter only under balancing
    for i in range(200):
        a.start_iteration(); b.start_iteration()
        assert a.current_id == b.current_id
        a.on_episode_end(i % 64, victory_points=1.0)
        b.on_episode_end(i % 64, victory_points=1.0)
    assert list(a.learner_side) == list(b.learner_side)
    assert list(a.is_mixed) == list(b.is_mixed)


def test_pressure_follows_the_self_play_split() -> None:
    p = _pool(True)
    assert p.pressure() == 0.0
    for _ in range(20):
        p.observe_selfplay(us_wins=40.0, games=400.0)   # US wins 10%
    assert p.weak_side() == 1 and p.pressure() == pytest.approx(1.0, abs=0.05)
    for _ in range(20):
        p.observe_selfplay(us_wins=360.0, games=400.0)  # US wins 90%
    assert p.weak_side() == -1 and p.pressure() > 0.9


def test_under_pressure_the_weak_seat_plays_more_and_more_envs_are_mixed() -> None:
    p = _pool(True)
    for _ in range(20):
        p.observe_selfplay(us_wins=20.0, games=400.0)   # the US is losing
    on_weak, mixed = 0, 0
    trials = 4000
    for i in range(trials):
        p.on_episode_end(i % 64, victory_points=None)
        on_weak += int(p.learner_side[i % 64] == 1)
        mixed += int(p.is_mixed[i % 64])
    assert on_weak / trials == pytest.approx(0.9, abs=0.03)
    assert mixed / trials == pytest.approx(0.8, abs=0.03)


def test_the_draw_prefers_opponents_the_weak_seat_beats_about_half_the_time() -> None:
    p = _pool(True, members=3)
    for _ in range(20):
        p.observe_selfplay(us_wins=20.0, games=400.0)   # weak seat: US
    easy, even, hard = p.ids
    for oid, rate in ((easy, 0.98), (even, 0.5), (hard, 0.02)):
        p.side_games[(oid, 1)] = 400.0
        p.side_wins[(oid, 1)] = 400.0 * rate
    probs = p.sampling_probs()
    assert probs[1] > probs[0] and probs[1] > probs[2]
    assert min(probs) >= p.pfsp_uniform_mix / 3 - 1e-9     # the floor keeps everyone in play


def test_the_seat_record_survives_a_resume() -> None:
    p = _pool(True)
    p.steps = [10, 20, 30]      # members are keyed by step on resume, so they must differ
    p.observe_selfplay(us_wins=30.0, games=400.0)
    p.side_wins[(p.ids[0], 1)] = 5.0
    p.side_games[(p.ids[0], 1)] = 9.0
    blob = p.state_dict()
    q = _pool(True)
    q.steps = [10, 20, 30]
    q.load_state_dict(blob, list(p.steps))
    assert q.sp_us == pytest.approx(p.sp_us)
    assert q.side_wins[(q.ids[0], 1)] == 5.0 and q.side_games[(q.ids[0], 1)] == 9.0


def _buffer_with(adv_us: torch.Tensor, adv_ussr: torch.Tensor) -> RolloutBuffer:
    buf = RolloutBuffer(buffer_size=2, num_envs=len(adv_us), obs_dim=4, device="cpu")
    buf.advantages = torch.stack([adv_us, adv_ussr])
    buf.players = torch.stack([torch.ones(len(adv_us), dtype=torch.int8),
                               -torch.ones(len(adv_ussr), dtype=torch.int8)])
    return buf


def test_per_seat_normalisation_restores_the_losing_seats_scale() -> None:
    gen = random.Random(3)
    big = torch.tensor([gen.gauss(0, 1.0) for _ in range(256)])
    small = torch.tensor([gen.gauss(0, 0.02) for _ in range(256)])   # the collapsing seat
    shared = _buffer_with(big, small)
    shared.normalise_advantages()
    assert float(shared.advantages[1].std()) < 0.05                   # drowned by the shared divisor
    per_seat = _buffer_with(big, small)
    per_seat.per_seat_adv_norm = True
    per_seat.normalise_advantages()
    assert float(per_seat.advantages[1].std()) == pytest.approx(1.0, abs=0.01)
    assert float(per_seat.advantages[0].std()) == pytest.approx(1.0, abs=0.01)
    # The pre-normalisation record the per-seat metrics come from.
    assert per_seat.side_advantage["ussr"]["std"] == pytest.approx(0.02, abs=0.005)


def test_shared_normalisation_is_unchanged_by_the_refactor() -> None:
    gen = random.Random(5)
    us = torch.tensor([gen.gauss(0.3, 1.0) for _ in range(128)])
    ussr = torch.tensor([gen.gauss(-0.1, 0.5) for _ in range(128)])
    buf = _buffer_with(us, ussr)
    flat = torch.cat([us, ussr])
    expected = (torch.stack([us, ussr]) - flat.mean()) / (flat.std() + 1e-8)
    buf.normalise_advantages()
    assert torch.allclose(buf.advantages, expected)
    assert buf.raw_advantage_std == pytest.approx(float(flat.std()), rel=1e-5)
