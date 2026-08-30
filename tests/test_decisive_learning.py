"""Tests for priority sampling and the decisive-decision probe."""

from typing import List

import numpy as np
import pytest
import torch
import ts_engine as ts

from ai.eval.decisive_probe import DecisiveStats, measure_decisive
from ai.training.rollout_buffer import RolloutBuffer
from bindings.action_encoder import ActionEncoder


def _buffer(advantages: List[float]) -> RolloutBuffer:
    b = RolloutBuffer(buffer_size=len(advantages), num_envs=1, device="cpu")
    for t, adv in enumerate(advantages):
        b.advantages[t, 0] = adv
    return b


def test_alpha_zero_is_a_permutation() -> None:
    """Uniform sampling must still visit every transition exactly once."""
    b = _buffer([0.0, 5.0, 0.1, 2.0])
    idx = b.priority_indices(0.0)
    assert sorted(int(i) for i in idx) == [0, 1, 2, 3]


def test_priority_favours_large_absolute_advantage() -> None:
    torch.manual_seed(0)
    b = _buffer([0.001] * 20 + [10.0])
    counts = np.zeros(21)
    for _ in range(200):
        for i in b.priority_indices(1.0):
            counts[int(i)] += 1
    # The single high-advantage transition should dominate the 20 low ones.
    assert counts[20] > counts[:20].sum(), f"high-advantage step drawn {counts[20]} vs {counts[:20].sum()}"


def test_priority_uses_magnitude_not_sign() -> None:
    """A large negative advantage is as informative as a large positive one."""
    torch.manual_seed(0)
    b = _buffer([-8.0, 0.001, 8.0, 0.001])
    counts = np.zeros(4)
    for _ in range(200):
        for i in b.priority_indices(1.0):
            counts[int(i)] += 1
    assert counts[0] > counts[1] * 5
    assert counts[2] > counts[3] * 5


def test_degenerate_advantages_fall_back_to_uniform() -> None:
    b = _buffer([0.0, 0.0, 0.0])
    idx = b.priority_indices(1.0)
    assert len(idx) == 3


# --- decisive probe ---------------------------------------------------------------------

def test_probe_ignores_chance_nodes() -> None:
    """A die roll is not a decision, so it must never be scored against the policy."""
    seen_chance = []

    def always_first(state: ts.GameState, player: ts.Player) -> int:
        if state.ctx().decision_type == ts.DecisionType.ROLL_DIE:
            seen_chance.append(1)
        legal = np.flatnonzero(ActionEncoder.get_legal_mask(state))
        return int(legal[0]) if len(legal) else 0

    stats = measure_decisive(always_first, num_games=3, max_steps=400)
    assert seen_chance, "fixture never reached a chance node; test would be vacuous"
    assert stats.decisions > 0


def test_forced_positions_are_separated_from_blunders() -> None:
    """Where every action loses, taking one is not a mistake and must not be counted."""
    def always_first(state: ts.GameState, player: ts.Player) -> int:
        legal = np.flatnonzero(ActionEncoder.get_legal_mask(state))
        return int(legal[0]) if len(legal) else 0

    stats = measure_decisive(always_first, num_games=6, max_steps=800)
    assert stats.loss_taken <= stats.loss_avoidable


def test_rates_are_nan_when_nothing_was_available() -> None:
    empty = DecisiveStats()
    assert np.isnan(empty.win_take_rate)
    assert np.isnan(empty.loss_avoid_rate)


def test_metrics_dict_carries_both_rates_and_counts() -> None:
    s = DecisiveStats(decisions=100, win_available=4, win_taken=3,
                      loss_avoidable=10, loss_taken=2)
    m = s.as_metrics()
    assert m["decisive_win_take_rate"] == pytest.approx(0.75)
    assert m["decisive_loss_avoid_rate"] == pytest.approx(0.8)
    assert m["decisive_win_available"] == 4.0
    assert m["decisive_loss_avoidable"] == 10.0
