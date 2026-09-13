"""Known-answer tests for the live critic-discrimination metrics.

Every one of these has an answer that can be worked out without running the code, which is the
point: a tracking metric that is silently wrong is worse than no metric, because it reads as
reassurance.
"""
from __future__ import annotations

import numpy as np
import pytest

from ai.training.critic_tracker import CriticTracker, brier_skill, roc_auc


def test_auc_perfect_random_and_inverted() -> None:
    scores = np.array([0.9, 0.8, 0.7, -0.7, -0.8, -0.9])
    labels = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    assert roc_auc(scores, labels) == pytest.approx(1.0)
    assert roc_auc(-scores, labels) == pytest.approx(0.0)


def test_auc_of_a_constant_critic_is_one_half() -> None:
    """The degenerate case this metric exists to catch: ties must average, not win."""
    scores = np.full(200, -0.8)
    labels = np.where(np.arange(200) < 20, 1.0, -1.0)
    assert roc_auc(scores, labels) == pytest.approx(0.5)


def test_auc_is_invariant_to_the_base_rate() -> None:
    """The whole argument for AUC over accuracy: a base-rate shift must not move it."""
    rng = np.random.default_rng(4)
    for n_pos in (500, 100, 20):
        pos = rng.normal(0.6, 0.4, n_pos)
        neg = rng.normal(-0.6, 0.4, 1000)
        scores = np.concatenate([pos, neg])
        labels = np.concatenate([np.ones(n_pos), -np.ones(1000)])
        assert roc_auc(scores, labels) == pytest.approx(0.98, abs=0.03)


def test_auc_is_invariant_to_monotone_rescaling() -> None:
    rng = np.random.default_rng(5)
    scores = rng.normal(size=400)
    labels = np.where(scores + rng.normal(0, 0.5, 400) > 0, 1.0, -1.0)
    base = roc_auc(scores, labels)
    assert roc_auc(scores * 0.01, labels) == pytest.approx(base)
    assert roc_auc(np.tanh(scores), labels) == pytest.approx(base)


def test_auc_undefined_with_one_class() -> None:
    assert np.isnan(roc_auc(np.array([0.1, 0.2]), np.array([1.0, 1.0])))


def test_brier_skill_zero_for_the_base_rate_forecast() -> None:
    labels = np.where(np.arange(1000) < 130, 1.0, -1.0)
    base = float((labels > 0).mean())
    assert brier_skill(np.full(1000, base), labels) == pytest.approx(0.0, abs=1e-9)


def test_brier_skill_positive_when_better_negative_when_worse() -> None:
    labels = np.where(np.arange(1000) < 300, 1.0, -1.0)
    good = np.where(labels > 0, 0.95, 0.05)
    bad = np.where(labels > 0, 0.05, 0.95)
    assert brier_skill(good, labels) > 0.9
    assert brier_skill(bad, labels) < 0.0


def test_tracker_samples_once_per_turn_and_resolves_on_outcome() -> None:
    tr = CriticTracker(num_envs=2, headline_turn=2)
    alive = np.array([True, True])
    # Two steps inside turn 1 must contribute one sample, not two.
    tr.observe(np.array([0.5, -0.5]), np.array([1, 1]), alive)
    tr.observe(np.array([0.6, -0.6]), np.array([1, 1]), alive)
    tr.observe(np.array([0.7, -0.7]), np.array([2, 2]), alive)
    assert tr.metrics(min_samples=1) == {}, "nothing resolved yet, so nothing to report"

    tr.resolve(0, us_won=True)
    tr.resolve(1, us_won=False)
    m = tr.metrics(min_samples=1)
    assert m["critic/samples"] == 4, "two turns x two envs"
    assert m["critic/auc"] == pytest.approx(1.0)


def test_tracker_drops_samples_when_the_outcome_is_unknown() -> None:
    tr = CriticTracker(num_envs=1)
    tr.observe(np.array([0.3]), np.array([1]), np.array([True]))
    tr.resolve(0, us_won=None)
    assert tr.metrics(min_samples=1) == {}


def test_tracker_reset_clears_pending_between_episodes() -> None:
    """A sample from a finished game must never be attached to the next game's result."""
    tr = CriticTracker(num_envs=1)
    tr.observe(np.array([0.9]), np.array([1]), np.array([True]))
    tr.resolve(0, us_won=True)
    tr.observe(np.array([-0.9]), np.array([1]), np.array([True]))
    tr.resolve(0, us_won=False)
    m = tr.metrics(min_samples=1)
    assert m["critic/samples"] == 2
    assert m["critic/auc"] == pytest.approx(1.0)


def test_tracker_turn_resets_are_not_treated_as_the_same_turn() -> None:
    """Auto-reset sends turn back to 1; that must start a fresh sample, not be skipped."""
    tr = CriticTracker(num_envs=1)
    alive = np.array([True])
    tr.observe(np.array([0.5]), np.array([1]), alive)
    tr.observe(np.array([0.4]), np.array([2]), alive)
    tr.resolve(0, us_won=True)
    tr.observe(np.array([-0.5]), np.array([1]), alive)
    tr.resolve(0, us_won=False)
    assert tr.metrics(min_samples=1)["critic/samples"] == 3


def test_dead_envs_contribute_nothing() -> None:
    tr = CriticTracker(num_envs=2)
    tr.observe(np.array([0.5, 0.5]), np.array([1, 1]), np.array([True, False]))
    tr.resolve(0, us_won=True)
    tr.resolve(1, us_won=True)
    assert tr.metrics(min_samples=1)["critic/samples"] == 1
