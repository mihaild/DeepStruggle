"""The split and the scoring rules behind the per-country trunk read-out.

These are the parts that decide whether a number means anything, and both have a failure mode
that looks like a good result: a leaky split inflates every score, and a raw exact-match accuracy
with no baseline reads as competence on countries that are simply empty in almost every position.
"""
from __future__ import annotations

import numpy as np

from ai.eval.state_readout import _exact_accuracy, _mode_baseline, split_by_env


def test_split_keeps_every_position_of_one_env_on_one_side() -> None:
    """The whole point: no game may appear in both halves."""
    env_ids = np.repeat(np.arange(20), 8)
    tr, te = split_by_env(env_ids)
    train_envs = set(env_ids[tr].tolist())
    test_envs = set(env_ids[te].tolist())
    assert train_envs & test_envs == set()
    assert train_envs | test_envs == set(range(20))
    assert len(tr) + len(te) == len(env_ids)


def test_split_is_deterministic_and_roughly_the_requested_fraction() -> None:
    env_ids = np.repeat(np.arange(100), 3)
    tr_a, te_a = split_by_env(env_ids, frac=0.7, seed=0)
    tr_b, te_b = split_by_env(env_ids, frac=0.7, seed=0)
    assert np.array_equal(tr_a, tr_b) and np.array_equal(te_a, te_b)
    assert len(set(env_ids[tr_a].tolist())) == 70


def test_split_never_leaves_the_training_side_empty() -> None:
    """A one-env collection is degenerate, but it must not produce an unsolvable ridge."""
    tr, te = split_by_env(np.zeros(5, dtype=np.int64))
    assert len(tr) > 0


def test_exact_accuracy_rounds_and_clips() -> None:
    """Influence is a non-negative integer, so 2.4 is a correct read of 2 and -0.3 of 0."""
    true = np.array([[2.0, 0.0], [3.0, 1.0]])
    pred = np.array([[2.4, -0.3], [3.6, 1.0]])
    acc = _exact_accuracy(pred, true)
    assert acc[0] == 0.5           # 2.4 -> 2 correct; 3.6 -> 4 wrong
    assert acc[1] == 1.0           # -0.3 clipped to 0 correct; 1.0 correct


def test_mode_baseline_is_the_floor_a_result_has_to_clear() -> None:
    """A country empty in 90% of positions gives 0.9 for free, before any probe runs."""
    train = np.array([[0.0]] * 9 + [[3.0]])
    test = np.array([[0.0]] * 9 + [[3.0]])
    assert _mode_baseline(train, test)[0] == 0.9
