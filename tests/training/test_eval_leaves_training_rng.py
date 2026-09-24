"""Snapshot evaluation must not move the random streams training samples from.

If it did, how often a run is evaluated would change what it learns: training is chaotic at 1e-9,
so one shifted draw puts the rest of the run on another trajectory. With the streams restored, the
evaluation interval (`--snapshot-every-steps`) is a reporting setting and the pool's growth rate is
set on its own (`--pool-every-steps`).
"""

from __future__ import annotations

import random

import numpy as np
import torch

from ai.training.generic_trainer import evaluate_and_log_snapshot, preserves_training_rng


def test_a_wrapped_function_leaves_every_stream_where_it_was() -> None:
    @preserves_training_rng
    def draws() -> float:
        torch.rand(5)
        if torch.cuda.is_available():
            torch.rand(5, device="cuda")
        np.random.rand(5)
        random.random()
        return 1.0

    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    assert draws() == 1.0
    after = (torch.rand(3), np.random.rand(3), random.random(),
             torch.rand(3, device="cuda") if torch.cuda.is_available() else None)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    fresh = (torch.rand(3), np.random.rand(3), random.random(),
             torch.rand(3, device="cuda") if torch.cuda.is_available() else None)
    assert torch.equal(after[0], fresh[0]) and np.array_equal(after[1], fresh[1])
    assert after[2] == fresh[2]
    a_cuda, f_cuda = after[3], fresh[3]
    if a_cuda is not None and f_cuda is not None:
        assert torch.equal(a_cuda, f_cuda)


def test_the_streams_are_restored_when_the_evaluation_raises() -> None:
    @preserves_training_rng
    def fails() -> None:
        torch.rand(5)
        raise RuntimeError("probe failed")

    torch.manual_seed(0)
    try:
        fails()
    except RuntimeError:
        pass
    got = torch.rand(3)
    torch.manual_seed(0)
    assert torch.equal(got, torch.rand(3))


def test_snapshot_evaluation_is_wrapped() -> None:
    assert getattr(evaluate_and_log_snapshot, "__wrapped__", None) is not None


def test_snapshots_every_10m_the_pool_every_5m_and_tf32_by_default() -> None:
    from ai.training.train import build_parser
    a = build_parser().parse_args([])
    assert (a.snapshot_every_steps, a.pool_every_steps, a.tf32) == (10_000_000, 5_000_000, True)


def test_tf32_can_be_turned_off() -> None:
    from ai.training.train import build_parser
    assert build_parser().parse_args(["--no-tf32"]).tf32 is False
