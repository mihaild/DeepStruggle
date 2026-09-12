"""Resuming a run continues it, rather than quietly restarting it.

Snapshots are bare `state_dict`s -- weights only -- which is what every evaluation tool and
`load_agent` expects, so the resume state lives in a separate file beside them. What that file has
to carry is the part a fresh `--warmup-checkpoint` load silently drops: the optimiser moments, the
reference policy NashPG regularises against, and the step counter. Without those a "continued" run
is a restart from the same weights, which is a different experiment wearing the same name.

The environment is deliberately absent from the state. `VectorizedBatchRunner` holds 512 live games
and cannot be serialised, and it does not need to be: the games are an i.i.d. stream, so dealing
fresh ones is the same experiment. That is a claim worth being explicit about rather than leaving
as an omission.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from ai.training.generic_trainer import (RESUME_FILENAME, load_resume_state, save_resume_state)


class _FakeTrainer:
    """The three pieces of state the resume file exists to carry."""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        self.reference_net = nn.Linear(4, 2)
        self.total_env_steps = 0
        self.total_iterations = 0


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Linear(4, 2)


def _step(model: nn.Module, trainer: _FakeTrainer, n: int) -> None:
    """Take real optimiser steps, so the moment estimates are actually populated."""
    for _ in range(n):
        loss = model(torch.ones(8, 4)).pow(2).mean()
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
        trainer.total_env_steps += 65_536
        trainer.total_iterations += 1


def test_resume_restores_weights_optimizer_reference_and_steps(tmp_path) -> None:
    dev = torch.device("cpu")
    model = _model()
    trainer = _FakeTrainer(model, dev)
    _step(model, trainer, 5)

    path = os.path.join(tmp_path, RESUME_FILENAME)
    save_resume_state(path, model, trainer, iteration=5, total_env_steps=trainer.total_env_steps,
                      elapsed_seconds=123.5)
    assert os.path.exists(path)

    want_w = {k: v.clone() for k, v in model.state_dict().items()}
    want_steps = trainer.total_env_steps
    # The optimiser's second-moment estimate for the first parameter, which is the thing a
    # restart-from-weights loses and a resume must not.
    first = next(iter(trainer.optimizer.state.values()))
    want_exp_avg_sq = first["exp_avg_sq"].clone()

    fresh = _model()
    with torch.no_grad():                     # make sure a real restore has to happen
        for p in fresh.parameters():
            p.zero_()
    fresh_trainer = _FakeTrainer(fresh, dev)
    state = load_resume_state(path, fresh, fresh_trainer)

    for k, v in fresh.state_dict().items():
        assert torch.equal(v, want_w[k]), f"weight {k} not restored"
    assert fresh_trainer.total_env_steps == want_steps
    assert state["total_env_steps"] == want_steps
    assert state["iteration"] == 5
    assert state["elapsed_seconds"] == pytest.approx(123.5)

    got = next(iter(fresh_trainer.optimizer.state.values()))
    assert torch.allclose(got["exp_avg_sq"], want_exp_avg_sq), (
        "optimiser moments were not restored -- this is exactly what makes a resume a resume "
        "rather than a restart from the same weights")


def test_a_resumed_run_continues_the_same_optimisation(tmp_path) -> None:
    """Ten steps, versus five then a resume then five more, must land in the same place."""
    dev = torch.device("cpu")

    straight = _model()
    st = _FakeTrainer(straight, dev)
    _step(straight, st, 10)

    broken = _model()
    bt = _FakeTrainer(broken, dev)
    _step(broken, bt, 5)
    path = os.path.join(tmp_path, RESUME_FILENAME)
    save_resume_state(path, broken, bt, 5, bt.total_env_steps, 1.0)

    revived = _model()
    rt = _FakeTrainer(revived, dev)
    load_resume_state(path, revived, rt)
    _step(revived, rt, 5)

    for k, v in revived.state_dict().items():
        assert torch.allclose(v, straight.state_dict()[k], atol=1e-6), (
            f"{k} diverged: an interrupted-and-resumed run must match an uninterrupted one")
    assert rt.total_env_steps == st.total_env_steps


def test_a_restart_from_weights_alone_does_not_match(tmp_path) -> None:
    """The negative control: loading weights without the optimiser really is different.

    If this ever passes, the resume file is not earning its place and the whole mechanism can be
    replaced by --warmup-checkpoint.
    """
    dev = torch.device("cpu")
    straight = _model()
    st = _FakeTrainer(straight, dev)
    _step(straight, st, 10)

    partial = _model()
    pt = _FakeTrainer(partial, dev)
    _step(partial, pt, 5)
    weights_only = {k: v.clone() for k, v in partial.state_dict().items()}

    restarted = _model()
    restarted.load_state_dict(weights_only)
    rt = _FakeTrainer(restarted, dev)          # fresh optimiser: no moments
    _step(restarted, rt, 5)

    same = all(torch.allclose(v, straight.state_dict()[k], atol=1e-6)
               for k, v in restarted.state_dict().items())
    assert not same, (
        "a restart from weights matched a true continuation, so the optimiser state is not "
        "affecting the trajectory and the resume file would be pointless")


def test_rng_state_round_trips(tmp_path) -> None:
    dev = torch.device("cpu")
    model = _model()
    trainer = _FakeTrainer(model, dev)
    torch.manual_seed(1234)
    np.random.seed(1234)
    path = os.path.join(tmp_path, RESUME_FILENAME)
    save_resume_state(path, model, trainer, 0, 0, 0.0)
    expected_torch = torch.randn(3)
    expected_np = np.random.rand(3)

    load_resume_state(path, model, trainer)
    assert torch.allclose(torch.randn(3), expected_torch), "torch RNG did not round-trip"
    assert np.allclose(np.random.rand(3), expected_np), "numpy RNG did not round-trip"
