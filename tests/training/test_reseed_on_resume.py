"""A resumed run diverges when given a new seed, and does not when it is not.

Restoring the RNG is right for continuing an interrupted run and wrong for branching one. The
distinction was silently absent: --seed set torch's RNG at startup and load_resume_state overwrote
it, so a "new seed" continuation replayed the original sampling and differed only in its deals.
"""

import os
import tempfile
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from ai.training.generic_trainer import load_resume_state, save_resume_state


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)


class _Trainer:
    """The surface save_resume_state and load_resume_state actually touch."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.reference_net = _Net()
        self.optimizer = torch.optim.Adam(_Net().parameters(), lr=1e-3)
        self.total_env_steps = 0
        self.total_iterations = 0


def _write(path: str, seed: Any) -> None:
    torch.manual_seed(1234)
    np.random.seed(1234)
    save_resume_state(path, _Net(), _Trainer(), iteration=7, total_env_steps=1000,
                      elapsed_seconds=12.0, seed=seed)


def _draw_after_load(path: str, seed: Any) -> tuple[float, float]:
    """The next torch and numpy draws after resuming, which is what actually differs."""
    torch.manual_seed(999)          # a distinct starting point, so a restore is visible
    np.random.seed(999)
    load_resume_state(path, _Net(), _Trainer(), seed=seed)
    return float(torch.rand(1).item()), float(np.random.rand())


class TestReseedOnResume:
    def test_a_new_seed_diverges_from_the_written_stream(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "resume_state.pt")
            _write(p, seed=20260916)
            same = _draw_after_load(p, seed=20260916)
            different = _draw_after_load(p, seed=20260918)
        assert same != different, (
            "a resume given a different seed replayed the original RNG stream; only the "
            "environment deals would have differed, which is half a seed change")

    def test_the_same_seed_restores_the_stream(self) -> None:
        """Two resumes at the same seed must be identical, or an ordinary continuation is not one."""
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "resume_state.pt")
            _write(p, seed=20260916)
            assert _draw_after_load(p, seed=20260916) == _draw_after_load(p, seed=20260916)

    def test_no_seed_given_restores_the_stream(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "resume_state.pt")
            _write(p, seed=20260916)
            assert _draw_after_load(p, seed=None) == _draw_after_load(p, seed=None)

    def test_a_state_predating_the_recorded_seed_honours_an_explicit_one(self) -> None:
        """A resume file written before the seed was recorded has no seed in it, and those are
        exactly the checkpoints a continuation branches from. An explicit --seed must still take
        effect."""
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "resume_state.pt")
            _write(p, seed=None)
            blob = torch.load(p, map_location="cpu", weights_only=False)
            del blob["seed"]                        # exactly what an older file looks like
            torch.save(blob, p)
            a = _draw_after_load(p, seed=20260918)
            b = _draw_after_load(p, seed=20260919)
        assert a != b, "two different seeds produced the same stream on a pre-seed resume file"

    def test_the_seed_is_recorded_so_the_next_resume_can_compare(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "resume_state.pt")
            _write(p, seed=20260916)
            assert torch.load(p, map_location="cpu", weights_only=False)["seed"] == 20260916
