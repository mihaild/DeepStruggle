"""A resumed run diverges when given a new seed, and does not when it is not.

Restoring the RNG is right for continuing an interrupted run and wrong for branching one. The
distinction was silently absent: --seed set torch's RNG at startup and load_resume_state overwrote
it, so a "new seed" continuation replayed the original sampling and differed only in its deals.
"""

import json
import os
import random
import tempfile
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from ai.training.generic_trainer import (load_resume_state, recorded_seeds, resolved_seeds, save_resume_state,
                                         seed_changes)


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


def _write_streams(path: str, seeds: Dict[str, Optional[int]]) -> None:
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    save_resume_state(path, _Net(), _Trainer(), iteration=7, total_env_steps=1000, elapsed_seconds=12.0,
                      seed=seeds["seed"], seeds=seeds)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    blob["env_game_rng_state"] = {"marker": 1}    # what an env with a game-seed generator saves
    torch.save(blob, path)


def _resume(path: str, seeds: Dict[str, Optional[int]]) -> tuple[tuple[float, float, float], Dict[str, Any]]:
    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)
    state = load_resume_state(path, _Net(), _Trainer(), seed=seeds["seed"], seeds=seeds)
    return (float(torch.rand(1).item()), float(np.random.rand()), random.random()), state


class TestEachSeedComponentOnResume:
    """Audit T-18: with --seed unchanged, --seed-sampling / --seed-env / --seed-pool were ignored on resume -- every
    stream was restored. Each component now reseeds its own stream, and only it."""

    def test_an_unchanged_resume_restores_every_stream(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "resume_state.pt")
            run = resolved_seeds(45)
            _write_streams(p, run)
            a, sa = _resume(p, run)
            b, _ = _resume(p, run)
        assert a == b, "an unchanged continuation must restore torch, numpy and Python's random"
        assert sa["env_game_rng_state"] == {"marker": 1} and not sa["pool_reseeded"]

    def test_seed_sampling_alone_reseeds_only_the_sampling_stream(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "resume_state.pt")
            _write_streams(p, resolved_seeds(45))
            same, _ = _resume(p, resolved_seeds(45))
            new, st = _resume(p, resolved_seeds(45, seed_sampling=46))
            again, _ = _resume(p, resolved_seeds(45, seed_sampling=46))
        assert all(x != y for x, y in zip(same, new)), "torch, numpy and Python's random all take the new seed"
        assert new == again, "reseeding is deterministic"
        assert st["env_game_rng_state"] == {"marker": 1} and not st["pool_reseeded"], "env and pool restored"

    def test_seed_env_and_seed_pool_reseed_their_own_streams(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "resume_state.pt")
            _write_streams(p, resolved_seeds(45))
            same, _ = _resume(p, resolved_seeds(45))
            env_draws, env_state = _resume(p, resolved_seeds(45, seed_env=46))
            pool_draws, pool_state = _resume(p, resolved_seeds(45, seed_pool=46))
        assert env_draws == same and env_state["env_game_rng_state"] is None and not env_state["pool_reseeded"]
        assert pool_draws == same and pool_state["pool_reseeded"]
        assert pool_state["env_game_rng_state"] == {"marker": 1}

    def test_a_branch_launched_with_overrides_continues_unchanged_under_them(self) -> None:
        """A run written under --seed 45 --seed-sampling 46 ... resumed with the same flags is a continuation."""
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "resume_state.pt")
            run = resolved_seeds(45, 46, 46, 46)
            _write_streams(p, run)
            blob = torch.load(p, map_location="cpu", weights_only=False)
            assert seed_changes(blob, p, run) == {}
            assert set(seed_changes(blob, p, resolved_seeds(45))) == {"seed_sampling", "seed_env", "seed_pool"}

    def test_an_older_state_reads_its_streams_from_the_runs_metadata(self) -> None:
        """A state written before `seeds` was saved: the run's metadata.json says which streams it used, skipping the
        leg an in-place continuation has just written for itself."""
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "resume_state.pt")
            _write_streams(p, resolved_seeds(45, 46, 46, 46))
            blob = torch.load(p, map_location="cpu", weights_only=False)
            del blob["seeds"]
            assert recorded_seeds(blob, p) == resolved_seeds(45), "no metadata: the default decomposition"
            legs = [{"seed": 45, "seed_sampling": 46, "seed_env": 46, "seed_pool": 46},
                    {"seed": 45, "seed_sampling": 47, "seed_env": 47, "seed_pool": 47}]
            with open(os.path.join(d, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump({**legs[-1], "legs": legs}, f)
            assert recorded_seeds(blob, p) == resolved_seeds(45, 47, 47, 47)
            assert recorded_seeds(blob, p, skip_current_leg=True) == resolved_seeds(45, 46, 46, 46)
