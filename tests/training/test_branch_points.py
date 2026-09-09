"""Any snapshot can be branched from, and asking for one that does not exist says so.

A run used to keep a single resume state, overwritten at every snapshot, so its only branch point
was its end. That is what stopped a seed replicate of arm E's 160M->240M stretch: the weights were
there, the optimiser moments were not. The failure mode to guard against now is the opposite one --
asking to branch at a step count the directory does not hold and silently getting the newest state
instead, which produces a run that looks entirely normal and answers a different question.
"""

import os
import tempfile
from typing import Iterator

import pytest
import torch
import torch.nn as nn

from ai.training.generic_trainer import (
    RESUME_AT_STEPS,
    RESUME_FILENAME,
    resolve_resume,
    resume_states_in,
)


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)


@pytest.fixture
def run_dir() -> Iterator[str]:
    """A run directory shaped like one a training run leaves behind."""
    with tempfile.TemporaryDirectory() as d:
        for steps in (5_046_272, 10_027_008, 15_007_744):
            torch.save({"total_env_steps": steps},
                       os.path.join(d, RESUME_AT_STEPS.format(steps=steps)))
            torch.save(_Net().state_dict(), os.path.join(d, f"snapshot_{steps}steps.pt"))
        torch.save({"total_env_steps": 15_007_744}, os.path.join(d, RESUME_FILENAME))
        yield d


class TestFindingBranchPoints:
    def test_every_per_snapshot_state_is_found(self, run_dir: str) -> None:
        assert sorted(resume_states_in(run_dir)) == [5_046_272, 10_027_008, 15_007_744]

    def test_the_run_level_state_is_not_mistaken_for_one(self, run_dir: str) -> None:
        """resume_state.pt has no step count in its name and must not parse as one."""
        assert all(RESUME_FILENAME not in p for p in resume_states_in(run_dir).values())

    def test_a_directory_without_any_is_empty_not_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            assert resume_states_in(d) == {}

    def test_a_missing_directory_is_empty_not_an_error(self) -> None:
        assert resume_states_in("/nonexistent/path/at/all") == {}


class TestResolving:
    def test_a_directory_resolves_to_the_run_level_state(self, run_dir: str) -> None:
        assert resolve_resume(run_dir) == os.path.join(run_dir, RESUME_FILENAME)

    def test_a_file_resolves_to_itself(self, run_dir: str) -> None:
        p = os.path.join(run_dir, RESUME_AT_STEPS.format(steps=10_027_008))
        assert resolve_resume(p) == p

    @pytest.mark.parametrize("sep", [":", "@"])
    def test_a_step_count_resolves_to_that_snapshot(self, run_dir: str, sep: str) -> None:
        got = resolve_resume(f"{run_dir}{sep}10027008")
        assert got == os.path.join(run_dir, RESUME_AT_STEPS.format(steps=10_027_008))

    def test_an_absent_step_count_raises_and_lists_what_exists(self, run_dir: str) -> None:
        """Silently falling back to the newest would run a different experiment and look fine."""
        with pytest.raises(FileNotFoundError) as e:
            resolve_resume(f"{run_dir}:160000000")
        msg = str(e.value)
        assert "160,000,000" in msg
        assert "5,046,272" in msg and "15,007,744" in msg, "the error must say what is available"

    def test_a_path_containing_a_colon_but_no_step_count_is_a_plain_path(self) -> None:
        """Only a trailing all-digit component means "branch here"; the rest is a path."""
        with tempfile.TemporaryDirectory() as d:
            odd = os.path.join(d, "run:name")
            os.makedirs(odd)
            assert resolve_resume(odd) == os.path.join(odd, RESUME_FILENAME)
