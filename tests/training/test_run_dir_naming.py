"""A run's short name belongs in the directory name.

The convention is `<engine>-<attempt>-<seed>` -- an `E`, an engine number, a two-digit attempt
and a two-digit seed, e.g. `E9-99-01`. A free-form directory name says nothing about which
engine trained it or which seed it used, and recovering that after the fact takes a
`git merge-base` against two commit hashes -- long enough that a set of cross-engine comparisons
once got written up as same-engine ones. The directory name is the copy every later command
quotes, so it is the copy that has to carry it.

No step budget appears in the name: one directory holds every budget of a lineage, and each
snapshot's own filename already carries the budget it was taken at.

The directory is rooted in the SHARED data tree, not in `data/` relative to the current
directory. `data/` is git-ignored, so a git worktree gets its own empty one and a run launched
from a worktree would write where nothing else looks, then lose it when the worktree is removed.
`tools/lib/data_root.py` resolves the main checkout from git; these tests therefore compare
against that root rather than a literal `data/checkpoints`.
"""
from __future__ import annotations

import pytest

from ai.training.generic_trainer import _resolve_run_dir
from tools.lib.data_root import checkpoints_dir

TS = "20260912_181622"


def test_run_name_becomes_the_directory_prefix() -> None:
    import os

    got = _resolve_run_dir(None, "E9-99-01", "v2", TS)
    assert got == os.path.join(checkpoints_dir(), "E9-99-01_20260912_181622")
    assert os.path.isabs(got), (
        "the run directory must be absolute, or it means a different place depending on which "
        "worktree the command was run from")


def test_without_a_run_name_the_old_default_still_applies() -> None:
    """Not an error: pre-existing scripts and smoke runs keep working."""
    import os

    assert _resolve_run_dir(None, None, "v2", TS) == os.path.join(
        checkpoints_dir(), "run_v2_20260912_181622")


def test_an_explicit_output_dir_is_honoured_when_it_carries_the_name() -> None:
    out = "/tmp/checkpoints/E9-99-01_rerun"
    assert _resolve_run_dir(out, "E9-99-01", "v2", TS) == out
    assert _resolve_run_dir(out + "/", "E9-99-01", "v2", TS) == out + "/"


def test_a_run_name_the_output_dir_contradicts_is_an_error() -> None:
    """The quiet version of this writes one run's weights into a directory named for another."""
    with pytest.raises(ValueError, match="is not in output_dir"):
        _resolve_run_dir("/tmp/checkpoints/some_training_run", "E9-99-01", "v2", TS)


@pytest.mark.parametrize("bad", ["E9-99", "some_training_run", "e9-99-01", "E9-99-01-80M", "9-99-01"])
def test_a_name_that_is_not_the_scheme_is_rejected(bad: str) -> None:
    """A trailing step budget is rejected along with the rest: one directory holds every budget
    of a lineage, so a steps field in the directory name is a claim that goes stale the moment
    the run is continued."""
    with pytest.raises(ValueError, match="is not <engine>-<attempt>-<seed>"):
        _resolve_run_dir(None, bad, "v2", TS)
