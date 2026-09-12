"""The short name from research/run_nomenclature.md belongs in the directory name.

A directory called `p1_scalar_nofilter` does not say which engine trained it, which seed it used,
or which row of the nomenclature table it is. Recovering that took a `git merge-base` against two
commit hashes, and in the meantime a set of cross-engine comparisons was written up as
same-engine ones. The name is the copy every later command quotes, so it is the copy that has to
carry it.
"""
from __future__ import annotations

import pytest

from ai.training.generic_trainer import _resolve_run_dir

TS = "20260912_181622"


def test_run_name_becomes_the_directory_prefix() -> None:
    assert _resolve_run_dir(None, "E3-12-21", "v2", TS) == \
        "data/checkpoints/E3-12-21_20260912_181622"


def test_without_a_run_name_the_old_default_still_applies() -> None:
    """Not an error: pre-existing scripts and smoke runs keep working."""
    assert _resolve_run_dir(None, None, "v2", TS) == "data/checkpoints/run_v2_20260912_181622"


def test_an_explicit_output_dir_is_honoured_when_it_carries_the_name() -> None:
    out = "/workspace/data/checkpoints/E3-12-21_rerun"
    assert _resolve_run_dir(out, "E3-12-21", "v2", TS) == out
    assert _resolve_run_dir(out + "/", "E3-12-21", "v2", TS) == out + "/"


def test_a_run_name_the_output_dir_contradicts_is_an_error() -> None:
    """The quiet version of this writes E3-12-21's weights into a directory named for E3-09."""
    with pytest.raises(ValueError, match="is not in output_dir"):
        _resolve_run_dir("data/checkpoints/p1_scalar_nofilter", "E3-12-21", "v2", TS)


@pytest.mark.parametrize("bad", ["E3-12", "p1_identity", "e3-12-21", "E3-12-21-80M", "3-12-21"])
def test_a_name_that_is_not_the_scheme_is_rejected(bad: str) -> None:
    """`E3-12-21-80M` is rejected too: one directory holds every budget of a lineage, so a steps
    field in the directory name is a claim that goes stale the moment the run is continued."""
    with pytest.raises(ValueError, match="is not <engine>-<attempt>-<seed>"):
        _resolve_run_dir(None, bad, "v2", TS)
