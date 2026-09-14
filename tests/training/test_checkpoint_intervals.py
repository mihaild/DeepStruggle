"""Snapshots are frequent, step-tagged resume files are not.

A resume file is 48MB against a snapshot's 13MB. Written at every snapshot with a 5M snapshot
interval, they are 1.5GB per 160M-step run and become 80% of the directory -- while what they are
*for*, branch points, is wanted every tens of millions of steps rather than every five.
"""
from __future__ import annotations

from typing import List, Optional

import pytest


def tagged_resume_steps(snapshot_steps: List[int], resume_every: int,
                        enabled: bool = True) -> List[int]:
    """The decision rule in generic_trainer's snapshot block, in isolation.

    Kept as a pure function so the interval logic has a test that does not need a GPU, a model
    or an hour of training to run.
    """
    written: List[int] = []
    last: Optional[int] = None
    for steps in snapshot_steps:
        if enabled and (last is None or steps - last >= resume_every):
            written.append(steps)
            last = steps
    return written


def test_resume_is_written_far_less_often_than_snapshots() -> None:
    snaps = list(range(5_000_000, 165_000_000, 5_000_000))  # 5M interval to 160M
    assert len(snaps) == 32
    written = tagged_resume_steps(snaps, resume_every=40_000_000)
    assert written == [5_000_000, 45_000_000, 85_000_000, 125_000_000], written
    assert len(written) == 4, "32 snapshots must not produce 32 resume files"


def test_interval_is_measured_from_what_was_written() -> None:
    """Not from the loop counter: an irregular snapshot cadence must not drift the interval."""
    snaps = [10_000_000, 12_000_000, 55_000_000, 56_000_000, 95_000_000]
    written = tagged_resume_steps(snaps, resume_every=40_000_000)
    assert written == [10_000_000, 55_000_000, 95_000_000]
    for a, b in zip(written, written[1:]):
        assert b - a >= 40_000_000


def test_first_snapshot_always_gets_one() -> None:
    """Otherwise a short run has no branch point at all."""
    assert tagged_resume_steps([3_000_000], resume_every=40_000_000) == [3_000_000]


def test_disabled_writes_none() -> None:
    snaps = list(range(5_000_000, 165_000_000, 5_000_000))
    assert tagged_resume_steps(snaps, resume_every=40_000_000, enabled=False) == []


@pytest.mark.parametrize("every", [20_000_000, 40_000_000, 80_000_000])
def test_spacing_holds_for_any_interval(every: int) -> None:
    snaps = list(range(5_000_000, 405_000_000, 5_000_000))
    written = tagged_resume_steps(snaps, resume_every=every)
    assert written, "some resume files must be written"
    for a, b in zip(written, written[1:]):
        assert b - a >= every


def test_the_cli_default_is_the_coarse_one() -> None:
    """Guards against the default silently reverting to 'every snapshot'."""
    import ai.training.train as train_mod

    parser = None
    for name in ("build_parser", "make_parser", "get_parser"):
        if hasattr(train_mod, name):
            parser = getattr(train_mod, name)()
            break
    if parser is None:
        pytest.skip("train.py does not expose its parser as a function")
    defaults = {a.dest: a.default for a in parser._actions}
    assert defaults.get("resume_every_steps") == 40_000_000
