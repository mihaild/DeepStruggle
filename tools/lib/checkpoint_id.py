"""One name for a checkpoint, derived from its path.

Snapshot filenames are not unique and cannot be. The step counts a run snapshots at are a
deterministic function of its configuration, so two runs with the same `--snapshot-every-steps`
and the same batch shape produce *byte-identically named* files: both `E4-01-01` and `E4-02-01`
contain a `snapshot_150011904steps.pt`. The identity of a checkpoint is its **path**, and every
tool that reduced it to a basename lost that.

The failure is quiet, which is why it kept happening. `tools/tournament.py` disambiguates
collisions by appending `#1` and `#2`, so nothing crashes and nothing is dropped -- but the report
then says `snapshot_150011904steps#1` beat `snapshot_150011904steps#2`, which is unreadable and
unciteable. A number that cannot be attributed to a run is not a measurement.

So: `E4-02-01@150M`. The run's short name, which is what `research/runs.md` indexes by, plus the
step count rounded to the scale people actually quote.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List

#: A run directory is `<short-name>_<YYYYMMDD>_<HHMMSS>`, e.g. `E4-02-01_20260919_040456`.
#: The short name is what runs.md indexes by, so it is what a label should carry.
_RUN_DIR_RE = re.compile(r"^(E\d+-\d{2}-\d{2})_\d{8}_\d{6}$")
_STEPS_RE = re.compile(r"(\d+)steps")


def _steps_suffix(filename: str) -> str:
    """`snapshot_150011904steps.pt` -> `150M`. Empty when the name carries no step count."""
    m = _STEPS_RE.search(filename)
    if not m:
        stem = os.path.splitext(filename)[0]
        # `snapshot_final`, `resume_state`, a hand-named warmup: keep whatever it says.
        return stem.replace("snapshot_", "").replace("resume_", "") or stem
    steps = int(m.group(1))
    if steps >= 1_000_000:
        return f"{steps / 1_000_000:.0f}M"
    return f"{steps / 1000:.0f}k"


def checkpoint_label(path: str) -> str:
    """A readable, run-attributed name for a checkpoint path.

    `.../E4-02-01_20260919_040456/snapshot_150011904steps.pt` -> `E4-02-01@150M`

    Falls back to the bare stem when the path carries no run directory -- a loose checkpoint such
    as `data/checkpoints/E4_1_warmup.pt` has nothing better to say, and inventing something would
    be worse than being plain.
    """
    path = str(path)
    filename = os.path.basename(path)
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))

    m = _RUN_DIR_RE.match(parent)
    if m:
        return f"{m.group(1)}@{_steps_suffix(filename)}"

    stem = os.path.splitext(filename)[0]
    # A run directory that does not match the naming scheme still beats nothing, as long as it is
    # not the shared checkpoints root -- `checkpoints/snapshot_5M` says less than `snapshot_5M`.
    if parent and parent not in ("checkpoints", "data", ""):
        return f"{parent}@{_steps_suffix(filename)}"
    return stem


def unique_labels(paths: Iterable[str]) -> List[str]:
    """Labels for a set of checkpoints, guaranteed distinct.

    Raises rather than quietly suffixing. Two checkpoints that still collide after being named by
    their run are either the same file twice -- which the caller did not mean -- or two runs
    sharing a short name, which `runs.md` forbids. Both are mistakes worth stopping for, and the
    `#1`/`#2` suffixing that used to paper over them is what made the reports unciteable.
    """
    paths = list(paths)
    labels = [checkpoint_label(p) for p in paths]
    seen: Dict[str, str] = {}
    for label, path in zip(labels, paths):
        prior = seen.get(label)
        if prior is not None:
            raise ValueError(
                f"two checkpoints share the label {label!r}:\n  {prior}\n  {path}\n"
                "A label is <run>@<steps>; a clash means the same file twice, or two runs with "
                "the same short name. Name the runs apart rather than disambiguating here.")
        seen[label] = path
    return labels
