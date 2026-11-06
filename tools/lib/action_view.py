"""Which action view a checkpoint decides in: E4, or E4.1's merged-influence view (P23).

In the E4.1 view a NODE slot at an op-choice node means "ops for influence, first point here".
A network trained in one view reads the other's masks wrongly -- a composed action is simply
absent from the E4 mask -- so every consumer must know which view each checkpoint was trained
in. Nothing in a snapshot file says so, and adding it would mean a new checkpoint format; the
run directory already says it:

* ``metadata.json`` records ``merged_influence`` and ``merged_influence_from_step``, the step
  count at which the run began deciding in the merged view (0 for a run that did so throughout,
  the resume step for an E4 lineage continued as E4.1);
* the snapshot's filename carries its step count.

A snapshot is in the merged view iff its run is and it was taken after that step. This is the
same move `checkpoint_id` makes for labels: derive from the path, never from a second record that
can drift.
"""

from __future__ import annotations

import json
import os
import re
from typing import Tuple

_SNAPSHOT_STEPS = re.compile(r"(?:snapshot|pool)_(\d+)steps\.pt$")


def run_view(run_dir: str) -> Tuple[bool, int]:
    """(merged_influence, merged_influence_from_step) for a run directory. E4 when unrecorded."""
    try:
        with open(os.path.join(run_dir, "metadata.json"), encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return False, 0
    return bool(meta.get("merged_influence", False)), int(meta.get("merged_influence_from_step", 0))


#: The action views (ts.ActionView): E4 is the base view, MERGED P23's E4.1, MACRO P42's until-control view.
VIEW_E4, VIEW_MERGED, VIEW_MACRO = 0, 1, 2


def run_macro_view(run_dir: str) -> Tuple[bool, int]:
    """(macro_influence, macro_influence_from_step) for a run directory: P42's view, recorded like the merged one."""
    try:
        with open(os.path.join(run_dir, "metadata.json"), encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return False, 0
    return bool(meta.get("macro_influence", False)), int(meta.get("macro_influence_from_step", 0))


def _in_view_from(path: str, from_step: int) -> bool:
    """Whether the checkpoint at `path` was taken after `from_step` (the step its run began deciding in a view)."""
    name = os.path.basename(path)
    m = _SNAPSHOT_STEPS.search(name)
    if m:
        return int(m.group(1)) > from_step
    if name == "snapshot_final.pt":
        return True
    return from_step == 0


def checkpoint_macro_influence(path: str) -> bool:
    """Whether the checkpoint at `path` was trained to decide in P42's macro view (the until-control slots)."""
    macro, from_step = run_macro_view(os.path.dirname(os.path.abspath(path)))
    return macro and _in_view_from(path, from_step)


def checkpoint_action_view(path: str) -> int:
    """The view (VIEW_E4 / VIEW_MERGED / VIEW_MACRO) the checkpoint at `path` decides in."""
    merged, macro = checkpoint_merged_influence(path), checkpoint_macro_influence(path)
    if merged and macro:
        raise ValueError(f"{path}: its run records both the merged and the macro view; they are exclusive")
    return VIEW_MERGED if merged else VIEW_MACRO if macro else VIEW_E4


def checkpoint_merged_influence(path: str) -> bool:
    """Whether the checkpoint at `path` was trained to decide in the merged-influence view."""
    merged, from_step = run_view(os.path.dirname(os.path.abspath(path)))
    if not merged:
        return False
    name = os.path.basename(path)
    m = _SNAPSHOT_STEPS.search(name)
    if m:
        return int(m.group(1)) > from_step
    if name == "snapshot_final.pt":
        return True  # a run's last weights, taken after any resume point
    # snapshot_0s.pt and anything unrecognised: the weights the run started from, which are
    # merged only if the run was merged from its first step.
    return from_step == 0
