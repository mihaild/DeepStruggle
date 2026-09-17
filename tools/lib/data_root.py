"""Where working artifacts live: one `data/` tree, shared by every worktree.

`data/` is git-ignored, so a git worktree does **not** share it with the main checkout — it gets
its own empty one, and anything written there is lost when the worktree is removed. That is how a
session came to hold 1.4 GB of checkpoints and five tournament reports inside
`.claude/worktrees/fix-profiler-bias/data/` while every earlier run sat in `/workspace/data`, with
the same relative path meaning two different places depending on which directory you were in.

The main checkout's `data/` is the canonical tree. It is found from git rather than from the
current directory: `git rev-parse --git-common-dir` points at the main repository's `.git` from
inside any worktree, so its parent is the main checkout.

Order of resolution:

1. `$TS_DATA_ROOT`, for a caller that means something else and says so;
2. `<main checkout>/data`, which is the case this exists for;
3. `data` relative to the current directory, if git cannot answer — no worse than before.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache


@lru_cache(maxsize=1)
def data_root() -> str:
    """Absolute path of the shared `data/` tree."""
    override = os.environ.get("TS_DATA_ROOT")
    if override:
        return os.path.abspath(override)
    try:
        common = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return os.path.abspath("data")
    main_checkout = os.path.dirname(os.path.abspath(common))
    return os.path.join(main_checkout, "data")


def data_path(*parts: str) -> str:
    """A path inside the shared data tree, e.g. `data_path("checkpoints", run_name)`."""
    return os.path.join(data_root(), *parts)


def checkpoints_dir() -> str:
    return data_path("checkpoints")


def tournaments_dir() -> str:
    return data_path("tournaments")
