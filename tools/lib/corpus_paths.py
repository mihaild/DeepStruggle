"""Where the ts-replayer corpus lives, resolved in one place.

The corpus is ~5 MB of downloaded human games and is git-ignored, so it is not repository
content and must not be re-fetched per checkout. Every git worktree under `.claude/worktrees/`
gets its own empty `data/`, so a corpus stored there would be downloaded again for each one --
300 requests at one per second, every time, for bytes already on the machine. It therefore
defaults to a shared per-user cache and is shared by every checkout and worktree.

Resolution order:

1. ``$TS_REPLAYER_CORPUS`` -- an explicit override, for CI or a scratch copy.
2. ``<repo>/data/datasets/ts_replayer`` -- only if it already exists, so checkouts that
   downloaded the corpus before this module keep working.
3. ``$XDG_CACHE_HOME/ts_ai/ts_replayer`` (or ``~/.cache/ts_ai/ts_replayer``) -- the default,
   and where the downloader writes.
"""

from __future__ import annotations

import os
import pathlib
from typing import List, Optional

CORPUS_ENV: str = "TS_REPLAYER_CORPUS"

_REPO_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[2]

#: How to obtain the corpus, quoted verbatim wherever its absence is reported.
DOWNLOAD_HINT: str = (
    "PYTHONPATH=. .venv/bin/python tools/download_ts_replayer.py"
)


def shared_cache_dir() -> pathlib.Path:
    """The per-user location every checkout and worktree shares."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(pathlib.Path.home(), ".cache")
    return pathlib.Path(base) / "ts_ai" / "ts_replayer"


def in_repo_dir() -> pathlib.Path:
    """The legacy in-repo location, kept working for checkouts that already use it."""
    return _REPO_ROOT / "data" / "datasets" / "ts_replayer"


def corpus_dir() -> pathlib.Path:
    """Where to read the corpus from. May not exist; see `require_corpus`."""
    override = os.environ.get(CORPUS_ENV)
    if override:
        return pathlib.Path(override)
    legacy = in_repo_dir()
    if legacy.is_dir() and any(legacy.glob("*.json.gz")):
        return legacy
    return shared_cache_dir()


def corpus_files() -> List[pathlib.Path]:
    """Every cached replay, sorted. Empty when the corpus has not been downloaded."""
    directory = corpus_dir()
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json.gz"))


def corpus_path(replay_id: int) -> pathlib.Path:
    return corpus_dir() / f"{replay_id}.json.gz"


def missing_corpus_reason() -> Optional[str]:
    """Why the corpus cannot be used, or None when it is present.

    Returned rather than raised so a caller can put it in a failure message. Tests should
    *fail* with this, not skip: a suite that skips when its data is absent passes everywhere
    while verifying nothing, which is the mistake this repository keeps repeating.
    """
    directory = corpus_dir()
    if not directory.is_dir():
        return (f"the ts-replayer corpus is not at {directory}. Download it once with:\n"
                f"    {DOWNLOAD_HINT}\n"
                f"It is shared by every checkout and worktree, so this is a one-time cost. "
                f"Set {CORPUS_ENV} to use a different location.")
    if not any(directory.glob("*.json.gz")):
        return (f"{directory} exists but holds no replays. Download them with:\n"
                f"    {DOWNLOAD_HINT}")
    return None
