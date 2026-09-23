"""Where the ts-replayer corpus lives, resolved in one place.

The corpus is ~5 MB of downloaded human games and is git-ignored, so it is not repository
content and must not be re-fetched per checkout. It lives in the **shared data tree**
(`tools.lib.data_root`, which finds the main checkout's `data/` even from inside a worktree), so
every checkout and worktree reads one copy. A worktree's own `data/` is empty, and a corpus stored
there would be downloaded again for each one -- 300 requests at one per second.

Resolution order:

1. ``$TS_REPLAYER_CORPUS`` -- an explicit override, for CI or a scratch copy.
2. ``<shared data root>/datasets/ts_replayer`` -- the default, and where the downloader writes.
   With ``$TS_DATA_ROOT`` unset that is the main checkout's ``data/``.

It used to default to ``~/.cache/ts_ai/ts_replayer``. All project data belongs in the shared data
tree (owner's rule), so the per-user cache is no longer read.
"""

from __future__ import annotations

import os
import pathlib
from typing import Dict, List, Optional, Tuple

CORPUS_ENV: str = "TS_REPLAYER_CORPUS"

#: How to obtain the corpus, quoted verbatim wherever its absence is reported. `build/release` is
#: on the path because importing `tools.lib` imports the engine.
DOWNLOAD_HINT: str = (
    "PYTHONPATH=.:build/release .venv/bin/python tools/download_ts_replayer.py"
)


def shared_corpus_dir() -> pathlib.Path:
    """The corpus inside the shared data tree, the same place from every checkout and worktree."""
    from tools.lib.data_root import data_path

    return pathlib.Path(data_path("datasets", "ts_replayer"))


def corpus_dir() -> pathlib.Path:
    """Where to read the corpus from. May not exist; see `require_corpus`."""
    override = os.environ.get(CORPUS_ENV)
    if override:
        return pathlib.Path(override)
    return shared_corpus_dir()


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


def game_fingerprint(path: pathlib.Path) -> Optional[str]:
    """A hash of the game's entries, or None if the file cannot be read.

    Only `all_turns` goes into the hash. `replay_id` and `source` differ between two records of
    the same game by construction, so including them would make every duplicate look distinct --
    which is exactly how the duplicates went unnoticed.
    """
    import gzip
    import hashlib
    import json

    try:
        with gzip.open(path, "rt") as fh:
            data = json.load(fh)
    except Exception:
        return None
    entries = data.get("all_turns")
    if not entries:
        return None
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def distinct_corpus_files() -> Tuple[List[pathlib.Path], Dict[str, List[int]]]:
    """The corpus with duplicate games removed, and the groups that were collapsed.

    ts-replayer serves the same game under several replay ids -- 300 files hold 266 distinct
    games, one of them recorded nine times. A duplicate is not harmless: it carries several times
    the weight in behaviour cloning and in every injection batch, and because ids differ it can
    land on both sides of an id-based train/held-out split, so held-out agreement partly measures
    memorisation. Deduplicate on content and split on the result.

    The lowest replay id in each group is kept, so the choice is stable across runs.
    """
    groups: Dict[str, List[pathlib.Path]] = {}
    unreadable: List[pathlib.Path] = []
    for path in corpus_files():
        key = game_fingerprint(path)
        if key is None:
            # Kept rather than dropped: an unreadable or empty file is a separate problem, and
            # this function's job is duplicates, not filtering.
            unreadable.append(path)
            continue
        groups.setdefault(key, []).append(path)

    def replay_id(path: pathlib.Path) -> int:
        try:
            return int(path.name.split(".")[0])
        except ValueError:
            return 1 << 30

    kept: List[pathlib.Path] = []
    collapsed: Dict[str, List[int]] = {}
    for key, paths in groups.items():
        paths.sort(key=replay_id)
        kept.append(paths[0])
        if len(paths) > 1:
            collapsed[key] = [replay_id(p) for p in paths]
    kept.extend(unreadable)
    kept.sort(key=replay_id)
    return kept, collapsed
