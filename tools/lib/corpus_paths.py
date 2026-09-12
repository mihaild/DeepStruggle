"""Where the ts-replayer corpus lives, resolved in one place.

The corpus is ~5 MB of downloaded human games and is git-ignored, so it is not repository
content and must not be re-fetched per checkout. Every checkout and every git worktree gets its
own empty `data/`, so a corpus stored there would be downloaded again for each one -- 300
requests at one per second, every time, for bytes already on the machine. It therefore defaults
to a shared per-user cache.

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
from typing import Dict, List, Optional, Tuple

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
