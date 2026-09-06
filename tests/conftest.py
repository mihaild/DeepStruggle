from __future__ import annotations

import os
from typing import Iterator

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-fuzz",
        action="store_true",
        default=False,
        help="Run differential fuzzing tests against external engines",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-fuzz"):
        return
    expr = getattr(config.option, "markexpr", "") or ""
    if "differential_fuzz" in expr:
        return
    for arg in config.args:
        if "test_differential_fuzzing" in arg:
            return

    skip_marker = pytest.mark.skip(
        reason="Differential fuzzers against external engines are not run by default. Use --run-fuzz, -m differential_fuzz, or specify tests/differential/test_differential_fuzzing.py"
    )
    for item in items:
        if "differential_fuzz" in item.keywords:
            item.add_marker(skip_marker)


def pytest_ignore_collect(collection_path, config: pytest.Config) -> bool | None:
    """Keep tests/differential out of collection unless it is asked for.

    Marking those tests skipped is not enough: the skip is applied after collection, and the
    modules fail at *import*, so a plain `pytest tests/` aborts with a collection error instead
    of running the suite. Ignoring the directory here rather than through an `--ignore` in
    pytest.ini keeps the escape hatches working -- `--run-fuzz`, `-m differential_fuzz`, or
    naming a path under tests/differential -- since those are all visible on `config`.
    """
    if "differential" not in collection_path.parts:
        return None
    if config.getoption("--run-fuzz"):
        return None
    if "differential_fuzz" in (getattr(config.option, "markexpr", "") or ""):
        return None
    if any("differential" in str(arg) for arg in config.args):
        return None
    return True


@pytest.fixture(scope="session")
def generated_replay_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A directory holding one freshly generated replay, and the server pointed at it.

    `data/replays/` is git-ignored, so nothing in the suite may assume it holds anything. The
    tempting alternative -- skip when it is empty -- is the failure mode this repository keeps
    rediscovering: a check that quietly measures nothing while passing everywhere. Three
    instances are on record (research/experiments.md section 1), pyrefly was found exiting 0
    having examined zero files, and the test written to catch a stale engine spent two commits
    silently skipping after a directory move.

    Generating instead also tests more than a committed file could. Invariant 6 requires every
    replay to be written by `generate_self_play_replay`, so a schema assertion over generated
    output covers the writer as it is now; a frozen fixture can drift from what the writer emits
    and the test would still pass. No checkpoint is needed -- the generator falls back to an
    untrained network when none is on disk -- and one game takes about a second. A single replay
    is also ~3.7 MB, which is reason enough not to commit one.
    """
    from web.server.replay import REPLAYS_DIR_ENV

    target = tmp_path_factory.mktemp("replays")
    out = str(target / "fixture_selfplay.tslog.json")

    previous = os.environ.get(REPLAYS_DIR_ENV)
    os.environ[REPLAYS_DIR_ENV] = str(target)
    try:
        from tools.lib.self_play import generate_self_play_replay

        _, saved = generate_self_play_replay(
            seed=2026,
            game_id="fixture_selfplay",
            output_path=out,
            device="cpu",
            verbose=False,
        )
        # A generation failure must surface as an error, never as an empty directory that the
        # dependent tests then skip over.
        assert os.path.exists(saved) and os.path.getsize(saved) > 0, (
            f"replay fixture was not written to {saved}"
        )
        yield str(target)
    finally:
        if previous is None:
            os.environ.pop(REPLAYS_DIR_ENV, None)
        else:
            os.environ[REPLAYS_DIR_ENV] = previous
