"""The tests must run against a build of the engine sources in this checkout.

A stale ts_engine.so left in build/ once shadowed the real one for every pytest run:
pytest.ini said `pythonpath = build .`, which takes precedence over PYTHONPATH, so the
suite ran against a months-old engine. The eighth-Action-Round fix appeared to fail in the
main checkout while passing in a worktree -- worktrees had no such stray file -- and the
engine sources were identical in both.

Nothing in the suite noticed, because a stale engine mostly behaves like a current one.
This checks the two things that would have caught it: the extension is not older than the
sources it was built from, and it comes from the configured build directory rather than
whatever else happens to be on sys.path.
"""

import os
import pathlib
from typing import Optional

import pytest
import ts_engine as ts

# tests/bindings/<this file> -> the checkout root is three levels up. This read
# `.parent.parent` while the file lived directly in tests/; the reorg moved it a level deeper
# and the constant did not follow, which pointed REPO at tests/ and quietly disabled both
# checks here -- the staleness one skipped for "no engine sources found" and the provenance
# one failed on every run. Derive it from the file's own depth so a future move is loud.
REPO = pathlib.Path(__file__).resolve().parents[2]
SOURCE_DIRS = ("engine", "bindings")
SOURCE_SUFFIXES = {".cpp", ".hpp", ".h", ".cc"}


def _newest_source() -> tuple[Optional[pathlib.Path], float]:
    """The newest source the extension is built *from*.

    engine/tests/ is not among them -- those compile into ts_tests, not the extension, so
    editing a C++ test never makes the .so stale and must not read as a missed rebuild. Nor is
    bindings/wasm/: it compiles only into the browser workbench's WebAssembly engine, so editing
    it cannot relink this .so, and a timestamp check would call it stale forever.
    """
    newest, newest_mtime = None, 0.0
    for directory in SOURCE_DIRS:
        root = REPO / directory
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            parts = path.relative_to(root).parts
            if "tests" in parts or (directory == "bindings" and parts[:1] == ("wasm",)):
                continue
            if path.suffix in SOURCE_SUFFIXES and path.is_file():
                mtime = path.stat().st_mtime
                if mtime > newest_mtime:
                    newest, newest_mtime = path, mtime
    return newest, newest_mtime


def test_extension_is_not_older_than_its_sources() -> None:
    """A binary older than the sources means someone forgot to rebuild, or it is a stray."""
    so_path = pathlib.Path(ts.__file__)
    newest, newest_mtime = _newest_source()
    if newest is None:
        pytest.skip("no engine sources found next to the tests")

    so_mtime = so_path.stat().st_mtime
    assert so_mtime >= newest_mtime, (
        f"ts_engine at {so_path} was built {newest_mtime - so_mtime:.0f}s before "
        f"{newest.relative_to(REPO)} was last modified. Rebuild with "
        f"'cmake --build build/release -j', and check for a stale .so shadowing it."
    )


def test_extension_comes_from_this_checkout() -> None:
    """Guards against importing an engine belonging to a different tree entirely."""
    so_path = pathlib.Path(ts.__file__).resolve()
    assert str(so_path).startswith(str(REPO) + os.sep), (
        f"ts_engine was imported from {so_path}, which is outside this checkout at {REPO}"
    )
