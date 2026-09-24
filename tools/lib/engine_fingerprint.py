"""One definition of "does this build match the engine sources".

Key invariant 10 says never to measure against a stale engine, and
`tools/scripts/check_engine_fresh.sh` enforces it for anything launched from the shell. Nothing
enforced it for pytest, which is the one place it is easiest to get wrong: `pytest.ini` sets

    pythonpath = build/release .

and a `pythonpath` in the ini file is prepended ahead of the environment's ``PYTHONPATH``. So a
stale `ts_engine` sitting in `build/release` wins over the build the caller asked for, silently,
with no output saying which extension was loaded. A checkout carrying a months-old extension
there reported 175 failures and 222 errors that had nothing to do with the code under test; the
opposite case is worse, because it is green.

The fingerprint hashes file *contents*, never mtimes: checking out a branch rewrites mtimes
without changing any source, and a timestamp check would report staleness every time.

The hash is defined here and used from both sides -- the shell script shells out to
``main()`` -- so the stamp one writes is always the stamp the other reads.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
from typing import List, Optional

#: Written next to the built extension by check_engine_fresh.sh after a successful build.
STAMP_NAME: str = ".engine_fingerprint"

#: Quoted verbatim wherever a mismatch is reported.
REBUILD_HINT: str = "tools/scripts/check_engine_fresh.sh"

_SUFFIXES = (".cpp", ".h", ".hpp")
_NAMES = ("CMakeLists.txt",)

_REPO_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[2]


def source_files(root: Optional[pathlib.Path] = None) -> List[pathlib.Path]:
    """Every file whose content can change what the compiled engine does, sorted.

    `engine/` and `bindings/`, plus the root `CMakeLists.txt`, which picks the compiler and the
    optimisation flags for both: it was missing, so a change of compiler or `-march` left every
    build reporting itself fresh. Python is not compiled in, and a change to it cannot make a
    built extension stale.
    """
    base = root or _REPO_ROOT
    found: List[pathlib.Path] = []
    top = base / "CMakeLists.txt"
    if top.is_file():
        found.append(top)
    for directory in ("engine", "bindings"):
        d = base / directory
        if not d.is_dir():
            continue
        for path in d.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix in _SUFFIXES or path.name in _NAMES:
                found.append(path)
    return sorted(found)


def fingerprint(root: Optional[pathlib.Path] = None) -> str:
    """A content hash of the engine and bindings sources."""
    base = root or _REPO_ROOT
    digest = hashlib.sha256()
    for path in source_files(base):
        digest.update(str(path.relative_to(base)).encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
        digest.update(b"\0")
    return digest.hexdigest()


def read_stamp(build_dir: pathlib.Path) -> Optional[str]:
    """The fingerprint recorded next to a built extension, or None when there is none."""
    try:
        return (build_dir / STAMP_NAME).read_text().strip() or None
    except OSError:
        return None


def write_stamp(build_dir: pathlib.Path, value: Optional[str] = None) -> str:
    """Record the current fingerprint next to a freshly built extension."""
    want = value or fingerprint()
    (build_dir / STAMP_NAME).write_text(want + "\n")
    return want


def staleness_reason(module_file: str, root: Optional[pathlib.Path] = None) -> Optional[str]:
    """Why the loaded `ts_engine` does not match the sources, or None when it does.

    `module_file` is the extension's own ``__file__``, so the answer is about the engine that
    was *actually imported* rather than about whichever build directory happens to be named in
    a config file. That distinction is the whole point: the failure this catches is importing
    one build while believing you configured another.

    Returned rather than raised so the caller can decide how loud to be.
    """
    build_dir = pathlib.Path(module_file).resolve().parent
    want = fingerprint(root)
    have = read_stamp(build_dir)

    if have is None:
        return (
            f"ts_engine was imported from {build_dir}, which carries no {STAMP_NAME}, so there\n"
            f"is no way to tell which sources it was built from. Build it through the checker,\n"
            f"which writes the stamp:\n\n"
            f"    {REBUILD_HINT}\n\n"
            f"Key invariant 10: never measure against a stale engine. If you copied an\n"
            f"extension into place by hand, stamp it too."
        )
    if have != want:
        return (
            f"ts_engine at {build_dir} was built from different sources than the ones in this\n"
            f"checkout.\n\n"
            f"    sources  {want}\n"
            f"    build    {have}\n\n"
            f"Rebuild before running anything that produces or consumes a measurement:\n\n"
            f"    {REBUILD_HINT}\n\n"
            f"Key invariant 10: a rebuilt engine can change the decision stream with no Python\n"
            f"change, so checkpoints, datasets and Elo numbers taken against the other build are\n"
            f"not comparable."
        )
    return None


def main(argv: Optional[List[str]] = None) -> int:
    """Print the fingerprint, so the shell script and this module cannot disagree."""
    args = list(argv if argv is not None else sys.argv[1:])
    if args and args[0] == "--stamp":
        if len(args) != 2:
            print("usage: engine_fingerprint.py --stamp <build-dir>", file=sys.stderr)
            return 2
        print(write_stamp(pathlib.Path(args[1])))
        return 0
    print(fingerprint())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
