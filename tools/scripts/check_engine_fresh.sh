#!/usr/bin/env bash
#
# Refuse to let an experiment run against a stale ts_engine.
#
# Every measurement in this repo -- a tournament, a dataset, a diagnostic, a replay conversion
# -- is only meaningful relative to the engine that produced it, and a rebuilt engine can change
# the decision stream without changing any Python. Call this before anything that generates or
# consumes a training artifact:
#
#     tools/scripts/check_engine_fresh.sh && PYTHONPATH=. .venv/bin/python tools/train.py ...
#
# Exit codes:
#   0  the build already matched the sources; carry on
#   1  the build was stale (it has now been rebuilt) -- rerun the command, and treat any
#      checkpoint, dataset or number produced before this point as measured on a different game
#   2  the build directory does not exist, or the rebuild failed
#
# Freshness is decided by hashing the engine and bindings sources rather than by comparing
# timestamps: checking out a branch rewrites mtimes without changing content, and that would
# report a false staleness every time.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${1:-$ROOT/build/release}"
STAMP="$BUILD_DIR/.engine_fingerprint"

# Defined in tools/lib/engine_fingerprint.py and shelled out to rather than reimplemented here.
# The stamp this script writes is read back by the pytest guard, and two hand-kept versions of
# the same hash would eventually disagree -- at which point one of them certifies nothing.
PY_BIN="${TS_PYTHON:-}"
if [[ -z "$PY_BIN" ]]; then
    if [[ -x "$ROOT/.venv/bin/python3" ]]; then
        PY_BIN="$ROOT/.venv/bin/python3"
    else
        PY_BIN="python3"
    fi
fi

fingerprint() {
    PYTHONPATH="$ROOT" "$PY_BIN" "$ROOT/tools/lib/engine_fingerprint.py"
}

# A stamp sitting next to an extension that is not there would certify nothing. This is the
# case that bit: build/release held a stale ts_engine and no stamp, and pytest imported it.
built_extension_exists() {
    compgen -G "$BUILD_DIR/ts_engine*.so" > /dev/null 2>&1
}

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "check_engine_fresh: no build directory at $BUILD_DIR" >&2
    echo "  configure it first:" >&2
    echo "  cmake -B build/release -S . -DPython_EXECUTABLE=\$(pwd)/.venv/bin/python3" >&2
    exit 2
fi

want="$(fingerprint)"
have="$(cat "$STAMP" 2>/dev/null || true)"

if [[ "$want" == "$have" && -n "$have" ]] && built_extension_exists; then
    echo "check_engine_fresh: engine build matches sources ($want)"
    exit 0
fi

echo "check_engine_fresh: engine build is stale or unstamped -- rebuilding" >&2
if ! cmake --build "$BUILD_DIR" -j >&2; then
    echo "check_engine_fresh: rebuild FAILED" >&2
    exit 2
fi
if ! built_extension_exists; then
    echo "check_engine_fresh: the build succeeded but no ts_engine*.so is in $BUILD_DIR," >&2
    echo "  so there is nothing here for pytest to import. Configure the build there, which" >&2
    echo "  is where LIBRARY_OUTPUT_DIRECTORY puts the extension:" >&2
    echo "  cmake -B $BUILD_DIR -S . -DPython_EXECUTABLE=\$(pwd)/.venv/bin/python3" >&2
    exit 2
fi
fingerprint > "$STAMP"

cat >&2 <<'WARN'

check_engine_fresh: the engine was rebuilt, so the command was not run.

Rerun it. Before you trust anything you already have, note that a rebuilt engine may play a
different game than the one your artifacts were produced under:

  - datasets in the (seed, actions) format silently truncate at the first newly-illegal action,
    and can lose most of their decisions without reporting anything
  - tournament results, Elo anchors and diagnostics predate the change and are not comparable
  - checkpoints still load, but were trained against the old decision stream

WARN
exit 1
