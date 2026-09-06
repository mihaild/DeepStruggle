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

fingerprint() {
    find "$ROOT/engine" "$ROOT/bindings" -type f \
        \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' -o -name 'CMakeLists.txt' \) -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 sha256sum \
        | sha256sum \
        | cut -d' ' -f1
}

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "check_engine_fresh: no build directory at $BUILD_DIR" >&2
    echo "  configure it first:" >&2
    echo "  cmake -B build/release -S . -DPython_EXECUTABLE=\$(pwd)/.venv/bin/python3" >&2
    exit 2
fi

want="$(fingerprint)"
have="$(cat "$STAMP" 2>/dev/null || true)"

if [[ "$want" == "$have" && -n "$have" ]]; then
    echo "check_engine_fresh: engine build matches sources ($want)"
    exit 0
fi

echo "check_engine_fresh: engine build is stale or unstamped -- rebuilding" >&2
if ! cmake --build "$BUILD_DIR" -j >&2; then
    echo "check_engine_fresh: rebuild FAILED" >&2
    exit 2
fi
fingerprint > "$STAMP"

cat >&2 <<'WARN'

check_engine_fresh: the engine was rebuilt, so the command was not run.

Rerun it. Before you trust anything you already have, note that a rebuilt engine may play a
different game than the one your artifacts were produced under:

  - datasets in the (seed, actions) format silently truncate at the first newly-illegal action
    (see data/datasets/archive/README.md for what that cost the last one)
  - tournament results, Elo anchors and diagnostics predate the change and are not comparable
  - checkpoints still load, but were trained against the old decision stream

WARN
exit 1
