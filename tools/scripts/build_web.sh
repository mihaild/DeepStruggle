#!/usr/bin/env bash
#
# Build the browser workbench: the WebAssembly engine from engine/ + bindings/, then the pages
# (the workbench and the Elo leaderboard, whose data is fitted here from leaderboard/).
#
#     tools/scripts/build_web.sh             # engine + pages (web/ui/dist)
#     tools/scripts/build_web.sh --engine    # the WebAssembly engine only (web/ui/public/engine)
#
# The page runs the engine in the browser, so after any engine change this is what makes the
# workbench play the new rules -- the Python build (check_engine_fresh.sh) does not. The local
# server shows a banner when the page's engine fingerprint no longer matches the sources.
#
# Needs Emscripten (tools/scripts/install_emsdk.sh, found in ~/.local/opt/emsdk or on PATH) and,
# for the page, node + npm.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENGINE_ONLY=0
[[ "${1:-}" == "--engine" ]] && ENGINE_ONLY=1

if ! command -v emcmake >/dev/null; then
    EMSDK_ENV="${EMSDK_DIR:-$HOME/.local/opt/emsdk}/emsdk_env.sh"
    if [[ -f "$EMSDK_ENV" ]]; then
        # shellcheck disable=SC1090
        source "$EMSDK_ENV" >/dev/null 2>&1
    fi
fi
if ! command -v emcmake >/dev/null; then
    echo "build_web: Emscripten not found -- install it with tools/scripts/install_emsdk.sh" >&2
    exit 2
fi

PY="${TS_PYTHON:-}"
if [[ -z "$PY" ]]; then
    if [[ -x "$ROOT/.venv/bin/python3" ]]; then PY="$ROOT/.venv/bin/python3"; else PY=python3; fi
fi
FINGERPRINT="$(PYTHONPATH="$ROOT" "$PY" "$ROOT/tools/lib/engine_fingerprint.py")"

BUILD="$ROOT/build/wasm"
# A build directory remembers the source tree it was configured for, and CMake refuses it from any
# other path -- the same checkout seen under another name included (a container's /workspace and
# the host's home directory). Such a cache is configured afresh rather than failing.
FRESH=()
# No cache yet (a fresh checkout, as on CI) is no cache to compare: sed's exit 2 on the missing file
# would otherwise end the script under `set -o pipefail`.
cached_src="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$BUILD/CMakeCache.txt" 2>/dev/null | head -n 1 || true)"
if [[ -n "$cached_src" && "$(realpath -e -- "$cached_src" 2>/dev/null)" != "$(realpath -e -- "$ROOT")" ]]; then
    echo "build_web: $BUILD was configured for $cached_src; configuring it afresh for $ROOT" >&2
    FRESH=(--fresh)
fi
# Likewise a build directory whose Emscripten has gone -- an install that moved or was removed --
# fails every reconfigure: the cache names its compiler, and CMakeFiles/<version>/CMakeSystem.cmake
# includes its toolchain file (and keeps it after a failed reconfigure has rewritten the cache).
# Afresh, emcmake's toolchain file names the Emscripten found now.
cached_cxx="$(sed -n 's/^CMAKE_CXX_COMPILER:[A-Z]*=//p' "$BUILD/CMakeCache.txt" 2>/dev/null | head -n 1 || true)"
system_toolchain="$(sed -n 's/^include("\(.*\)")$/\1/p' "$BUILD"/CMakeFiles/*/CMakeSystem.cmake 2>/dev/null | head -n 1 || true)"
for gone in "$cached_cxx" "$system_toolchain"; do
    if [[ ${#FRESH[@]} -eq 0 && "$gone" == /* && ! -e "$gone" ]]; then
        echo "build_web: $BUILD names a toolchain file that does not exist here ($gone); configuring it afresh" >&2
        FRESH=(--fresh)
    fi
done
emcmake cmake ${FRESH[@]+"${FRESH[@]}"} -B "$BUILD" -S "$ROOT" -DCMAKE_BUILD_TYPE=Release \
    -DTS_ENGINE_FINGERPRINT="$FINGERPRINT" >/dev/null
cmake --build "$BUILD" --target ts_engine_wasm -j "$(nproc)"
echo "build_web: engine $FINGERPRINT -> web/ui/public/engine/"

if [[ $ENGINE_ONLY == 1 ]]; then
    exit 0
fi
# The leaderboard page's data: fitted from the records in leaderboard/, never committed. Plain
# python, no torch or engine needed -- on CI this is the runner's python3.
# TS_PLAYDEK_ESTIMATE names tools/playdek_estimate.py's output; without it the page has no Playdek scale.
PLAYDEK_ARGS=()
if [[ -n "${TS_PLAYDEK_ESTIMATE:-}" ]]; then
    PLAYDEK_ARGS=(--playdek "$TS_PLAYDEK_ESTIMATE")
fi
PYTHONPATH="$ROOT" "$PY" "$ROOT/tools/leaderboard.py" fit --output "$ROOT/web/ui/public/leaderboard.json" \
    ${PLAYDEK_ARGS[@]+"${PLAYDEK_ARGS[@]}"}
cd "$ROOT/web/ui"
# Install when missing, and again when package-lock.json has changed since the last install
# (npm records that install in node_modules/.package-lock.json): a checkout whose node_modules
# predates a new dependency would otherwise fail to build with a missing import.
if [[ ! -f node_modules/.package-lock.json || package-lock.json -nt node_modules/.package-lock.json ]]; then
    npm ci --no-audit --no-fund
fi
npm run build
echo "build_web: page -> web/ui/dist/"
