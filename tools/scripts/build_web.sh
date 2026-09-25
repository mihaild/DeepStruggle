#!/usr/bin/env bash
#
# Build the browser workbench: the WebAssembly engine from engine/ + bindings/, then the page.
#
#     tools/scripts/build_web.sh             # engine + page (web/ui/dist)
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
emcmake cmake -B "$BUILD" -S "$ROOT" -DCMAKE_BUILD_TYPE=Release \
    -DTS_ENGINE_FINGERPRINT="$FINGERPRINT" >/dev/null
cmake --build "$BUILD" --target ts_engine_wasm -j "$(nproc)"
echo "build_web: engine $FINGERPRINT -> web/ui/public/engine/"

if [[ $ENGINE_ONLY == 1 ]]; then
    exit 0
fi
cd "$ROOT/web/ui"
if [[ ! -d node_modules ]]; then
    npm ci --no-audit --no-fund
fi
npm run build
echo "build_web: page -> web/ui/dist/"
