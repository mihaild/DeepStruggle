#!/usr/bin/env bash
#
# Install Emscripten -- the compiler for the browser workbench's WebAssembly engine -- without root.
#
#     tools/scripts/install_emsdk.sh              # into ~/.local/opt/emsdk
#     EMSDK_DIR=/some/dir EMSDK_VERSION=6.0.10 tools/scripts/install_emsdk.sh
#
# Pinned, not "latest": a different Emscripten is a different compiler, and the browser engine
# is checked bit for bit against the native one (tests/web/test_wasm_engine.py). Bump it
# deliberately, and rerun that test when you do. tools/scripts/build_web.sh finds it here.
set -euo pipefail

DIR="${EMSDK_DIR:-$HOME/.local/opt/emsdk}"
VERSION="${EMSDK_VERSION:-6.0.10}"

if [[ ! -d "$DIR/.git" ]]; then
    git clone --depth 1 https://github.com/emscripten-core/emsdk.git "$DIR"
fi
"$DIR/emsdk" install "$VERSION"
"$DIR/emsdk" activate "$VERSION" >/dev/null
# shellcheck disable=SC1091
source "$DIR/emsdk_env.sh" >/dev/null 2>&1
echo "install_emsdk: $(emcc --version | head -1)"
echo "  build the browser engine and page with: tools/scripts/build_web.sh"
