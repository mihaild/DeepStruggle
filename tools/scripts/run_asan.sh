#!/usr/bin/env bash
# Run a command (usually python) against the AddressSanitizer build of ts_engine.
#
#     tools/scripts/run_asan.sh .venv/bin/python -m pytest -q tests/bindings
#
# The engine is built with clang, so the runtime is compiler-rt's shared ASan library, which the
# extension was linked against with -shared-libasan (see the sanitizer build in CLAUDE.md). A
# Python interpreter is not itself instrumented, so the runtime has to be preloaded: it must be
# the first library in the process, ahead of anything that allocates.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${ASAN_BUILD_DIR:-$ROOT/build_san}"

CLANGXX="$(command -v clang++ || true)"
[[ -z "$CLANGXX" && -x "$HOME/.local/bin/clang++" ]] && CLANGXX="$HOME/.local/bin/clang++"
if [[ -z "$CLANGXX" ]]; then
    echo "run_asan: clang++ not found (apt-get install clang, or tools/scripts/install_clang_userspace.sh)" >&2
    exit 1
fi
LIBASAN="$("$CLANGXX" -print-file-name=libclang_rt.asan-x86_64.so)"
if [[ ! -f "$LIBASAN" ]]; then
    echo "run_asan: $CLANGXX has no shared ASan runtime ($LIBASAN); install compiler-rt" >&2
    echo "  (apt-get install libclang-rt-dev, or tools/scripts/install_clang_userspace.sh)" >&2
    exit 1
fi
if ! compgen -G "$BUILD_DIR/ts_engine*.so" >/dev/null; then
    echo "run_asan: no ts_engine in $BUILD_DIR -- configure the sanitizer build first (CLAUDE.md)" >&2
    exit 1
fi

export LD_PRELOAD="$LIBASAN"
export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0:verify_asan_link_order=0}"
export PYTHONPATH="$BUILD_DIR:${PYTHONPATH:-$ROOT}"

exec "$@"
