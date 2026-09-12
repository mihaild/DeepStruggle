#!/usr/bin/env bash
set -e

# Find libasan path dynamically
LIBASAN=$(gcc -print-file-name=libasan.so 2>/dev/null || true)
if [ -z "$LIBASAN" ] || [ ! -f "$LIBASAN" ]; then
    LIBASAN=$(find /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu -name "libasan.so*" 2>/dev/null | head -n 1)
fi

if [ -z "$LIBASAN" ]; then
    echo "Error: libasan.so not found on the system. Please ensure gcc/libasan is installed." >&2
    exit 1
fi

export LD_PRELOAD="$LIBASAN"
export ASAN_OPTIONS="detect_leaks=0:verify_asan_link_order=0"
export PYTHONPATH="$(pwd)/build/asan:${PYTHONPATH:-$(pwd)}"

exec "$@"
