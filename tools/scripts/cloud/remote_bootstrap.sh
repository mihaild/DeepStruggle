#!/usr/bin/env bash
# Prepare a rented host to run training: ship the source, build the engine, prove it works.
#
#   remote_bootstrap.sh <ssh-target> [remote-dir]
#
# Refuses to finish quietly if the engine does not import or does not match the sources, because
# a host that looks provisioned but has a stale or broken extension produces a run whose numbers
# are wrong rather than one that fails.
set -euo pipefail

TARGET="${1:?usage: remote_bootstrap.sh <ssh-target> [remote-dir]}"
REMOTE_DIR="${2:-/workspace/ts}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

echo "==> bootstrapping $TARGET into $REMOTE_DIR (from $LOCAL_ROOT)"

# --- 1. source -----------------------------------------------------------------------------
# Everything the build and the training need, and nothing that is large, host-specific, or
# regenerable. .venv and build/ are excluded deliberately: a venv is not portable, and the
# extension must be rebuilt against the image's Python anyway.
ssh "$TARGET" "mkdir -p '$REMOTE_DIR'"

# rsync when it exists, tar-over-ssh when it does not. rsync is absent and not installable on
# the workstation this was written on, so a hard dependency on it would fail at first use.
EXCLUDES=(.git .venv build build_release data web/ui/node_modules web/ui/dist
          __pycache__ .triton_cache external)
if command -v rsync >/dev/null 2>&1; then
    RS_EXC=()
    for e in "${EXCLUDES[@]}"; do RS_EXC+=(--exclude "$e/"); done
    rsync -az --delete "${RS_EXC[@]}" "$LOCAL_ROOT/" "$TARGET:$REMOTE_DIR/"
else
    TAR_EXC=()
    for e in "${EXCLUDES[@]}"; do TAR_EXC+=(--exclude "./$e"); done
    tar czf - -C "$LOCAL_ROOT" "${TAR_EXC[@]}" . \
        | ssh "$TARGET" "tar xzf - -C '$REMOTE_DIR'"
fi

# --- 2. build ------------------------------------------------------------------------------
ssh "$TARGET" bash -s <<REMOTE
set -euo pipefail
cd "$REMOTE_DIR"
PY=\$(command -v python3.14 || command -v python3)
if [ -x /opt/venv/bin/python ]; then PY=/opt/venv/bin/python; fi
echo "==> building engine with \$PY"
cmake -B build/release -S . -DPython_EXECUTABLE="\$PY" >/dev/null
cmake --build build/release -j "\$(nproc)" >/dev/null
echo "==> engine built"
REMOTE

# --- 3. prove it ---------------------------------------------------------------------------
# check_engine_fresh.sh compares the built module against the engine/ and bindings/ sources, so
# this catches a partial build as well as a stale one.
ssh "$TARGET" bash -s <<REMOTE
set -euo pipefail
cd "$REMOTE_DIR"
PY=/opt/venv/bin/python
[ -x "\$PY" ] || PY=\$(command -v python3.14 || command -v python3)
tools/scripts/check_engine_fresh.sh
PYTHONPATH=.:build/release "\$PY" -c "
import ts_engine as ts, torch
print('  engine OK, OBS_SIZE =', int(ts.OBS_SIZE))
print('  torch', torch.__version__, 'cuda available:', torch.cuda.is_available())
print('  gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
print('  cores:', __import__('os').cpu_count())
"
REMOTE

echo "==> $TARGET ready"
