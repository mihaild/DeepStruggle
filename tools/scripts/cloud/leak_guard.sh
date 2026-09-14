#!/usr/bin/env bash
# Warn when a vast.ai instance outlives the script that created it.
#
# A benchmark instance should exist for twenty minutes. One that is still running an hour later
# is billing silently, and nothing else in the system will say so -- this already happened once,
# when `destroy instance` aborted on its interactive confirmation prompt while its output was
# redirected to /dev/null.
#
#   leak_guard.sh [max-minutes]
#
# Emits one line per check only when something is wrong, so it is quiet in the normal case.
set -euo pipefail
MAX_MIN="${1:-40}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/workspace/.venv/bin/python

while true; do
    "$HERE/vast.sh" show instances --raw 2>/dev/null | "$PY" "$HERE/_leak_check.py" "$MAX_MIN" || true
    sleep 300
done
