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
#
# **Set max-minutes to outlive the work, not to a generic default.** A host deliberately kept
# alive for a training run is not a leak, and a guard that warns about it every five minutes
# teaches you to ignore the one warning that matters. Size it to the expected run length plus a
# margin: a 160M-step arm is roughly 3.5 hours, so 240 minutes rather than 40.
set -euo pipefail
MAX_MIN="${1:-40}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/workspace/.venv/bin/python

while true; do
    "$HERE/vast.sh" show instances --raw 2>/dev/null | "$PY" "$HERE/_leak_check.py" "$MAX_MIN" || true
    sleep 300
done
