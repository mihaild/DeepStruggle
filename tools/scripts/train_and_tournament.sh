#!/usr/bin/env bash
# Generic end-to-end training and tournament pipeline runner.
# Supports warm-started or tabula-rasa RL with periodic snapshots,
# blunder-aware rewards, live evaluation, and post-training massive tournament.

set -euo pipefail

ARCH="${1:-v3}"
DURATION="${2:-7200}"                 # Default: 2 hours (7200s)
SNAPSHOT_INTERVAL="${3:-1200}"        # Default: 20 minutes (1200s)
WARMUP_CHECKPOINT="${4:-data/checkpoints/coldwar_net_v3_warmup.pt}"

EXTRA_ARGS=()
if [ -n "$WARMUP_CHECKPOINT" ] && [ -f "$WARMUP_CHECKPOINT" ]; then
    EXTRA_ARGS+=(--warmup-checkpoint "$WARMUP_CHECKPOINT")
fi

echo "================================================================================"
echo " Starting Generic Training & Tournament Pipeline"
echo " Arch: $ARCH | Duration: ${DURATION}s | Snapshot Every: ${SNAPSHOT_INTERVAL}s"
echo " Warmup Checkpoint: ${WARMUP_CHECKPOINT:-None}"
echo "================================================================================"

TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/mihaild/prog/ts_ai/.triton_cache}" \
PYTHONPATH=. .venv/bin/python tools/train.py \
  --arch "$ARCH" \
  --duration-seconds "$DURATION" \
  --snapshot-interval-seconds "$SNAPSHOT_INTERVAL" \
  --reward-scheme blunder_aware \
  --eval-opponents heuristic random data/checkpoints/run_v2_blunder_aware_9h/snapshot_21601s.pt \
  --eval-games-per-side 50 \
  --post-tournament \
  --post-tournament-models heuristic random data/checkpoints/run_v2_blunder_aware_9h/snapshot_21601s.pt \
  --post-tournament-games 500 \
  "${EXTRA_ARGS[@]}"
