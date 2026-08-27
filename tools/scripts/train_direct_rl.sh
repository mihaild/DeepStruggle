#!/usr/bin/env bash
# Runs ColdWarNet reinforcement learning directly without memory-heavy BC warmup datasets,
# automatically creating checkpoints in data/checkpoints/run_[version]_[start date]_[start time]
# and running a massive tournament benchmark upon completion.

set -e

ARCH="${1:-v3}"
DURATION="${2:-7200}"       # Default 2 hours (7200s)
SNAPSHOT_INTERVAL="${3:-600}" # Default 10 minutes (600s)

echo "Starting generic RL training: arch=$ARCH, duration=${DURATION}s, snapshot_interval=${SNAPSHOT_INTERVAL}s"

PYTHONPATH=. .venv/bin/python tools/train.py \
  --arch "$ARCH" \
  --duration-seconds "$DURATION" \
  --snapshot-interval-seconds "$SNAPSHOT_INTERVAL" \
  --reward-scheme blunder_aware \
  --eval-opponents heuristic random data/checkpoints/run_v2_blunder_aware_9h/snapshot_21601s.pt \
  --eval-games-per-side 50 \
  --post-tournament \
  --post-tournament-models heuristic random data/checkpoints/run_v2_blunder_aware_9h/snapshot_21601s.pt \
  --post-tournament-games 500
