#!/usr/bin/env bash
# Runs ColdWarNet reinforcement learning directly without memory-heavy BC warmup datasets,
# automatically creating a timestamped checkpoint directory and running a massive tournament
# benchmark upon completion.
#
# Usage: train_direct_rl.sh [arch] [duration_s] [snapshot_interval_s] [opponent.pt]
#
# The opponent checkpoint is optional and is a path on your machine. Without it the run is
# measured against the built-in heuristic and random bots.

set -e

ARCH="${1:-v2}"               # v1 | v2 | mlp
DURATION="${2:-7200}"         # Default 2 hours (7200s)
SNAPSHOT_INTERVAL="${3:-600}" # Default 10 minutes (600s)
OPPONENT_CHECKPOINT="${4:-}"  # Optional: extra opponent for eval and tournament

OPPONENTS=(heuristic random)
if [ -n "$OPPONENT_CHECKPOINT" ]; then
    if [ ! -f "$OPPONENT_CHECKPOINT" ]; then
        echo "opponent checkpoint not found: $OPPONENT_CHECKPOINT" >&2
        exit 1
    fi
    OPPONENTS+=("$OPPONENT_CHECKPOINT")
fi

echo "Starting generic RL training: arch=$ARCH, duration=${DURATION}s, snapshot_interval=${SNAPSHOT_INTERVAL}s"
echo "Opponents: ${OPPONENTS[*]}"

PYTHONPATH=. .venv/bin/python tools/train.py \
  --arch "$ARCH" \
  --duration-seconds "$DURATION" \
  --snapshot-interval-seconds "$SNAPSHOT_INTERVAL" \
  --reward-scheme blunder_aware \
  --eval-opponents "${OPPONENTS[@]}" \
  --eval-games-per-side 50 \
  --post-tournament \
  --post-tournament-models "${OPPONENTS[@]}" \
  --post-tournament-games 500
