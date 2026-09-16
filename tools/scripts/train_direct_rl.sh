#!/usr/bin/env bash
# Runs ColdWarNet reinforcement learning directly without memory-heavy BC warmup datasets,
# automatically creating a timestamped checkpoint directory and running a massive tournament
# benchmark upon completion.
#
# Usage: train_direct_rl.sh [arch] [train_steps] [snapshot_every_steps] [opponent.pt]
#
# Budgets are in env steps. The snapshot interval also sets how fast the self-play opponent
# pool grows, so two arms of an A/B must share it.
#
# The opponent checkpoint is optional and is a path on your machine. Without it the run is
# measured against the built-in heuristic and random bots.

set -e

ARCH="${1:-v2}"               # v1 | v2 | mlp
TRAIN_STEPS="${2:-80000000}"          # Default: the standard 80M env steps
SNAPSHOT_EVERY_STEPS="${3:-5000000}"  # Default: every 5M steps
OPPONENT_CHECKPOINT="${4:-}"  # Optional: extra opponent for eval and tournament

OPPONENTS=(heuristic random)
if [ -n "$OPPONENT_CHECKPOINT" ]; then
    if [ ! -f "$OPPONENT_CHECKPOINT" ]; then
        echo "opponent checkpoint not found: $OPPONENT_CHECKPOINT" >&2
        exit 1
    fi
    OPPONENTS+=("$OPPONENT_CHECKPOINT")
fi

echo "Starting generic RL training: arch=$ARCH, budget=${TRAIN_STEPS} steps, snapshot every ${SNAPSHOT_EVERY_STEPS} steps"
echo "Opponents: ${OPPONENTS[*]}"

PYTHONPATH=. .venv/bin/python tools/train.py \
  --arch "$ARCH" \
  --train-steps "$TRAIN_STEPS" \
  --snapshot-every-steps "$SNAPSHOT_EVERY_STEPS" \
  --reward-scheme blunder_aware \
  --eval-opponents "${OPPONENTS[@]}" \
  --eval-games-per-side 50 \
  --post-tournament \
  --post-tournament-models "${OPPONENTS[@]}" \
  --post-tournament-games 500
