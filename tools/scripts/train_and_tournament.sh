#!/usr/bin/env bash
# Generic end-to-end training and tournament pipeline runner.
# Supports warm-started or tabula-rasa RL with periodic snapshots,
# blunder-aware rewards, live evaluation, and post-training massive tournament.
#
# Usage: train_and_tournament.sh [arch] [train_steps] [snapshot_every_steps] [warmup.pt] [opponent.pt]
#
# Both budgets are in env steps, not seconds. A wall-clock budget cannot make two arms
# comparable, because steps/sec depends on the policy, and the snapshot interval also sets
# how fast the self-play opponent pool grows.
#
# The last two are optional and are paths on your machine: a warmup checkpoint to start from,
# and an extra opponent to evaluate and run the tournament against. With neither, the run starts
# from a fresh network and is measured against the built-in heuristic and random bots.

set -euo pipefail

ARCH="${1:-v2}"                       # v1 | v2 | mlp
TRAIN_STEPS="${2:-80000000}"          # Default: the standard 80M env steps
SNAPSHOT_EVERY_STEPS="${3:-5000000}"  # Default: every 5M steps, which also grows the pool
WARMUP_CHECKPOINT="${4:-}"            # Optional: path to a BC warmup checkpoint
OPPONENT_CHECKPOINT="${5:-}"          # Optional: extra opponent for eval and tournament

# A checkpoint that was asked for and is not there is an error, not something to quietly drop:
# the run would otherwise start tabula rasa while its log said it was warm-started.
EXTRA_ARGS=()
if [ -n "$WARMUP_CHECKPOINT" ]; then
    if [ ! -f "$WARMUP_CHECKPOINT" ]; then
        echo "warmup checkpoint not found: $WARMUP_CHECKPOINT" >&2
        echo "Build one first with: tools/train.py --mode warmup ..." >&2
        exit 1
    fi
    EXTRA_ARGS+=(--warmup-checkpoint "$WARMUP_CHECKPOINT")
fi

OPPONENTS=(heuristic random)
if [ -n "$OPPONENT_CHECKPOINT" ]; then
    if [ ! -f "$OPPONENT_CHECKPOINT" ]; then
        echo "opponent checkpoint not found: $OPPONENT_CHECKPOINT" >&2
        exit 1
    fi
    OPPONENTS+=("$OPPONENT_CHECKPOINT")
fi

echo "================================================================================"
echo " Starting Generic Training & Tournament Pipeline"
echo " Arch: $ARCH | Budget: ${TRAIN_STEPS} steps | Snapshot Every: ${SNAPSHOT_EVERY_STEPS} steps"
echo " Warmup Checkpoint: ${WARMUP_CHECKPOINT:-None}"
echo " Opponents: ${OPPONENTS[*]}"
echo "================================================================================"

TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$(pwd)/.triton_cache}" \
PYTHONPATH=. .venv/bin/python tools/train.py \
  --arch "$ARCH" \
  --train-steps "$TRAIN_STEPS" \
  --snapshot-every-steps "$SNAPSHOT_EVERY_STEPS" \
  --reward-scheme blunder_aware \
  --eval-opponents "${OPPONENTS[@]}" \
  --eval-games-per-side 50 \
  --post-tournament \
  --post-tournament-models "${OPPONENTS[@]}" \
  --post-tournament-games 500 \
  "${EXTRA_ARGS[@]}"
