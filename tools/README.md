# Twilight Struggle AI CLI Tools (`tools/`)

This directory contains standalone, reusable CLI tools for training, evaluating, tournament benchmarking, replay generation, and checkpoint inspection.

---

## 1. `tools/train.py` (Unified Training Pipeline)
Launches neural network reinforcement learning or supervised warm-up with live snapshot tournament evaluation.

```bash
# 1-Hour RL fine-tuning run with blunder-aware rewards and 10-minute snapshot evaluations
PYTHONPATH=. .venv/bin/python tools/train.py \
  --arch v2 \
  --warmup-checkpoint checkpoints/run_v2_blunder_aware_9h/snapshot_21601s.pt \
  --duration-seconds 3600 \
  --snapshot-interval-seconds 600 \
  --reward-scheme blunder_aware \
  --num-envs 512 \
  --buffer-size 128 \
  --batch-size 4096 \
  --eval-games-per-side 50 \
  --eval-opponents random heuristic checkpoints/run_v2_blunder_aware_9h/snapshot_21601s.pt \
  --output-dir checkpoints/my_new_run
```

---

## 2. `tools/tournament.py` (Massive Parallel Tournament & Elo Calculator)
Runs ultra-fast vectorized round-robin tournaments (300–800 games/sec) across arbitrary models, computing Bradley-Terry Elo ratings, win rate matrices (Total / USSR / US), and side-specific loss cause breakdowns.

```bash
# Run a 2,000-game-per-matchup tournament across all checkpoints in a directory:
PYTHONPATH=. .venv/bin/python tools/tournament.py \
  --checkpoint-dir checkpoints/run_v2_blunder_aware_9h \
  --games-per-side 1000 \
  --batch-chunk-size 1000 \
  --anchor-model HeuristicBot \
  --anchor-elo 1500.0 \
  --output-report checkpoints/run_v2_blunder_aware_9h/massive_tournament_report.md \
  --output-json checkpoints/run_v2_blunder_aware_9h/massive_tournament_results.json
```

---

## 3. `tools/evaluate.py` (Head-to-Head Match Evaluator)
Runs a head-to-head match between any two models or bots, reporting separate US and USSR win rates and exact loss causes.

```bash
# Evaluate model against HeuristicBot for 100 games (50 US / 50 USSR)
PYTHONPATH=. .venv/bin/python tools/evaluate.py \
  --agent-a checkpoints/run_v2_blunder_aware_9h/snapshot_21601s.pt \
  --agent-b heuristic \
  --games-per-side 50
```

---

## 4. `tools/generate_replay.py` (Game Replay Generator)
Generates full step-by-step game replays (`.tslog.json`) and prints the direct web workbench URL.

```bash
# Generate a self-play game for a checkpoint:
PYTHONPATH=. .venv/bin/python tools/generate_replay.py \
  --model checkpoints/run_v2_blunder_aware_9h/snapshot_21601s.pt \
  --seed 42 \
  --temperature 0.2 \
  --game-id snapshot_21601s_selfplay

# View in Web Workbench at:
# http://localhost:8000/?replay=snapshot_21601s_selfplay.tslog.json
```

---

## 5. `tools/inspect_checkpoints.py` (Checkpoint Registry Inspector)
Scans `checkpoints/` and displays all saved models, sizes, timestamps, and detected architectures (`ColdWarNet` V1 vs `ColdWarNetV2`).

```bash
PYTHONPATH=. .venv/bin/python tools/inspect_checkpoints.py
```
