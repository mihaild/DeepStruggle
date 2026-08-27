# Twilight Struggle AI CLI Tools (`tools/`)

This directory contains standalone, reusable developer CLI tools for training, tournament benchmarking, match simulation, replay generation, demonstration dataset creation, and checkpoint inspection.

---

## 1. `tools/train.py` (Unified Training Pipeline)
Launches neural network reinforcement learning (NashPG) or supervised demonstration warmup with live snapshot tournament evaluation.

```bash
# RL training run with blunder-aware rewards and 20-minute snapshot evaluations
TRITON_CACHE_DIR=.triton_cache PYTHONPATH=. .venv/bin/python tools/train.py \
  --arch v3 \
  --warmup-checkpoint data/checkpoints/coldwar_net_v3_warmup.pt \
  --duration-seconds 7200 \
  --snapshot-interval-seconds 1200 \
  --reward-scheme blunder_aware \
  --num-envs 512 \
  --eval-games-per-side 50 \
  --eval-opponents random heuristic \
  --output-dir data/checkpoints/my_new_run
```

---

## 2. `tools/tournament.py` (Unified Tournament & Head-to-Head Evaluator)
Fast vectorized tournament and matchup evaluator (300–800 games/sec). Adapts automatically based on the number of models passed:

### A. 2-Model Head-to-Head Matchup Mode:
```bash
# Evaluate model against HeuristicBot for 100 games (50 US / 50 USSR) with loss causes
PYTHONPATH=. .venv/bin/python tools/tournament.py \
  --models data/checkpoints/.../snapshot_3602s.pt heuristic \
  --games-per-side 50
```

### B. Multi-Model Round-Robin Tournament Mode:
```bash
# Run a 1,000-game-per-matchup round-robin tournament across all checkpoints in a directory:
PYTHONPATH=. .venv/bin/python tools/tournament.py \
  --checkpoint-dir data/checkpoints/run_v3_20260827_205207 \
  --games-per-side 500 \
  --anchor-model HeuristicBot \
  --anchor-elo 1500.0 \
  --output-report data/checkpoints/run_v3_20260827_205207/massive_tournament_report.md
```

---

## 3. `tools/play_match.py` (Unified Match Runner & Replay Generator)
Plays matches between any pair of agents, supports two distinct checkpoints, provides interactive CLI terminal play, and generates standardized `.tslog.json` replays for the Web Workbench.

```bash
# 1. Pit two different neural checkpoints against each other:
PYTHONPATH=. .venv/bin/python tools/play_match.py \
  --us data/checkpoints/run_v3_20260827_205207/snapshot_3602s.pt \
  --ussr data/checkpoints/run_v2_blunder_aware_9h/snapshot_21601s.pt \
  --game-id v3_vs_v2

# 2. Interactive Terminal Play (Human vs AI Bot):
PYTHONPATH=. .venv/bin/python tools/play_match.py \
  --us human \
  --ussr heuristic

# 3. Rich Strategic Commentary & Regional Scoring Breakdown:
PYTHONPATH=. .venv/bin/python tools/play_match.py \
  --us strategic \
  --ussr event_heavy \
  --commentary

# View any generated replay in the Web Workbench at:
# http://localhost:8000/?replay=<filename>.tslog.json
```

---

## 4. `tools/generate_dataset.py` (Vectorized Demonstration Dataset Generator)
Charns out thousands of games in parallel using 500 C++ environments and multi-temperature exploration schedules, dumping compressed `.jsonl.gz` datasets for supervised BC warmup training.

```bash
PYTHONPATH=. .venv/bin/python tools/generate_dataset.py \
  --total-games 5000 \
  --batch-size 500 \
  --output data/datasets/warmup_5k_games.jsonl.gz
```

---

## 5. `tools/inspect_checkpoints.py` (Checkpoint Registry Inspector)
Scans `data/checkpoints/` and displays all saved models, file sizes, modification timestamps, and detected network architectures (`ColdWarNet` V1, V2, or V3).

```bash
PYTHONPATH=. .venv/bin/python tools/inspect_checkpoints.py
```

---

## 6. Shared Helpers Library (`tools/lib/`)
Contains internal simulation, evaluation, and logging modules imported by the CLI tools:
- `tools/lib/player_agent.py`: Unified agent loader (`load_agent`) and policy inference wrappers.
- `tools/lib/batch_tournament.py`: High-throughput C++ batch tournament runner and Bradley-Terry MLE solver.
- `tools/lib/tournament_evaluator.py`: Diagnostic loss cause classifier (`classify_game_ending_reason`).
- `tools/lib/self_play.py`: Single-game trajectory runner.
- `tools/lib/scoring_formatter.py`: Regional scoring calculation formatter.
- `tools/lib/checkpoint_utils.py`: Architecture detection and checkpoint discovery utilities.
