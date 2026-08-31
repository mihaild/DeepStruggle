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

### Budgeting a run, and what evaluation costs

`--duration-seconds` budgets **training time only**. Snapshot evaluation and start-pool
refreshes are timed separately and excluded, so the flag means what it says.

For an A/B, budget by steps instead:

```bash
# Two arms that are exactly comparable: identical step budget, one flag apart
PYTHONPATH=. .venv/bin/python tools/train.py --arch v2 --train-steps 60000000 ... --start-pool-frac 1.0
PYTHONPATH=. .venv/bin/python tools/train.py --arch v2 --train-steps 60000000 ... --start-pool-frac 0.0
```

A wall-clock budget cannot make two arms comparable, because steps/sec depends on the
policy: the arm whose games run longer has costlier evaluations and gets less training.
That is directional rather than random, and it confounded a real 3-hour A/B, whose arms
finished 1024 and 473 iterations on identical settings. `--train-steps` removes it. Because
a step budget says nothing about elapsed time, the progress line projects a wall-clock ETA
from the observed rate plus measured overhead, so a step budget can still be aimed at a
target duration:

```
[1,310,720/1,500,000 steps, ETA 130s] It   20 | Steps: 1,310,720 (15,356 st/s) | ...
```

`--eval-max-snapshot-opponents` (default 4) bounds evaluation cost. Each snapshot is
otherwise added to the opponent list permanently, making evaluation quadratic in run
length; the final evaluation of a 3-hour run faced 14 opponents and took 957s against a
900s snapshot interval, leaving about one training iteration per interval. Baselines from
`--eval-opponents` are never dropped. Pass `0` for the old unbounded behaviour.

Snapshot evaluation runs through the same vectorized path as `tools/tournament.py`
(~30x the one-game-at-a-time loop it replaced).

Every iteration is logged to `<output-dir>/training_metrics.jsonl` and mirrored to TensorBoard
event files in `<output-dir>/tb/` (`--no-tensorboard` disables the mirror; the JSONL is always
written, and a missing/broken `tensorboard` install only prints a warning). Watch a live run with:

```bash
.venv/bin/python -m tensorboard.main --logdir data/checkpoints/my_new_run/tb
```

Beyond the loss terms, each iteration records `explained_variance` (`1 - Var(G - V) / Var(G)` for
the win-value head — the primary read on whether the critic is learning), advantage-distribution
health (`adv_std`, `adv_std_raw`, `adv_frac_near_zero`), game length (`mean_turn`, `median_turn`,
`episodes_completed`), the ending-reason mix (`ending_frac_*`), and `entropy_fixed_probe` — mean
masked policy entropy on a pool of ~2,000 (observation, mask) pairs frozen at the start of the run,
which unlike the on-policy `entropy` cannot be masked by state-distribution drift.

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
