---
name: ts-tournament
description: >-
  Run tournaments and head-to-head match evaluations between Twilight Struggle neural network models
  and heuristic bots. Activate this skill whenever the user asks to evaluate a model, compare two
  checkpoints, run a tournament, calculate Bradley-Terry Elo ratings, diagnose loss causes, or benchmark agents.
---

# Twilight Struggle AI: Tournament & Evaluation Runbook

This skill provides step-by-step instructions and standard execution commands for benchmarking Twilight Struggle models using vectorized parallel simulation (300–800 games/sec).

> [!IMPORTANT]
> **Dynamic Checkpoints & Architecture Versions**:
> The checkpoint paths (e.g. `data/checkpoints/run_v3_...`) and architecture versions (`v1`, `v2`, `v3`, etc.) shown throughout this runbook are **illustrative examples**.
> Checkpoints, runs, and champion models will evolve over time. Always run:
> ```bash
> PYTHONPATH=. .venv/bin/python tools/inspect_checkpoints.py
> ```
> to discover currently available checkpoints, their timestamps, and their detected architectures before launching an evaluation.

---

## 1. Quick Reference: Evaluation & Tournament Commands

### A. 2-Model Head-to-Head Matchup Mode
Evaluates a single pair of models with granular side-by-side performance and exact loss cause diagnostics:

```bash
# Evaluate a snapshot against HeuristicBot for 100 games (50 US / 50 USSR):
# (Replace with your target checkpoint path discovered via tools/inspect_checkpoints.py)
PYTHONPATH=. .venv/bin/python tools/tournament.py \
  --models data/checkpoints/<target_run>/<snapshot>.pt heuristic \
  --games-per-side 50 \
  --device cpu
```

```bash
# Compare two different neural checkpoints directly:
PYTHONPATH=. .venv/bin/python tools/tournament.py \
  --models data/checkpoints/<run_A>/<snapshot_A>.pt data/checkpoints/<run_B>/<snapshot_B>.pt \
  --games-per-side 100 \
  --device cpu
```

### B. Massive Round-Robin Tournament (Entire Directory)
Runs full round-robin matches across all checkpoints in a folder plus baseline heuristics, generating a Bradley-Terry MLE Elo leaderboard:

```bash
# Run tournament across any target run directory:
PYTHONPATH=. .venv/bin/python tools/tournament.py \
  --checkpoint-dir data/checkpoints/<target_run_directory> \
  --games-per-side 500 \
  --batch-chunk-size 1000 \
  --anchor-model HeuristicBot \
  --anchor-elo 1500.0 \
  --output-report data/checkpoints/<target_run_directory>/massive_tournament_report.md \
  --output-json data/checkpoints/<target_run_directory>/massive_tournament_results.json \
  --device cpu
```

### C. Checkpoint Discovery & Architecture Inspection
Quickly displays all saved `.pt` models, file sizes, timestamps, and detected network architectures (ColdWarNet V1, V2, or V3):

```bash
PYTHONPATH=. .venv/bin/python tools/inspect_checkpoints.py
```

---

## 2. Understanding Tournament Diagnostics & Output

### 1. Bradley-Terry MLE Elo Ratings
- **Anchor**: By default, `HeuristicBot` is anchored to `1500.0` Elo.
- **Interpretation**:
  - `RandomBot`: Typically ~1340 – 1350 Elo.
  - `HeuristicBot`: 1500.0 Elo (Baseline standard).
  - `Supervised Warmup Checkpoints`: Typically ~1650 – 1700 Elo (>85% win rate vs Heuristic).
  - `Trained RL SOTA Checkpoints`: Typically ~1800 – 1950+ Elo.

### 2. Side-Specific Win Rate Asymmetry
Twilight Struggle has a well-known historical balance dynamic:
- **USSR Early Advantage**: USSR typically has higher win rates in early turns (Turns 1–3).
- **US Late-Game Advantage**: US tends to dominate late-war turns (Turns 7–10).
- When analyzing evaluation reports, always inspect both **US Win Rate** and **USSR Win Rate** separately.

### 3. Diagnostic Loss Causes Breakdown
`tools/tournament.py` and `TournamentEvaluator` classify every game ending into 5 canonical, exhaustive categories:
- **`DEFCON 1 (own decision)`**: The player made a 1-step avoidable decision that reduced DEFCON to 1 (e.g. couping a battleground country at DEFCON 2, couping when *Cuban Missile Crisis* continuous effect is active, or playing an event that lowers DEFCON at DEFCON 2).
- **`DEFCON 1 (opponent decision)`**: The player was forced to drop DEFCON to 1 by an opponent's decision (e.g. playing an opponent's card for Operations whose event triggers and degrades DEFCON, or opponent conducts an event coup).
- **`20 VP`**: Game ended early due to the victory point track reaching $+20$ or hBc20$ margin (including Europe Control, regional scoring cards, event VP swings, space race, or held scoring cards at turn end).
- **`final scoring`**: Game reached the end of Turn 10 and was decided on regional final scoring and Chinese Civil War bonus.
- **`wargames`**: Game ended early by the execution of Card #100 *Wargames* event.

---

## 3. Important Evaluation Flags & Options

Flag | Default | Description
:--- | :--- | :---
`--models` | `None` | List of models to benchmark (`heuristic`, `random`, or `.pt` paths).
`--checkpoint-dir` | `None` | Automatically discovers all `.pt` snapshots in the directory.
`--games-per-side` | `500` | Number of games per role (total $2 \times N$ games per matchup pair).
`--batch-chunk-size`| `1000` | Parallel vectorized environment chunk size.
`--anchor-model` | `HeuristicBot` | Model used as fixed reference for Elo scaling.
`--anchor-elo` | `1500.0` | Target rating for the anchor model.
`--output-report` | `None` | Path to write comprehensive Markdown tournament report.
`--output-json` | `None` | Path to write machine-readable JSON results.
`--device` | `cpu` | Device for inference (`cpu` or `cuda`).
