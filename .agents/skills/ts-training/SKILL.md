---
name: ts-training
description: >-
  Train Twilight Struggle neural network agents using Behavioral Cloning (BC) demonstration
  warmup and vectorized Nash Policy Gradient (NashPG) self-play RL. Activate this skill whenever
  the user asks to train a model, train starting from a warmup dataset, run behavioral cloning / clone behavior,
  fine-tune a checkpoint, run RL self-play, or generate demonstration datasets.
---

# Twilight Struggle AI: Model Training Runbook

This skill provides step-by-step instructions and standard execution commands for training ColdWarNet neural networks on the C++ simulation engine.

> [!CAUTION]
> **MANDATORY: NEVER WRITE AD-HOC PYTHON SCRIPTS TO TRAIN MODELS OR CLONE BEHAVIOR!**
> Do **NOT** create scratch scripts, one-off `.py` files, or ad-hoc scripts importing `ai.training.behavioral_cloning`.
> ALL training workflows (Behavioral Cloning warmup, RL self-play, or hybrid BC warmup + RL) MUST be run using the unified CLI:
> `tools/train.py`

> [!IMPORTANT]
> **Dynamic Versions, Checkpoints & Datasets**:
> Model architectures (`v1`, `v2`, `v3`, `v4`...), checkpoint directories, and demonstration datasets evolve continuously over time:
> - **Architecture**: Check `ai/models/` or current SOTA flag (e.g. `--arch v3`). Never assume a fixed architecture version across runs.
> - **Warmup Seed Checkpoint**: Inspect available models via `python tools/inspect_checkpoints.py` to select the current champion or warmup checkpoint.
> - **Datasets**: Check `data/datasets/` for the latest demonstration dataset (e.g. `data/datasets/*.jsonl.gz`).
> - The paths shown in the examples below use representative placeholders `<arch>`, `<warmup_seed.pt>`, and `<dataset.jsonl.gz>`.

---

## 1. "Train Model Starting from Warmup Dataset" Workflows

Depending on the user's intent, there are two standard ways to train starting from a warmup dataset:

### Option A: Pure BC Warmup (Supervised Pre-training Only)
When the user asks to "warm up a model", "pre-train on dataset", or "clone behavior":
```bash
TRITON_CACHE_DIR=.triton_cache PYTHONPATH=. .venv/bin/python tools/train.py \
  --mode warmup \
  --arch <arch> \
  --warmup-dataset data/datasets/<latest_dataset>.jsonl.gz \
  --bc-epochs 2 \
  --batch-size 1024 \
  --output-dir data/checkpoints/coldwar_net_<arch>_warmup.pt
```
*This streams the dataset without high RAM usage (<70 MB), runs BC epochs, and saves the standalone warmup checkpoint.*

### Option B: Full RL Training Starting from Warmup Dataset (BC Warmup -> NashPG RL)
When the user asks to train a model starting from a warmup dataset. Budgets are in env
steps, not hours -- if the user asks for $N$ hours, convert using the run's observed
steps/sec and say what you assumed:
```bash
TRITON_CACHE_DIR=.triton_cache PYTHONPATH=. .venv/bin/python tools/train.py \
  --arch <arch> \
  --warmup-dataset data/datasets/<latest_dataset>.jsonl.gz \
  --bc-epochs 2 \
  --train-steps 160000000 \
  --snapshot-every-steps 5000000 \
  --reward-scheme blunder_aware \
  --eval-opponents heuristic random \
  --eval-games-per-side 50 \
  --post-tournament
```
*`tools/train.py` automatically runs BC warmup first, saves the warmed-up checkpoint in the run directory, loads the weights into active and reference policies, and immediately begins NashPG RL self-play!*

---

## 2. Standard Training Commands

### A. Full Time-Bounded RL Run Starting from Existing Checkpoint
Runs vectorized NashPG self-play across 512 parallel C++ environments, with blunder-aware reward shielding, live snapshot evaluation, and automated post-training massive tournament:

```bash
TRITON_CACHE_DIR=.triton_cache PYTHONPATH=. .venv/bin/python tools/train.py \
  --arch <arch> \
  --train-steps 160000000 \
  --snapshot-every-steps 5000000 \
  --warmup-checkpoint data/checkpoints/<latest_warmup_or_champion>.pt \
  --reward-scheme blunder_aware \
  --eval-opponents heuristic random data/checkpoints/<historical_champion>.pt \
  --eval-games-per-side 50 \
  --post-tournament \
  --post-tournament-models heuristic random data/checkpoints/<historical_champion>.pt \
  --post-tournament-games 500
```
> [!NOTE]
> If `--output-dir` is not explicitly passed, `tools/train.py` automatically generates a compliant timestamped directory using the current architecture: `data/checkpoints/run_<arch>_YYYYMMDD_HHMMSS`.

### B. Fast Smoke-Test / Sanity Check (30 Seconds)
Use this whenever verifying code changes, new loss functions, or architecture modifications before launching a long training session:

```bash
TRITON_CACHE_DIR=.triton_cache PYTHONPATH=. .venv/bin/python tools/train.py \
  --arch <arch> \
  --train-steps 200000 \
  --snapshot-every-steps 100000 \
  --num-envs 64 \
  --eval-opponents heuristic random \
  --eval-games-per-side 5 \
  --post-tournament \
  --post-tournament-models heuristic random \
  --post-tournament-games 10
```

### C. Demonstration Dataset Generation
Generates thousands of games in parallel using 500 C++ environments and multi-temperature exploration schedules:

```bash
PYTHONPATH=. .venv/bin/python tools/generate_dataset.py \
  --total-games 5000 \
  --batch-size 500 \
  --output data/datasets/warmup_<date>_<N>k_games.jsonl.gz
```

---

## 3. Mandatory Rules & Invariants for Training

1. **Pre-Training Commit & Metadata Record**:
   Before launching any training run:
   - Create a clean git commit on the active branch recording all codebase modifications (without pushing / advancing remote master).
   - Ensure a short description of the changes and training intent is provided via `--description "<text>"`.
   - `tools/train.py` automatically captures the git commit hash, commit message, training mode, and description, writing `metadata.json` into the checkpoint directory.

2. **Never Write Ad-Hoc Scripts**:
   All training, BC warmup, and fine-tuning **MUST** be invoked through `tools/train.py`. Writing standalone Python scripts or scratch files violates repository architecture.

2. **Mandatory Checkpoint Directory Naming**:
   All checkpoint runs **MUST** follow the versioned timestamp pattern:
   `data/checkpoints/run_[version]_[start date]_[start time]`
   *(e.g., `data/checkpoints/run_v3_20260827_205207`). Never use hardcoded or ad-hoc names like `run_2h`.*

3. **Environment Variables**:
   Always prefix training invocations with:
   `TRITON_CACHE_DIR=.triton_cache PYTHONPATH=.`
   This prevents Triton cache permission collisions in shared environments.

4. **Reward Scheme Selection**:
   - `blunder_aware` (Default & Strongly Recommended): Shields the agent from learning spurious event traps (+1.0 for setting DEFCON traps, -1.0 for unprovoked DEFCON suicide, 0.0 for falling into opponent traps).
   - `zero_sum`: Terminal win/loss only (+1.0 / -1.0).
   - `shaped`: Zero-sum with heuristic milestones (VP swing shaping).

5. **Architecture Evolution**:
   - Always check current architectures available in `ai/models/` (e.g. `v3` dual pointer co-attention, `v2` cross-attention).
   - Ensure the `--arch` flag matches the network architecture being trained or fine-tuned.

6. **Memory-Bounded Dataset Streaming**:
   When training BC, never load full datasets into monolithic memory tensors. Always use bounded streaming via `WarmupDataset.stream_batches`.

---

## 4. Monitoring & Verifying a Training Run

1. **Step Throughput**:
   On a modern CPU with 512 parallel environments, throughput should reach **4,000 – 6,000 steps/sec**.
2. **Loss Convergence**:
   - Policy Loss (`L_policy`): Typically fluctuates between -0.05 and +0.05.
   - Value Loss (`L_value_win`): Should steadily decrease towards < 0.15.
   - KL Divergence from Reference Policy (`D_KL`): Kept bounded below 0.20 by the NashPG reference penalty.
3. **Live Snapshots**:
   Snapshots are written every `--snapshot-every-steps` as `snapshot_[N]steps.pt` and evaluated against heuristic baselines. That cadence also sets how fast the self-play opponent pool grows, so two arms of an A/B must share it or they are not a one-factor comparison.
4. **Post-Training Tournament**:
   When `--post-tournament` is enabled, `tools/tournament.py` automatically executes a massive round-robin tournament across all snapshots and baselines upon training completion, dumping `massive_tournament_report.md`.
