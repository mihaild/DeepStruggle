---
name: ts-training
description: >-
  Train Twilight Struggle neural network agents using Behavioral Cloning (BC) demonstration
  warmup and vectorized Nash Policy Gradient (NashPG) self-play RL. Activate this skill whenever
  the user asks to train a model, fine-tune a checkpoint, run RL self-play, train with a reward scheme
  (blunder_aware), or generate demonstration datasets.
---

# Twilight Struggle AI: Model Training Runbook

This skill provides step-by-step instructions and standard execution commands for training ColdWarNet neural networks (V1, V2, V3) on the C++ simulation engine.

---

## 1. Quick Reference: Standard Training Commands

### A. Full Time-Bounded RL Self-Play Run (Recommended Default)
Runs vectorized NashPG self-play across 512 parallel C++ environments, with blunder-aware reward shielding, live snapshot evaluation every $N$ seconds, and an automated post-training massive tournament:

```bash
# 2-Hour training run for ColdWarNet V3 starting from supervised warmup seed:
TRITON_CACHE_DIR=.triton_cache PYTHONPATH=. .venv/bin/python tools/train.py \
  --arch v3 \
  --duration-seconds 7200 \
  --snapshot-interval-seconds 1200 \
  --warmup-checkpoint data/checkpoints/coldwar_net_v3_warmup.pt \
  --reward-scheme blunder_aware \
  --eval-opponents heuristic random data/checkpoints/run_v2_blunder_aware_9h/snapshot_21601s.pt \
  --eval-games-per-side 50 \
  --post-tournament \
  --post-tournament-models heuristic random data/checkpoints/run_v2_blunder_aware_9h/snapshot_21601s.pt \
  --post-tournament-games 500
```
> [!NOTE]
> If `--output-dir` is not explicitly passed, `tools/train.py` automatically generates a compliant timestamped directory: `data/checkpoints/run_v3_YYYYMMDD_HHMMSS`.

### B. Fast Smoke-Test / Sanity Check (30 Seconds)
Use this whenever verifying code changes, new loss functions, or architecture modifications before launching a long training session:

```bash
TRITON_CACHE_DIR=.triton_cache PYTHONPATH=. .venv/bin/python tools/train.py \
  --arch v3 \
  --duration-seconds 30 \
  --snapshot-interval-seconds 15 \
  --num-envs 64 \
  --eval-opponents heuristic random \
  --eval-games-per-side 5 \
  --post-tournament \
  --post-tournament-models heuristic random \
  --post-tournament-games 10
```

### C. Phase 0: Supervised BC Demonstration Warmup
Pre-trains a raw model directly on demonstration datasets (`data/datasets/warmup_5k_games.jsonl.gz`) using memory-bounded streaming (<70 MB RAM):

```bash
TRITON_CACHE_DIR=.triton_cache PYTHONPATH=. .venv/bin/python tools/train.py \
  --mode warmup \
  --arch v3 \
  --warmup-dataset data/datasets/warmup_5k_games.jsonl.gz \
  --bc-epochs 2 \
  --batch-size 1024 \
  --output-dir data/checkpoints/coldwar_net_v3_warmup.pt
```

### D. Demonstration Dataset Generation
Generates 5,000 games in parallel using 500 C++ environments and multi-temperature schedules:

```bash
PYTHONPATH=. .venv/bin/python tools/generate_dataset.py \
  --total-games 5000 \
  --batch-size 500 \
  --output data/datasets/warmup_5k_games.jsonl.gz
```

---

## 2. Mandatory Rules & Invariants for Training

1. **Mandatory Checkpoint Directory Naming**:
   All checkpoint runs **MUST** follow:
   `data/checkpoints/run_[version]_[start date]_[start time]`
   *(e.g., `data/checkpoints/run_v3_20260827_205207`). Never use ad-hoc names like `run_2h`.*

2. **Environment Variables**:
   Always prefix training invocations with:
   `TRITON_CACHE_DIR=.triton_cache PYTHONPATH=.`
   This avoids Triton cache permission collisions in shared environments.

3. **Reward Scheme Selection**:
   - `blunder_aware` (Default & Strongly Recommended): Shields the agent from learning spurious event traps (+1.0 for setting DEFCON traps, -1.0 for unprovoked DEFCON suicide, 0.0 for falling into opponent traps).
   - `zero_sum`: Terminal win/loss only (+1.0 / -1.0).
   - `shaped`: Zero-sum with heuristic milestones (VP swing shaping).

4. **Architecture Selection**:
   - `--arch v3` (Current SOTA): Dual Pointer Co-Attention between graph countries and hand cards.
   - `--arch v2`: Cross-Attention between countries and hand cards.
   - `--arch v1`: Flat node + card fusion.

5. **Memory-Bounded Dataset Streaming**:
   When training BC, never load full datasets into monolithic memory tensors. Always use `WarmupDataset.stream_batches`.

---

## 3. Monitoring & Verifying a Training Run

1. **Step Throughput**:
   On a modern CPU with 512 parallel environments, throughput should reach **4,000 – 6,000 steps/sec**.
2. **Loss Convergence**:
   - Policy Loss (`L_policy`): Typically fluctuates between -0.05 and +0.05.
   - Value Loss (`L_value_win`): Should steadily decrease towards < 0.15.
   - KL Divergence from Reference Policy (`D_KL`): Kept bounded below 0.20 by the NashPG reference penalty.
3. **Live Snapshots**:
   Snapshots are written every `--snapshot-interval-seconds` as `snapshot_[N]s.pt` and evaluated against heuristic baselines.
4. **Post-Training Tournament**:
   When `--post-tournament` is enabled, `tools/tournament.py` automatically executes a massive round-robin tournament across all snapshots and baselines upon training completion, dumping `massive_tournament_report.md`.
