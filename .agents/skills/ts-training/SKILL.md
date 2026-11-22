---
name: ts-training
description: >-
  Train Twilight Struggle neural network agents using Behavioral Cloning (BC) demonstration
  warmup and vectorized Nash Policy Gradient (NashPG) self-play RL. Activate this skill whenever
  the user asks to train a model, train starting from a warmup dataset, run behavioral cloning / clone behavior,
  fine-tune a checkpoint, run RL self-play, or generate demonstration datasets.
---

# Twilight Struggle AI: Model Training Runbook

How to launch training on the C++ engine. The rules here are the repository's invariants
(`CLAUDE.md`, `AGENTS.md` §4); the full flag reference is `tools/README.md`.

> [!CAUTION]
> **NEVER WRITE AD-HOC PYTHON SCRIPTS TO TRAIN MODELS OR CLONE BEHAVIOR.**
> All training — BC warmup, RL self-play, or BC warmup followed by RL — goes through `tools/train.py`.

---

## 1. Before launching

1. **Fresh engine**: prefix every launch with `tools/scripts/check_engine_fresh.sh &&` (it rebuilds
   and exits 1 when the build is stale, so the launch does not run against old rules).
2. **Commit first** (locally, never push) and pass `--description "<what changed and why>"`;
   `tools/train.py` writes the commit, mode and description into the run's `metadata.json`.
3. **Derive the flags from a healthy run, never from a template** (the examples below show shape
   only):
   ```bash
   tools/scripts/launch_flags.py <a-recent-healthy-run-dir>              # what it actually used
   tools/scripts/launch_flags.py <reference-run> --diff <your-new-run>   # after launching: what changed
   ```
   The architecture and recipe codes (A/R in the run name) are flag sets in
   `research/architectures_and_recipes.md`; the current lineage is `--arch ladder` with its
   required `--ladder-*` flags.
4. **An RL run needs the opponent pool** (`--opponent-frac 0.3 --opponent-self-pool`, pool size 12):
   without it one seat runs away and the critic collapses. The startup banner must say
   `[opponent pool] ... frac=0.3`.
5. **Name the run**: `--run-name E<n>-A<n>-R<n>-S<n>[@<step>M+<changes>]`
   (`research/method/run_nomenclature.md`); the directory becomes
   `data/checkpoints/<name>_<date>_<time>`, and `tools/scripts/run_codes.py --run <dir>` checks
   the name against the recorded flags. Without a run name the directory is `run_<arch>_<date>_<time>`,
   for smoke tests only.

## 2. Commands

### RL self-play
```bash
tools/scripts/check_engine_fresh.sh && TRITON_CACHE_DIR=.triton_cache PYTHONPATH=. .venv/bin/python tools/train.py \
  --arch ladder <A-code flags> <R-code flags> \
  --run-name E<n>-A<n>-R<n>-S<n> --description "..." \
  --train-steps <N> --snapshot-every-steps 10000000 \
  --eval-opponents heuristic random --eval-games-per-side 50
```
Budgets are in env steps, never hours; if the user asks for hours, convert with the run's observed
steps/sec and say what you assumed. For an A/B, both arms share `--train-steps` and
`--pool-every-steps`. A continuation of a run under the same name is `--resume <its-dir>` and
writes into the same directory. Watch a launched run with
`tools/scripts/watch_run.py <run-dir> --target-steps <N>`, which reports START / PROGRESS / STALL /
NOSTART / CRASH / DONE, since silence and success look alike.

### Optional: BC warmup (not what the ladder does)
```bash
TRITON_CACHE_DIR=.triton_cache PYTHONPATH=. .venv/bin/python tools/train.py \
  --mode warmup --arch ladder <A-code flags> --warmup-dataset <dataset.jsonl.gz> \
  --bc-epochs 2 --batch-size 1024 --output-dir <warmup.pt>
```
Then `--warmup-checkpoint <warmup.pt>` on the RL run, or pass `--warmup-dataset` directly to an RL
run to warm up and continue in one go. Say so in the run's description: a warm start is a
difference from the lineage. Datasets stream with bounded memory (`WarmupDataset.stream_batches`),
never loaded whole.

### Smoke test (seconds)
```bash
TRITON_CACHE_DIR=.triton_cache PYTHONPATH=. .venv/bin/python tools/train.py \
  --arch v2 --train-steps 200000 --snapshot-every-steps 100000 --num-envs 64 \
  --eval-opponents heuristic random --eval-games-per-side 5
```

### Demonstration datasets (rebuild after every engine change)
```bash
PYTHONPATH=.:build/release .venv/bin/python tools/generate_dataset.py \
  --models <checkpoint.pt> --total-games 5000 --output-path <dataset.jsonl.gz>   # self-play: a syntax prior
PYTHONPATH=.:build/release .venv/bin/python tools/build_human_dataset.py          # human corpus: the strategy prior
```

## 3. What a run writes

- `snapshot_<N>steps.pt` every `--snapshot-every-steps` (default 10M), evaluated live; evaluation
  leaves training's RNG untouched, so the interval is reporting only. The opponent pool grows on
  its own schedule, `--pool-every-steps` (default 5M, `pool_<N>steps.pt`).
- `training_metrics.jsonl`, mirrored to `<run-dir>/tb` for TensorBoard; `metadata.json` with the
  flags, commit and description.
- With `--post-tournament`, a round robin over the snapshots at the end (`final_tournament_report.md`).
  Ratings that are compared across runs come from the leaderboard instead (the `ts-tournament` skill).
- Reward schemes (`--reward-scheme`): `terminal` (default: winner +1, loser -1), `shaped`,
  `useful_actions`, `curriculum`. `blunder_aware` and the blunder window were removed on 2026-11-22
  (`--no-blunder-window` / `--no-held-scoring-blunder` are still accepted, with no effect).
