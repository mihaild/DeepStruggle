# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This repo already has a detailed **`AGENTS.md`** (plus per-directory `engine/AGENTS.md`, `bindings/AGENTS.md`, `bot/AGENTS.md`, `web/server/AGENTS.md`) — read those for full architecture diagrams and directory listings. This file summarizes what you need to be productive immediately.

## What this is

The complete AI, simulation engine, web workbench, and training infrastructure for a Deluxe Edition **Twilight Struggle** (110-card) implementation. Three layers:

1. **C++20 engine** (`engine/`) — zero-allocation simulation core, exposed to Python via nanobind.
2. **Python AI/training stack** (`ai/`, `bindings/`, `tools/`) — neural net, RL training (NashPG), rewards, bots.
3. **Web workbench** (`web/`) — FastAPI game server + Vite/TypeScript SVG map UI for human vs bot/replay play.

## Build & environment setup

```bash
# Python venv
python3 -m venv .venv && source .venv/bin/activate
pip install nanobind fastapi "uvicorn[standard]" websockets pytest numpy pydantic httpx torch torchvision
# z3-solver (MIT) reconstructs the hands behind a ts-replayer log; the converter falls
# back to per-turn heuristics without it, so it is optional but wanted for that work.
pip install z3-solver

# Web UI
cd web/ui && npm install && npm run build && cd ../..

# C++ engine + nanobind extension (root CMake orchestrates engine/ and bindings/)
cmake -B build/release -S . -DPython_EXECUTABLE=$(pwd)/.venv/bin/python3
cmake --build build/release -j
```

The extension lands at `build/release/ts_engine.cpython-*.so` — the binary root, not
`build/release/bindings/` — and is imported via `PYTHONPATH=.:build/release`.

**Never run an experiment against a stale engine** (key invariant #10). Guard every command
that generates or consumes a checkpoint, dataset, Elo anchor or diagnostic:

```bash
tools/scripts/check_engine_fresh.sh && PYTHONPATH=.:build/release .venv/bin/python tools/train.py ...
```

It exits 0 when the built module matches the `engine/`/`bindings/` sources, and rebuilds and
exits 1 when it does not, so the chained command does not run. A rebuilt engine can change the
decision stream with no Python change: datasets in the `(seed, actions)` format then truncate
silently (the archived warmup set kept 27% of its decisions — `data/datasets/archive/README.md`)
and old checkpoints keep loading and running forward passes, so nothing announces the problem.

Sanitizer build (AddressSanitizer + UBSan), for engine work:
```bash
cmake -B build_san -S . -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build_san -j
```

## Tests & type checking

**`tests/` is split into groups, and not every group is relevant to every change.** Match the
suite to what you touched — most importantly, **do not run `tests/web` for engine, bindings, AI or
tools changes.** It tells you nothing about them and fails on missing build artifacts (`web/ui/dist`,
a Playwright browser) that are deliberately not in the repo, so a red web suite on a backend change
is pure noise.

| you changed | run |
|:---|:---|
| `engine/`, `bindings/` | C++ suite, then `tests/bindings tests/engine_logic` (~30s) |
| `tools/lib/ts_replayer_*` | the above + `tests/replayer` (~6.5 min) |
| `ai/`, `bot/`, `tools/` | the above + `tests/training` (~7 min) |
| `web/**` | **also** `tests/web`, after building the UI (below) |

```bash
# 0. Always first — never measure against a stale engine (key invariant #10)
tools/scripts/check_engine_fresh.sh

# 1. C++ engine: fast inner loop for engine/ work
./build/release/engine/ts_tests
./build/release/engine/ts_benchmark
./build/release/engine/ts_fuzz --games 10000
./build/release/engine/ts_fuzz --steps 5000000 --seed 42

# 2. Backend Python — everything except the web UI. The default suite for backend work.
#    ~7 min, dominated by tests/replayer.
PYTHONPATH=. .venv/bin/python -m pytest -q tests/bindings tests/engine_logic tests/replayer tests/training

#    Narrower loops while iterating:
PYTHONPATH=. .venv/bin/python -m pytest -q tests/bindings tests/engine_logic   # 331 tests, ~10s
PYTHONPATH=. .venv/bin/python -m pytest -q tests/replayer                      # 422 tests, ~6min
PYTHONPATH=. .venv/bin/python -m pytest -q tests/training                      # 187 tests, ~34s
PYTHONPATH=. .venv/bin/python -m pytest -q tests/engine_logic/test_all_110_cards.py::TestName::test_case

# 3. Web/UI — ONLY for changes under web/. Two prerequisites, neither committed:
cd web/ui && npm install && npm run build && cd ../..   # produces web/ui/dist
.venv/bin/python -m playwright install chromium         # for the E2E tests
PYTHONPATH=. .venv/bin/python -m pytest -q tests/web

# 4. Static typing — MUST return 0 errors after any Python change. Pass paths explicitly.
.venv/bin/pyrefly check ai tools tests web bindings
```

> **Worktrees and the venv.** `.venv/` lives in the main checkout, not in a git worktree under
> `.claude/worktrees/`, so use the main checkout's path from there (e.g. `/workspace/.venv/bin/python`).
> Invoke pytest as `python -m pytest`, not via the `.venv/bin/pytest` console script: that script
> carries an absolute shebang from wherever the venv was first created, which breaks if the venv or
> the repository is ever moved or copied.

What each group covers: `bindings/` (nanobind surface), `engine_logic/` (game rules driven through
the bindings — prefer adding new rule coverage to `engine/tests/*.cpp` instead), `replayer/` (the
`ts_replayer` log-conversion pipeline), `training/` (RL/reward/NashPG stack), `web/` (server,
bot-client, Playwright E2E), `differential/` (cross-engine fuzzing, WIP).

**Run pyrefly with explicit paths, never bare.** A bare `pyrefly check` silently checks **nothing**
when the repository is a git worktree under `.claude/worktrees/`: pyrefly honours
`.git/info/exclude`, which the Claude Code harness populates with `**/.claude/worktrees/`, so every
file is excluded — and it exits 0 having examined zero files, which satisfies "must return 0 errors"
while testing nothing. `No Python files matched patterns ...` on the last line means you have
measured nothing. The bare form works in the main checkout, which is why this is easy to miss.

**Do not run the differential suites.** `tests/differential/` is gated behind the `differential_fuzz`
marker / `--run-fuzz` flag and is ignored at collection (`tests/conftest.py`) because those modules
fail at *import* and would otherwise abort the whole run. It is WIP and not informative in its
current state, so it is not part of the check a change is expected to pass — do not spend time
reviving it. `pyrefly.toml` likewise excludes `tests/differential/**` and `external/**`.

## Running training / tournaments / matches / web play

**Never write ad-hoc scripts to invoke training, tournaments, or match simulation directly — always go through the unified CLIs below** (see Key invariants #9).

```bash
# Phase 0: BC warmup from a demonstration dataset
TRITON_CACHE_DIR=.triton_cache PYTHONPATH=. .venv/bin/python tools/train.py \
  --mode warmup --arch v3 --warmup-dataset <regenerated-dataset.jsonl.gz> \
  --bc-epochs 2 --batch-size 1024 --output-dir data/checkpoints/coldwar_net_v3_warmup.pt

# Phase 1-3: RL self-play + live snapshot evals + post-training tournament
TRITON_CACHE_DIR=.triton_cache PYTHONPATH=. .venv/bin/python tools/train.py \
  --arch v3 --duration-seconds 7200 --snapshot-interval-seconds 1200 \
  --warmup-checkpoint data/checkpoints/coldwar_net_v3_warmup.pt \
  --reward-scheme blunder_aware \
  --eval-opponents heuristic random <checkpoint.pt> --eval-games-per-side 50 \
  --post-tournament --post-tournament-models heuristic random <checkpoint.pt> \
  --post-tournament-games 500
# or: ./tools/scripts/train_and_tournament.sh v3 7200 1200 data/checkpoints/coldwar_net_v3_warmup.pt

# Standalone tournament / Elo evaluation over a checkpoint directory
PYTHONPATH=. .venv/bin/python tools/tournament.py \
  --checkpoint-dir data/checkpoints/run_v3_<ts> --games-per-side 500 \
  --anchor-model HeuristicBot --anchor-elo 1500.0 \
  --output-report data/checkpoints/run_v3_<ts>/massive_tournament_report.md

# Single match / replay generation (also supports --us human for interactive CLI play)
PYTHONPATH=. .venv/bin/python tools/play_match.py --us heuristic --ussr strategic

# Web workbench: server + bot opponent + browser
PYTHONPATH=. .venv/bin/python -m uvicorn web.server.main:app --host 0.0.0.0 --port 8000
PYTHONPATH=. .venv/bin/python -m web.bot_client --game-id game-1 --role USSR --type neural --model-path <checkpoint.pt>
# then open http://localhost:8000/?game_id=game-1&role=US
```

Other `tools/` CLIs: `generate_dataset.py` (vectorized demonstration dataset generation), `inspect_checkpoints.py` (scans `data/checkpoints/`, detects V1/V2/V3 architecture), `download_ts_replayer.py` (one-time cached fetch of the human game corpus into `data/datasets/ts_replayer/`). Full flag reference in `tools/README.md`.

**Human replay conversion (`tools/lib/ts_replayer_*.py`):** the corpus of human games is converted to engine decisions by rebuilding each log entry's position, driving it through the engine, and checking the result against the log's own next board; the hands, which the log never states in full, are solved for a whole game at once as a z3 constraint problem. Nothing is guessed — a decision the log does not determine is a bug to diagnose, not a gap to fill (see `AGENTS.md` §4 invariant 11, and `tools/README.md` §6).

**Test data is never committed.** `data/replays/` and `data/checkpoints/` are git-ignored, so no
test may assume they hold anything — and never guard such a test with a skip, which passes
everywhere while checking nothing. Use the `generated_replay_dir` fixture (`tests/conftest.py`),
which generates a replay into a temp dir and points the server at it via `TS_REPLAYS_DIR`. Schema
drift is caught separately by `tests/bindings/replay_schema.golden.json`; regenerate it with
`python tests/bindings/test_replay_schema_golden.py` when the schema changes on purpose.

## Architecture

**Engine (`engine/`, C++20):** `ts::GameState` is trivially copyable and capped at 4 KB — no heap allocation in the simulation core. Turns/events are decomposed into a stream of 4-byte `MicroAction` structs (`DecisionType`, target/card/country id, sub-choice, flags) processed by a state machine (`state_machine.cpp`) uniformly across `Phase::HEADLINE` and `Phase::ACTION_ROUND`. Card event handlers live in `src/events/{early,mid,late}_war.cpp` split by era. `ActionMask::generate_flat_mask_212` / `decode_flat_action_212` define the canonical **212-dim flat action space**. All simulation randomness goes through the deterministic SplitMix64 PRNG in `state.rng_state` (`ts::Prng`) — never `std::rand` or similar, since replays must be bit-for-bit reproducible from a seed.

**Bindings (`bindings/`):** nanobind module (`ts_bindings.cpp`) exports `ts_engine` and a vectorized batch runner. `action_encoder.py` is the bidirectional Python codec for the 212-dim flat action space; `ts_env.py` wraps single and vectorized (batched, C++-driven) environments for RL training.

**AI (`ai/`):** `models/coldwar_net.py` (ColdWarNet: GNN GraphConv + card + global ResNet with masked action heads, V1/V2/V3 variants auto-detected from checkpoints). `training/nash_pg.py` implements NashPG — PPO-style loss with KL regularization against a frozen reference policy snapshot (`π_ref`), refreshed periodically, to converge toward Nash equilibrium without cycling. `training/rollout_buffer.py` does trajectory storage + GAE (zero-sum, alternating between players). `rewards/reward_calculator.py` holds the reward strategies (`BlunderAwareRewardCalculator`, `ZeroSumTerminalReward`, `ShapedZeroSumReward`, `UsefulActionsReward`) — perspective-aligned and zero-sum across US/USSR.

**Bots (`bot/`):** all inherit `BaseBot` (`select_action` for dict/JSON-based servers, `select_flat_action` for the fast 212-dim vectorized path, `reset`). Baselines: `random_bot`, `heuristic_bot`, `exploratory_bot`, `strategic_bot` (DEFCON-2 containment focus), `event_heavy_bot`, `human_bot` (interactive CLI); `neural_bot` loads ColdWarNet checkpoints.

**Web (`web/`):** `web/server/main.py` is the FastAPI app — REST endpoints for game/replay/metadata, plus `/ws/game/{game_id}?role=US|USSR|OBSERVER`. `session.py`'s `GameSession` owns the `ts_engine.GameState`, validates actions against the active `DecisionContext`, steps the engine, and broadcasts `STATE_UPDATE`. `replay.py` reads/writes `.tslog.json` under `data/replays/` (symlinked to `replays/`). `web/ui/` is the Vite + TypeScript SVG map frontend. `web/bot_client.py` connects a `BaseBot` to a running game over WebSocket.

**Data layout:** `data/checkpoints/` (model weights, symlinked as `checkpoints/`), `data/replays/` (`.tslog.json` game logs, symlinked as `replays/`), `data/datasets/` (compressed demonstration `.jsonl.gz` for BC warmup). `rules/` holds `rules.json`/`cards.json`/`map.json` and related spec files, the formal spec the engine implements — tracked in git, except `rules/Rules_Final.pdf` (the official rulebook, GMT Games copyright), which stays git-ignored. `external/struggler` is a git submodule used as an independent reference engine for differential testing.

## Key invariants (see `AGENTS.md` §4 for full detail)

1. `ts::GameState` must stay trivially copyable and ≤ 4 KB — no heap allocation in the engine core.
2. Action space is always the 212-dim flat space (`ActionEncoder` / `generate_flat_mask_212`) — don't introduce parallel encodings.
3. NashPG regularizes the active policy against a frozen `π_ref` snapshot with fixed η; don't couple the reference update to the active policy's gradient step.
4. All simulation randomness must go through `state.rng_state` (SplitMix64) for reproducibility.
5. All Python must be fully type-annotated (`TypedDict`s in `web/server/replay_types.py` for serialized JSON); run `.venv/bin/pyrefly check` after any Python change and keep it at 0 errors.
6. All self-play/replay generation must go through `tools.lib.self_play.generate_self_play_replay` to keep `.tslog.json` schema consistent — don't hand-roll replay writers.
7. Checkpoint directories must follow `data/checkpoints/run_[version]_[YYYYMMDD]_[HHMMSS]` — no ad-hoc names.
8. Demonstration dataset loading must use bounded streaming (`WarmupDataset.stream_batches` / `stream_transitions`), never a monolithic in-memory load.
9. Never write ad-hoc scripts for training/tournaments/matches — use `tools/train.py`, `tools/tournament.py`, `tools/play_match.py` respectively.
10. Never measure against a stale engine — run `tools/scripts/check_engine_fresh.sh` before generating or consuming any checkpoint, dataset, or benchmark number, and re-take anything measured before a rebuild.

Each of `engine/AGENTS.md`, `bindings/AGENTS.md`, `bot/AGENTS.md`, `web/server/AGENTS.md`, and root `AGENTS.md` carries a "keep documentation synchronized" rule — when you change behavior in one of those directories, update its `AGENTS.md` (and root `AGENTS.md` if the change is architecturally significant) in the same change.
