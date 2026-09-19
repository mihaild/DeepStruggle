# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

`AGENTS.md` (plus `engine/AGENTS.md`, `bindings/AGENTS.md`, `bot/AGENTS.md`,
`web/server/AGENTS.md`) carries the full architecture and directory listings. This file is the
short version: what you need to be productive immediately.

## What this is

The AI, simulation engine, web workbench, and training infrastructure for a Deluxe Edition
**Twilight Struggle** (110-card) implementation. Three layers:

1. **C++20 engine** (`engine/`) — zero-allocation simulation core, exposed to Python via nanobind.
2. **Python AI/training stack** (`ai/`, `bindings/`, `tools/`) — network, RL training (NashPG), rewards, bots.
3. **Web workbench** (`web/`) — FastAPI game server + Vite/TypeScript SVG map UI for human vs bot/replay play.

## Build & environment setup

```bash
# Python venv — requirements.txt is the source of truth for dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Web UI
cd web/ui && npm install && npm run build && cd ../..

# C++ engine + nanobind extension (root CMake orchestrates engine/ and bindings/)
cmake -B build/release -S . -DPython_EXECUTABLE=$(pwd)/.venv/bin/python3
cmake --build build/release -j
```

Three requirements degrade rather than break if absent: `z3-solver` (the human-log converter then
falls back to per-turn heuristics and no longer reproduces the corpus), `tensorboard` (training
still writes `training_metrics.jsonl`) and `playwright` (browser E2E tests only).

The extension lands at `build/release/ts_engine.cpython-*.so` — the binary root, not
`build/release/bindings/` — and is imported via `PYTHONPATH=.:build/release`.

**The type stubs are generated, not written.** Building regenerates `bindings/ts_engine/` — a stub
*package*, `__init__.pyi` plus one file per nanobind submodule — from the module just built, so it
cannot drift from the bindings. `tests/bindings/test_stub_is_generated.py` fails if the committed
copy no longer matches. Never hand-edit those files; rebuild and commit what changed. To express
something introspection cannot recover, such as the shape of a value returned as an untyped
`nb::dict`, edit `bindings/ts_engine.pyi.pattern`.

**Never run an experiment against a stale engine.** Guard every command that generates or consumes
a checkpoint, dataset, Elo anchor or diagnostic:

```bash
tools/scripts/check_engine_fresh.sh && PYTHONPATH=.:build/release .venv/bin/python tools/train.py ...
```

It exits 0 when the built module matches the `engine/`/`bindings/` sources, and rebuilds and exits
1 when it does not, so the chained command does not run. A rebuilt engine can change the decision
stream with no Python change: datasets in the `(seed, actions)` format then truncate silently, and
old checkpoints keep loading and running forward passes, so nothing announces the problem.

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
tools changes.** It tells you nothing about them and fails on missing build artifacts
(`web/ui/dist`, a Playwright browser) that are deliberately not in the repo, so a red web suite on
a backend change is pure noise.

| you changed | run |
|:---|:---|
| `engine/`, `bindings/` | C++ suite, then `tests/bindings tests/engine_logic` (~30s) |
| `tools/lib/ts_replayer_*` | the above + `tests/replayer` |
| `ai/`, `bot/`, `tools/` | the above + `tests/training` |
| `web/**` | **also** `tests/web`, after building the UI (below) |

### The ts-replayer corpus

`tests/replayer` (and a few tests in `tests/engine_logic`) run against the **real** human corpus,
not synthetic data — that is the point of them, so it cannot be generated. It is ~5 MB,
git-ignored, and downloaded once per **machine** into `~/.cache/ts_ai/ts_replayer`
(`$XDG_CACHE_HOME` honoured), **not** under `data/`, so every checkout and worktree shares one
copy instead of re-fetching 300 throttled requests. `tools/lib/corpus_paths.py` resolves the
location. **A missing corpus fails these tests; it does not skip them** — a `skipif` on missing
input passes on every machine while checking nothing.

```bash
PYTHONPATH=. .venv/bin/python tools/download_ts_replayer.py
```

`pytest-xdist` is installed; `-n auto` takes the backend suite from minutes to about a minute. The
single whole-corpus test, `test_every_game_in_the_corpus_converts`, is marked `corpus_full` and
**deselected by default** because it dominates `tests/replayer` on its own. Run it before merging
any change to `tools/lib/ts_replayer_*`:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q -m corpus_full tests/replayer
```

### The commands

```bash
# 0. Always first — never measure against a stale engine
tools/scripts/check_engine_fresh.sh

# 1. C++ engine: fast inner loop for engine/ work
./build/release/engine/ts_tests
./build/release/engine/ts_benchmark
./build/release/engine/ts_fuzz --games 10000
./build/release/engine/ts_fuzz --steps 5000000 --seed 42

# 2. Backend Python — everything except the web UI. The default suite for backend work.
PYTHONPATH=. .venv/bin/python -m pytest -q -n auto tests/bindings tests/engine_logic tests/replayer tests/training

#    Narrower loops while iterating:
PYTHONPATH=. .venv/bin/python -m pytest -q tests/bindings tests/engine_logic
PYTHONPATH=. .venv/bin/python -m pytest -q tests/replayer
PYTHONPATH=. .venv/bin/python -m pytest -q tests/training
PYTHONPATH=. .venv/bin/python -m pytest -q tests/engine_logic/test_all_110_cards.py::TestName::test_case

# 3. Web/UI — ONLY for changes under web/. Two prerequisites, neither committed:
cd web/ui && npm install && npm run build && cd ../..   # produces web/ui/dist
.venv/bin/python -m playwright install chromium         # for the E2E tests
PYTHONPATH=. .venv/bin/python -m pytest -q tests/web

# 4. Static typing — MUST return 0 errors after any Python change. Pass paths explicitly.
.venv/bin/pyrefly check ai tools tests web bindings
```

> Invoke pytest as `python -m pytest`, not via the `.venv/bin/pytest` console script: that script
> carries an absolute shebang from wherever the venv was first created, which breaks if the venv or
> the repository is ever moved or copied. For the same reason, use the path of whichever checkout
> actually holds `.venv/` — it is not necessarily the directory you are working in.

What each group covers: `bindings/` (nanobind surface), `engine_logic/` (game rules driven through
the bindings — prefer adding new rule coverage to `engine/tests/*.cpp`), `replayer/` (the
`ts_replayer` log-conversion pipeline), `training/` (RL/reward/NashPG stack), `web/` (server,
bot-client, Playwright E2E), `differential/` (cross-engine fuzzing, WIP).

**Run pyrefly with explicit paths, never bare.** pyrefly honours `.git/info/exclude`, so in a
checkout whose exclude file covers the working directory a bare `pyrefly check` matches **zero**
files and exits 0 — which satisfies "must return 0 errors" while checking nothing. `No Python
files matched patterns ...` on the last line means you have measured nothing.

**Do not run the differential suites.** `tests/differential/` is gated behind the
`differential_fuzz` marker / `--run-fuzz` flag, and `pytest_ignore_collect` in `tests/conftest.py`
keeps it out of collection entirely — those modules fail at *import*, and a skip applied after
collection would be too late. It is WIP and not part of the check a change is expected to pass
(tracked as TEST-1 in `BUGS.md`). `pyrefly.toml` likewise excludes it and `external/**`.

**Two demonstration datasets, built not committed.** Both land under the git-ignored `data/` tree
and both must be rebuilt after an engine change:

```bash
# self-play, from a chosen checkpoint
PYTHONPATH=.:build/release .venv/bin/python tools/generate_dataset.py \
  --models <checkpoint.pt> --total-games 5000 --output-path <dataset.jsonl.gz>

# human games from the ts-replayer corpus
PYTHONPATH=.:build/release .venv/bin/python tools/build_human_dataset.py
```

The self-play set is a *syntax* prior — it can only distil the policy that produced it, and it
carries that policy's biases. The human set is the only *strategy* prior available, and its value
targets are masked on the part of the corpus whose recording stops before the end of the game.

## Running training / tournaments / matches / web play

**Never write ad-hoc scripts to invoke training, tournaments, or match simulation directly — always
go through the unified CLIs below** (invariant "no ad-hoc scripts").

**A warm start is optional and is NOT what the ladder does.** Every late E3 arm was a cold start;
E4-01 and E4-02 were warm-started only because the numbered "Phase 0 / Phase 1-3" framing below
reads as a mandatory pipeline. It is not one. Omit `--warmup-checkpoint` unless the arm is
specifically about warm starting, and note it in the run description when you use it, because it
is a difference from the lineage.

```bash
# Phase 0 (OPTIONAL): BC warmup from a demonstration dataset
TRITON_CACHE_DIR=.triton_cache PYTHONPATH=. .venv/bin/python tools/train.py \
  --mode warmup --arch v2 --warmup-dataset <dataset.jsonl.gz> \
  --bc-epochs 2 --batch-size 1024 --output-dir <warmup.pt>

# Phase 1-3: RL self-play + live snapshot evals + post-training tournament
TRITON_CACHE_DIR=.triton_cache PYTHONPATH=. .venv/bin/python tools/train.py \
  --arch v2 --train-steps 160000000 --snapshot-every-steps 5000000 \
  --reward-scheme blunder_aware \
  --opponent-frac 0.3 --opponent-self-pool --opponent-pool-size 12 \
  --identity-dim 16 --per-entity-heads 64 --graph-layers 0 --self-transform \
  --run-name E<engine>-<attempt>-<seed> \
  --eval-opponents heuristic random <checkpoint.pt> --eval-games-per-side 50 \
  --post-tournament --post-tournament-models heuristic random <checkpoint.pt> \
  --post-tournament-games 500
#   add --warmup-checkpoint <warmup.pt> ONLY for a deliberately warm-started arm
# or: ./tools/scripts/train_and_tournament.sh v2 160000000 5000000 <warmup.pt>

# Standalone tournament / Elo evaluation over a checkpoint directory
PYTHONPATH=. .venv/bin/python tools/tournament.py \
  --checkpoint-dir <run-dir> --games-per-side 500 \
  --anchor-model HeuristicBot --anchor-elo 1500.0 \
  --output-report <run-dir>/tournament_report.md

# Single match / replay generation (also supports --us human for interactive CLI play)
PYTHONPATH=. .venv/bin/python tools/play_match.py --us heuristic --ussr strategic

# Web workbench: server + bot opponent + browser
PYTHONPATH=. .venv/bin/python -m uvicorn web.server.main:app --host 0.0.0.0 --port 8000
PYTHONPATH=. .venv/bin/python -m web.bot_client --game-id game-1 --role USSR --type neural --model-path <checkpoint.pt>
# then open http://localhost:8000/?game_id=game-1&role=US
```

Other `tools/` CLIs: `generate_dataset.py` and `build_human_dataset.py` (demonstration datasets),
`inspect_checkpoints.py` (architecture detected from the weights), `download_ts_replayer.py`
(cached fetch of the human corpus), `behavioral_test.py` (scripted probes). Full flag reference in
`tools/README.md`.

**Human replay conversion (`tools/lib/ts_replayer_*.py`):** the corpus of human games is converted
to engine decisions by rebuilding each log entry's position, driving it through the engine, and
checking the result against the log's own next board; the hands, which the log never states in
full, are solved for a whole game at once as a z3 constraint problem. Nothing is guessed — a
decision the log does not determine is a bug to diagnose, not a gap to fill (see `AGENTS.md` §4
and `tools/README.md` §6).

**Test data is never committed.** The replay and checkpoint directories under `data/` are
git-ignored, so no test may assume they hold anything — and never guard such a test with a skip,
which passes everywhere while checking nothing. Use the `generated_replay_dir` fixture
(`tests/conftest.py`), which generates a replay into a temp dir and points the server at it via
`TS_REPLAYS_DIR`. Schema drift is caught separately by
`tests/bindings/replay_schema.golden.json`; regenerate it with
`python tests/bindings/test_replay_schema_golden.py` when the schema changes on purpose.

## Architecture

**Engine (`engine/`, C++20):** `ts::GameState` is trivially copyable and capped at 4 KB — no heap
allocation in the simulation core. Turns/events are decomposed into a stream of 4-byte
`MicroAction` structs (`DecisionType`, target/card/country id, sub-choice, flags) processed by a
state machine (`state_machine.cpp`) uniformly across `Phase::HEADLINE` and `Phase::ACTION_ROUND`.
Card event handlers live in `src/events/{early,mid,late}_war.cpp` split by era.
`ActionMask::generate_flat_mask_212` / `decode_flat_action_212` define the canonical **212-dim flat
action space**. All simulation randomness goes through the deterministic SplitMix64 PRNG in
`state.rng_state` (`ts::Prng`) — never `std::rand` or similar, since replays must be bit-for-bit
reproducible from a seed.

**Bindings (`bindings/`):** the nanobind module (`ts_bindings.cpp`) exports `ts_engine` and a
vectorized batch runner. `action_encoder.py` is the bidirectional Python codec for the 212-dim flat
action space; `ts_env.py` wraps single and vectorized (batched, C++-driven) environments for RL.

**Never change the observation without asking.** The layout is a representation decision and it is
the owner's, not a detail to be improved in passing. This covers adding a feature, removing one,
changing what a slot means, and changing the width. Two reasons it is worse than an ordinary
refactor:

* A network reads fixed slices, so a changed observation does not fail — a checkpoint keeps loading
  and simply misreads. The *width* half of this is now caught: every model checks its input width
  in `extract_features`, and `bindings.ts_env.check_obs_width` checks a model against the engine
  before a probe runs. A change of *content* at the same width is invisible to everything.
* It invalidates every checkpoint and every `(seed, actions)` dataset, and resets the Elo ladder,
  so the cost is paid by every measurement that came before.

Propose the change, say what it costs in floats, and wait. Two standing preferences from the owner:
**do not spend a large vector on a rare mechanism** — a per-country or per-card bit for one card is
not worth 84 or 110 floats — and **a partial version of a feature is worse than none**.

**There is one observation layout: v2.3, 3,824 floats** — 84x26 board, 110x14 card, 100 global.
The earlier layouts are gone, along with every way of asking for one:
`ts.extract_observation(state, perspective)` and `ts.VectorizedBatchRunner(n, seed)` take no
`layout` argument, `TsVectorizedEnv` takes no `layout` or `obs_flags`, and `tools/train.py` has no
`--obs-layout` or `--engine-flag`. `ts.OBS_SIZE` is the width. That is not tidying: a defaulted
`layout` parameter was the mechanism behind five separate instances of one bug, because handing a
model the wrong layout returns a number instead of raising. With one layout there is no argument to
get wrong. Checkpoints from the retired layouts cannot be loaded and are not being converted — they
predate the starred-card fix, so they were trained against a different game; `check_obs_width` and
`check_checkpoint_layout` refuse them by width rather than letting them misread.
`obs_flags::STAGED_CARDS` survives in `engine/include/ts/game_state.hpp` (exported as
`OBS_FLAG_STAGED_CARDS`) as a reserved bit, so the bit is never reused with a different meaning;
no code path varies on it.

**AI (`ai/`):** `models/coldwar_net_v2.py` is the baseline (ColdWarNetV2: GNN GraphConv +
card/country cross-attention + global ResNet with masked action heads); `models/coldwar_net.py`
holds V1, kept because checkpoints predating V2 name it, and the architecture is detected from a
checkpoint's own weights. `training/nash_pg.py` implements NashPG — PPO-style loss with KL
regularization against a frozen reference policy snapshot (`π_ref`), refreshed periodically, to
converge toward Nash equilibrium without cycling. `training/rollout_buffer.py` does trajectory
storage + GAE (zero-sum, alternating between players); `training/train.py` is the CLI entry point
behind `tools/train.py`. `rewards/reward_calculator.py` holds the reward strategies
(`BlunderAwareRewardCalculator`, `ZeroSumTerminalReward`, `ShapedZeroSumReward`,
`UsefulActionsReward`) — perspective-aligned and zero-sum across US/USSR. `ai/eval/` is the probe
suite (calibration, ablations, behavioral and human-corpus comparisons) and `ai/search/pimcts.py`
is the perfect-information MCTS used for evaluation.

**Bots (`bot/`):** all inherit `BaseBot` (`select_action` for dict/JSON-based servers,
`select_flat_action` for the fast 212-dim vectorized path, `reset`). Baselines: `random_bot`,
`heuristic_bot`, `exploratory_bot`, `strategic_bot` (DEFCON-2 containment focus),
`event_heavy_bot`, `human_bot` (interactive CLI); `neural_bot` loads ColdWarNet checkpoints.

**Web (`web/`):** `web/server/main.py` is the FastAPI app — REST endpoints for game/replay/metadata,
plus `/ws/game/{game_id}?role=US|USSR|OBSERVER`. `session.py`'s `GameSession` owns the
`ts_engine.GameState`, validates actions against the active `DecisionContext`, steps the engine, and
broadcasts `STATE_UPDATE`. `replay.py` reads and writes `.tslog.json` under the replay directory
(`TS_REPLAYS_DIR` overrides it). `web/ui/` is the Vite + TypeScript SVG map frontend.
`web/bot_client.py` connects a `BaseBot` to a running game over WebSocket.

**Data layout:** `data/` is git-ignored and holds working artifacts — model checkpoints,
`.tslog.json` game logs, and compressed demonstration `.jsonl.gz` datasets. `rules/` holds
`rules.json`/`cards.json`/`map.json` and related spec files, the formal spec the engine implements
— tracked in git, except `rules/Rules_Final.pdf` (the official rulebook, GMT Games copyright),
which stays git-ignored. `external/struggler` is a git submodule used as an independent reference
engine for differential testing.

## Key invariants (see `AGENTS.md` §4 for full detail)

1. `ts::GameState` must stay trivially copyable and ≤ 4 KB — no heap allocation in the engine core.
2. Action space is always the 212-dim flat space (`ActionEncoder` / `generate_flat_mask_212`) — don't introduce parallel encodings.
3. NashPG regularizes the active policy against a frozen `π_ref` snapshot with fixed η; don't couple the reference update to the active policy's gradient step.
4. All simulation randomness must go through `state.rng_state` (SplitMix64) for reproducibility.
5. All Python must be fully type-annotated (`TypedDict`s in `web/server/replay_types.py` for serialized JSON); run `pyrefly check` with explicit paths after any Python change and keep it at 0 errors.
6. All self-play/replay generation must go through `tools.lib.self_play.generate_self_play_replay` to keep `.tslog.json` schema consistent — don't hand-roll replay writers.
7. A checkpoint directory is named from the run's short name plus its start timestamp, via `tools/train.py --run-name` — no ad-hoc names, which say nothing about which engine or seed produced the run.
8. Demonstration dataset loading must use bounded streaming (`WarmupDataset.stream_batches` / `stream_transitions`), never a monolithic in-memory load.
9. Never write ad-hoc scripts for training/tournaments/matches — use `tools/train.py`, `tools/tournament.py`, `tools/play_match.py` respectively.
10. Record the commit and a description in the run's `metadata.json` before launching a training run (`tools/train.py --description` does this).
11. Never change the C++ engine without asking the owner first.
12. Human replay conversion is never approximated — a decision the log does not determine is diagnosed, not filled in.
13. Never measure against a stale engine — run `tools/scripts/check_engine_fresh.sh` before generating or consuming any checkpoint, dataset, or benchmark number, and re-take anything measured before a rebuild.
14. **An RL run needs an opponent pool.** `--opponent-frac 0.3 --opponent-self-pool
    --opponent-pool-size 12`, as every run since E3-30 has used. Without it the policy trains only
    against its own current self, one seat runs away, and the critic loses all skill — twice now,
    from two different causes (`research/archive/E3_ladder/log/P15_X4b_collapse_is_pool_starvation.md`,
    `research/log/E4_pool_starvation_recurrence.md`). Before launching, diff the intended flags
    against a recent healthy run's `metadata.json` rather than trusting the template above; the
    startup banner must say `[opponent pool] ... frac=0.3`.
15. **Diff the WHOLE `metadata.json`, not the flags you are thinking about.** Invariant 14 was
    written for the opponent pool and then applied only to the opponent pool: E4 was launched with
    the bare architecture defaults while all eleven late E3 arms carried `identity_dim 16`,
    `per_entity_heads 64`, `graph_layers 0`, `self_transform` -- so no E3/E4 comparison is clean
    (`research/findings/training/e4_architecture_discontinuity.md`). The flags you are not
    thinking about are the ones that drift.

Each of `engine/AGENTS.md`, `bindings/AGENTS.md`, `bot/AGENTS.md`, `web/server/AGENTS.md`, and root
`AGENTS.md` carries a "keep documentation synchronized" rule — when you change behavior in one of
those directories, update its `AGENTS.md` (and root `AGENTS.md` if the change is architecturally
significant) in the same change.
