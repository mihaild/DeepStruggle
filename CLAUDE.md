# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

`AGENTS.md` (plus `engine/AGENTS.md`, `bindings/AGENTS.md`, `bot/AGENTS.md`,
`web/server/AGENTS.md`, `web/ui/AGENTS.md`) carries the full architecture and directory listings.
This file is the short version: what you need to be productive immediately, and every rule.

## What this is

The AI, simulation engine, web workbench, and training infrastructure for a Deluxe Edition
**Twilight Struggle** (110-card) implementation. Three layers:

1. **C++20 engine** (`engine/`) — zero-allocation simulation core, exposed to Python via nanobind.
2. **Python AI/training stack** (`ai/`, `bindings/`, `tools/`) — networks, RL training (NashPG), search, rewards, bots.
3. **Web workbench** (`web/`) — a browser page (Vite/TypeScript/SVG) that runs the engine as
   WebAssembly and models as ONNX; `web/server/` is only a local files server.

## Build & environment setup

```bash
# Python venv — requirements.txt is the source of truth for dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# C++ engine + nanobind extension (root CMake orchestrates engine/ and bindings/)
# Needs clang and LLVM's LTO tools: apt-get install clang llvm lld
#   (no root: tools/scripts/install_clang_userspace.sh)
cmake -B build/release -S . -DPython_EXECUTABLE=$(pwd)/.venv/bin/python3
cmake --build build/release -j

# Web page: the WebAssembly engine, then npm ci + build into web/ui/dist
# (needs Emscripten: tools/scripts/install_emsdk.sh)
tools/scripts/build_web.sh
```

**The engine is built with clang, and CMake refuses any other compiler** — about +20% over GCC on
the same sources, with bit-identical games. `tools/scripts/check_engine_fresh.sh` reconfigures a
build directory created under GCC. The batch runner's parallel loops call GCC's `libgomp` directly
(not `#pragma omp`) so the process keeps the one OpenMP pool it shares with torch; clang's own
OpenMP adds a second pool, measured 5-7% slower. `tests/bindings/test_build_toolchain.py` fails on
a non-clang engine, a second pool, or loops that silently run serially (`bindings/AGENTS.md` §3).

`z3-solver` is required for human-log conversion: unavailable or unsuccessful hand solving is a
conversion failure before any decisions are emitted. Two requirements degrade if absent:
`tensorboard` (training still writes `training_metrics.jsonl`) and `playwright` (browser E2E tests only).

The extension lands at `build/release/ts_engine.cpython-*.so` — the binary root, not
`build/release/bindings/` — and is imported via `PYTHONPATH=.:build/release`.

**The type stubs are generated, not written.** Building regenerates `bindings/ts_engine/` (a stub
package) from the module just built; `tests/bindings/test_stub_is_generated.py` fails if the
committed copy no longer matches. Never hand-edit it: rebuild and commit what changed. What
introspection cannot recover (e.g. the shape of an untyped `nb::dict`) goes in
`bindings/ts_engine.pyi.pattern`.

**Never run an experiment against a stale engine.** Guard every command that generates or consumes
a checkpoint, dataset, Elo anchor or diagnostic:

```bash
tools/scripts/check_engine_fresh.sh && PYTHONPATH=.:build/release .venv/bin/python tools/train.py ...
```

It exits 0 when the built module matches the `engine/`/`bindings/` sources, and rebuilds and exits
1 when it does not, so the chained command does not run. A rebuilt engine can change the decision
stream with no Python change: `(seed, actions)` datasets then truncate silently, and old checkpoints
keep loading and running, so nothing announces the problem.

Sanitizer build (AddressSanitizer + UBSan), for engine work:
```bash
cmake -B build_san -S . -DCMAKE_BUILD_TYPE=Debug -DPython_EXECUTABLE=$(pwd)/.venv/bin/python3 \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined -shared-libasan"
cmake --build build_san -j
ASAN_OPTIONS=detect_leaks=0 ./build_san/engine/ts_tests
tools/scripts/run_asan.sh .venv/bin/python -m pytest -q tests/bindings   # preloads clang's ASan runtime
```
`-shared-libasan` is for the extension only (Python is not instrumented, so `run_asan.sh` preloads
compiler-rt's shared runtime); the executables keep the static one.

## Tests & type checking

**Match the suite to what you touched.** In particular, **do not run the whole of `tests/web` for
AI or tools changes**: it tells you nothing about them and fails on build artifacts that are
deliberately not in the repo (`web/ui/dist`, a Playwright browser), so a red web suite on a backend
change is noise. The exception is an engine or bindings change: the page runs the engine as
WebAssembly, and `tests/web/test_wasm_engine.py` proves that build still plays the native games.

| you changed | run |
|:---|:---|
| `engine/`, `bindings/` | C++ suite, then `tests/bindings tests/engine_logic` (~30s), then `tools/scripts/build_web.sh --engine` and `tests/web/test_wasm_engine.py` |
| `tools/lib/ts_replayer_*` | the above + `tests/replayer` |
| `ai/`, `bot/`, `tools/` | the above + `tests/training` |
| `web/**` | **also** `tests/web`, after `tools/scripts/build_web.sh` |
| `research/**` | `tests/docs` (fast, no build needed) — every relative link must resolve |
| any `.py` | **also** `tests/lint` (fast, no build needed) — no new import inside a function or class; the per-file baseline only goes down |

### The ts-replayer corpus

`tests/replayer` (and a few tests in `tests/engine_logic`) run against the **real** human corpus,
which cannot be generated. It is ~5 MB, git-ignored, and downloaded once into
`datasets/ts_replayer` in the **shared data tree** (the main checkout's `data/`, found from git even
inside a worktree by `tools/lib/data_root.py`; `$TS_REPLAYER_CORPUS` overrides it). **A missing
corpus fails these tests; it does not skip them** — a `skipif` on missing input passes on every
machine while checking nothing.

```bash
PYTHONPATH=.:build/release .venv/bin/python tools/download_ts_replayer.py
```

The whole-corpus test, `test_every_game_in_the_corpus_converts`, is marked `corpus_full` and
**deselected by default** because it dominates `tests/replayer`. Run it before merging any change
to `tools/lib/ts_replayer_*`:

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

# 2. Backend Python — everything except the web UI (-n auto: pytest-xdist, about a minute)
PYTHONPATH=. .venv/bin/python -m pytest -q -n auto tests/bindings tests/engine_logic tests/replayer tests/training tests/docs tests/lint

#    Narrower loops while iterating:
PYTHONPATH=. .venv/bin/python -m pytest -q tests/bindings tests/engine_logic
PYTHONPATH=. .venv/bin/python -m pytest -q tests/replayer
PYTHONPATH=. .venv/bin/python -m pytest -q tests/training
PYTHONPATH=. .venv/bin/python -m pytest -q tests/lint          # imports at module top (ratchet: tests/lint/nested_imports_baseline.json)
PYTHONPATH=. .venv/bin/python -m pytest -q tests/engine_logic/test_all_110_cards.py::TestName::test_case

# 3. Web/UI — for changes under web/ AND to engine/ or bindings/. Prerequisites:
tools/scripts/build_web.sh                              # wasm engine + web/ui/dist (Emscripten)
.venv/bin/python -m playwright install chromium         # for the E2E tests
PYTHONPATH=. .venv/bin/python -m pytest -q tests/web

# 4. Static typing — MUST return 0 errors after any Python change. Pass paths explicitly.
.venv/bin/pyrefly check ai tools tests web bindings
```

**Invoke pytest as `python -m pytest`**, not via `.venv/bin/pytest`: that script carries an
absolute shebang from wherever the venv was created and breaks if the venv or repository moves.
Likewise use the path of whichever checkout actually holds `.venv/`, which need not be the one you
are working in.

What each group covers: `bindings/` (nanobind surface), `engine_logic/` (game rules driven through
the bindings — prefer adding new rule coverage to `engine/tests/*.cpp`), `replayer/` (the
`ts_replayer` log-conversion pipeline), `training/` (RL/reward/NashPG/search stack), `docs/` (the
research record's internal links resolve), `web/` (local server, wasm parity, Playwright E2E),
`differential/` (cross-engine fuzzing, WIP).

**Run pyrefly with explicit paths, never bare.** pyrefly honours `.git/info/exclude`, so in a
checkout whose exclude file covers the working directory a bare `pyrefly check` matches **zero**
files and exits 0. `No Python files matched patterns ...` on the last line means you measured
nothing.

**Do not run the differential suites.** `tests/differential/` fails at *import*, so
`pytest_ignore_collect` in `tests/conftest.py` keeps it out of collection unless `--run-fuzz` /
`-m differential_fuzz` is given. It is WIP and not part of the check a change must pass (TEST-1 in
`BUGS.md`); `pyrefly.toml` excludes it and `external/**`.

**Test data is never committed.** The replay and checkpoint directories under `data/` are
git-ignored, so no test may assume they hold anything — and never guard such a test with a skip,
which passes everywhere while checking nothing. Use the `generated_replay_dir` fixture
(`tests/conftest.py`), which generates a replay into a temp dir and points the server at it via
`TS_REPLAYS_DIR`. Schema drift is caught by `tests/bindings/replay_schema.golden.json`; regenerate
it with `python tests/bindings/test_replay_schema_golden.py` when the schema changes on purpose.

**Two demonstration datasets, built not committed.** Both land under `data/` and both must be
rebuilt after an engine change:

```bash
# self-play, from a chosen checkpoint
PYTHONPATH=.:build/release .venv/bin/python tools/generate_dataset.py \
  --models <checkpoint.pt> --total-games 5000 --output-path <dataset.jsonl.gz>

# human games from the ts-replayer corpus
PYTHONPATH=.:build/release .venv/bin/python tools/build_human_dataset.py
```

The self-play set is a *syntax* prior (it can only distil the policy that produced it, biases
included); the human set is the only *strategy* prior, and its value targets are masked on games
whose recording stops before the end.

## Running training / tournaments / matches / web play

**Never write ad-hoc scripts to invoke training, tournaments, or match simulation — always go
through the unified CLIs below** (invariant 9).

**Derive a launch command from a run that worked, not from a template.** Copied templates have
dropped lineage-critical flags four times (invariant 15); the flags you forget are the ones you are
not thinking about. So ask the tool, before launching and again after (it reads `metadata.json`, so
the second run confirms what the run actually received):

```bash
tools/scripts/launch_flags.py <a-recent-healthy-run-dir>                  # what it actually used
tools/scripts/launch_flags.py <reference-run> --diff <your-new-run>       # what you changed
```

Defaults are read from `build_parser()`, so a flag added tomorrow is covered. The A (architecture)
and R (recipe) codes in a run name are flag sets listed in `research/architectures_and_recipes.md`.

**A warm start is optional and is NOT what the ladder does.** Every late E3 arm was a cold start;
E4-01/E4-02 were warm-started only because an earlier "Phase 0 / Phase 1-3" framing read as
mandatory (a branch `@<step>M+...` resumes its parent's state, which is not a warm start). Omit
`--warmup-checkpoint` unless the arm is about warm starting, and say so in its description when
you use it.

```bash
# RL self-play + live snapshot evals (+ optional post-training tournament). Shape only: take the
# architecture/recipe flags from launch_flags.py on the parent run, never from here.
TRITON_CACHE_DIR=.triton_cache PYTHONPATH=. .venv/bin/python tools/train.py \
  --arch ladder <A-code flags> <R-code flags, incl. the opponent pool of invariant 14> \
  --run-name E<n>-A<n>-R<n>-S<n> --description "..." \
  --train-steps <N> --snapshot-every-steps 10000000 \
  --eval-opponents heuristic random --eval-games-per-side 50
#   OPTIONAL BC warmup: --mode warmup --warmup-dataset <dataset.jsonl.gz> --bc-epochs 2 --output-dir <warmup.pt>,
#   then --warmup-checkpoint <warmup.pt> ONLY for a deliberately warm-started arm

# Standalone tournament / Elo evaluation over a checkpoint directory
PYTHONPATH=. .venv/bin/python tools/tournament.py \
  --checkpoint-dir <run-dir> --games-per-side 500 \
  --anchor-model HeuristicBot --anchor-elo 1500.0 \
  --output-report <run-dir>/tournament_report.md

# The Elo leaderboard (one scale per engine epoch; records in leaderboard/, tools/README.md §2b):
# register a network and a player, play it against the epoch's main players, show the table
PYTHONPATH=. .venv/bin/python tools/leaderboard.py add-network <name> <file.pt> --description "..."
PYTHONPATH=. .venv/bin/python tools/leaderboard.py add-player '<name>[~gumbel(sims=256,k=8)]'
tools/scripts/check_engine_fresh.sh && PYTHONPATH=.:build/release .venv/bin/python \
  tools/leaderboard_play.py --epoch E8 --players '<player-id>' --games-per-side 1000
PYTHONPATH=. .venv/bin/python tools/leaderboard.py show --main-only   # or --lineage <run-name>
# Publish to Hugging Face (repo mirrors data/checkpoints/; .onnx only for the final snapshot,
# the final SWA, soups and leaderboard networks): tools/README.md §2c
PYTHONPATH=.:build/release .venv/bin/python tools/publish_hf.py run <run-dir> --dry-run

# Single match / replay generation (also --us human for interactive CLI play)
PYTHONPATH=. .venv/bin/python tools/play_match.py --us heuristic --ussr strategic

# Web workbench: rebuild the page after an engine change (it runs its own copy of the engine)
tools/scripts/build_web.sh
PYTHONPATH=.:build/release .venv/bin/python -m web.server.main --port 8000
# then open http://localhost:8000/ -- or publish web/ui/dist (GitHub Pages: .github/workflows/pages.yml)
```

**A leaderboard report link is the checkpoint analysis, never a plan.** The `report` of a network
or a player in `leaderboard/` is the behaviour report `tools/scripts/checkpoint_report.py` generated
for exactly that network (`--checkpoints`) or player (`--players`) -- openings, how games end, what
each side does with every card, and its searched-leak census first when it has one -- in
`research/log/per_checkpoint/`. The plan or research log the checkpoint was made for is not its
report, nor is the leak census written alone (`--games 0`, a searched-leak report, which its
network's behaviour report links); until a report is generated, leave the link null.
`tools/leaderboard.py validate` (run by `tests/training`) refuses anything else.

Other `tools/` CLIs: `generate_dataset.py` and `build_human_dataset.py` (demonstration datasets),
`inspect_checkpoints.py` (architecture detected from the weights), `download_ts_replayer.py`
(cached fetch of the human corpus), `behavioral_test.py` (scripted probes). Full flag reference in
`tools/README.md`.

**Human replay conversion (`tools/lib/ts_replayer_*.py`):** each log entry's position is rebuilt,
driven through the engine and checked against the log's own next board; the hands, which the log
never states in full, are solved for a whole game at once as a z3 constraint problem. Nothing is
guessed — a decision the log does not determine is a bug to diagnose (invariant 12,
`AGENTS.md` §4, `tools/README.md` §6).

## Architecture

**Engine (`engine/`, C++20):** `ts::GameState` is trivially copyable and capped at 4 KB. Turns and
events are a stream of 4-byte `MicroAction`s (`DecisionType`, target/card/country id, sub-choice,
flags) processed by one state machine (`state_machine.cpp`) across `Phase::HEADLINE` and
`Phase::ACTION_ROUND`. Card event handlers live in `src/events/{early,mid,late}_war.cpp`.
`ActionMask::generate_flat_mask_212` / `decode_flat_action_212` define the canonical **flat action
space** (220 wide since P17; the names are historical). All randomness goes through the SplitMix64
PRNG in `state.rng_state` (`ts::Prng`), so replays are bit-for-bit reproducible from a seed.

**Bindings (`bindings/`):** `ts_bindings.cpp` exports `ts_engine`, the vectorized batch runner, the
C++ MCTS tree (`BatchedSearch`) and the rollout root's batch (`RolloutBatch`, shared with the
WebAssembly build). `action_encoder.py` is the Python codec for the flat action space; `ts_env.py`
wraps single and vectorized environments for RL.

**Never change the observation without asking.** The layout is a representation decision and it is
the owner's: adding a feature, removing one, changing a slot's meaning or the width. Why it is worse
than an ordinary refactor:

* A network reads fixed slices, so a changed observation does not fail — a checkpoint keeps loading
  and misreads. Width changes are caught (`extract_features` checks every model's input width,
  `bindings.ts_env.check_obs_width` checks a model against the engine); a *content* change at the
  same width is invisible to everything.
* It invalidates every checkpoint and every `(seed, actions)` dataset and resets the Elo ladder.

Propose the change, say what it costs in floats, and wait. Two standing preferences from the owner:
**do not spend a large vector on a rare mechanism** (a per-country or per-card bit for one card is
not worth 84 or 110 floats) and **a partial version of a feature is worse than none**.

**There is one observation layout: v2.3, 3,824 floats** (84x26 board, 110x14 card, 100 global;
`ts.OBS_SIZE`). Nothing takes a `layout` argument — a defaulted one caused five instances of one
bug, because a wrong layout returns a number instead of raising. Checkpoints from retired layouts
predate the starred-card fix and are refused by width (`check_obs_width`,
`check_checkpoint_layout`). `obs_flags::STAGED_CARDS` (`OBS_FLAG_STAGED_CARDS`) survives only as a
reserved bit, so it is never reused with another meaning.

**Optional feature blocks are appended, never inserted.** An approved addition is an
`obs_features` bit (`engine/include/ts/game_state.hpp`) whose block is written *after* the 3,824
base floats, so existing checkpoints keep running. A model's set is recorded in its weights
(LadderNet's `obs_features` buffer) and in `metadata.json`, and chosen with
`tools/train.py --obs-features`, whose default since **E8 is `headline_reveal`** (owner, 2026-10-20;
the flag with no names gives none, what every earlier run read; `launch_flags.py` compares each run
against its own epoch's default, so earlier codes hold). The engine has no default:
`ts.extract_observation_features(state, perspective, features)` requires the set
(`bindings.ts_env.model_obs_features(model)`),
`ts.obs_size_for(features)` is the width, and the batch runner holds a set per env and side
(`set_obs_features(us, ussr)`), zero-padding rows to the widest so models with different sets share
a batch. Training pools require every member, including fixed and league opponents, to have the
learner's exact feature bits; equal observation widths are insufficient. Blocks so far:
`OPS_BUDGET` (bit 1, 3 floats; P30 C4) and `HEADLINE_REVEAL` (bit 2, no width:
it shows the box-4 holder the opponent's revealed headline in the base's card rows, for models trained with it;
2026-10-20). Adding a block still needs the owner.

**AI (`ai/`):** the current lineage is LadderNet (`models/ladder_net.py`, `--arch ladder`), the
configurable P21 backbone whose required structural flags make up the A codes;
`models/coldwar_net_v2.py` (ColdWarNetV2, and `ColdWarNetMLP` for `--arch mlp`) and
`models/coldwar_net.py` (V1) stay so older checkpoints load — the architecture is detected from a
checkpoint's own weights. `training/nash_pg.py` implements NashPG: a PPO-style loss with KL
regularization against a frozen reference snapshot (`π_ref`), refreshed periodically, to converge
toward Nash equilibrium without cycling. `training/rollout_buffer.py` holds trajectories and
zero-sum alternating GAE; `training/train.py` is the CLI behind `tools/train.py`.
`rewards/reward_calculator.py` holds the perspective-aligned, zero-sum reward schemes.
`ai/search/` holds the searchers (`rollout_root.py`, the rollout root; `gumbel_root.py` and
`batched_mcts.py`; `pimcts.py`, perfect-information MCTS for evaluation) and `ai/eval/` the probe
suite.

**Bots (`bot/`):** all inherit `BaseBot` (`select_action` for dict/JSON callers,
`select_flat_action` for the flat vectorized path, `reset`). Baselines: `random_bot`,
`heuristic_bot`, `heuristic_mcts_bot` (search with a hand-written leaf evaluation; sees both
hands), `exploratory_bot`, `strategic_bot`, `event_heavy_bot`, `human_bot` (interactive CLI);
`neural_bot` loads checkpoints.

**Web (`web/`):** `web/ui/` (Vite + TypeScript + SVG) holds the engine as WebAssembly
(`bindings/wasm/ts_engine_wasm.cpp`), the game session (`src/game/`) and model analysis
(`src/analysis/`: ONNX in onnxruntime-web, from a local checkpoint, a Hugging Face repo or a dropped
file, plus the rollout root). Display state and saves are one C++ implementation
(`bindings/state_json.cpp`) shared with the Python bindings, and `tests/web/test_wasm_engine.py`
holds the two engines to bit-identical games. `web/server/main.py` only lists checkpoints
(exporting each to ONNX on first request, `tools/export_onnx.py`) and replays; `replay.py` reads and
writes `.tslog.json` (`TS_REPLAYS_DIR` overrides the directory).

**Data layout:** `data/` is git-ignored and holds checkpoints, `.tslog.json` game logs and
`.jsonl.gz` datasets. `rules/` holds the formal spec the engine implements
(`rules.json`/`cards.json`/`map.json`, ...), tracked except `rules/Rules_Final.pdf` (GMT Games
copyright). `external/struggler` is a git submodule used as a reference engine for differential
testing.

## Key invariants (`AGENTS.md` §4 has the full detail)

1. `ts::GameState` must stay trivially copyable and ≤ 4 KB — no heap allocation in the engine core.
2. Action space is always the flat space (`ActionEncoder` / `generate_flat_mask_212`; 220 wide since P17, the name is historical) — don't introduce parallel encodings. The E4.1 merged-influence view (P23) is not one: same 220 slots, and its composed actions are *defined* as the E4 steps they name. A checkpoint's view is read from its run directory (`tools/lib/action_view.py`); every harness must build masks in the deciding agent's view.
3. NashPG regularizes the active policy against a frozen `π_ref` snapshot with fixed η; don't couple the reference update to the active policy's gradient step.
4. All simulation randomness must go through `state.rng_state` (SplitMix64) for reproducibility.
5. All Python must be fully type-annotated (`TypedDict`s in `web/server/replay_types.py` for serialized JSON); run `pyrefly check` with explicit paths after any Python change and keep it at 0 errors.
6. All self-play/replay generation must go through `tools.lib.self_play.generate_self_play_replay` to keep `.tslog.json` schema consistent — don't hand-roll replay writers.
7. A checkpoint directory is named from the run's name plus its start timestamp, via `tools/train.py --run-name` — no ad-hoc names. A name is `E<n>-A<n>-R<n>-S<n>` (engine, architecture code, recipe code, seed) plus its branch points, `@<step>M+<changes>` (`research/method/run_nomenclature.md`); the A and R codes are flag sets in `research/architectures_and_recipes.md`, and `tools/scripts/run_codes.py --run <dir>` checks a launched run's name against its recorded flags.
8. Demonstration dataset loading must use bounded streaming (`WarmupDataset.stream_batches` / `stream_transitions`), never a monolithic in-memory load.
9. Never write ad-hoc scripts for training/tournaments/matches — use `tools/train.py`, `tools/tournament.py`, `tools/play_match.py` respectively.
10. Record the commit and a description in the run's `metadata.json` before launching a training run (`tools/train.py --description` does this).
11. Never change the C++ engine without asking the owner first.
12. Human replay conversion is never approximated — a decision the log does not determine is diagnosed, not filled in.
13. Never measure against a stale engine — run `tools/scripts/check_engine_fresh.sh` before generating or consuming any checkpoint, dataset, or benchmark number, and re-take anything measured before a rebuild.
14. **An RL run needs an opponent pool:** `--opponent-frac 0.3 --opponent-self-pool
    --opponent-pool-size 12`, as every run since E3-30 has used. Without it the policy trains only
    against its current self, one seat runs away and the critic loses all skill — twice now, from
    two different causes (`research/archive/E3_ladder/log/P15_X4b_collapse_is_pool_starvation.md`,
    `research/log/E4_pool_starvation_recurrence.md`). The startup banner must say
    `[opponent pool] ... frac=0.3`.
15. **Derive a launch command from a healthy run, never from a template.** Run
    `tools/scripts/launch_flags.py <reference-run> --diff <your-run>` before launching and again
    after, and be able to justify every line it prints. Four lineage-critical flags drifted
    otherwise -- the opponent pool, the architecture flags, a warm start the ladder does not use,
    and `ref_update_freq` (200,000 against E3's 5,000,000) -- each found only after the compute was
    spent (`research/findings/training/e4_architecture_discontinuity.md`). A rule you must remember
    to apply to each new field is not a rule, which is why this one is a command.

Each of `engine/AGENTS.md`, `bindings/AGENTS.md`, `bot/AGENTS.md`, `web/server/AGENTS.md`,
`web/ui/AGENTS.md` and root `AGENTS.md` carries a "keep documentation synchronized" rule — when you
change behavior in one of those directories, update its `AGENTS.md` (and root `AGENTS.md` if the
change is architecturally significant) in the same change.
