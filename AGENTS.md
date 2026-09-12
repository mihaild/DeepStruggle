# Twilight Struggle AI: Project Guide & Agent Instructions

This repository contains the complete AI, simulation engine, web workbench, and training infrastructure for the Deluxe Edition of **Twilight Struggle** (110 Cards).

---

## 1. Project Architecture & Components

```mermaid
graph TD
    subgraph WebWorkbench ["Web Workbench (web/)"]
        UI["Vite + TypeScript + SVG Deluxe Map (web/ui/)"]
        HUD["Decision HUD & Action Dispatcher"]
        Replayer["Replay & Timeline Player"]
        Server["FastAPI REST & WebSocket Game Server (web/server/)"]
        WS["WebSocket Game Session Manager"]
        Logger["Replay Recorder (.tslog.json)"]
        UI <-->|WebSocket: JSON State / Actions| WS
        WS <--> Server
        Server <--> Logger
    end

    subgraph NeuralAI ["Neural Network & RL (ai/)"]
        ColdWarNet["ColdWarNet (ai/models/)"]
        NashPG["NashPG Trainer (ai/training/)"]
        BC["Behavioral Cloning (ai/training/)"]
        Rewards["BlunderAware / Shaped Rewards (ai/rewards/)"]
    end

    subgraph Bot ["Bot Clients & Heuristic Agents (bot/)"]
        BaseBot["Generic BaseBot Class (bot/base_bot.py)"]
        NeuralBotClient["NeuralBot (PyTorch V1/V2)"]
        Heuristic["HeuristicBot Baseline"]
        Random["RandomBot Baseline"]
        Exploratory["ExploratoryBot Baseline"]
        Strategic["StrategicBot Realist Agent"]
        EventHeavy["EventHeavyBot Agent"]
        Human["HumanBot Interactive CLI"]
        BaseBot --> Heuristic
        BaseBot --> Random
        BaseBot --> NeuralBotClient
        BaseBot --> Exploratory
        BaseBot --> Strategic
        BaseBot --> EventHeavy
        BaseBot --> Human
    end

    subgraph Tools ["Generic CLI Tools & Shared Library (tools/)"]
        TrainRunner["Unified Training CLI (tools/train.py)"]
        TourneyRunner["Unified Tournament & Evaluator (tools/tournament.py)"]
        MatchRunner["Unified Match & Replay Player (tools/play_match.py)"]
        DatasetGen["Demonstration Generator (tools/generate_dataset.py)"]
        Inspector["Checkpoint Inspector (tools/inspect_checkpoints.py)"]
        SharedLib["Shared Simulation & Analytics Engine (tools/lib/)"]
        Scripts["Shell Automation (tools/scripts/)"]
    end

    subgraph Bridge ["Native Python Bindings & Envs (bindings/)"]
        Nanobind["nanobind Extension Module (ts_engine)"]
        ActionCodec["212-dim ActionEncoder (bindings/action_encoder.py)"]
        VecEnv["Vectorized C++ Batch Runner Wrapper (bindings/ts_env.py)"]
    end

    subgraph CoreEngine ["C++ Simulation Core (engine/)"]
        Core["ts::Engine (C++20 Zero-Allocation, 2.45M step/s)"]
        State["ts::GameState (4 KB Trivially Copyable)"]
        Mask["ts::ActionMask (212-dim Flat Action Space)"]
        Obs["ts::Observation (3824-dim Float Tensor)"]
        Map["ts::MapData (84 Countries Graph)"]
        Cards["ts::CardData (110 Cards Event Logic)"]
    end

    WebBotRunner["WebSocket bot_client.py (web/)"] <-->|WebSocket| WS
    WS <--> Nanobind
    Nanobind <--> CoreEngine
    VecEnv <--> Nanobind
    VecEnv <--> NashPG
    NashPG <--> ColdWarNet
    Rewards <--> NashPG
    ColdWarNet --> NeuralBotClient
    NeuralBotClient --> WebBotRunner
    SharedLib <--> ColdWarNet
```

---

## 2. Directory Structure

```
.
├── CMakeLists.txt              # Minimal root CMake configuration (orchestrates engine & bindings)
├── .gitignore                  # Git ignore rules for build, venv, node, logs, and rules/
├── .python-version             # Python runtime version pinned for environment
├── pyrefly.toml / pytest.ini   # Static typing and test configuration
│
├── engine/                     # Core C++20 simulation engine (see engine/AGENTS.md)
│   ├── CMakeLists.txt          # Engine library, unit tests, fuzzer, benchmark targets
│   ├── AGENTS.md               # Specific instructions for maintaining the C++ engine
│   ├── include/ts/             # Public C++ headers (GameState, MicroAction, ActionMask, etc.)
│   ├── src/                    # Implementation files (scoring, ops, cards, state machine)
│   └── tests/                  # C++ test suites (ts_tests, ts_fuzz, ts_benchmark)
│
├── bindings/                   # Native Python bridge via nanobind & Python environment wrappers
│   ├── CMakeLists.txt          # nanobind module build configuration
│   ├── AGENTS.md               # Specific instructions for maintaining Python bindings
│   ├── ts_bindings.cpp         # nanobind module exporting ts_engine & VectorizedBatchRunner
│   ├── ts_engine.pyi           # Python type stubs for IDE and static typing
│   ├── action_encoder.py       # 212-dim Flat Action <-> MicroAction bidirectional codec
│   └── ts_env.py               # Single & Vectorized batched C++ simulation wrapper
│
├── ai/                         # Neural Network, Training & Reward Strategy (ai/AGENTS.md)
│   ├── models/                 # Neural network architectures
│   │   ├── coldwar_net.py      # ColdWarNet (GNN GraphConv + Card + Global ResNet + Masked Heads)
│   │   └── coldwar_net_architecture.svg # Architecture diagram
│   ├── rewards/                # Perspective-aligned reward strategies
│   │   └── reward_calculator.py# BlunderAwareRewardCalculator, ZeroSumTerminalReward, ShapedZeroSumReward
│   └── training/               # Training pipelines & algorithms
│       ├── rollout_buffer.py   # Trajectory storage, GAE advantage calculator & value/advantage diagnostics
│       ├── behavioral_cloning.py # Phase 0 supervised pre-training
│       ├── nash_pg.py          # NashPG (Nash Policy Gradient with iterative KL regularization) + FixedEntropyProbe
│       ├── warmup_dataset_loader.py # Memory-bounded demonstration streaming
│       └── train.py            # CLI training entry point
│
├── bot/                        # Bot clients & baseline heuristics (see bot/AGENTS.md)
│   ├── AGENTS.md               # Specific instructions for developing AI bots
│   ├── base_bot.py             # Generic BaseBot abstract class defining bot interface
│   ├── random_bot.py           # RandomBot baseline
│   ├── heuristic_bot.py        # HeuristicBot rule-based baseline
│   ├── neural_bot.py           # NeuralBot client using ColdWarNet checkpoints (V1/V2)
│   ├── exploratory_bot.py      # ExploratoryBot diverse exploration agent
│   ├── strategic_bot.py        # StrategicBot DEFCON-2 containment & commentary agent
│   ├── event_heavy_bot.py      # EventHeavyBot event-prioritizing agent
│   └── human_bot.py            # HumanBot interactive CLI terminal player
│
├── web/                        # Web Workbench (UI + Backend Server + Bot Client)
│   ├── bot_client.py           # WebSocket network bot runner for browser matches
│   ├── ui/                     # Vite + TypeScript + SVG Deluxe Map
│   │   ├── index.html
│   │   ├── package.json / vite.config.ts
│   │   └── src/                # Map view, HUD, tracks, debug panel, replay controls
│   └── server/                 # FastAPI Game Server and Replay Manager
│       ├── AGENTS.md           # Instructions for server maintainers
│       ├── main.py             # FastAPI app, REST routes, WebSocket endpoint /ws/game/{id}
│       ├── session.py          # GameSession class, action router, state broadcasting
│       ├── replay.py           # ReplayLogger and ReplayManager
│       └── replay_types.py     # TypedDict specifications for state, action, and logs
│
├── tools/                      # Reusable agent & developer CLI tools (see tools/README.md)
│   ├── README.md               # Tool descriptions, CLI flags, and usage examples
│   ├── train.py                # Unified RL training & fine-tuning runner
│   ├── tournament.py           # Unified tournament & head-to-head evaluator
│   ├── play_match.py           # Unified match runner & replay generator (.tslog.json)
│   ├── generate_dataset.py     # High-throughput vectorized demonstration generator (.jsonl.gz)
│   ├── inspect_checkpoints.py  # Checkpoint discovery, architecture detection & inspector
│   ├── download_ts_replayer.py # One-time fetch of the human game corpus (cached, throttled)
│   ├── lib/                    # Reusable simulation, evaluation & analytics backend
│   │   ├── ts_replayer_parse.py   # The human log's grammar (entries, moves, rolls, reveals)
│   │   ├── ts_replayer_convert.py # Verified log -> engine decisions, entry by entry
│   │   ├── ts_replayer_hands.py   # Both hands for a whole game, solved as a z3 constraint problem
│   │   ├── player_agent.py     # Unified Agent loader (random, heuristic, neural)
│   │   ├── tournament_evaluator.py # Matchup runner & loss cause classifier
│   │   ├── self_play.py        # Self-play simulation & .tslog.json recorder
│   │   ├── batch_tournament.py # Vectorized batch tournament runner & Bradley-Terry MLE
│   │   ├── scoring_formatter.py # Regional scoring audit calculation helper
│   │   └── checkpoint_utils.py # Checkpoint scanning and architecture detection
│   └── scripts/                # Shell automation scripts
│       ├── train_and_tournament.sh # Unified training & tournament bash runner
│       ├── train_direct_rl.sh  # Direct RL self-play runner
│       └── run_asan.sh         # AddressSanitizer execution script
│
├── data/                       # Datasets, Checkpoints & Recorded Replays
│   ├── checkpoints/            # Model weights (arm_*/, coldwar_net_v2_warmup.pt)
│   ├── replays/                # Saved game logs (*.tslog.json)
│   └── datasets/               # Demonstration datasets; archive/ holds superseded ones
│
├── external/                   # External integrations & differential engines
│   ├── README.md               # Integration guide
│   └── struggler/              # External reference engine (Python)
│
├── tests/                      # Python pytest suite (1360 tests with external/struggler on
│                                # PYTHONPATH per the invocation below; 969 without it, since the
│                                # 2 differential-fuzzing files then fail to import), split by what's under test
│   ├── conftest.py             # --run-fuzz option; skips differential_fuzz tests by default
│   ├── bindings/                # Binding-layer smoke tests only (the nanobind surface itself)
│   │   ├── test_bindings.py       # Validates Python nanobind module
│   │   ├── test_types_and_json_schemas.py # Validates replay JSON schema and pyrefly typing
│   │   └── test_engine_build_is_current.py # Guards against a stale .so shadowing the real build
│   ├── engine_logic/            # Game-rule tests driven through the bindings (110 cards, headline
│   │   │                         # resolution, space race, coups, etc.) -- migration debt: new engine
│   │   │                         # rule coverage belongs in engine/tests/*.cpp, not here (see AGENTS.md
│   │   │                         # "keep documentation synchronized" and CLAUDE.md's own stated preference)
│   │   ├── test_all_110_cards.py  # Comprehensive unit tests for all 110 cards
│   │   └── test_card_fixes.py     # Dedicated verification suite for card rules fixes
│   ├── replayer/                 # tools/lib/ts_replayer_*.py conversion pipeline (61 targeted
│   │   │                         # edge-case files, one human-replay-corpus mismatch each, plus parsing)
│   │   └── test_ts_replayer_parse.py # Grammar/arithmetic checks on the raw log parser
│   ├── training/                 # RL/reward/eval stack: NashPG, credit assignment, GAE, PIMCTS, etc.
│   │   ├── test_credit_assignment.py # Unit tests for reward propagation & Rule 4.3 headlines
│   │   └── test_neural_and_nashpg.py # Unit & integration tests for ColdWarNet, ActionMask, NashPG
│   ├── web/                     # Server, bot-client, and browser E2E tests
│   │   ├── test_server_and_bot.py  # Validates REST APIs, bot-vs-bot WebSocket simulation
│   │   ├── test_web_workbench.py   # FastAPI client and UI metadata endpoint tests
│   │   └── test_e2e_space_race.py  # Playwright E2E browser tests
│   └── differential/             # Cross-engine fuzzing -- WIP/unstable, gated behind the
│       │                         # differential_fuzz marker / --run-fuzz flag, not run by default
│       ├── test_differential_fuzzing.py # cross-engine fuzz: native vs struggler
│       ├── test_unified_differential.py # Parameterized differential runs vs each external engine
│       └── engine_interface.py, native_adapter.py, struggler_adapter.py
│                                # Shared EngineProtocol adapters
│
├── rules/                      # Formal spec the engine implements (tracked, except the PDF)
│   ├── Rules_Final.pdf         # [GIT IGNORED] Official rulebook -- GMT Games copyright, not ours to commit
│   ├── rules.md / rules.json   # Formal mathematical rules specification
│   ├── cards.json / primitives # 110 cards metadata and state machine primitives
│   ├── flags.json              # 47 persistent continuous effect & state bits
│   └── map.json / map.md       # 84-country graph topology & coordinates
│
└── build/                      # Build outputs (.gitignored)
    ├── release/                # Standard release CMake build
    └── asan/                   # AddressSanitizer / UBSan debug build
```

---

## 3. Developer & Agent Workflows

### 3.1 Initial Environment Setup
```bash
# 1. Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install nanobind fastapi "uvicorn[standard]" websockets pytest pytest-xdist numpy pydantic httpx torch torchvision

# 2. Web UI dependencies & production build
cd web/ui && npm install && npm run build && cd ../..
```

### 3.2 Build C++ Engine & Nanobind Extension
```bash
# Standard Release Build
cmake -B build/release -S . -DPython_EXECUTABLE=$(pwd)/.venv/bin/python3
cmake --build build/release -j
```

`cmake --build` produces `build/release/ts_engine.cpython-*.so`, which is what
`PYTHONPATH=.:build/release` imports as `ts_engine`. Note the output path: the module lands in
the **binary root**, not in `build/release/bindings/`, which holds only intermediate objects.

**Before any experiment, check the build is not stale** (invariant 13):

```bash
tools/scripts/check_engine_fresh.sh && PYTHONPATH=.:build/release .venv/bin/python tools/train.py ...
```

The script hashes the `engine/` and `bindings/` sources and compares that against the stamp it
wrote next to the built module. It exits 0 when they match, and rebuilds and exits **1** when
they do not — so a chained command stops rather than running against yesterday's rules. It
hashes content instead of comparing timestamps because checking out a branch rewrites mtimes
without changing anything, which would report staleness on every switch.

### 3.3 Generic Scheme for Training & Tournament Pipelines

```mermaid
graph TD
    subgraph Phase0 ["Phase 0: Supervised BC Warmup"]
        Demonstrations["Demonstration Dataset (regenerate per engine; see data/datasets/archive/README.md)"]
        Streamer["WarmupDataset.stream_batches (B=1024, Reservoir Buffer, RAM < 70MB)"]
        WarmupModel["Warmup Checkpoint: data/checkpoints/coldwar_net_v2_warmup.pt"]
        Demonstrations --> Streamer --> WarmupModel
    end

    subgraph Phase1 ["Phase 1: Vectorized NashPG RL Pipeline (tools/train.py)"]
        VecEnv["C++ VectorizedBatchRunner (512 Envs, 5,000+ step/s)"]
        ActiveNet["Active Policy π_θ (Stratified Temperatures 0.10 - 0.50)"]
        RefNet["Frozen Reference Policy π_ref (Outer-loop KL Anchor)"]
        Rollout["RolloutBuffer (512 envs × 128 steps = 65,536 transitions)"]
        BlunderCalc["BlunderAwareRewardCalculator (Event Traps +1.0, Unprovoked -1.0, Shielding 0.0)"]
        GAE["Alternating Zero-Sum GAE (λ=0.98, γ=0.999)"]
        NashLoss["PPO Loss + η·D_KL(π_θ || π_ref) - c_ent·H(π_θ)"]

        WarmupModel --> ActiveNet
        ActiveNet <--> VecEnv
        VecEnv --> BlunderCalc --> Rollout --> GAE --> NashLoss --> ActiveNet
        ActiveNet -.->|Periodic Snapshot| RefNet
    end

    subgraph Phase2 ["Phase 2: Live Snapshot Evaluations"]
        Snapshots["Snapshots Saved Every N Seconds (snapshot_1200s.pt, snapshot_2400s.pt...)"]
        LiveEval["Live Match Evaluator vs HeuristicBot, RandomBot, & Historical Champions"]
        ActiveNet --> Snapshots --> LiveEval
    end

    subgraph Phase3 ["Phase 3: Post-Training Massive Tournament (tools/tournament.py)"]
        RoundRobin["Massive Vectorized Round-Robin (1,000 Games / Pair, 200+ games/s)"]
        EloCalc["Bradley-Terry MLE Elo Rating Matrix (Anchor: HeuristicBot = 1500.0)"]
        Causes["Diagnostic Loss Causes (Provoked vs Unprovoked DEFCON, Sudden Death VP, Scoring)"]
        Reports["Comprehensive Markdown & JSON Reports (final_tournament_report.md)"]

        Snapshots --> RoundRobin
        RoundRobin --> EloCalc --> Reports
        RoundRobin --> Causes --> Reports
    end
```

#### Standard Execution Commands
```bash
# 1. Phase 0: Bounded Streaming BC Warmup (Full 5,000 games in ~3 minutes, <70 MB RAM)
TRITON_CACHE_DIR=.triton_cache PYTHONPATH=. .venv/bin/python tools/train.py \
  --mode warmup \
  --arch v2 \
  --warmup-dataset <regenerated-dataset.jsonl.gz> \
  --bc-epochs 2 \
  --batch-size 1024 \
  --output-dir data/checkpoints/coldwar_net_v2_warmup.pt

# 2. Phase 1 & 2 & 3: Unified RL Training + Live Snapshots + Post-Training Tournament
# (or simply use ./tools/scripts/train_and_tournament.sh v2 7200 1200 data/checkpoints/coldwar_net_v2_warmup.pt)
TRITON_CACHE_DIR=.triton_cache PYTHONPATH=. .venv/bin/python tools/train.py \
  --arch v2 \
  --duration-seconds 7200 \
  --snapshot-interval-seconds 1200 \
  --warmup-checkpoint data/checkpoints/coldwar_net_v2_warmup.pt \
  --reward-scheme blunder_aware \
  --eval-opponents heuristic random data/checkpoints/run_v2_blunder_aware_9h/snapshot_21601s.pt \
  --eval-games-per-side 50 \
  --post-tournament \
  --post-tournament-models heuristic random data/checkpoints/run_v2_blunder_aware_9h/snapshot_21601s.pt \
  --post-tournament-games 500

# 2a. A/B experiments: budget by steps, not by wall clock. Evaluation cost scales with how
#     long the policy's games run, so a time budget hands the two arms different amounts of
#     training (one real 3-hour A/B finished 1024 vs 473 iterations on identical settings).
#     --duration-seconds now counts training only; evaluation and pool refreshes are excluded.
#     --eval-max-snapshot-opponents bounds evaluation, which is otherwise quadratic in run length.
TRITON_CACHE_DIR=.triton_cache PYTHONPATH=. .venv/bin/python tools/train.py \
  --arch v2 --train-steps 60000000 --eval-max-snapshot-opponents 4 \
  --output-dir data/checkpoints/ab_arm_on --start-pool-frac 1.0

# 2b. Watch a live (or finished) run: every training_metrics.jsonl metric is mirrored to
#     <output-dir>/tb/ as TensorBoard event files. Pass --no-tensorboard to write JSONL only.
.venv/bin/python -m tensorboard.main --logdir data/checkpoints/run_v2_<timestamp>/tb

# 3. Standalone Post-Tournament & Elo Evaluation Across Any Checkpoint Directory
PYTHONPATH=. .venv/bin/python tools/tournament.py \
  --checkpoint-dir data/checkpoints/run_v2_20260827_205207 \
  --games-per-side 500 \
  --anchor-model HeuristicBot \
  --anchor-elo 1500.0 \
  --output-report data/checkpoints/run_v2_20260827_205207/massive_tournament_report.md
```

### 3.4 Launch Web Workbench & Play Against NeuralBot
```bash
# 1. Start backend server (serves web UI on port 8000)
PYTHONPATH=. .venv/bin/python -m uvicorn web.server.main:app --host 0.0.0.0 --port 8000

# 2. In a separate terminal, launch NeuralBot for the opponent (e.g. USSR)
PYTHONPATH=. .venv/bin/python -m web.bot_client --game-id game-1 --role USSR --type neural --model-path data/checkpoints/run_v2_20260827_205207/snapshot_3602s.pt

# 3. Open browser at:
# http://localhost:8000/?game_id=game-1&role=US
```

### 3.5 Reusable Agent CLI Tools (`tools/`)

#### Standard CLI Tools (`tools/`)
1. **Unified Training Runner (`tools/train.py`)**:
   Starts time-bounded RL with live snapshot tournaments, blunder-aware reward shielding, and optional post-training massive tournament benchmarking.
2. **Unified Tournament & Evaluator (`tools/tournament.py`)**:
   Runs ultra-fast parallel tournaments and matchups (300-800 games/sec). If passed 2 models, outputs a granular head-to-head report with loss causes; if passed multiple models or a directory, outputs the full round-robin leaderboard and Bradley-Terry Elo matrix.
3. **Unified Match Runner & Replay Generator (`tools/play_match.py`)**:
   Runs matches between any pair of agents (supporting distinct checkpoints, heuristics, interactive human CLI play via `--us human`, and commentary), saving standardized `.tslog.json` replays with direct Web Workbench viewer URLs.
4. **Vectorized Demonstration Generator (`tools/generate_dataset.py`)**:
   Charns out thousands of games in parallel using 500 C++ environments and multi-temperature schedules, dumping compressed `.jsonl.gz` datasets for supervised BC warmup.
5. **Checkpoint Registry Inspector (`tools/inspect_checkpoints.py`)**:
   Scans `data/checkpoints/` and displays all saved models, sizes, timestamps, and the architecture detected from the weights.
6. **Human Game Corpus (`tools/download_ts_replayer.py` + `tools/lib/ts_replayer_*.py`)**:
   Downloads community-uploaded human games from ts-replayer.fly.dev and converts them into
   engine decisions. The conversion is verified rather than parsed: each entry is rebuilt from
   the position the log states, driven through the engine, and its outcome compared against the
   log's own next board. The hands, which the log never states in full, are solved for a whole
   game at once as a constraint problem (z3, MIT-licensed and optional). 300 of 300 games
   convert in full — 29,820 of 30,620 entries, 144,844 decisions. See `tools/README.md` §6.

---

## 4. Key Maintenance Invariants for Agents

1. **Zero Heap Allocations in Engine Core**:
   `ts::GameState` must remain trivially copyable (`std::is_trivially_copyable_v<GameState>`) and within 4 KB.
2. **212-Dimensional Flat Action Space**:
   All neural network policy heads and action masks operate over the exact 212 flat action space mapped by `bindings.ActionEncoder` and `ActionMask::generate_flat_mask_212`.
3. **NashPG Reference Regularization**:
   The active policy $\pi_\theta$ is regularized against the frozen outer-loop snapshot $\pi_{\text{ref}}^{(k)}$ with fixed $\eta$, ensuring monotonic convergence to Nash equilibrium without strategy cycling.
4. **Deterministic PRNG**:
   All simulation randomness uses `state.rng_state` with SplitMix64 (`ts::Prng`).
5. **Strict Type Annotations & Mandatory Type Checking**:
   All Python types must be explicitly annotated (using `TypedDict` definitions in `web/server/replay_types.py` for all serialized JSON structures, replays, game states, audit logs, and metrics). Whenever modifying or adding Python code, static type checks MUST be executed via `.venv/bin/pyrefly check` and all typing validation tests must pass cleanly without errors.
6. **Unified Self-Play & Replay Generation**:
   All self-play simulation and `.tslog.json` replay recording across training pipelines, evaluation benchmarks, and CLI scripts MUST use the unified `generate_self_play_replay` function in `tools.lib.self_play` to guarantee 100% adherence to standard `.tslog.json` schema (`ReplayLogDict`).
7. **Mandatory Checkpoint Directory Naming Convention**:
   A checkpoint directory MUST carry the run's short name from `research/run_nomenclature.md`:
   `data/checkpoints/[engine]-[attempt]-[seed]_[start date]_[start time]`
   (e.g. `data/checkpoints/E3-12-21_20260912_181622`). Pass `tools/train.py --run-name E3-12-21`
   and the directory is built for you; the name is validated and also recorded in
   `metadata.json`. Omit the steps field — one directory holds every budget of a lineage, and
   each snapshot's filename already carries its own.
   Add the table row *before* launching, since the name is a claim about which row the run is.
   `data/checkpoints/run_[version]_[start date]_[start time]` remains the fallback for smoke runs
   that are not arms. Hardcoded, ad-hoc names (e.g. `run_v2_2h`, `p1_scalar_nofilter`) are
   forbidden: they say nothing about engine or seed, and a set of cross-engine comparisons was
   written up as same-engine ones because recovering that took a `git merge-base` on two hashes.
8. **Bounded Dataset Streaming & OOM Prevention**:
   Any operation reading or training on demonstration datasets (`WarmupDataset`) MUST use bounded streaming (`stream_batches` / `stream_transitions`). Loading entire multi-million transition datasets into monolithic in-memory tensors without bounds is forbidden to prevent system Out-Of-Memory (OOM) failures.
9. **Mandatory Unified CLI Invariant — No Ad-Hoc Scripts**:
   Agents must NEVER write ad-hoc Python scripts, scratch files, or one-off code to call Behavioral Cloning (`ai/training/behavioral_cloning.py`), train neural models, run tournaments, or simulate matches.
   - For all model training (both Behavioral Cloning warmup and RL self-play): **ALWAYS execute `tools/train.py`** (`tools/train.py --mode warmup --warmup-dataset <path>` for BC warmup, or `tools/train.py --warmup-dataset <path>` for BC warmup + RL self-play).
   - For all tournaments and evaluations: **ALWAYS execute `tools/tournament.py`**.
   - For all match simulation and replays: **ALWAYS execute `tools/play_match.py`**.
   Writing one-off scripts to invoke training or simulation functions directly violates repository architecture.
10. **Pre-Training Commit & Checkpoint Metadata Invariant**:
   Before launching any model training run:
   - A git commit MUST be created recording the current codebase state (commit locally on the active branch without pushing or advancing remote master).
   - The commit hash, training mode, and a short description of what was changed and the training goal MUST be recorded in `metadata.json` within the checkpoint directory (`data/checkpoints/run_.../metadata.json`). The training CLI (`tools/train.py --description "..."`) automatically records these metadata fields at startup.
11. **Engine Change Restriction Invariant**:
   Agents must **NEVER make any changes to the C++ engine (`engine/`) without explicitly asking the user and obtaining prior confirmation**. Every engine modification requires prior user approval without exception.
12. **Human Replay Conversion Is Never Approximated**:
   The ts-replayer corpus is training data for a model meant to learn how people play, so a
   decision the log does not determine must NEVER be invented, defaulted, or filled in by a
   heuristic — it is a failure to diagnose and fix. The same goes for the engine: report a
   "safety net" fallback rather than relying on one, and fail loudly instead of recovering an
   approximation. Two exceptions exist, both explicitly approved and both narrow: guessing the
   cards Our Man in Tehran looks at, and the individually diagnosed entries in
   `_KNOWN_SCORE` / `_LOG_MISCOUNTED` / `_INVALID_PLAYS` where the log itself is wrong. When
   reporting a conversion mismatch, cite the exact replay, turn and action round — and per
   invariant 11, propose engine fixes rather than making them.

13. **Never Measure Against A Stale Engine**:
   Every artifact in this repository — a checkpoint, a demonstration dataset, an Elo anchor, a
   diagnostic table, a converted replay — is only meaningful relative to the engine that
   produced it, and rebuilding the engine can change the decision stream without a line of
   Python changing. Run `tools/scripts/check_engine_fresh.sh` before generating or consuming
   any of them, and when it reports the build was stale, treat every number taken beforehand as
   measured on a different game until it is re-taken.
   Two failure modes make this worse than it sounds, and both have already happened here:
   - Datasets in the `(seed, [flat_action, ...])` format do not fail when the engine moves under
     them. `WarmupDataset.stream_transitions` stops a game at the first newly-illegal action and
     continues to the next, so the set silently shrinks — and always by losing the *tail* of
     each game, which biases what remains toward openings. The archived 5,000-game set retained
     **27% of its decisions and 5% of its games intact**; see
     `data/datasets/archive/README.md`.
   - Checkpoints keep loading. Old weights still accept the current observation and run a
     forward pass, so nothing announces that they were trained against different rules. Loading
     cleanly is not evidence of comparability.


### Test data that is not in the repository

`data/replays/` and `data/checkpoints/` are git-ignored, so **no test may assume either holds
anything**. Do not guard such a test with a skip: a check that skips when its input is missing
passes on every machine while verifying nothing, which is this repository's most repeated bug
(`research/metrics.md` §1, plus a `pyrefly` invocation found checking zero files and a
stale-engine test disabled for two commits by a directory move).

Use the `generated_replay_dir` fixture in `tests/conftest.py` instead. It generates a replay with
`generate_self_play_replay` into a temporary directory and points the server at it through
`TS_REPLAYS_DIR`, so the test runs against real output of the current writer. It needs no
checkpoint — the generator falls back to an untrained network — and costs about a second. Two
reasons this beats committing a sample replay: one replay is ~3.7 MB, and a frozen file can drift
from what the writer emits while the test keeps passing.

Generated data cannot catch a change to the schema itself, since the writer and the TypedDicts
move together. That is what `tests/bindings/replay_schema.golden.json` is for — the schema pinned
as source, a few KB, reviewable as a diff. Regenerate it deliberately when the schema changes:

```bash
PYTHONPATH=.:build/release .venv/bin/python tests/bindings/test_replay_schema_golden.py
```

To get real replays for the workbench during development, generate them; they are not repo content:

```bash
PYTHONPATH=. .venv/bin/python tools/play_match.py --us heuristic --ussr strategic --game-id scratch
```

---

## 5. Run Test Suites

`tests/` is split by what is under test, and **the groups are not all relevant to every change.**
Pick by what you touched; the table is the whole rule.

| you changed | run | cost |
|:---|:---|---:|
| `engine/`, `bindings/` | C++ suite + `tests/bindings tests/engine_logic` | ~30s |
| `tools/lib/ts_replayer_*` | the above + `tests/replayer` | ~6.5 min |
| `ai/`, `bot/`, `tools/` | the above + `tests/training` | ~7 min |
| `web/server/`, `web/ui/`, `web/bot_client.py` | **also** `tests/web` (see prerequisites) | +5s |

**Do not run `tests/web` for engine, bindings, AI or tools changes.** It cannot tell you anything
about them, and it fails for reasons that have nothing to do with your change: the frontend tests
need `web/ui/dist` built and the E2E tests need a Playwright browser installed, neither of which is
in the repository. A red web suite on an engine change is noise that trains you to ignore failures.

### The ts-replayer corpus

`tests/replayer` (and four tests in `tests/engine_logic`) run against the **real** human corpus,
not synthetic data — that is the point of them, so it cannot be generated. It is ~5 MB, git-ignored,
and downloaded once per **machine**:

```bash
PYTHONPATH=. .venv/bin/python tools/download_ts_replayer.py
```

It is cached in `~/.cache/ts_ai/ts_replayer` (`$XDG_CACHE_HOME` honoured), **not** under `data/`,
so every checkout and every git worktree shares one copy — a worktree has its own empty `data/`,
and a corpus stored there would be re-fetched, 300 requests at one per second, for bytes already
on the machine. `tools/lib/corpus_paths.py` resolves the location: `$TS_REPLAYER_CORPUS` first,
then an existing `data/datasets/ts_replayer` for checkouts that predate this, then the shared cache.

**A missing corpus fails these tests; it does not skip them.** They previously carried
`skipif(corpus not downloaded)` against a hardcoded absolute path, so on any machine but the one
that path was written for, 289 test functions skipped and the suite reported green.

### Run the suite in parallel

`pytest-xdist` is installed; `-n auto` uses every core and is worth it everywhere:

| | serial | `-n auto` (24 cores) |
|:---|---:|---:|
| `tests/replayer` | 3m 19s | **21s** |
| whole backend suite | ~7m | **61s** |

The single whole-corpus test, `test_every_game_in_the_corpus_converts`, is marked `corpus_full`
and **deselected by default** — on its own it was 167s of the 364s that `tests/replayer` used to
take. Run it before merging any change to `tools/lib/ts_replayer_*`:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q -m corpus_full tests/replayer   # ~2m 48s
```

A `converted_game` session fixture in `tests/conftest.py` memoizes `convert_game` by replay id
for tests that want it. It is available rather than mandatory: with `-n auto` the suite already
runs in 21s, so rewriting ~120 existing call sites to use it would buy little for the risk.

```bash
# ---------------------------------------------------------------------------------------------
# 0. ALWAYS FIRST -- never measure against a stale engine (invariant 13)
# ---------------------------------------------------------------------------------------------
tools/scripts/check_engine_fresh.sh

# ---------------------------------------------------------------------------------------------
# 1. C++ ENGINE -- the fast inner loop for engine/ work (368 tests, seconds)
# ---------------------------------------------------------------------------------------------
./build/release/engine/ts_tests
./build/release/engine/ts_benchmark
./build/release/engine/ts_fuzz --games 10000            # invariant fuzzer
./build/release/engine/ts_fuzz --steps 5000000 --seed 42

# ---------------------------------------------------------------------------------------------
# 2. BACKEND PYTHON -- everything except the web UI. The default suite for engine, bindings,
#    replayer, AI and tools work. ~7 minutes, dominated by tests/replayer.
# ---------------------------------------------------------------------------------------------
PYTHONPATH=. .venv/bin/python -m pytest -q -n auto tests/bindings tests/engine_logic tests/replayer tests/training

#    Narrower loops while iterating (run the full backend suite before calling the work done):
PYTHONPATH=. .venv/bin/python -m pytest -q tests/bindings tests/engine_logic   # 331 tests, ~10s
PYTHONPATH=. .venv/bin/python -m pytest -q tests/replayer                      # 422 tests, ~6min
PYTHONPATH=. .venv/bin/python -m pytest -q tests/training                      # 187 tests, ~34s
PYTHONPATH=. .venv/bin/python -m pytest -q tests/engine_logic/test_all_110_cards.py::TestName::test_case

# ---------------------------------------------------------------------------------------------
# 3. WEB / UI -- ONLY for changes under web/. Needs two build artifacts that are not committed:
# ---------------------------------------------------------------------------------------------
cd web/ui && npm install && npm run build && cd ../..   # produces web/ui/dist
.venv/bin/python -m playwright install chromium         # for the E2E tests
PYTHONPATH=. .venv/bin/python -m pytest -q tests/web

# ---------------------------------------------------------------------------------------------
# 4. STATIC TYPING -- after ANY Python change, must be 0 errors (invariant 5)
# ---------------------------------------------------------------------------------------------
.venv/bin/pyrefly check ai tools tests web bindings
```

> **Worktrees and the venv.** `.venv/` lives in the main checkout, not in a git worktree under
> `.claude/worktrees/`, so use the main checkout's path from there (e.g. `/workspace/.venv/bin/python`).
> Invoke pytest as `python -m pytest`, not via the `.venv/bin/pytest` console script: that script
> carries an absolute shebang from wherever the venv was first created, which breaks if the venv or
> the repository is ever moved or copied.

**Pass pyrefly the source directories explicitly.** A bare `pyrefly check` silently checks ZERO
files when the repo is a git worktree under `.claude/worktrees/`: pyrefly honours
`.git/info/exclude`, which the harness fills with `**/.claude/worktrees/`, so everything is
excluded and it exits 0 having examined nothing. `No Python files matched patterns` on the last
line means the check measured nothing, not that the code is clean.

**Do NOT run `tests/differential/`.** It is gated behind the `differential_fuzz` marker /
`--run-fuzz` and is ignored at collection (`tests/conftest.py`), because those modules fail at
*import* and would otherwise abort the whole run. It is WIP, not informative in its current state,
and not part of the check a change is expected to pass. Do not spend time reviving it.
