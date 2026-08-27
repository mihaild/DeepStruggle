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

    subgraph Bot ["Bot Clients & Agents (bot/)"]
        BaseBot["Generic BaseBot Class (bot/base_bot.py)"]
        Runner["WebSocket bot_client.py"]
        NeuralBotClient["NeuralBot (PyTorch)"]
        Heuristic["HeuristicBot Baseline"]
        Random["RandomBot Baseline"]
        AgentPlayer["Agent Interactive Player CLI"]
        BaseBot --> Heuristic
        BaseBot --> Random
        BaseBot --> NeuralBotClient
        BaseBot --> AgentPlayer
    end

    subgraph Tools ["Generic CLI Tools & Evaluation (tools/)"]
        TrainRunner["Unified Training CLI (tools/train.py)"]
        TourneyRunner["Massive Tournament CLI (tools/tournament.py)"]
        EvalCLI["Head-to-Head Evaluator (tools/evaluate.py)"]
        ReplayGen["Replay Generator (tools/generate_replay.py)"]
        Inspector["Checkpoint Inspector (tools/inspect_checkpoints.py)"]
        GenericEval["Generic Evaluation & Self-Play (tools/eval/)"]
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
        Obs["ts::Observation (4293-dim Float Tensor)"]
        Map["ts::MapData (84 Countries Graph)"]
        Cards["ts::CardData (110 Cards Event Logic)"]
    end

    Runner <-->|WebSocket: JSON State / Actions| WS
    WS <--> Nanobind
    Nanobind <--> CoreEngine
    VecEnv <--> Nanobind
    VecEnv <--> NashPG
    NashPG <--> ColdWarNet
    Rewards <--> NashPG
    ColdWarNet --> NeuralBotClient
    NeuralBotClient --> Runner
    GenericEval <--> ColdWarNet
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
│       ├── rollout_buffer.py   # Trajectory storage & GAE advantage calculator
│       ├── behavioral_cloning.py # Phase 0 supervised pre-training
│       ├── nash_pg.py          # NashPG (Nash Policy Gradient with iterative KL regularization)
│       ├── warmup_dataset_loader.py # Memory-bounded demonstration streaming
│       └── train.py            # CLI training entry point
│
├── bot/                        # Bot clients & interactive players (see bot/AGENTS.md)
│   ├── AGENTS.md               # Specific instructions for developing AI bots
│   ├── base_bot.py             # Generic BaseBot abstract class defining bot interface
│   ├── random_bot.py           # RandomBot baseline
│   ├── heuristic_bot.py        # HeuristicBot rule-based baseline
│   ├── neural_bot.py           # NeuralBot client using ColdWarNet checkpoints
│   ├── agent_player.py         # Rich CLI & interactive agent player interface
│   └── bot_client.py           # WebSocket network bot runner for live matches
│
├── web/                        # Web Workbench (UI + Backend Server)
│   ├── ui/                     # Vite + TypeScript + SVG Deluxe Map (was frontend/)
│   │   ├── index.html
│   │   ├── package.json / vite.config.ts
│   │   └── src/                # Map view, HUD, tracks, debug panel, replay controls
│   └── server/                 # FastAPI Game Server and Replay Manager (was server/)
│       ├── AGENTS.md           # Instructions for server maintainers
│       ├── main.py             # FastAPI app, REST routes, WebSocket endpoint /ws/game/{id}
│       ├── session.py          # GameSession class, action router, state broadcasting
│       ├── replay.py           # ReplayLogger and ReplayManager
│       └── replay_types.py     # TypedDict specifications for state, action, and logs
│
├── tools/                      # Reusable agent & developer CLI tools (see tools/README.md)
│   ├── README.md               # Tool descriptions, CLI flags, and usage examples
│   ├── train.py                # Unified RL training & fine-tuning runner
│   ├── tournament.py           # Massive vectorized round-robin tournament & Elo matrix evaluator
│   ├── evaluate.py             # Head-to-head match evaluation CLI
│   ├── generate_replay.py      # Self-play or head-to-head replay generator (.tslog.json)
│   ├── inspect_checkpoints.py  # Checkpoint discovery, architecture detection & metadata inspector
│   ├── generate_warmup_dataset.py # Demonstration rollout generator
│   ├── eval/                   # Generic agent evaluation & self-play utilities
│   │   ├── player_agent.py     # Unified Agent loader (random, heuristic, neural)
│   │   ├── tournament_evaluator.py # Matchup runner & loss cause classifier
│   │   ├── self_play.py        # Self-play simulation & .tslog.json recorder
│   │   ├── batch_tournament.py # Vectorized batch tournament runner
│   │   └── arena.py            # Tournament evaluator vs baselines
│   └── scripts/                # Shell automation scripts
│       ├── train_and_tournament.sh # Unified training & tournament bash runner
│       ├── train_direct_rl.sh  # Direct RL self-play runner
│       └── run_asan.sh         # AddressSanitizer execution script
│
├── data/                       # Datasets, Checkpoints & Recorded Replays
│   ├── checkpoints/            # Model weights (run_v3_*, coldwar_net_v3_warmup.pt)
│   ├── replays/                # Saved game logs (*.tslog.json)
│   └── datasets/               # Demonstration datasets (warmup_5k_games.jsonl.gz)
│
├── external/                   # External integrations & differential engines
│   ├── README.md               # Integration guide
│   └── struggler/              # External reference engine (Rust/Python)
│
├── tests/                      # Python pytest integration test suite (368 tests)
│   ├── test_credit_assignment.py # Unit tests for reward propagation & Rule 4.3 headlines
│   ├── test_neural_and_nashpg.py # Unit & integration tests for ColdWarNet, ActionMask, NashPG
│   ├── test_all_110_cards.py   # Comprehensive unit tests for all 110 cards
│   ├── test_all_110_cards_differential.py # Exhaustive 110-card cross-engine validation (131 tests)
│   ├── test_struggler_differential.py # Cross-engine differential test suite
│   ├── test_card_fixes.py      # Dedicated verification suite for card rules fixes
│   ├── test_bindings.py        # Validates Python nanobind module
│   ├── test_server_and_bot.py  # Validates REST APIs, bot-vs-bot WebSocket simulation
│   ├── test_types_and_json_schemas.py # Validates replay JSON schema and pyrefly typing
│   ├── test_web_workbench.py   # FastAPI client and UI metadata endpoint tests
│   └── test_e2e_space_race.py  # Playwright E2E browser tests
│
├── rules/                      # [GIT IGNORED] General game rules, PDF, map & card descriptions
│   ├── Rules_Final.pdf         # Official Twilight Struggle Deluxe Edition rulebook
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
pip install nanobind fastapi "uvicorn[standard]" websockets pytest numpy pydantic httpx torch torchvision

# 2. Web UI dependencies & production build
cd web/ui && npm install && npm run build && cd ../..
```

### 3.2 Build C++ Engine & Nanobind Extension
```bash
# Standard Release Build
cmake -B build/release -S . -DPython_EXECUTABLE=$(pwd)/.venv/bin/python3
cmake --build build/release -j
```

### 3.3 Generic Scheme for Training & Tournament Pipelines

```mermaid
graph TD
    subgraph Phase0 ["Phase 0: Supervised BC Warmup"]
        Demonstrations["Demonstration Dataset (5,000 Games, data/datasets/warmup_5k_games.jsonl.gz)"]
        Streamer["WarmupDataset.stream_batches (B=1024, Reservoir Buffer, RAM < 70MB)"]
        WarmupModel["Warmup Checkpoint: data/checkpoints/coldwar_net_v3_warmup.pt (90% vs Heuristic)"]
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
  --arch v3 \
  --warmup-dataset data/datasets/warmup_5k_games.jsonl.gz \
  --bc-epochs 2 \
  --batch-size 1024 \
  --output-dir data/checkpoints/coldwar_net_v3_warmup.pt

# 2. Phase 1 & 2 & 3: Unified RL Training + Live Snapshots + Post-Training Tournament
# (or simply use ./tools/scripts/train_and_tournament.sh v3 7200 1200 data/checkpoints/coldwar_net_v3_warmup.pt)
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

# 3. Standalone Post-Tournament & Elo Evaluation Across Any Checkpoint Directory
PYTHONPATH=. .venv/bin/python tools/tournament.py \
  --checkpoint-dir data/checkpoints/run_v3_20260827_205207 \
  --games-per-side 500 \
  --anchor-model HeuristicBot \
  --anchor-elo 1500.0 \
  --output-report data/checkpoints/run_v3_20260827_205207/massive_tournament_report.md
```

### 3.4 Launch Web Workbench & Play Against NeuralBot
```bash
# 1. Start backend server (serves web UI on port 8000)
PYTHONPATH=. .venv/bin/python -m uvicorn web.server.main:app --host 0.0.0.0 --port 8000

# 2. In a separate terminal, launch NeuralBot for the opponent (e.g. USSR)
PYTHONPATH=. .venv/bin/python -m bot.bot_client --game-id game-1 --role USSR --type neural --model-path data/checkpoints/run_v3_20260827_205207/snapshot_3602s.pt

# 3. Open browser at:
# http://localhost:8000/?game_id=game-1&role=US
```

### 3.5 Reusable Agent CLI Tools (`tools/`)

#### Standard CLI Tools (`tools/`)
1. **Unified Training Runner (`tools/train.py`)**:
   Starts time-bounded RL with live snapshot tournaments, blunder-aware reward shielding, and optional post-training massive tournament benchmarking.
2. **Massive Vectorized Tournament & Elo Rating (`tools/tournament.py`)**:
   Runs ultra-fast parallel tournaments (300-800 games/sec) across all checkpoints in a directory, calculating Bradley-Terry Elo ratings, total/USSR/US winning matrices, and side-specific loss cause breakdowns.
3. **Generate Game Replay (`tools/generate_replay.py`)**:
   Runs self-play or head-to-head matches and logs standardized `.tslog.json` replays with the exact Web Workbench viewer URL.
4. **Head-to-Head Match Evaluator (`tools/evaluate.py`)**:
   Runs a fast match between any two agents, breaking down US and USSR win rates and exact loss reasons.
5. **Checkpoint Registry Inspector (`tools/inspect_checkpoints.py`)**:
   Scans `data/checkpoints/` and displays all saved models, sizes, timestamps, and detected architectures.

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
   All self-play simulation and `.tslog.json` replay recording across training pipelines, evaluation benchmarks, and CLI scripts MUST use the unified `generate_self_play_replay` function in `tools.eval.self_play` to guarantee 100% adherence to standard `.tslog.json` schema (`ReplayLogDict`).
7. **Mandatory Checkpoint Directory Naming Convention**:
   All model checkpoint directories MUST follow the standard pattern:
   `data/checkpoints/run_[version]_[start date]_[start time]`
   (e.g., `data/checkpoints/run_v3_20260826_231500` or `data/checkpoints/run_v2_20260825_093352`). Hardcoded, ad-hoc directory names (e.g. `run_v3_2h`) are strictly forbidden.
8. **Bounded Dataset Streaming & OOM Prevention**:
   Any operation reading or training on demonstration datasets (`WarmupDataset`) MUST use bounded streaming (`stream_batches` / `stream_transitions`). Loading entire multi-million transition datasets into monolithic in-memory tensors without bounds is forbidden to prevent system Out-Of-Memory (OOM) failures.

---

## 5. Run Test Suites
```bash
# C++ Unit Tests (304 tests) & Performance Benchmark
./build/release/engine/ts_tests
./build/release/engine/ts_benchmark

# Python Integration Tests (368 tests including Neural & NashPG suite)
PYTHONPATH=.:external/struggler/src .venv/bin/pytest -v tests/

# Static Type Checking with Pyrefly (must return 0 errors)
.venv/bin/pyrefly check
```
