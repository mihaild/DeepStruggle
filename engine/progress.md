# Twilight Struggle AI Engine — Progress Report (`engine/progress.md`)

This report provides a complete, chronological record of what has been implemented and validated in the **Twilight Struggle Engine (`engine/`)**, as well as the remaining components planned for the broader `ts_ai` platform.

---

## 1. Executive Summary

- **Component**: C++20 Simulation Engine for Twilight Struggle Deluxe Edition (110 Cards).
- **Location**: `engine/` (includes `engine/include/`, `engine/src/`, `engine/tests/`, `engine/CMakeLists.txt`).
- **Memory Footprint**: `sizeof(GameState) \approx 1.1\text{ KB} \le 4\text{ KB}`, 64-byte aligned, zero dynamic allocations, trivially copyable (`std::is_trivially_copyable_v<GameState> = true`).
- **Throughput Measured**: **2,092,261 simulation steps/sec** on a single CPU core (target was $\ge 100,000$).
- **Test Status**:
  - Unit & Integration Tests (`ts_tests`): **20/20 Passed**.
  - Invariant Fuzzer (`ts_fuzz`): **100,000+ steps rolled out across full games with 0 deadlocks and 0 invariant violations**.
  - High-Speed Benchmark (`ts_benchmark`): **500,000 steps in 0.239s**.

---

## 2. Detailed Work Completed (Chronological History)

### Phase 1: Specifications, Rules & Topology Mapping
- Thoroughly parsed and analyzed `specification.txt`, `action_interface.txt`, `game_state.txt`, `rules.md`, `rules.json`, `map.md`, `map.json`, `flags.json`, `cards.json`, and `card_primitives.json`.
- Mapped all 84 countries, 6 geographic regions, 29 battleground countries, subregions (Western/Eastern Europe with Austria & Finland intersection, Southeast Asia), and superpower adjacency links (USA: 4 countries, USSR: 8 countries).
- Formulated the 64-bit continuous effects bitfield (`effect_bits`), encoding all 43 persistent flags and the turn-cleanup mask (`0x00000780A3CBE3C0ULL`).
- Created and obtained user approval on the master technical implementation plan in `implementation_plan.md`.

### Phase 2: Core Memory Layout & Primitives
- Created `engine/include/ts/types.hpp` with strongly typed enums for `Player`, `Phase`, `WarEra`, `CardLocation`, `DecisionType`, `ActionType`, `Region`, `SubRegion`, `PlayMode`, `TimingBranch`, and `OpMode`.
- Created `engine/include/ts/constants.hpp` with full constants for 84 countries, all 110 card IDs, and 64-bit flag masks.
- Created `engine/include/ts/micro_action.hpp` defining the 4-byte `MicroAction` struct (`alignas(4)`).
- Created `engine/include/ts/prng.hpp` implementing the 64-bit SplitMix64 deterministic PRNG.
- Created `engine/include/ts/game_state.hpp` defining the contiguous `GameState`, `DecisionContext` stack (max depth 3), `ActionToken` (16 bytes), `ActionHistoryBuffer`, and `ObservationBuffer`.

### Phase 3: Graph Topology, Card Metadata & Rule Engines
- Created `engine/include/ts/map_data.hpp` and `engine/src/map_data.cpp` with static lookup tables and precomputed 128-bit adjacency bitmasks (`uint64_t[2]`) for $O(1)$ connectivity and region filtering.
- Created `engine/include/ts/card_data.hpp` and `engine/src/card_data.cpp` with static metadata for all 110 cards (Ops, side, era, one-time asterisk, scoring card flags).
- Created `engine/include/ts/scoring.hpp` and `engine/src/scoring.cpp` implementing:
  - Precise country control: $US \ge stab \land US - USSR \ge stab$.
  - Regional Presence, Domination, Control, battleground bonuses, and superpower adjacency bonuses.
  - Formosan Resolution Taiwan battleground modifier.
  - Shuttle Diplomacy battleground subtraction.
  - Southeast Asia scoring (#38).
  - Military Ops deficit VP evaluation.
  - Final Scoring at Turn 10.
  - Europe Control instant victory check ($VP = \pm 20, Phase = GAME\_OVER$).
- Created `engine/include/ts/space_race.hpp` and `engine/src/space_race.cpp` implementing the 8 Space Race milestones, Ops prerequisites, roll success checks, VP awards, and milestone abilities (e.g. Animal in Space 2 attempts/turn, Man in Space headline priority, Space Walk safe discard, Space Station 8 ARs).
- Created `engine/include/ts/ops.hpp` and `engine/src/ops.cpp` implementing:
  - Influence placement with dynamic cost transitions (2 Ops dropping to 1 Op as soon as enemy control breaks).
  - Coup execution with DEFCON degradation, instant DEFCON suicide checks, CMC checks, NATO/Reformer/US-Japan pact protection, and Death Squads / SALT roll modifiers.
  - Realignment rolls with adjacency and control modifiers.
  - Legal target bitmask generators for influence, coup, and realignment.

### Phase 4: Card Event Handlers (All 110 Cards)
- Created `engine/src/events/early_war.cpp` for Cards 1–35, 103–106:
  - Duck and Cover, Five Year Plan, Socialist Governments, Fidel, Vietnam Revolts, Blockade, Korean War, Romanian Abdication, Arab-Israeli War, Comecon, Nasser, Warsaw Pact, De Gaulle, Captured Nazi Scientist, Truman Doctrine, Olympic Games, NATO, Independent Reds, Marshall Plan, Indo-Pakistani War, Containment, CIA Created, US-Japan Pact, Suez Crisis, East European Unrest, Decolonization, Red Scare/Purge, UN Intervention, De-Stalinization, Nuclear Test Ban, Formosan Resolution, Defectors, The Cambridge Five, Special Relationship, NORAD.
- Created `engine/src/events/mid_war.cpp` for Cards 36–81, 107–108:
  - Brush War, Cuban Missile Crisis, Nuclear Subs, Quagmire, SALT Negotiations, Bear Trap, Summit, How I Learned to Stop Worrying, Junta, Kitchen Debates, Missile Envy, We Will Bury You, Brezhnev Doctrine, Portuguese Empire Crumbles, South African Unrest, Allende, Willy Brandt, Muslim Revolution, ABM Treaty, Cultural Revolution, Flower Power, U-2 Incident, OPEC, Lone Gunman, Colonial Rear Guards, Panama Canal Returned, Camp David Accords, Puppet Governments, Grain Sales, John Paul II, Latin American Death Squads, OAS Founded, Nixon Plays the China Card, Sadat Expels Soviets, Shuttle Diplomacy, Voice of America, Liberation Theology, Ussuri River Skirmish, Ask Not What Your Country Can Do For You, Alliance for Progress, One Small Step, Che, Our Man in Tehran.
- Created `engine/src/events/late_war.cpp` for Cards 82–102, 109–110:
  - Iranian Hostage Crisis, The Iron Lady, Reagan Bombs Libya, Star Wars, North Sea Oil, The Reformer, Marine Barracks Bombing, Soviets Shoot Down KAL-007, Glasnost, Ortega Elected in Nicaragua, Terrorism, Iran-Contra Scandal, Chernobyl, Latin American Debt Crisis, Tear Down this Wall, "An Evil Empire", Aldrich Ames Remix, Pershing II Deployed, Wargames, Solidarity, Iran-Iraq War, Yuri and Samantha, AWACS Sale to Saudis.
- Created `engine/src/card_dispatcher.cpp` linking event triggers, sub-decision step handlers, and action masking filters across all cards.

### Phase 5: Micro-Decision State Machine & Observations
- Created `engine/include/ts/action_mask.hpp` and `engine/src/action_mask.cpp` generating legal action masks for `SELECT_CARD`, `SELECT_PLAY_MODE`, `CHOOSE_TIMING_BRANCH`, `SELECT_OP_MODE`, `POINT_NODE`, and `CHOOSE_BRANCH`.
- Created `engine/include/ts/state_machine.hpp` and `engine/src/state_machine.cpp` managing the complete turn loop:
  - Setup Phase (USSR 6 EE, US 7 WE).
  - Headline Phase simultaneous selection, Space 4 reveal priority, HV determination, tie breaking (US first), Defectors cancellation.
  - Action Round loop with Quagmire/Bear Trap escape rolls, timing branches (`OPS_FIRST` vs `EVENT_FIRST`), NORAD triggers at AR end, and hand exhaustion auto-pass.
  - Phase E..I turn-end evaluation, Military Ops deficit scoring, held scoring card loss check, China Card flip, turn advancement, and Mid/Late War deck integration.
- Created `engine/include/ts/observation.hpp` and `engine/src/observation.cpp` extracting `ObservationBuffer` float feature tensors for neural network consumption.
- Created `engine/include/ts/serialization.hpp` and `engine/src/serialization.cpp` providing binary state snapshotting and JSON export.
- Created `engine/include/ts/engine.hpp` and `engine/src/engine.cpp` providing the unified public API for `ts::Engine`.

### Phase 6: Project Reorganization & Validation
- Reorganized all engine-related code, headers, build configs, and test suites into `engine/`.
- Updated top-level `CMakeLists.txt` to include `engine/` via `add_subdirectory(engine)`.
- Authored `engine/AGENTS.md` and `engine/progress.md`.
- Successfully built in sandbox with GCC/Clang C++20 (`-O3 -march=native`).
- Ran and verified all test suites (`ts_tests`, `ts_fuzz`, `ts_benchmark`).

---

## 3. Verification & Validation Metrics

| Test Target | Executable | Steps / Trials | Result | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Unit & Integration Suite** | `./build/engine/ts_tests` | 20 test cases | **20/20 PASSED** | Validates Map topology, battlegrounds, Scoring math, Ops costs, Coups, NATO, Card events, DEFCON suicide priority, Re-entrancy stack |
| **Invariant Fuzzer** | `./build/engine/ts_fuzz` | 100,000 steps | **PASSED (0 errors)** | Random rollouts over 6 full games: strictly zero deadlocks, zero out-of-bounds VP/DEFCON/MilOps/Space/Stack violations |
| **Throughput Benchmark** | `./build/engine/ts_benchmark` | 500,000 steps | **2,092,261 steps/sec** | Single physical core, zero allocation, over 20x the required 100k target |

---

## 4. What Remains / Future Roadmap

The simulation engine is complete, optimized, and tested. The following components represent planned extensions for the broader `ts_ai` repository:

1. **Python / PyBind11 Bindings (`python/` or `bindings/`)**:
   - Expose `ts::Engine`, `ts::GameState`, `ts::MicroAction`, and `extract_observation` directly to Python NumPy buffers with zero-copy semantics.
   - Implement Gymnasium / PettingZoo compliant environment wrappers (`TwilightStruggleEnv`).
2. **Search Engine & MCTS (`mcts/` or `search/`)**:
   - Implement batched Monte Carlo Tree Search (MCTS) leveraging zero-allocation `GameState` cloning (`std::memcpy`).
   - Implement Root Parallelization and Leaf Parallelization with GPU tensor evaluation queues.
3. **Neural Network Architecture (`nn/` or `model/`)**:
   - Policy & Value network (Dual-headed ResNet / Transformer / Graph Neural Network on the 84-country map graph).
   - Spatial feature embeddings for countries, adjacency edges, and hand card representations.
4. **Training Pipeline (`training/`)**:
   - Self-play worker cluster generating rollout trajectories into a circular replay buffer.
   - PPO / AlphaZero policy-value loss optimization in PyTorch with mixed-precision training.
5. **Evaluation & Tournament Framework (`eval/`)**:
   - Heuristic baseline bots (Greedy Ops, DEFCON protection, regional influence maximizers).
   - ELO rating tracker and tournament matchmaking runner.
