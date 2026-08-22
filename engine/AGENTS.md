# Twilight Struggle AI: C++ Simulation Core Guide

This directory contains the zero-allocation, high-throughput simulation engine for the Deluxe Edition of **Twilight Struggle** (110 Cards).

---

## 1. Core Architectural Pillars

1. **Zero Heap Allocation in Simulation Core**: The core data structures (`GameState`, `DecisionContext`, `MicroAction`) use fixed-size memory layouts. `GameState` is strictly trivially copyable and under 4 KB (`sizeof(GameState) <= 4096`).
2. **Micro-Decision State Machine**: All multi-step actions (Ops, events with multiple target choices, headline selection, space races) are decomposed into fine-grained atomic steps (`DecisionType`).
3. **High Simulation Throughput**: Single-core simulation speed exceeds **2,000,000 steps/second**, meeting reinforcement learning and MCTS training requirements.
4. **Deterministic Bit-for-Bit State**: Built-in 64-bit SplitMix64 PRNG (`Prng`) ensures bit-for-bit replayability from integer seeds.
5. **Unified Sub-Decision Processing**: State machine handles sub-decisions (e.g. `SELECT_OP_MODE`, `POINT_NODE`, `CHOOSE_TIMING_BRANCH`, `CHOOSE_BRANCH`) uniformly across both `Phase::HEADLINE` and `Phase::ACTION_ROUND`.

---

## 2. Mandatory Documentation Maintenance Rule for Agents

> [!IMPORTANT]
> **Keep Engine Documentation Synchronized**:
> Whenever adding or modifying card handlers (`events/*.cpp`), updating `GameState`, adding tests in `tests/`, or adjusting fuzzer/sanitizer flags, you **MUST** update this file and root [`AGENTS.md`](file:///home/mihaild/prog/ts_ai/AGENTS.md).

---

## 3. Directory Structure

```
engine/
├── CMakeLists.txt              // Build target for ts_engine library, ts_tests, ts_fuzz, ts_benchmark
├── AGENTS.md                   // Developer and agent documentation (this file)
├── progress.md                 // Progress report and future work items
├── include/ts/                 // Public and internal engine headers
│   ├── types.hpp               // Enums: Player, Phase, WarEra, CardLocation, DecisionType, ActionType, Region, SubRegion, PlayMode, TimingBranch, OpMode
│   ├── constants.hpp           // Country IDs, Card IDs (1..110), 64-bit Effect Flags, Bit Masks
│   ├── micro_action.hpp        // 4-byte packed MicroAction struct (alignas(4))
│   ├── game_state.hpp          // Contiguous trivially copyable GameState struct
│   ├── prng.hpp                // Deterministic SplitMix64 PRNG
│   ├── map_data.hpp            // 84 countries static metadata, graph adjacency 128-bit bitmasks
│   ├── card_data.hpp           // 110 cards static metadata (Ops, side, era, asterisk, scoring)
│   ├── scoring.hpp             // Region scoring formulas (Presence/Domination/Control), victory evaluation
│   ├── ops.hpp                 // Influence placement, Coup mechanics, Realignment rolls, dynamic costs
│   ├── space_race.hpp          // Space race tracks, milestone rewards, and special abilities
│   ├── card_handlers.hpp       // Card event handlers and sub-decision dispatch declarations
│   ├── action_mask.hpp         // Legal action mask generator per DecisionType
│   ├── state_machine.hpp       // Turn and Action Round lifecycle, Headline resolution
│   ├── observation.hpp         // Neural observation feature extractor (ObservationBuffer)
│   ├── serialization.hpp       // Binary snapshot and JSON serialization
│   └── engine.hpp              // Top-level ts::Engine public interface
├── src/                        // Engine implementations
│   ├── map_data.cpp            // Static country array and adjacency lookup tables
│   ├── card_data.cpp           // Static card metadata array
│   ├── scoring.cpp             // Regional scoring and final scoring math
│   ├── ops.cpp                 // Ops execution and target validation
│   ├── space_race.cpp          // Space race advancement
│   ├── events/
│   │   ├── early_war.cpp       // Cards 1-35, 103-106 event handlers
│   │   ├── mid_war.cpp         // Cards 36-81, 107-108 event handlers
│   │   └── late_war.cpp        // Cards 82-102, 109-110 event handlers
│   ├── card_dispatcher.cpp     // Event trigger dispatch, sub-decision step router, action mask filters
│   ├── action_mask.cpp         // ActionMask::generate_mask implementation
│   ├── state_machine.cpp       // State machine turn loop, setup, headline, and AR transitions
│   ├── observation.cpp         // Feature extractor implementation
│   ├── serialization.cpp       // Serializer binary & JSON implementations
│   └── engine.cpp              // ts::Engine API implementation
└── tests/                      // Engine test suites
    ├── test_framework.hpp      // Lightweight assertion & test registry framework
    ├── game_test_wrapper.hpp   // High-level full game execution wrapper & policy harness
    ├── test_main.cpp           // Test runner
    ├── test_map.cpp            // Topology, battlegrounds count, adjacency symmetry tests
    ├── test_scoring.cpp        // Regional formulas, Europe control instant win tests
    ├── test_bugs_regression.cpp// Specific regression tests for card bugs
    ├── test_ops.cpp            // Dynamic cost drop, Coup DEFCON degradation, NATO tests
    ├── test_cards_early.cpp    // Early war card unit tests
    ├── test_cards_mid.cpp      // Mid war card unit tests
    ├── test_cards_late.cpp     // Late war card unit tests
    ├── test_defcon_suicide.cpp // DEFCON suicide priority in OPS_FIRST vs EVENT_FIRST
    ├── test_reentrancy.cpp     // Re-entrant DecisionContext stack tests
    ├── test_card_interactions.cpp // Multi-card interaction test suite
    ├── test_state_lifecycle.cpp// Turn, headline, and phase lifecycle tests
    ├── test_states.cpp         // Continuous state effects and modifier tests
    ├── test_card_edge_cases.cpp// Complete edge-case tests across all cards
    ├── test_full_game.cpp      // 10-turn full game integration tests ending in final scoring
    ├── test_fuzz.cpp           // Invariant fuzzer (--games <N>, --steps <N>, --seed <S>)
    └── test_benchmark.cpp      // 500k-step throughput benchmark
```

---

## 4. Micro-Decision Pipeline & Action Representation

The engine splits complex turns into a sequential stream of atomic 4-byte `MicroAction` structures:

```cpp
struct alignas(4) MicroAction {
    DecisionType decision_type; // 1 byte: SELECT_CARD, SELECT_PLAY_MODE, CHOOSE_TIMING_BRANCH, SELECT_OP_MODE, POINT_NODE, CHOOSE_BRANCH
    uint8_t      primary_id;    // 1 byte: Card ID (1..110), Country ID (0..83), Branch ID (0..7), PlayMode, OpMode, TimingBranch, or CONFIRM_DONE (0x80)
    uint8_t      secondary_id;  // 1 byte: Sub-choice / quantity / manual die roll
    uint8_t      flags;         // 1 byte: Additional modifiers / opponent manual die roll
};
```

---

## 5. How to Build, Test, and Benchmark

### Standard Build:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Build with Sanitizers (AddressSanitizer + UndefinedBehaviorSanitizer):
```bash
cmake -B build_san -S . \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build_san -j
```

### Run Unit Tests (275 tests):
```bash
./build/engine/ts_tests
# Or with sanitizers:
./build_san/engine/ts_tests
```

### Run Invariant Fuzzer:
```bash
# Run 10,000 full games:
./build/engine/ts_fuzz --games 10000

# Run 5,000,000 steps:
./build/engine/ts_fuzz --steps 5000000 --seed 42

# Under ASan + UBSan:
./build_san/engine/ts_fuzz --games 10000
```

### Run Performance Benchmark:
```bash
./build/engine/ts_benchmark
```

---

## 6. Guidelines for Extending the Engine

1. **Never Allocate Heap Memory in State Types**: Always ensure `static_assert(std::is_trivially_copyable_v<GameState>);` passes.
2. **Deterministic PRNG**: Use `Prng::roll_d6(state.rng_state)` or `Prng::random_index(state.rng_state, n)` whenever game state dice or shuffles are executed.
3. **Pass / Confirm Handling**: Multi-step cards (e.g. *Suez Crisis*, *Muslim Revolution*, *Independent Reds*, *Special Relationship*) must support early pass (`action.primary_id == 0` or `CONFIRM_DONE`) when no eligible targets remain.
4. **Always Update Tests When Changing Card Logic**: Add unit test cases in `engine/tests/` for any new card behaviors, interactions, or edge cases.
