# Python Bindings & Environment Wrappers (`bindings/`)

This directory contains the Python native extension module (`ts_engine`) built with **`nanobind`**, along with the environment wrappers and action codec that bridge the C++20 simulation core with Python.

---

## 1. File Overview

- [`CMakeLists.txt`](CMakeLists.txt): Nanobind module build target configuring `ts_engine` compilation and linkage against `ts_engine_core`.
- [`ts_bindings.cpp`](ts_bindings.cpp): Complete nanobind module definition. Exposes:
  - Enums: `Player`, `Phase`, `DecisionType`, `Resolution`, `TimingBranch`, `OpMode`, `CardLocation`, `WarEra`, `Region`, `RegionalStatus`, `RollType`.
  - Structs: `MicroAction`, `CountryState`, `DecisionContext`, `GameState`.
  - Static Engine API: `init_game()`, `step()`, `step_flat()`, `try_step()`, `try_step_flat()`,
    `auto_advance_step()`, `is_terminal()`, `get_terminal_utility()`, `get_legal_action_mask()`, `get_flat_action_mask()`, `get_legal_action_indices()`, and the held-scoring predicates.
  - Neural Network & Vectorized RL: `extract_observation()`, `decode_flat_action()`, `encode_micro_action()`, `ActionMask`, `VectorizedBatchRunner` (contiguous multi-game simulation).
  - **P23 / E4.1 merged-influence view** (`engine/AGENTS.md` §4): `get_flat_action_mask(state, merged_influence=False)` (module and `Engine`), `ActionMask.generate_flat_mask(state, merged_influence=False)`, `Engine.step_flat` / `try_step_flat(..., merged_influence=False)`, and `ActionMask.is_merged_influence_action(state, idx)`. The runner holds the view **per env and side** -- `set_merged_influence(us, ussr)` and `set_merged_influence_env(i, us, ussr)` -- because a tournament seats an E4 and an E4.1 agent in one game; its cached mask is built for whoever decides next and a step is applied in the actor's view. Default everywhere is off, which is exactly E4. `bindings/ts_env.TsVectorizedEnv` mirrors the per-side flags so `reset_all(base_seed)`, which builds a new runner, keeps them.
  - `GameState.raw_bytes()`: the state's memory, for exact equality tests (it is trivially copyable).
  - Hand-knowledge helpers over `CardLocation`: `in_hand_of()`, `known_to_opponent()`, `hand_of()`, `hand_holder()`, `revealed()`, plus `reveal_hand()` / `reveal_both_hands()`. Use these instead of comparing a location to a hand constant — see [`engine/AGENTS.md`](../engine/AGENTS.md) §7.
  - Metadata helpers: `MapData`, `CardData`, `StateMachine`, `Operations`, `Scoring`, `CardHandlers`.
  - `EffectBits` submodule: the named 64-bit continuous-effect flags.
  - Dictionary serializer: `state_to_dict()` / `GameState.to_dict()` -- the display state.
  - JSON text of the same: `GameState.to_display_json()`, `GameState.to_save_json()` (byte for
    byte `json.dumps(to_save_dict(), sort_keys=True, separators=(",", ":"))`) and
    `state_from_save_json()`. `game_ending_reason(state)` names why a game ended
    (`tools/lib/tournament_evaluator.classify_game_ending_reason` delegates to it), and
    `selftest_digest(games)` is the whole-game digest the WebAssembly build must reproduce.
- [`state_json.hpp`](state_json.hpp) / [`state_json.cpp`](state_json.cpp): **the display state,
  the save and its loader, and the ending reason, written once** as a small dependency-free JSON
  value tree. The nanobind functions above convert the tree to Python objects; the WebAssembly
  build prints it. Two hand-kept copies would drift -- the browser showing or saving a position
  differently from Python -- which is why the conversions moved here out of `ts_bindings.cpp`.
- [`wasm/ts_engine_wasm.cpp`](wasm/ts_engine_wasm.cpp): the engine for the **browser workbench**,
  a flat C API compiled by Emscripten (`tools/scripts/build_web.sh`; under `emcmake` this
  directory builds `ts_engine_wasm` into `web/ui/public/engine/` instead of the Python module).
  New game, raw-bytes undo snapshots, display/save JSON, loading a save (refused unless it
  round-trips), stepping a MicroAction with the workbench's forced-die drain (all or nothing),
  masks in both action views, flat-action decode, observations, names, debug setters, the action
  layout and the engine fingerprint baked in at build.
- [`selftest.hpp`](selftest.hpp): one whole-game digest (full save, both observations, the mask,
  every position) compiled into both builds; `tests/web/test_wasm_engine.py` compares the numbers.
  - Save format: `GameState.to_save_dict()` / `state_from_save_dict()`, tagged `ts_save_v2`. Named
    fields, not a byte blob, so a save written by one build opens in another: an unknown key is
    ignored and a missing one keeps its default. It restores the board, the tracks, the card
    locations, the RNG, **and the decision state machine** (`ctx_stack`, `ctx_stack_depth`), plus
    the headline owners and the die-roll record.

    The decision stack is not optional and v1 omitted it, which is what the version bump marks.
    Without it a state saved mid-decision restored pointing at a *different* decision — measured at
    91 of 97 mid-game positions on one seed — while every visible board field matched, so nothing
    looked wrong. `engine/src/observation.cpp` reads `ctx()`, walks `ctx_stack` and encodes
    `ctx_stack_depth`, so the restored state also produced a different observation.

    Deliberately **not** saved: `action_history` and `turn_aggregates`. Both are written and never
    read — no hit in `observation.cpp`, and in `engine/src` only the two increments at
    `ops.cpp:322,412` — so they are diagnostics, and restoring them would grow the format to
    preserve a display detail. If a rule ever starts reading either, it must be added here.
- [`ts_engine/`](ts_engine/__init__.pyi): **Generated** type stub *package* — `__init__.pyi` plus one file per nanobind
  submodule (`EffectBits.pyi`) — for IDE completion and `pyrefly`. Regenerated by the build from the
  module just built, so it cannot drift from the bindings; never hand-edit it. To express something
  introspection cannot recover (the shape of a value returned as an untyped `nb::dict`, say), edit
  [`ts_engine.pyi.pattern`](ts_engine.pyi.pattern) and rebuild.
- [`action_encoder.py`](action_encoder.py): Bidirectional codec translating between the flat 212-dimensional policy action space and hierarchical `ts::MicroAction` engine structs.
- [`ts_env.py`](ts_env.py): Vectorized (`TsVectorizedEnv`) and single-game (`TsSingleEnv`) batch runner
**`step()` raises; `try_step()` returns a bool.** A refused action is a caller bug, not a game
event: since P14 `step` validates against the same mask a caller is expected to sample from, so a
refusal can only mean the action was chosen against a different state -- which is exactly the
staleness bug the env guard reports. `step()`/`step_flat()` therefore raise `RuntimeError` naming
what the engine was asking for, and `try_step()`/`try_step_flat()` keep the bool for the callers
that genuinely probe legality (the fuzzers, and mask-vs-step differential sweeps). Prefer
`tools.lib.game_step.step_checked`, which wraps `try_step*` with a richer diagnostic.

Flipping the default surfaced four Python tests that had been walking a game by feeding indices
from the 128-wide **per-decision** mask (`get_legal_action_indices`) to `step_flat`, which reads the
**flat 212-dim** space. Every step was refused, every refusal discarded, and the loops ran to their
iteration caps against a frozen state. One always hit a `pytest.skip` it could never escape, and
`test_dmcts`'s `_midgame_state` returned the opening position, so every determinization test was
checking a freshly dealt hand. The two mask spaces are not interchangeable; `get_flat_action_mask`
is the one `step_flat` reads.

  environments for PyTorch RL loops. `TsVectorizedEnv.step` returns an `info` dict whose
  `completed_episodes` entries carry:
  - `turn` — the terminal game turn, read before the auto-reset. Note it reads 11 for a completed
    game, because `finish_end_turn` increments before testing its bound.
  - `ply` — game length in the continuous player-slot numeration of
    [`ai/game_length.py`](../ai/game_length.py), where 1 is the USSR's turn-1 headline and 154 is a
    game that played all ten turns out. Prefer it to `turn`, which cannot separate a game abandoned
    at turn 7 AR1 from one that ran to turn 7 AR7.
  - `ending_reason` — one of `ENDING_REASON_KEYS` (`20vp`, `final_scoring`, `defcon1_self`,
    `defcon1_provoked`, `held_scoring`, `wargames`), classified via
    `tools.lib.tournament_evaluator.classify_game_ending_reason` (imported lazily to keep torch out
    of the env's import chain).

  The same values are exposed per-env as `info["terminal_turns"]` / `info["terminal_plies"]` /
  `info["ending_reasons"]`.

  **The runner's cached observation and mask.** `VectorizedBatchRunner` keeps one observation and
  mask per env, and three things keep them current:
  * `step_flat_all` rebuilds each env it steps;
  * `reset_game` rebuilds the env it resets;
  * `set_state` rebuilds nothing, so whoever calls it must refresh.

  `TsVectorizedEnv` refreshes only after `_apply_start_position` actually injects a start position
  through `set_state`. That applies in `step`'s auto-reset, in `reset_env` and in `reset_all`.
  Building an observation costs ~7.9 µs per env against ~0.8 µs for a game step. Refreshing all envs
  whenever any game ended doubled that cost on most steps, while `reset_env` and `reset_all`
  injected without any refresh. `tests/training/test_env_refresh.py` pins both halves.

---

## 2. Mandatory Documentation Maintenance Rule for Agents

> [!IMPORTANT]
> **Keep Python Bindings Documentation Synchronized**:
> Whenever adding or changing exported types, updating nanobind signatures, or altering serializer dictionaries, you **MUST** update this file and root [`AGENTS.md`](../AGENTS.md).

---

## 3. How to Compile & Test

**Threads: one OpenMP pool per process, torch's.** The batch runner's parallel loops go through
`gomp_parallel_for` in `ts_bindings.cpp`, which calls libgomp's `GOMP_parallel` directly, and
the extension links GCC's `libgomp.so.1` by exact file. Torch ships libgomp and the loader shares
it by soname, so the engine and torch use one runtime and one thread pool (and one
`OMP_WAIT_POLICY`, which `tools/train.py` and `tools/tournament.py` set to PASSIVE). Do **not** go back to
`#pragma omp`: under clang it links LLVM's libomp, a second pool of spinning workers beside
torch's (measured 5-7% slower on a rollout loop), and clang's `-fopenmp=libgomp` compiles the
loops *serially* without a warning. `tests/bindings/test_build_toolchain.py` catches all three:
a non-clang engine (`ts_engine.BUILD_COMPILER`), a thread pool added beside torch's, and loops
that do not spread across threads.

```bash
cmake -B build/release -S . -DPython_EXECUTABLE=$(pwd)/.venv/bin/python3   # clang, found by CMake
cmake --build build/release -j

# Run Python binding tests. Invoke pytest as a module, never via .venv/bin/pytest:
# that console script carries an absolute shebang and breaks if the venv is moved.
PYTHONPATH=.:build/release .venv/bin/python -m pytest -q tests/bindings
```
