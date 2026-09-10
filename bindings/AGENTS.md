# Python Bindings & Environment Wrappers (`bindings/`)

This directory contains the Python native extension module (`ts_engine`) built with **`nanobind`**, along with high-performance Python environment wrappers and action codecs that bridge the C++20 simulation core with Python.

---

## 1. File Overview

- [`CMakeLists.txt`](CMakeLists.txt): Nanobind module build target configuring `ts_engine` compilation and linkage against `ts_engine_core`.
- [`ts_bindings.cpp`](ts_bindings.cpp): Complete nanobind module definition. Exposes:
  - Enums: `Player`, `Phase`, `DecisionType`, `PlayMode`, `TimingBranch`, `OpMode`, `CardLocation`, `WarEra`.
  - Structs: `MicroAction`, `CountryState`, `DecisionContext`, `GameState`.
  - Static Engine API: `init_game()`, `step()`, `step_flat()`, `is_terminal()`, `get_terminal_utility()`, `get_legal_action_mask()`, `get_flat_action_mask()`, `get_legal_action_indices()`.
  - Neural Network & Vectorized RL: `extract_observation()`, `get_flat_action_mask()`, `decode_flat_action()`, `encode_micro_action()`, `ActionMask`, `VectorizedBatchRunner` (contiguous multi-game simulation).
  - Metadata helpers: `MapData`, `CardData`.
  - Dictionary serializer: `state_to_dict()` for full zero-copy state inspection.
- [`ts_engine.pyi`](ts_engine.pyi): Python type stub declarations for IDE completions and static type checkers (`pyrefly`).
- [`action_encoder.py`](action_encoder.py): Bidirectional codec translating between the flat 212-dimensional policy action space and hierarchical `ts::MicroAction` engine structs.
- [`ts_env.py`](ts_env.py): Vectorized (`TsVectorizedEnv`) and single-game (`TsSingleEnv`) batch runner environments interfacing with PyTorch RL loops. `TsVectorizedEnv.step` returns an `info` dict whose `completed_episodes` entries carry `turn` (terminal game turn, read before the auto-reset) and `ending_reason` — one of `ENDING_REASON_KEYS` (`20vp`, `final_scoring`, `defcon1_self`, `defcon1_provoked`, `held_scoring`, `wargames`), classified via `tools.lib.tournament_evaluator.classify_game_ending_reason` (imported lazily to keep torch out of the env's import chain). Each entry also carries `ply` — the game's length in the continuous player-slot numeration defined by [`ai/game_length.py`](../ai/game_length.py), where 1 is the USSR's turn-1 headline and 154 is a game that played all ten turns out. `turn` alone cannot express length: it cannot separate a game abandoned at turn 7 AR1 from one that ran to turn 7 AR7, and it reads 11 for a completed game because `finish_end_turn` increments before testing its bound. The same values are exposed per-env as `info["terminal_turns"]` / `info["terminal_plies"]` / `info["ending_reasons"]`.

---

## 2. Mandatory Documentation Maintenance Rule for Agents

> [!IMPORTANT]
> **Keep Python Bindings Documentation Synchronized**:
> Whenever adding or changing exported types, updating nanobind signatures, or altering serializer dictionaries, you **MUST** update this file and root [`AGENTS.md`](../AGENTS.md).

---

## 3. How to Compile & Test

### Standard Build:
```bash
cmake -B build/release -S . -DPython_EXECUTABLE=$(pwd)/.venv/bin/python3
cmake --build build/release -j

# Run Python binding tests
PYTHONPATH=. .venv/bin/pytest -v tests/bindings/
```
