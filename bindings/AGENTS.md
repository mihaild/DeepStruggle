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
- [`ts_env.py`](ts_env.py): Vectorized (`TsVectorizedEnv`) and single-game (`TsSingleEnv`) batch runner environments interfacing with PyTorch RL loops.

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
PYTHONPATH=. .venv/bin/pytest -v tests/test_bindings.py
```
