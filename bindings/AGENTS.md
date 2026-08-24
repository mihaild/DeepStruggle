# Python Bindings (nanobind) Developer Guide

This directory contains the Python native extension module (`ts_engine`) built with **`nanobind`**. It exposes the C++20 `ts::Engine` and `ts::GameState` directly to Python with minimal overhead.

---

## 1. File Overview

- [`ts_bindings.cpp`](file:///home/mihaild/prog/ts_ai/bindings/ts_bindings.cpp): Complete nanobind module definition. Exposes:
  - Enums: `Player`, `Phase`, `DecisionType`, `PlayMode`, `TimingBranch`, `OpMode`, `CardLocation`, `WarEra`.
  - Structs: `MicroAction`, `CountryState`, `DecisionContext`, `GameState`.
  - Static Engine API: `init_game()`, `step()`, `step_flat()`, `is_terminal()`, `get_terminal_utility()`, `get_legal_action_mask()`, `get_flat_action_mask()`, `get_legal_action_indices()`.
  - Neural Network & Vectorized RL: `extract_observation()`, `get_flat_action_mask()`, `decode_flat_action()`, `encode_micro_action()`, `ActionMask`, `VectorizedBatchRunner` (contiguous multi-game simulation).
  - Metadata helpers: `MapData`, `CardData`.
  - Dictionary serializer: `state_to_dict()` for full zero-copy state inspection.

---

## 2. Mandatory Documentation Maintenance Rule for Agents

> [!IMPORTANT]
> **Keep Python Bindings Documentation Synchronized**:
> Whenever adding or changing exported types, updating nanobind signatures, or altering serializer dictionaries, you **MUST** update this file and root [`AGENTS.md`](file:///home/mihaild/prog/ts_ai/AGENTS.md).

---

## 3. Key Rules for Maintaining Bindings

1. **Enum Arithmetic**:
   Always pass `nb::is_arithmetic()` when registering enums (e.g. `nb::enum_<ts::DecisionType>(m, "DecisionType", nb::is_arithmetic())`). This ensures Python code can cast to `int` and compare with integers directly.
2. **Reference vs Copy for Nested State**:
   When exposing methods returning internal state references (like `state.ctx()`), specify `nb::rv_policy::reference` to avoid unnecessary copies.
3. **Array and Container Conversions**:
   Include `<nanobind/ndarray.h>`, `<nanobind/stl/string.h>`, `<nanobind/stl/vector.h>`, `<nanobind/stl/array.h>` when converting between C++ containers and Python lists/tuples/numpy arrays.

---

## 4. How to Compile & Test

### Standard Build:
```bash
cmake -B build -S . -DPython_EXECUTABLE=$(pwd)/.venv/bin/python3
cmake --build build -j

# Run Python binding tests (outputs to build/ts_engine.so)
PYTHONPATH=build:. .venv/bin/pytest -v tests/test_bindings.py
```
