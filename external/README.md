# External Engine Integrations & Differential Testing

This directory houses external reference engines and differential testing integrations for the Twilight Struggle simulation.

---

## `external/struggler/`
* **Source**: Independent Rust/Python implementation of Twilight Struggle game logic (`https://github.com/alekpinel/struggler`).
* **Purpose**: Serves as an oracle and cross-engine differential validation baseline for all 110 cards, scoring rules, state machine edge cases, and board influence mechanics.
* **Differential Test Suites**:
  * [`tests/differential/test_unified_differential.py`](../tests/differential/test_unified_differential.py): Parameterized cross-engine mechanics and card verification suite.
  * [`tests/differential/struggler_adapter.py`](../tests/differential/struggler_adapter.py): State and action translation adapter between `ts::GameState` and Struggler.

### Running Struggler Differential Tests
```bash
PYTHONPATH=.:build/release:external/struggler/src .venv/bin/pytest --run-fuzz tests/differential/test_unified_differential.py -k struggler
```

---

## Cross-Engine Differential Fuzzing (`tests/differential/test_differential_fuzzing.py`)
* **Purpose**: Side-by-side fuzzing tests cross-validating `ts_ai` against the external implementation.
* **Coverage**:
  * Randomized board topologies & influence across 84 countries.
  * DEFCON levels (2..5) and regional restriction enforcement.
  * Action legality parity (coups and realignments).
  * Regional scoring arithmetic fuzzing across all regions.
  * Randomized multi-step action trajectories (placements, coups, realignments).

### Running All Differential & Fuzzing Tests
```bash
PYTHONPATH=.:build/release:external/struggler/src .venv/bin/pytest --run-fuzz tests/differential/
```

---

## Status

**These suites are work in progress and are not part of the check a change is expected to pass.**
They are excluded from collection (`tests/conftest.py`) because the modules fail at *import*, and
`pyrefly.toml` excludes them too. `CLAUDE.md` says not to spend time reviving them.

A second reference engine, `ts-blockchain` (a JavaScript implementation for the Saito Game
Engine), was integrated here and has been removed: its adapter, its Node.js bridge and its
submodule were carried through every refactor for a suite nobody runs.
