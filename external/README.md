# External Engine Integrations & Differential Testing

This directory houses external reference engines and differential testing integrations for the Twilight Struggle simulation.

---

## 1. `external/struggler/`
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

## 2. `external/ts-blockchain/`
* **Source**: Open-source JavaScript implementation of Twilight Struggle written for the Saito Game Engine module (`https://github.com/trevelyan/ts-blockchain`).
* **Purpose**: Cross-engine differential validation baseline for action legality, influence placement, coups, realignments, space race, regional scoring, all 110 card events, and persistent game effects.
* **Adapter & Differential Testing**:
  * [`tests/differential/blockchain_bridge.js`](../tests/differential/blockchain_bridge.js): Headless Node.js bridge executing `twilight.js` over stdio JSON-RPC.
  * [`tests/differential/blockchain_adapter.py`](../tests/differential/blockchain_adapter.py): State, action, and bridge adapter between `ts::GameState` and `ts-blockchain`.
  * [`tests/differential/test_unified_differential.py`](../tests/differential/test_unified_differential.py): Cross-engine action legality and state resolution differential test suite.

### Running Blockchain Differential Tests
```bash
PYTHONPATH=.:build/release .venv/bin/pytest --run-fuzz tests/differential/test_unified_differential.py -k blockchain
```

---

## 3. Cross-Engine Differential Fuzzing (`tests/differential/test_differential_fuzzing.py`)
* **Purpose**: Side-by-side fuzzing tests cross-validating `ts_ai` against **both** external implementations (`struggler` and `ts-blockchain`) simultaneously.
* **Coverage**:
  * Randomized board topologies & influence across 84 countries.
  * DEFCON levels (2..5) and regional restriction enforcement.
  * Tripartite action legality parity (coups and realignments).
  * Regional scoring arithmetic fuzzing across all regions.
  * Randomized multi-step action trajectories (placements, coups, realignments).

### Running All Differential & Fuzzing Tests
```bash
PYTHONPATH=.:build/release:external/struggler/src .venv/bin/pytest --run-fuzz tests/differential/
```
