# External Engine Integrations & Differential Testing

This directory houses external reference engines and differential testing integrations for the Twilight Struggle simulation.

---

## 1. `external/struggler/`
* **Source**: Independent Rust/Python implementation of Twilight Struggle game logic (`https://github.com/alekpinel/struggler`).
* **Purpose**: Serves as an oracle and cross-engine differential validation baseline for all 110 cards, scoring rules, state machine edge cases, and board influence mechanics.
* **Differential Test Suites**:
  * [`tests/test_all_110_cards_differential.py`](../tests/test_all_110_cards_differential.py): 131 exhaustive cross-engine card verification tests.
  * [`tests/test_struggler_differential.py`](../tests/test_struggler_differential.py): Core mechanics, setup, reachability, coups, and scoring.
  * [`tests/struggler_adapter.py`](../tests/struggler_adapter.py): State and action translation adapter between `ts::GameState` and Struggler.

### Running Struggler Differential Tests
```bash
PYTHONPATH=.:build/release:external/struggler/src .venv/bin/pytest tests/test_all_110_cards_differential.py tests/test_struggler_differential.py
```

---

## 2. `external/ts-blockchain/`
* **Source**: Open-source JavaScript implementation of Twilight Struggle written for the Saito Game Engine module (`https://github.com/trevelyan/ts-blockchain`).
* **Purpose**: Cross-engine differential validation baseline for action legality, influence placement, coups, realignments, space race, regional scoring, all 110 card events, and persistent game effects.
* **Adapter & Differential Testing**:
  * [`tests/blockchain_bridge.js`](../tests/blockchain_bridge.js): Headless Node.js bridge executing `twilight.js` over stdio JSON-RPC.
  * [`tests/blockchain_adapter.py`](../tests/blockchain_adapter.py): State, action, and bridge adapter between `ts::GameState` and `ts-blockchain`.
  * [`tests/test_all_110_cards_blockchain.py`](../tests/test_all_110_cards_blockchain.py): 177 exhaustive tests for all 110 card events, mechanics, and 47 persistent effect flags.
  * [`tests/test_blockchain_differential.py`](../tests/test_blockchain_differential.py): Cross-engine action legality and state resolution differential test suite.

### Running Blockchain Differential Tests
```bash
PYTHONPATH=.:build/release .venv/bin/pytest tests/test_all_110_cards_blockchain.py tests/test_blockchain_differential.py
```

---

## 3. Cross-Engine Differential Fuzzing (`tests/test_differential_fuzzing.py`)
* **Purpose**: Side-by-side fuzzing tests cross-validating `ts_ai` against **both** external implementations (`struggler` and `ts-blockchain`) simultaneously.
* **Coverage**:
  * Randomized board topologies & influence across 84 countries.
  * DEFCON levels (2..5) and regional restriction enforcement.
  * Tripartite action legality parity (coups and realignments).
  * Regional scoring arithmetic fuzzing across all regions.
  * Randomized multi-step action trajectories (placements, coups, realignments).

### Running All 430 Differential & Fuzzing Tests
```bash
PYTHONPATH=.:build/release:external/struggler/src .venv/bin/pytest -v \
  tests/test_all_110_cards_differential.py \
  tests/test_all_110_cards_blockchain.py \
  tests/test_blockchain_differential.py \
  tests/test_differential_fuzzing.py \
  tests/test_struggler_differential.py
```
