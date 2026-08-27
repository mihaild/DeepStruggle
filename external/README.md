# External Engine Integrations & Differential Testing

This directory houses external reference engines and differential testing integrations for the Twilight Struggle simulation.

---

## 1. `external/struggler/`
* **Source**: Independent Rust/Python implementation of Twilight Struggle game logic.
* **Purpose**: Serves as an oracle and cross-engine differential validation baseline for all 110 cards, scoring rules, state machine edge cases, and board influence mechanics.
* **Differential Test Suites**:
  * [`tests/test_all_110_cards_differential.py`](../tests/test_all_110_cards_differential.py): 131 exhaustive cross-engine card verification tests.
  * [`tests/test_struggler_differential.py`](../tests/test_struggler_differential.py): Fuzzing and differential rollout comparison.
  * [`tests/struggler_adapter.py`](../tests/struggler_adapter.py): State and action translation adapter between `ts::GameState` and Struggler.

### Running Differential Tests
```bash
PYTHONPATH=.:external/struggler/src .venv/bin/pytest tests/test_all_110_cards_differential.py
```
