# Bot Clients & AI Agents Guide (`bot/`)

This directory contains pure bot client implementations, baseline heuristics, and decision policies for **Twilight Struggle**. All bots inherit from [`BaseBot`](base_bot.py).

---

## 1. Overview of Bot Clients

- [`base_bot.py`](base_bot.py): Abstract `BaseBot` class defining the standard interface for Twilight Struggle agents:
  - `select_action(state_dict, legal_actions_dict) -> Optional[dict]`: Dict-based action selection for WebSocket / JSON servers.
  - `select_flat_action(state, player) -> int`: Fast 212-dim flat index selection for vectorized environments.
  - `reset()`: Resets internal memory or search state between games.
- [`random_bot.py`](random_bot.py): `RandomBot` baseline stochastic player making uniform random choices over legal moves.
- [`heuristic_bot.py`](heuristic_bot.py): `HeuristicBot` rule-based expert player prioritizing key battlegrounds, scoring card timing, and ops efficiency.
- [`neural_bot.py`](neural_bot.py): `NeuralBot` deep reinforcement learning player driven by `ColdWarNet` PyTorch checkpoints (auto-detects V1, V2, and V3 architectures).
- [`exploratory_bot.py`](exploratory_bot.py): `ExploratoryBot` agent designed to explore diverse decision paths, Space Race, Realignments, and Coups.
- [`strategic_bot.py`](strategic_bot.py): `StrategicBot` high-level strategic agent prioritizing DEFCON-2 containment, coups, realignments, and Space Race safety with rich strategy/commentary generation.
- [`event_heavy_bot.py`](event_heavy_bot.py): `EventHeavyBot` agent maximizing card event play and event-first timing.
- [`human_bot.py`](human_bot.py): `HumanBot` interactive terminal CLI player prompting the user for numbered choices.

---

## 2. Launching Bots

### Playing Against Bots via Web Workbench:
The WebSocket client is located in [`web/bot_client.py`](../web/bot_client.py):
```bash
# 1. Start backend server
PYTHONPATH=. .venv/bin/python -m uvicorn web.server.main:app --host 0.0.0.0 --port 8000

# 2. Launch NeuralBot for USSR
PYTHONPATH=. .venv/bin/python -m web.bot_client --game-id game-1 --role USSR --type neural --model-path data/checkpoints/.../snapshot_3602s.pt

# 3. Open browser at http://localhost:8000/?game_id=game-1&role=US
```

### Running Offline Matches Between Bots:
Matches can be run offline with full replay generation via [`tools/play_match.py`](../tools/play_match.py):
```bash
# Run a match between HeuristicBot and StrategicBot
PYTHONPATH=. .venv/bin/python tools/play_match.py --us heuristic --ussr strategic
```

---

## 3. Mandatory Documentation Maintenance Rule for Agents

> [!IMPORTANT]
> **Keep Bot Documentation Synchronized**:
> Whenever adding new bot strategies or modifying bot interfaces, you **MUST** update this file and root [`AGENTS.md`](../AGENTS.md).
