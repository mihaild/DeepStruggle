# Bot Clients & AI Agents Guide (`bot/`)

This directory contains bot client implementations, interactive agent interfaces, and baseline heuristics for **Twilight Struggle**.

---

## 1. Overview of Bot Clients

- [`base_bot.py`](base_bot.py): Abstract `BaseBot` class defining the standard interface for Twilight Struggle agents:
  - `select_action(state_dict, legal_actions_dict) -> Optional[dict]`: WebSocket/REST dict-based action selection.
  - `select_flat_action(state, player) -> int`: Fast 212-dim flat index selection for vectorized environments.
  - `reset()`: Resets internal memory or search state between games.
- [`random_bot.py`](random_bot.py): `RandomBot` baseline stochastic player making uniform random choices over legal moves.
- [`heuristic_bot.py`](heuristic_bot.py): `HeuristicBot` rule-based expert player prioritizing key battlegrounds, scoring card timing, and ops efficiency.
- [`neural_bot.py`](neural_bot.py): `NeuralBot` deep reinforcement learning player driven by `ColdWarNet` PyTorch checkpoints.
- [`bot_client.py`](bot_client.py): WebSocket network client for connecting bots to live game sessions on the server.
- [`agent_player.py`](agent_player.py): Interactive terminal player with color-coded map display and manual move selection.

> [!NOTE]
> Standalone game generation and auditing scripts (`generate_event_heavy_game.py`, `generate_strategic_game.py`, `play_and_audit_full_game.py`, etc.) have been moved to [`tools/`](../tools/).

---

## 2. Launching Bots

### Playing Against NeuralBot via Web Workbench:
```bash
# 1. Start server
PYTHONPATH=. .venv/bin/python -m uvicorn web.server.main:app --host 0.0.0.0 --port 8000

# 2. Launch NeuralBot for USSR
PYTHONPATH=. .venv/bin/python -m bot.bot_client --game-id game-1 --role USSR --type neural --model-path data/checkpoints/run_v3_20260827_205207/snapshot_3602s.pt

# 3. Open browser at http://localhost:8000/?game_id=game-1&role=US
```

### Playing Against HeuristicBot:
```bash
PYTHONPATH=. .venv/bin/python -m bot.bot_client --game-id game-1 --role USSR --type heuristic
```

---

## 3. Mandatory Documentation Maintenance Rule for Agents

> [!IMPORTANT]
> **Keep Bot Documentation Synchronized**:
> Whenever adding new bot strategies, changing CLI options, or updating connection parameters, you **MUST** update this file and root [`AGENTS.md`](../AGENTS.md).
