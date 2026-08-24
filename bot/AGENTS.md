# Bot Clients & AI Agents Guide

This directory contains the bot clients, interactive agent players, and evaluation harnesses for **Twilight Struggle**.

---

## 1. Overview of Bot Clients

- [`bot_client.py`](file:///home/mihaild/prog/ts_ai/bot/bot_client.py): Unified WebSocket bot runner supporting:
  - `RandomBot`: Baseline stochastic player.
  - `HeuristicBot`: Rule-based baseline player prioritizing battlegrounds and scoring cards.
  - `NeuralBot`: Deep reinforcement learning agent driven by `ColdWarNet` (NashPG / Behavioral Cloning).
- [`neural_bot.py`](file:///home/mihaild/prog/ts_ai/bot/neural_bot.py): Neural network player client that loads trained PyTorch weights and selects actions via masked inference.
- [`agent_player.py`](file:///home/mihaild/prog/ts_ai/bot/agent_player.py): Rich terminal CLI interface with color-coded map and card inspections.

---

## 2. Launching Bots

### Playing Against NeuralBot via Web Workbench:
```bash
# 1. Start server
PYTHONPATH=. .venv/bin/python -m uvicorn server.main:app --host 0.0.0.0 --port 8000

# 2. Launch NeuralBot for USSR
PYTHONPATH=. .venv/bin/python -m bot.bot_client --game-id game-1 --role USSR --type neural --model-path checkpoints/coldwar_net.pt

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
> Whenever adding new bot strategies, changing CLI options, or updating connection parameters, you **MUST** update this file and root [`AGENTS.md`](file:///home/mihaild/prog/ts_ai/AGENTS.md).
