# Bot Client & AI Agent Guide

This directory contains standalone bot clients that connect to the game server over WebSocket, receive game states, evaluate legal actions, and play moves.

---

## 1. File Overview

- [`bot_client.py`](file:///home/mihaild/prog/ts_ai/bot/bot_client.py):
  - `BaseBot`: Abstract base class defining `select_action(state, legal_actions) -> dict`.
  - `RandomBot`: Selects uniformly random valid actions from `legal_actions['valid_ids']`. Supports early pass (`0x80`).
  - `HeuristicBot`: Rule-based baseline bot:
    - Prioritizes East Germany/Poland for USSR and West Germany/Italy for US during Setup.
    - Plays Scoring cards when held, or highest Ops cards.
    - Selects friendly events or uses Ops for influence/coups.
    - Prioritizes battlegrounds in current regions.
  - `run_bot_client()`: Async WebSocket client loop connecting to `ws://localhost:8000/ws/game/{id}?role={role}`.

---

## 2. Implementing New Bot Strategies

To add a new bot strategy (e.g. MCTS bot, neural policy bot):

1. Inherit from `BaseBot`:
   ```python
   class MCTSBot(BaseBot):
       def select_action(self, state: dict, legal_actions: dict) -> dict:
           valid_ids = legal_actions.get("valid_ids", [])
           d_type = legal_actions.get("decision_type", 0)
           # Perform search or policy evaluation ...
           return {
               "decision_type": d_type,
               "primary_id": best_id,
               "secondary_id": 0,
               "flags": 0
           }
   ```
2. Register the bot strategy in `bot_client.py` CLI parser.

---

## 3. How to Run

```bash
# Play against Heuristic Bot on game-1 as USSR
PYTHONPATH=. .venv/bin/python -m bot.bot_client --game-id game-1 --role USSR --type heuristic

# Play with artificial delay (e.g. 0.5s per step for human observation)
PYTHONPATH=. .venv/bin/python -m bot.bot_client --game-id game-1 --role USSR --delay 0.5
```
