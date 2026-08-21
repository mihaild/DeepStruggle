# FastAPI Game Server & Replay System Guide

This directory contains the backend server for Twilight Struggle, providing real-time game orchestration, WebSocket communication, REST endpoints, and replay file recording.

---

## 1. File Overview

- [`main.py`](file:///home/mihaild/prog/ts_ai/server/main.py):
  - FastAPI application entry point.
  - REST endpoints:
    - `POST /api/games/new`: Initializes a new game session.
    - `GET /api/games/{game_id}`: Fetches current full game state.
    - `GET /api/replays`: Lists all saved `.tslog.json` replay files.
    - `GET /api/replays/{filename}`: Loads a specific replay file.
    - `GET /api/metadata/map`: Returns Deluxe 84-country map data.
    - `GET /api/metadata/cards`: Returns all 110 cards metadata.
  - WebSocket endpoint: `/ws/game/{game_id}?role=US|USSR|OBSERVER`.
  - Static file mounting: Serves `frontend/dist` when compiled.

- [`session.py`](file:///home/mihaild/prog/ts_ai/server/session.py):
  - `GameSession` class managing the `ts_engine.GameState` instance.
  - Validates incoming `PLAY_ACTION` messages against the active `DecisionContext`.
  - Executes `ts_engine.Engine.step()` and generates human-readable event descriptions.
  - Broadcasts `STATE_UPDATE` JSON payloads to all connected WebSocket clients.
  - Supports `DEBUG_OVERRIDE` actions for manual influence/DEFCON/VP manipulation.

- [`replay.py`](file:///home/mihaild/prog/ts_ai/server/replay.py):
  - `ReplayLogger`: Appends micro-actions, turn/AR milestones, and state snapshots.
  - `ReplayManager`: Discovers and loads `.tslog.json` files from `replays/`.

---

## 2. WebSocket Protocol Specification

### Client $\to$ Server:
```json
{
  "type": "PLAY_ACTION",
  "action": {
    "decision_type": 5,
    "primary_id": 14,
    "secondary_id": 0,
    "flags": 0
  }
}
```

### Server $\to$ Client:
```json
{
  "type": "STATE_UPDATE",
  "state": {
    "turn": 1,
    "action_round": 1,
    "defcon": 5,
    "victory_points": 0,
    "phasing_player": "USSR",
    "decision_context": { ... },
    "legal_actions": { ... },
    "countries": { ... },
    "hands": { ... },
    "action_logs": [ ... ]
  }
}
```

---

## 3. How to Run & Test

```bash
# Start server
PYTHONPATH=. .venv/bin/python -m uvicorn server.main:app --host 0.0.0.0 --port 8000

# Run server integration tests
PYTHONPATH=. .venv/bin/pytest -v tests/test_server_and_bot.py
```
