# FastAPI Game Server & Replay System Guide (`web/server/`)

This directory contains the backend server for Twilight Struggle, providing real-time game orchestration, WebSocket communication, REST metadata endpoints, and replay file management.

---

## 1. File Overview

- [`main.py`](main.py):
  - FastAPI application entry point.
  - REST endpoints:
    - `POST /api/games/new`: Initializes a new game session with optional RNG seed.
    - `GET /api/games/{game_id}`: Fetches current full game state dictionary.
    - `POST /api/games/{game_id}/cancel_action` and `POST /api/games/{game_id}/undo`: Both call
      `GameSession.handle_cancel_action`, discarding the in-progress decision and returning the
      restored state.
    - `GET /api/replays`: Lists the saved `.tslog.json` replay logs; appending a filename to that
      path loads one, with its full state snapshots.
    - `GET /api/metadata/map`: Returns Deluxe 84-country map data and regional topology.
    - `GET /api/metadata/cards`: Returns all 110 cards metadata (Ops, sides, eras, rules text).
  - WebSocket endpoint: `/ws/game/{game_id}?role=US|USSR|OBSERVER`.
  - Static file mounting: Serves compiled `web/ui/dist` on `/`.

- [`session.py`](session.py):
  - `GameSession` class managing the `ts_engine.GameState` instance.
  - Validates incoming `PLAY_ACTION` messages against the active `DecisionContext`.
  - Executes `ts_engine.Engine.step()` and generates human-readable event logs.
  - Broadcasts `STATE_UPDATE` JSON payloads to all connected WebSocket clients.
  - Supports `DEBUG_OVERRIDE` actions for manual influence/DEFCON/VP manipulation.

- [`replay.py`](replay.py):
  - `ReplayLogger`: Appends micro-actions, turn/AR milestones, and state snapshots as `.tslog.json`.
  - `ReplayManager`: Discovers and loads those files.
  - `replays_dir()` resolves the location per call — `$TS_REPLAYS_DIR` if set, else `data/replays`
    — so a test fixture can point the server at a directory it has just generated a replay into.
    Prefer it to the `REPLAYS_DIR` constant, which cannot see an override set after import.

- [`replay_types.py`](replay_types.py):
  - Strongly typed `TypedDict` definitions for all serialized game states, action logs, audit structures, and WebSocket envelopes.

---

## 2. Mandatory Documentation Maintenance Rule for Agents

> [!IMPORTANT]
> **Keep Server Documentation Synchronized**:
> Whenever adding or altering REST endpoints, changing WebSocket message schemas, updating session routing, or changing replay formats, you **MUST** update this file and root [`AGENTS.md`](../../AGENTS.md).

---

## 3. How to Run & Test

```bash
# Start server
PYTHONPATH=. .venv/bin/python -m uvicorn web.server.main:app --host 0.0.0.0 --port 8000

# Run server and bot integration tests. Invoke pytest as a module, never via .venv/bin/pytest:
# that console script carries an absolute shebang and breaks if the venv is moved.
# tests/web additionally needs a built UI bundle (cd web/ui && npm run build) and, for the E2E
# tests, a Playwright Chromium. It is not the suite to run for engine, bindings, AI or tools work.
PYTHONPATH=. .venv/bin/python -m pytest -q tests/web/test_server_and_bot.py tests/web/test_web_workbench.py
```
