# FastAPI Game Server & Replay System Guide (`web/server/`)

This directory contains the backend server for Twilight Struggle, providing real-time game orchestration, WebSocket communication, REST metadata endpoints, and replay file management.

---

## Stepping the engine

`GameSession.handle_action` does not step the engine directly. It goes through
`tools.lib.game_step`: `step_checked` (which raises `IllegalActionError` instead of returning a
`False` somebody will discard) and then `drain_chance`, which resolves the chance nodes the action
landed on. That is the same pair `GameLoop` uses for `play_match` and `self_play`, and the three
have to agree or a replay written by one cannot be re-driven by another.

Two rules follow:

* **Never re-implement the drain.** The session used to own a copy that looped `while` the game sat
  on a chance node, discarding the engine's return value, so a refused roll spun forever. A client
  sending an out-of-range `secondary_id` was enough to hang the request.
* **`secondary_id` on the action that reaches a chance node is the manual die** (the UI's
  `selectedDieRoll`: 0 for auto, 1..6 to force). It is a workbench affordance for testing the
  engine and is passed through as `drain_chance(forced_die=...)`. Anything outside 0..6 is refused
  by the engine, and the handler answers `False` after rolling back.

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
  - Optional per-step **trace**: `policy` (what the model believed at that node) and `critic`
    (both value heads from both perspectives on the state the step's snapshot shows), plus
    `metadata.trace` naming the model, engine build and settings that produced them. Written by
    `tools/lib/self_play.py` and `tools/play_match.py --trace`, and after the fact by
    `tools/annotate_replay.py`; see `ai/eval/policy_readout.py`. `policy.top` lists **every**
    legal action by default (`trace_top_k=0`), because the workbench paints each probability
    onto the card, mode button or country it belongs to. **Every reader must treat both
    keys as absent by default** — a heuristic bot has no distribution, a human game has no
    model, and replays predating the trace have neither. The live server never puts a trace in a
    `STATE_UPDATE` unasked: a distribution over a bot's legal actions is a read on its hand,
    which is what the per-role `observation_b64` exists to withhold. The one exception is the
    opt-in live analysis below, sent only to the socket that requested it.
  - `start_position` in the metadata marks a game begun from a loaded position (a shared link)
    rather than from the seed; such a replay cannot be re-driven from the seed alone.

- [`analysis.py`](analysis.py): **live model analysis** for the workbench.
  - `list_models()` / `resolve_model_path()`: checkpoints are named relative to the checkpoints
    tree (`$TS_CHECKPOINTS_DIR`, else the shared `data/checkpoints`); `resume_*.pt` and anything
    outside the tree are refused, so a URL cannot point the server at an arbitrary file.
  - `MODEL_CACHE`: a few `NeuralAgent`s loaded via `NeuralAgent.from_checkpoint`, which detects
    the architecture from the weights and the **action view** (E4 / E4.1 merged-influence) from
    the run directory. Device from `$TS_ANALYSIS_DEVICE`, default `cpu` (a position costs a few
    ms; the GPU usually belongs to a training run).
  - `analyze()`: the policy over every legal action **in the model's own view** (`read_policy`,
    `deterministic=True`, so the torch RNG is untouched) and the critic from both perspectives
    (`read_critic`). Each choice carries the MicroAction a click would send, so the UI attaches
    probabilities by matching what a button/card/country sends, never by flat offsets. A
    *composed* E4.1 choice ("Ops → Influence, first point in X") names the commit half.
  - `flat_to_steps()`: how "play favourite" applies a flat action. A composed E4.1 action is
    applied as the two E4 steps it is defined as (as `Engine::step_flat` composes it), each
    logged and undoable like a click; half of one is never left applied.
  - `encode_position()` / `decode_position()`: the address-bar token — `to_save_dict` JSON,
    zlib, base64url, ~1 KB mid-game. A token that does not round-trip through
    `state_from_save_dict` exactly is refused rather than opened approximately.

  WebSocket messages (in `main.py`): `SET_ANALYSIS_MODEL {model: <rel path> | null}` opts this
  socket in or out (errors come back as `ANALYSIS_ERROR {message}`); every later `STATE_UPDATE`
  to that socket carries `analysis` (a `LiveAnalysisDict`) for the same position as its `state`,
  and `analysis_model`. `PLAY_FLAT {flat_idx, forced_die, expect_position?}` plays a flat action
  in the socket's model's view; with `expect_position` it is ignored unless the game still holds
  that position (auto-play, a click and a second tab can all race for the same node). Every `state` now carries `position`, the token for the board it describes.
  `GameSession.version` is bumped on every change of position; a readout is cached per
  (model, version) and a superseded broadcast is dropped rather than sent late.

- **`POST /api/games/{game_id}/position`** `{position}`: put a shared position on the board.
  Returns `changed: false` (and does nothing) when the game already holds exactly that
  position, which is what lets a page reload keep the game's undo history; otherwise the
  history and log restart there. A malformed token is a 400.
- **`GET /api/analysis/models`**: every loadable checkpoint, grouped by run, newest run first.
  - `ReplayManager`: Discovers and loads those files.
  - `replays_dir()` resolves the location per call — `$TS_REPLAYS_DIR` if set, else `data/replays`
    — so a test fixture can point the server at a directory it has just generated a replay into.
    Prefer it to the `REPLAYS_DIR` constant, which cannot see an override set after import.

- **`GET /api/metadata/action_space`** (in [`main.py`](main.py)): the 212-dim flat action space
  — offsets, `confirm_done_index`, and the `DecisionType` ids — served from `ActionEncoder` and
  the engine enum. The workbench needs it to turn a trace's flat index back into the card, mode
  or country it refers to. **Do not copy these offsets into the frontend**: a second hand-kept
  table would eventually disagree with the encoder, and every probability would then be painted
  on the wrong thing while still looking plausible. The mapping is pinned by
  `tests/web/test_action_space_metadata.py`, which replays a real game through it.

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
