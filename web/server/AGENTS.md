# Local Files Server & Replay System Guide (`web/server/`)

The workbench runs **in the browser** (see [`web/ui/AGENTS.md`](../ui/AGENTS.md)): the engine is
the WebAssembly build of `engine/` + `bindings/`, the game session and the model readout are
TypeScript, and models run as ONNX in onnxruntime-web. The same page is published to GitHub Pages
with no server at all. This directory is what a *local* checkout adds on top: the files a static
page cannot reach -- this machine's checkpoints and replays -- and the replay writer the Python
tools use.

It holds **no game**: no sessions, no WebSocket, no bot clients. The game lives in
`web/ui/src/game/` and `web/ui/src/analysis/` (the action log is pinned by a golden recorded from
the old Python session, the readout against `read_policy` / `read_critic`).

---

## 1. File Overview

- [`main.py`](main.py): the FastAPI app.
  - `GET /api/local/info` -- the engine fingerprint of the sources here, and the roots. The page
    compares it with the fingerprint baked into its WebAssembly engine and shows **ENGINE STALE**
    when the page was built from other sources (rebuild with `tools/scripts/build_web.sh`).
  - `GET /api/local/models` -- every network under the checkpoints tree, grouped by run, newest
    run first (`resume_*.pt` are training state and are left out).
  - `GET /api/local/models/onnx?path=<run>/<snapshot>.pt` -- that checkpoint as ONNX. The first
    request exports it with `tools/export_onnx.py` (a few seconds: the export is verified against
    torch on real positions and refused if ONNX Runtime disagrees); later ones are served from the
    cache. 404 for anything outside the tree, 422 for a checkpoint that does not export.
  - `GET /api/local/replays`, `GET /api/local/replays/<file>` -- the replay directory's
    `*.tslog.json` files. A file that cannot be read is listed after the replays as
    `{filename, error}` (the page shows it, disabled) rather than left out; opening one is a 422.
  - Serves the built page (`web/ui/dist`) at `/`.
  - CORS: only pages on `localhost` / `127.0.0.1` / `[::1]` (any port) may read the API from
    another origin. The page is served from here and the Vite dev server proxies `/api`, so neither
    needs it; nothing else should read this machine's checkpoints. The server binds 127.0.0.1 by
    default; the images listen on 0.0.0.0 inside the container and are published on the host's
    loopback (`deploy/web/compose.yaml`, `web/Dockerfile`'s run line).

- [`local_files.py`](local_files.py): what those endpoints read.
  - `models_root()` (`$TS_CHECKPOINTS_DIR`, else the shared `data/checkpoints`) and
    `resolve_model_path()`, which refuses anything outside the tree or not a network file.
  - `onnx_for()`: the export cache (`$TS_ONNX_CACHE_DIR`, else `data/onnx_cache`), keyed by the
    checkpoint's size and mtime and the engine fingerprint -- a retrained snapshot or a rebuilt
    engine gets a fresh export. **One export at a time, process-wide**: torch's ONNX exporter keeps
    global state, and two exports in parallel threads fail each other.

- [`replay.py`](replay.py):
  - `ReplayLogger`: Appends micro-actions, turn/AR milestones, and state snapshots as `.tslog.json`.
    Written by `tools/lib/self_play.py`, `tools/play_match.py` and the tournament tools; the
    browser workbench exports its own games in the same format (*Export Replay*).
  - Optional per-step **trace**: `policy` (what the model believed at that node) and `critic`
    (both value heads from both perspectives on the step's snapshot; only the side to move there,
    the next step's `player`, reads a trained value, and the workbench dims the other), plus
    `metadata.trace` naming the model, engine build and settings. Written by
    `tools/lib/self_play.py`, `tools/play_match.py --trace` and, after the fact,
    `tools/annotate_replay.py` (`ai/eval/policy_readout.py`). `policy.top` lists **every** legal
    action by default (`trace_top_k=0`) so the workbench can paint each probability where it
    belongs. **Every reader must treat both keys as absent by default** — a heuristic bot has no
    distribution, a human game no model, and old replays neither.
  - `start_position` in the metadata marks a game begun from a loaded position (a shared link)
    rather than from the seed; such a replay cannot be re-driven from the seed alone.
  - `ReplayManager`: discovers and loads those files, confined to the replay directory.
    `replays_dir()` resolves the location per call — `$TS_REPLAYS_DIR` if set, else
    `data/replays` — so a test fixture can point the server at a directory it has just generated
    a replay into. (The import-time `REPLAYS_DIR` constant, which could not see an override, is
    gone.)

- [`replay_types.py`](replay_types.py):
  - Strongly typed `TypedDict` definitions for all serialized game states, action logs, replays
    and audit structures. `GameStateDict` is what `GameState.to_dict()` returns -- the display
    state written once in `bindings/state_json.cpp` and shared with the page's engine -- and
    `SaveStateDict` what `GameState.to_save_dict()` returns. Every key either writes must be
    declared: `tests/bindings/test_state_dict_keys_are_typed.py` walks real positions' dicts and
    the source's literal keys (`replay_schema.golden.json`, generated from the TypedDicts, cannot
    see an engine key they lack). A new engine key goes into its TypedDict, then the golden is
    regenerated.

---

## 2. Mandatory Documentation Maintenance Rule for Agents

> [!IMPORTANT]
> **Keep Server Documentation Synchronized**:
> Whenever adding or altering endpoints, changing what the page is served, or changing replay
> formats, you **MUST** update this file and root [`AGENTS.md`](../../AGENTS.md).

---

## 3. How to Run & Test

```bash
# Build the page and its WebAssembly engine, then serve them with this machine's files
tools/scripts/build_web.sh
PYTHONPATH=.:build/release .venv/bin/python -m web.server.main --port 8000

# Tests (python -m pytest, never .venv/bin/pytest). tests/web needs the built page and, for the
# E2E tests, a Playwright Chromium (.venv/bin/python -m playwright install chromium).
PYTHONPATH=.:build/release .venv/bin/python -m pytest -q tests/web
```
