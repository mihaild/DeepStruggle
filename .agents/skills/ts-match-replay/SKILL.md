---
name: ts-match-replay
description: >-
  Simulate individual Twilight Struggle matches between agents, play interactively via CLI terminal,
  generate standardized .tslog.json game replays, and launch Web Workbench browser matches. Activate this skill
  whenever the user asks to play a match, watch a game replay, play interactively in the terminal, or connect
  a bot to the web UI.
---

# Twilight Struggle AI: Match Simulator & Replay Guide

This skill provides step-by-step instructions for running offline matches between any two agents, playing interactively in the terminal, and inspecting replays via the Web Workbench.

> [!IMPORTANT]
> **Dynamic Checkpoints**:
> Checkpoint paths evolve as new models are trained.
> - Run `python tools/inspect_checkpoints.py` to view all saved models and architectures.
> - You can pass `--us neural` or `--ussr neural` without a path to **automatically select the latest trained checkpoint**.
> - Specific paths shown in examples below (e.g. `<checkpoint_path>.pt`) should be replaced with current models.

---

## 1. Quick Reference: Match & Replay Commands

### A. Simulating a Match Between Two Checkpoints
```bash
# Pit two checkpoints against each other:
# (Replace with paths discovered via tools/inspect_checkpoints.py, or use 'neural' for latest)
PYTHONPATH=. .venv/bin/python tools/play_match.py \
  --us data/checkpoints/<run_A>/<snapshot_A>.pt \
  --ussr data/checkpoints/<run_B>/<snapshot_B>.pt \
  --game-id match_vA_vs_vB \
  --seed 42
```
*Output*: Dumps `data/replays/match_vA_vs_vB.tslog.json` and prints the direct Web Workbench viewer URL:
`http://localhost:8000/?replay=match_vA_vs_vB.tslog.json`

### B. Interactive Terminal Play (Human vs AI Bot)
Allows a human to play Twilight Struggle directly in the terminal with numbered move prompts:

```bash
# Play as US against the latest trained NeuralBot as USSR:
PYTHONPATH=. .venv/bin/python tools/play_match.py \
  --us human \
  --ussr neural \
  --game-id my_human_game
```
*The replay is automatically saved to `data/replays/` so you can review your game on the web map afterward.*

### C. Match with Strategic Commentary & Regional Scoring Audits
Embeds strategic rationale, chain-of-thought lines, and regional scoring calculations directly into the replay metadata:

```bash
PYTHONPATH=. .venv/bin/python tools/play_match.py \
  --us strategic \
  --ussr event_heavy \
  --commentary \
  --game-id strategic_vs_event
```

---

## 2. Web Workbench: Live Browser Play Against AI

### Step 1: Start Backend Game Server
```bash
PYTHONPATH=. .venv/bin/python -m uvicorn web.server.main:app --host 0.0.0.0 --port 8000
```

### Step 2: Connect Bot Client via WebSocket
In a separate terminal, start the bot for the opponent side:
```bash
# Connect NeuralBot as USSR (replace with target model checkpoint):
PYTHONPATH=. .venv/bin/python -m web.bot_client \
  --game-id game-1 \
  --role USSR \
  --type neural \
  --model-path data/checkpoints/<target_run>/<snapshot>.pt

# Or connect HeuristicBot as USSR:
PYTHONPATH=. .venv/bin/python -m web.bot_client --game-id game-1 --role USSR --type heuristic
```

### Step 3: Open Browser
Navigate to:
`http://localhost:8000/?game_id=game-1&role=US`

---

## 3. Supported Agent Identifiers

When using `--us`, `--ussr`, or `--agent` in `tools/play_match.py`:
- `heuristic`: Rule-based battleground and scoring timing expert ([`HeuristicBot`](../../bot/heuristic_bot.py)).
- `random`: Uniform stochastic player ([`RandomBot`](../../bot/random_bot.py)).
- `strategic`: DEFCON-2 realist containment agent with rich strategy notes ([`StrategicBot`](../../bot/strategic_bot.py)).
- `event_heavy`: Event-prioritizing agent with EVENT_FIRST timing ([`EventHeavyBot`](../../bot/event_heavy_bot.py)).
- `exploratory`: Exploration agent testing Space Race, coups, and edge cases ([`ExploratoryBot`](../../bot/exploratory_bot.py)).
- `human`: Interactive terminal player prompting for stdin input ([`HumanBot`](../../bot/human_bot.py)).
- `neural`: Auto-discovers and loads the latest trained `.pt` checkpoint.
- Path to any `.pt` file (e.g. `data/checkpoints/<run>/<snapshot>.pt`): Loads that specific checkpoint, auto-detecting V1, V2, or V3 architecture.
