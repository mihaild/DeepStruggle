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

---

## 1. Quick Reference: Match & Replay Commands

### A. Simulating a Match Between Two Different Checkpoints
```bash
PYTHONPATH=. .venv/bin/python tools/play_match.py \
  --us data/checkpoints/run_v3_20260827_205207/snapshot_3602s.pt \
  --ussr data/checkpoints/run_v2_blunder_aware_9h/snapshot_21601s.pt \
  --game-id v3_vs_v2 \
  --seed 42
```
*Output*: Dumps `data/replays/v3_vs_v2.tslog.json` and prints the direct Web Workbench viewer URL:
`http://localhost:8000/?replay=v3_vs_v2.tslog.json`

### B. Interactive Terminal Play (Human vs AI Bot)
Allows a human to play Twilight Struggle directly in the terminal with numbered move prompts:

```bash
# Play as US against a trained NeuralBot as USSR:
PYTHONPATH=. .venv/bin/python tools/play_match.py \
  --us human \
  --ussr data/checkpoints/run_v3_20260827_205207/snapshot_3602s.pt \
  --game-id my_human_game
```
*Replay is automatically saved so you can review your game on the web map afterward.*

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
# Connect NeuralBot as USSR:
PYTHONPATH=. .venv/bin/python -m web.bot_client \
  --game-id game-1 \
  --role USSR \
  --type neural \
  --model-path data/checkpoints/run_v3_20260827_205207/snapshot_3602s.pt

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
- Path to any `.pt` file (e.g. `data/checkpoints/.../snapshot_1200s.pt`): Loads that specific checkpoint, auto-detecting V1, V2, or V3 architecture.
