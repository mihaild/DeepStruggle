# Bot Client & AI Agent Guide

This directory contains standalone bot clients and interactive agent interfaces that connect to the game server over WebSocket, receive game states, evaluate legal actions, and play moves.

---

## 1. File Overview

- [`generate_strategic_game.py`](file:///home/mihaild/prog/ts_ai/bot/generate_strategic_game.py):
  - Generates deep, high-level reasoned games maintaining DEFCON at 2 for most of the match, prioritizing Realignments in contested non-battleground/battleground theaters, and managing toxic card disposals via Space Race.
  - Automatically writes replays to `.tslog.json` and human-readable `.log` files.

- [`generate_event_heavy_game.py`](file:///home/mihaild/prog/ts_ai/bot/generate_event_heavy_game.py):
  - Generates event-dense match trajectories maximizing historical event trigger rates (preferring `PlayMode::EVENT` and `TimingBranch::EVENT_FIRST`).
  - Outputs full replay to `replays/event_heavy_game.tslog.json` and in-character commentary to `.log`.

- [`bot_client.py`](file:///home/mihaild/prog/ts_ai/bot/bot_client.py):
  - `BaseBot`: Abstract base class defining `select_action(state, legal_actions) -> dict`.
  - `RandomBot`: Selects uniformly random valid actions from `legal_actions['valid_ids']`. Supports early pass (`0x80`).
  - `HeuristicBot`: Rule-based baseline bot:
    - Prioritizes East Germany/Poland for USSR and West Germany/Italy for US during Setup.
    - Plays Scoring cards when held, or highest Ops cards.
    - Selects friendly events or uses Ops for influence/coups.
    - Prioritizes battlegrounds in current scoring regions.
  - `run_bot_client()`: Async WebSocket client loop connecting to `ws://localhost:8000/ws/game/{id}?role={role}`.

- [`agent_player.py`](file:///home/mihaild/prog/ts_ai/bot/agent_player.py):
  - Rich CLI & Python API for LLM agents or human developers to query state and play actions by friendly name or index.

---

## 2. Mandatory Documentation Maintenance Rule for Agents

> [!IMPORTANT]
> **Keep Bot Documentation Synchronized**:
> Whenever adding new bot policies, modifying heuristic rules, extending CLI parameters, or altering the agent interactive interface, you **MUST** update this file and root [`AGENTS.md`](file:///home/mihaild/prog/ts_ai/AGENTS.md).

---

## 3. Implementing New Bot Strategies

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

## 4. How to Run Bots

```bash
# Play against Heuristic Bot on game-1 as USSR
PYTHONPATH=. .venv/bin/python -m bot.bot_client --game-id game-1 --role USSR --type heuristic

# Play with artificial delay (e.g. 0.5s per step for human observation)
PYTHONPATH=. .venv/bin/python -m bot.bot_client --game-id game-1 --role USSR --delay 0.5
```

---

## 5. Agent Interactive Play Tool (`bot.agent_player`)

[`agent_player.py`](file:///home/mihaild/prog/ts_ai/bot/agent_player.py) provides a rich CLI & Python API for LLM agents or human developers to query state and play actions by name or index.

### 5.1 CLI Commands

```bash
# 1. Inspect current state, active decision & legal choices
PYTHONPATH=. .venv/bin/python -m bot.agent_player status

# 2. View global board breakdown (regional influence and country control)
PYTHONPATH=. .venv/bin/python -m bot.agent_player board

# 3. Take an action by numbered index (e.g. 1) or by name
PYTHONPATH=. .venv/bin/python -m bot.agent_player play 1
PYTHONPATH=. .venv/bin/python -m bot.agent_player play "Poland"
PYTHONPATH=. .venv/bin/python -m bot.agent_player play "Duck and Cover"
PYTHONPATH=. .venv/bin/python -m bot.agent_player play "OPS"
PYTHONPATH=. .venv/bin/python -m bot.agent_player play "INFLUENCE"
PYTHONPATH=. .venv/bin/python -m bot.agent_player play "done"

# 4. Play action with manual die roll (1..6)
PYTHONPATH=. .venv/bin/python -m bot.agent_player play "EVENT" --roll 2

# 5. Undo last action
PYTHONPATH=. .venv/bin/python -m bot.agent_player undo

# 6. Start a new game with seed
PYTHONPATH=. .venv/bin/python -m bot.agent_player new-game --seed 42

# 7. Auto-play N steps using baseline bot
PYTHONPATH=. .venv/bin/python -m bot.agent_player auto --steps 10
```

### 5.2 Python API for Agents

```python
from bot.agent_player import AgentGameClient, format_state_summary, format_board_overview

client = AgentGameClient(use_local=True)
state = client.get_state()

# Print formatted overview
print(format_state_summary(state, client))

# Resolve action by friendly name
state_after = client.resolve_choice("Poland")
```
