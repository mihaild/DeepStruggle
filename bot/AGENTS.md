# Bot Clients & AI Agents Guide (`bot/`)

This directory contains pure bot client implementations, baseline heuristics, and decision policies for **Twilight Struggle**. All bots inherit from [`BaseBot`](base_bot.py).

---

## 1. Overview of Bot Clients

- [`base_bot.py`](base_bot.py): Abstract `BaseBot` class defining the standard interface for Twilight Struggle agents:
  - `select_action(state_dict, legal_actions_dict) -> Optional[dict]`: Dict-based action selection for JSON-driven callers.
  - `select_flat_action(state, player) -> int`: Fast 212-dim flat index selection for vectorized environments.
  - `reset()`: Resets internal memory or search state between games.
- [`random_bot.py`](random_bot.py): `RandomBot` baseline stochastic player making uniform random choices over legal moves.
- [`heuristic_bot.py`](heuristic_bot.py): `HeuristicBot` rule-based expert player prioritizing key battlegrounds, scoring card timing, and ops efficiency.
- [`neural_bot.py`](neural_bot.py): `NeuralBot` deep reinforcement learning player driven by `ColdWarNet` PyTorch checkpoints. The architecture (V1 or V2) is detected from the weights; a checkpoint from a retired architecture is refused outright by `tools.lib.player_agent.reject_retired_architecture` rather than partially loaded.
- [`exploratory_bot.py`](exploratory_bot.py): `ExploratoryBot` agent designed to explore diverse decision paths, Space Race, Realignments, and Coups.
- [`strategic_bot.py`](strategic_bot.py): `StrategicBot` high-level strategic agent prioritizing DEFCON-2 containment, coups, realignments, and Space Race safety with rich strategy/commentary generation.
- [`event_heavy_bot.py`](event_heavy_bot.py): `EventHeavyBot` agent maximizing card event play and event-first timing.
- [`human_bot.py`](human_bot.py): `HumanBot` interactive terminal CLI player prompting the user for numbered choices.

---

## 2. Launching Bots

### Playing Against a Model in the Web Workbench:
The workbench runs in the browser and plays neural models itself (ONNX exports of checkpoints),
so no bot process is involved: pick a model in *Model Analysis* and set *Auto-play* to the side it
should play (see [`web/ui/AGENTS.md`](../web/ui/AGENTS.md)). The network bot client that used to
connect a `BaseBot` to a server-side game (`web/bot_client.py`) was removed with that server.

### Running Offline Matches Between Bots:
Matches can be run offline with full replay generation via [`tools/play_match.py`](../tools/play_match.py):
```bash
# Run a match between HeuristicBot and StrategicBot
PYTHONPATH=. .venv/bin/python tools/play_match.py --us heuristic --ussr strategic
```

---

## 3. Mandatory Documentation Maintenance Rule for Agents

> [!IMPORTANT]
> **Keep Bot Documentation Synchronized**:
> Whenever adding new bot strategies or modifying bot interfaces, you **MUST** update this file and root [`AGENTS.md`](../AGENTS.md).
