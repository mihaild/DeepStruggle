# Bot Clients & AI Agents Guide (`bot/`)

Bot implementations, baseline heuristics and decision policies for **Twilight Struggle**. All bots
inherit from [`BaseBot`](base_bot.py).

---

## 1. Overview of Bot Clients

- [`base_bot.py`](base_bot.py): the abstract `BaseBot` interface:
  - `select_action(state_dict, legal_actions_dict) -> Optional[dict]`: dict-based selection for JSON-driven callers.
  - `select_flat_action(state, player) -> int`: a flat action index, for vectorized environments.
  - `reset()`: clears internal memory or search state between games.
- [`random_bot.py`](random_bot.py): `RandomBot`, uniform over legal moves.
- [`heuristic_bot.py`](heuristic_bot.py): the terminal match's `HeuristicBot`, a fixed rule list
  (battlegrounds, scoring-card timing, ops efficiency). Its JSON influence scoring reads countries
  by their display names. Tournament evaluation uses the separate native-state `HeuristicAgent`
  in `tools/lib/player_agent.py` as its default Elo anchor.
- [`heuristic_mcts_bot.py`](heuristic_mcts_bot.py): `HeuristicMCTSBot` (`heuristic_mcts[:sims]` in `tools/play_match.py`), MCTS over the real engine with `ai.search.heuristic_eval` at the leaf and no checkpoint. 18/20 against HeuristicBot from 16 simulations on (`research/findings/training/heuristic_mcts_baseline.md`). **It searches the true state and sees the opponent's hand**: fine for a benchmark, never a claim about play under real information.
- [`neural_bot.py`](neural_bot.py): `NeuralBot`, loaded through `tools.lib.player_agent.NeuralAgent.from_checkpoint`: the architecture (V1, V2 or LadderNet) is detected from the weights and a retired one is refused rather than partially loaded. It builds E4 masks and reads the base observation, so a checkpoint in the merged-influence view or with observation feature blocks is refused (`tests/training/test_neural_bot_checkpoint.py`).
- [`exploratory_bot.py`](exploratory_bot.py): `ExploratoryBot`, explores diverse paths: Space Race, realignments, coups.
- [`strategic_bot.py`](strategic_bot.py): `StrategicBot`, DEFCON-2 containment, coups, realignments and Space Race safety, with strategy commentary.
- [`event_heavy_bot.py`](event_heavy_bot.py): `EventHeavyBot`, maximises event play and event-first timing.
- [`human_bot.py`](human_bot.py): `HumanBot`, interactive terminal player prompting for numbered choices.

---

## 2. Launching Bots

**Against a model in the web workbench** no bot process is involved: the page plays ONNX exports
of checkpoints itself (pick a model in *Model Analysis*, set *Auto-play*; see
[`web/ui/AGENTS.md`](../web/ui/AGENTS.md)).

**Offline matches** with replay generation go through [`tools/play_match.py`](../tools/play_match.py):
```bash
PYTHONPATH=. .venv/bin/python tools/play_match.py --us heuristic --ussr strategic
```

---

## 3. Mandatory Documentation Maintenance Rule for Agents

> [!IMPORTANT]
> **Keep Bot Documentation Synchronized**:
> Whenever adding new bot strategies or modifying bot interfaces, you **MUST** update this file and root [`AGENTS.md`](../AGENTS.md).
