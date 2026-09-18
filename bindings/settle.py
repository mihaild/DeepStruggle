"""One place that decides how far past a decision nobody controls the engine is advanced.

Before this there were eleven: `drain_chance_nodes` in `ai/search/pimcts.py`, `settle` in
`ai/search/batched_mcts.py`, `_drain` in `tools/lib/game_loop.py`, `_drain_chance_nodes` in
`tools/lib/tournament_evaluator.py`, four byte-identical `_drain`s across `ai/eval/`, and more in
the tests -- against one C++ `Engine.auto_advance_step`. Most were the same six lines copied.

**The duplication was not the cost; the disagreement was.** Two settlers that stop at different
depths put two components on different decisions while both believe they are synchronised. That is
exactly how `BatchedMCTSConfig.advance_root` came to return actions illegal in the caller's state:
the Python side drained chance nodes, the C++ side also skipped single-legal-action nodes, and the
tree rooted one decision further on than the caller stood. Every one of the five production
searchers carries `advance_root=False` to work around it, and `play_match.py`'s comment records
what it cost before anyone noticed -- the engine refusing an action "1,355 times in one game".

A mode is a property of the *measurement*, not of the engine: a run settled one way is not
comparable with a run settled another, the same way sampling temperature is not comparable. State
it explicitly when it matters.
"""

from __future__ import annotations

import enum

import ts_engine as ts


class SettleMode(enum.Enum):
    """How far to advance past decisions the player has no say in."""

    #: Advance nothing. Every decision the engine raises is presented, chance nodes included.
    #: For probes that inspect a specific node and must not have it skipped out from under them.
    NONE = "none"

    #: Resolve die rolls only. A chance node is the engine's to settle, never a policy's -- a
    #: policy asked for one picks its own dice. This is what the eight hand-rolled drains did.
    #:
    #: `tools.lib.game_step.drain_chance` is the checked superset of this, and stays separate on
    #: purpose: it counts what it resolved, takes the workbench's forced die, and steps through
    #: `step_checked` so a refusal names its context. Those belong to the interactive and replay
    #: paths. This module sits below `tools/`, so it cannot import them; the loop is six lines and
    #: duplicating it knowingly beats inverting the layering.
    CHANCE = "chance"

    #: Everything the player has no say in: chance nodes, single-legal-action decisions, and
    #: deterministic event targets. Fewer steps and fewer crossings of the binding boundary,
    #: and the engine does it in one call rather than a Python loop.
    FORCED = "forced"


# Depth is not the only axis. `tools.lib.game_loop.SettlePolicy` carries a second one:
# RECORD_FORCED settles to exactly the depth FORCED does, but plays each forced action itself so
# every step passes through the replay recorder. `Engine.auto_advance_step` reports how many steps
# it took and not WHICH, so a replay written under FORCED cannot be re-driven. That distinction
# belongs to the loop that owns the recorder, not here -- this module answers "how far", and
# game_loop answers "and who plays them". Both take the depths from the enum above so they cannot
# disagree about what "forced" means.


#: What a caller gets when it does not say. Settling everything is right for anything that asks a
#: policy to move: a decision with one legal action is not a decision, and paying a Python round
#: trip to hand it back is waste. Callers that need a specific node intact ask for NONE.
DEFAULT = SettleMode.FORCED


def settle(state: ts.GameState, mode: SettleMode = DEFAULT) -> None:
    """Advance `state` past whatever `mode` says the player does not control."""
    if mode is SettleMode.NONE:
        return
    if mode is SettleMode.FORCED:
        ts.Engine.auto_advance_step(state)
        return
    while (not ts.Engine.is_terminal(state)
           and state.ctx().decision_player == ts.Player.NONE
           and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
        ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))


def drain_chance_nodes(state: ts.GameState) -> None:
    """Resolve pending die rolls. Kept as a name because it says what it does at call sites."""
    settle(state, SettleMode.CHANCE)
