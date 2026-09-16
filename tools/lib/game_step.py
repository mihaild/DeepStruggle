"""The step contract, in one place.

Five loops drive a game forward -- `TsVectorizedEnv`, `play_match`, `BatchMatchRunner`, the
searcher, and the web session -- and each answered two questions its own way, with nothing
enforcing either: who settles the state, and who checks the engine's return value. In practice
nobody checked the return value.

Both gaps fired together in one recorded game. The searcher settled its clone and returned an
action for a later decision; `play_match`, which does not settle, submitted it; the engine refused
it and changed nothing; the loop discarded the `False` and re-offered the same node 1,355 times
until the step cap. The replay recorded every refusal as though it had happened, which is why it
read as a card played twice and a coup with no resolution.

This module is the nucleus of `GameDriver` (see `research/plans/P13_one_game_driver.md`), not a
stopgap: the driver grows around it rather than replacing it.

**A rejected action is a caller bug, never a game event.** Every action in the 212-wide legal mask
decodes to a MicroAction whose `decision_type` matches the context -- verified over 177,258
state/action pairs -- so the engine only refuses an action the caller had no business submitting:
it chose outside the mask, or it handed the engine a state the engine was not asking about.
Raising makes that visible at the point of the mistake instead of ten thousand steps later.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

import ts_engine as ts


class IllegalActionError(RuntimeError):
    """The engine refused an action. Raised instead of looping or recording a phantom move."""


def _describe(state: ts.GameState, action: Union[int, ts.MicroAction]) -> str:
    ctx = state.ctx()
    want = ts.DecisionType(int(ctx.decision_type))
    if isinstance(action, int):
        try:
            ma = ts.decode_flat_action(state, action)
            got = ts.DecisionType(int(ma.decision_type))
        except Exception:
            got = None
        detail = f"flat action {action}" + (f" (decodes to {got})" if got is not None else "")
    else:
        detail = (f"MicroAction(decision_type={ts.DecisionType(int(action.decision_type))}, "
                  f"primary={int(action.primary_id)})")
    return (f"engine refused {detail}; it is asking for {want} "
            f"from {ctx.decision_player}")


def step_checked(state: ts.GameState, action: Union[int, ts.MicroAction],
                 auto_advance: bool = False, *, context: Optional[str] = None) -> None:
    """Advance the game by one action, or raise.

    `auto_advance` settles the state afterwards, past decisions with no discretion. It is an
    explicit choice because the five loops disagreed about it silently, and that disagreement is
    what let a searcher answer a different question from the one the caller asked.

    `context` is appended to the error, so a failure names the loop it came from.
    """
    if isinstance(action, (int, np.integer)):
        ok = ts.Engine.try_step_flat(state, int(action), auto_advance)
    else:
        ok = ts.Engine.try_step(state, action, auto_advance)
    if not ok:
        msg = _describe(state, int(action) if isinstance(action, (int, np.integer)) else action)
        if context:
            msg = f"{msg} [{context}]"
        raise IllegalActionError(msg)


def assert_legal(state: ts.GameState, action: int, *, context: Optional[str] = None) -> None:
    """Check an action against the mask before submitting it.

    Cheaper to diagnose than a refusal from inside the engine, and it catches the case where a
    caller is about to submit an action chosen against a DIFFERENT state -- which is how the
    searcher's stall began.
    """
    from bindings.action_encoder import ActionEncoder

    mask = np.asarray(ActionEncoder.get_legal_mask(state))
    if not (0 <= action < len(mask)) or not mask[action]:
        msg = (f"action {action} is not in the legal mask "
               f"({int(mask.sum())} legal, decision={ts.DecisionType(int(state.ctx().decision_type))})")
        if context:
            msg = f"{msg} [{context}]"
        raise IllegalActionError(msg)


def drain_chance(state: ts.GameState, *, forced_die: int = 0,
                 context: Optional[str] = None) -> int:
    """Resolve the chance nodes the game is sitting on, returning how many were resolved.

    A chance node is a `ROLL_DIE` that no player owns. It is not a decision: no policy, bot or
    human is asked for one, and no replay records it, because a reader regenerates it from the
    RNG. Every writer and every reader of a replay has to agree on this, so it lives here rather
    than being re-implemented per caller -- it was re-implemented four times, and the web copy
    could not fail safely.

    `forced_die` is the workbench's manual-roll affordance (`selectedDieRoll` in the UI, 0 for
    auto, 1..6 to force). 0 means "roll normally"; anything outside 0..6 is refused by the engine
    and raises here rather than looping.
    """
    resolved = 0
    while (not ts.Engine.is_terminal(state)
           and state.ctx().decision_player == ts.Player.NONE
           and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
        step_checked(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, forced_die, 0, 0),
                     context=context or "chance node")
        resolved += 1
    return resolved
