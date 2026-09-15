"""One loop, with the action source injected.

    state = init(seed)
    while not terminal(state):
        action = source.choose(state)
        state  = step(state)

Five loops used to do this -- `TsVectorizedEnv`, `play_match`, `BatchMatchRunner`, the searcher and
`GameSession` -- each written for a different purpose and each answering two questions its own way,
with nothing enforcing either: who settles the state, and who checks the engine's return value. A
mismatch between two of those answers produced 1,355 refused actions recorded in a replay as though
they had happened. See `research/plans/P13_one_game_driver.md`.

Here both questions have one answer. Stepping goes through `step_checked`, so a refused action
raises at the point of the mistake. Settling is an explicit `SettlePolicy` rather than a habit.

Callers differ only in **where actions come from** and **what gets recorded**.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np

import ts_engine as ts
from bindings.action_encoder import ActionEncoder
from tools.lib.game_step import IllegalActionError, step_checked


class SettlePolicy(enum.Enum):
    """Who answers decisions the player has no say in."""

    #: Nobody. Every decision the players own, including single-option ones, reaches the
    #: source. Chance nodes are drained regardless unless `drain_chance=False`.
    NONE = "none"
    #: The engine, in C++. Fastest -- 0.76x the wall time of the Python equivalent -- but it
    #: reports only how many steps it took, not which, so those steps CANNOT be recorded. Right
    #: for training, which records no replay; wrong for anything that does.
    ENGINE = "engine"
    #: The loop, by playing the single legal action itself. Slower, and every step passes through
    #: the recorder -- which is what makes a replay re-drivable without knowing who produced it.
    RECORD_FORCED = "record_forced"


class ActionSource(Protocol):
    """Where a move comes from: a policy, a search, a bot, a human, or a recording."""

    def choose(self, state: ts.GameState, /) -> Optional[ts.MicroAction]:
        """Return the action to play, or None to abandon the game (a forfeit).

        Positional-only: the loop always calls this positionally, and pinning the parameter's
        *name* would force every implementation to spell it the same way for no benefit.
        """
        ...


@dataclass
class StepRecord:
    """One step, as the recorder sees it."""

    index: int
    player: str
    action: ts.MicroAction
    forced: bool          # played by the settle policy rather than chosen by a source
    turn: int
    action_round: int
    #: The phase as it was when the action was taken, named as the replay format writes it. It is
    #: the pre-action phase because an action that ends a turn would otherwise be filed under the
    #: next one.
    phase_name: str = ""
    #: The flat (212-dim) index of the action, when it has one. A recorded replay needs it: the
    #: readers re-apply steps with `step_flat`, and one lacking the index is *skipped* by the
    #: fidelity test rather than failed, so dropping it would quietly retire that check.
    flat: Optional[int] = None
    #: The state AFTER the action and after the chance nodes it lands on, if a snapshot callback
    #: was supplied. That is the replay format's contract -- see the module docstring.
    state_after: Optional[Any] = None
    #: The state as it was before the action. Only what needs to *name* the action (a card in a
    #: hand it has since left) reads this; the replay snapshot is `state_after`.
    state_before: Optional[Any] = None


@dataclass
class LoopResult:
    steps: int
    terminal: bool
    utility: float
    forfeited_by: Optional[str] = None
    hit_step_cap: bool = False
    records: List[StepRecord] = field(default_factory=list)


def _player_name(p: ts.Player) -> str:
    return "US" if p == ts.Player.US else ("USSR" if p == ts.Player.USSR else "NONE")


def acting_player(state: ts.GameState) -> ts.Player:
    ctx = state.ctx()
    return ctx.decision_player if ctx.decision_player != ts.Player.NONE else state.phasing_player


class GameLoop:
    """Drive one game to its end, taking actions from per-side sources."""

    def __init__(self, state: ts.GameState,
                 sources: Dict[ts.Player, ActionSource],
                 *,
                 settle: SettlePolicy = SettlePolicy.RECORD_FORCED,
                 recorder: Optional[Callable[[StepRecord, ts.GameState], None]] = None,
                 snapshot: Optional[Callable[[ts.GameState], Any]] = None,
                 drain_chance: bool = True,
                 max_steps: int = 4000,
                 keep_records: bool = False) -> None:
        self.state = state
        self.sources = sources
        self.settle = settle
        self.recorder = recorder
        self.snapshot = snapshot
        self.drain_chance = drain_chance
        self.max_steps = max_steps
        self.keep_records = keep_records

    # -- settling -------------------------------------------------------------------------

    def _settle(self, result: LoopResult) -> None:
        """Advance past decisions with no discretion, according to the policy."""
        if self.settle is SettlePolicy.NONE:
            return
        if self.settle is SettlePolicy.ENGINE:
            ts.Engine.auto_advance_step(self.state)
            return
        # RECORD_FORCED: play each forced action ourselves so the recorder sees it.
        for _ in range(self.max_steps):
            if ts.Engine.is_terminal(self.state):
                return
            self._drain()
            if ts.Engine.is_terminal(self.state):
                return
            mask = np.asarray(ActionEncoder.get_legal_mask(self.state))
            legal = np.flatnonzero(mask)
            if len(legal) != 1:
                return
            self._play(int(legal[0]), forced=True, result=result)

    # -- stepping -------------------------------------------------------------------------

    def _drain(self) -> None:
        """Resolve chance nodes: a ROLL_DIE nobody owns.

        Not a decision, so no source is asked and no StepRecord is made. A reader regenerates
        these from the RNG, which is why they are absent from every saved replay; recording them
        would desynchronise every existing reader, all of which drain.
        """
        if not self.drain_chance:
            return
        while (not ts.Engine.is_terminal(self.state)
               and self.state.ctx().decision_player == ts.Player.NONE
               and self.state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
            step_checked(self.state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0),
                         context="GameLoop chance node")

    def _play(self, action, *, forced: bool, result: LoopResult) -> None:
        player = _player_name(acting_player(self.state))
        is_flat = isinstance(action, (int, np.integer))
        micro = ts.decode_flat_action(self.state, int(action)) if is_flat else action
        flat = int(action) if is_flat else ActionEncoder.encode(self.state, micro)
        rec = StepRecord(index=result.steps, player=player, action=micro, forced=forced,
                         turn=int(self.state.turn), action_round=int(self.state.action_round),
                         phase_name=str(self.state.current_phase).replace("Phase.", ""),
                         flat=flat,
                         state_before=self.snapshot(self.state) if self.snapshot else None)
        # Step first, record last. Recording before stepping is how refused actions reached
        # replays: play_match logged the step and only then discovered the engine had refused it.
        # Here a refusal raises out of step_checked and nothing is recorded at all.
        step_checked(self.state, micro, context="GameLoop")
        self._drain()
        result.steps += 1
        if self.snapshot is not None:
            rec.state_after = self.snapshot(self.state)
        if self.recorder is not None:
            self.recorder(rec, self.state)
        if self.keep_records:
            result.records.append(rec)

    # -- the loop -------------------------------------------------------------------------

    def run(self) -> LoopResult:
        result = LoopResult(steps=0, terminal=False, utility=0.0)
        self._drain()
        self._settle(result)

        while not ts.Engine.is_terminal(self.state):
            if result.steps >= self.max_steps:
                result.hit_step_cap = True
                break
            mover = acting_player(self.state)
            source = self.sources.get(mover)
            if source is None:
                raise IllegalActionError(
                    f"no action source for {_player_name(mover)} at "
                    f"{ts.DecisionType(int(self.state.ctx().decision_type))}")
            action = source.choose(self.state)
            if action is None:
                result.forfeited_by = _player_name(mover)
                break
            self._play(action, forced=False, result=result)
            self._settle(result)

        result.terminal = ts.Engine.is_terminal(self.state)
        if result.terminal:
            result.utility = float(ts.Engine.get_terminal_utility(self.state))
        return result
