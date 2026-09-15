"""The loop must play a whole game, record faithfully, and fail loudly on a bad action.

The point of collapsing five loops into one is that the contract stops being per-caller. These
tests pin that contract: RECORD_FORCED records every step so a replay re-drives without knowing who
produced it, a refused action raises instead of looping, and a missing source is an error rather
than a silent stall.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest

import ts_engine as ts
from bindings.action_encoder import ActionEncoder
from tools.lib.game_loop import (ActionSource, GameLoop, SettlePolicy, StepRecord,
                                 acting_player)
from tools.lib.game_step import IllegalActionError


class FirstLegal:
    """Deterministic source: always the lowest-indexed legal action."""

    def choose(self, state: ts.GameState):
        legal = np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(state)))
        if not len(legal):
            return None
        return ts.decode_flat_action(state, int(legal[0]))


class Forfeits:
    def choose(self, state: ts.GameState):
        return None


class AlwaysIllegal:
    """Returns an action for a decision the engine is not asking about."""

    def choose(self, state: ts.GameState):
        want = int(state.ctx().decision_type)
        other = 1 if want != 1 else 2
        return ts.MicroAction(ts.DecisionType(other), 0, 0, 0)


def _fresh(seed: int = 4242) -> ts.GameState:
    s = ts.GameState()
    ts.Engine.init_game(s, seed)
    return s


def _both(source: ActionSource) -> Dict[ts.Player, ActionSource]:
    return {ts.Player.US: source, ts.Player.USSR: source}


def test_it_plays_a_whole_game() -> None:
    loop = GameLoop(_fresh(), _both(FirstLegal()), settle=SettlePolicy.RECORD_FORCED)
    res = loop.run()
    assert res.terminal, f"game did not finish: {res}"
    assert not res.hit_step_cap
    assert res.steps > 100, f"only {res.steps} steps"
    assert res.utility in (-1.0, 0.0, 1.0)


def test_record_forced_records_every_step_so_a_replay_re_drives() -> None:
    """The property that makes a replay portable: apply its actions in order, with no settling."""
    seen: list[StepRecord] = []
    loop = GameLoop(_fresh(), _both(FirstLegal()), settle=SettlePolicy.RECORD_FORCED,
                    recorder=lambda rec, st: seen.append(rec))
    res = loop.run()
    assert len(seen) == res.steps

    # Re-drive from scratch with NO settling. Every recorded action must be accepted.
    replay = _fresh()
    for rec in seen:
        assert ts.Engine.step(replay, rec.action), (
            f"recorded action at step {rec.index} was refused on replay")
    assert ts.Engine.is_terminal(replay) == res.terminal
    assert float(ts.Engine.get_terminal_utility(replay)) == res.utility


def test_engine_settling_cannot_be_recorded_faithfully() -> None:
    """Why RECORD_FORCED exists, stated as a test rather than a comment.

    `auto_advance_step` reports how many steps it took, not which, so a replay recorded under
    ENGINE settling is missing them and cannot be re-driven without settling identically.
    """
    seen: list[StepRecord] = []
    loop = GameLoop(_fresh(), _both(FirstLegal()), settle=SettlePolicy.ENGINE,
                    recorder=lambda rec, st: seen.append(rec))
    loop.run()

    replay = _fresh()
    refused = 0
    for rec in seen:
        if not ts.Engine.step(replay, rec.action):
            refused += 1
    assert refused > 0, (
        "ENGINE settling unexpectedly produced a re-drivable record; if that is now true, "
        "RECORD_FORCED may no longer be necessary")


def test_a_refused_action_raises_instead_of_looping() -> None:
    loop = GameLoop(_fresh(), _both(AlwaysIllegal()), settle=SettlePolicy.NONE)
    with pytest.raises(IllegalActionError):
        loop.run()


def test_a_forfeit_ends_the_game_and_says_who() -> None:
    state = _fresh()
    ts.Engine.auto_advance_step(state)
    mover = acting_player(state)
    sources: Dict[ts.Player, ActionSource] = {
        ts.Player.US: FirstLegal(), ts.Player.USSR: FirstLegal()}
    sources[mover] = Forfeits()
    res = GameLoop(state, sources, settle=SettlePolicy.NONE).run()
    assert res.forfeited_by == ("US" if mover == ts.Player.US else "USSR")
    assert not res.terminal


def test_a_missing_source_is_an_error_not_a_stall() -> None:
    state = _fresh()
    ts.Engine.auto_advance_step(state)
    mover = acting_player(state)
    sources: Dict[ts.Player, ActionSource] = {
        m: FirstLegal() for m in (ts.Player.US, ts.Player.USSR) if m != mover}
    with pytest.raises(IllegalActionError):
        GameLoop(state, sources, settle=SettlePolicy.NONE).run()


def test_the_step_cap_is_reported_rather_than_silent() -> None:
    loop = GameLoop(_fresh(), _both(FirstLegal()), settle=SettlePolicy.RECORD_FORCED,
                    max_steps=20)
    res = loop.run()
    assert res.hit_step_cap and res.steps == 20 and not res.terminal
