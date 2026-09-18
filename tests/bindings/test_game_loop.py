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


def _drain(state: ts.GameState) -> None:
    """What every reader of the format does: resolve the chance nodes, which are never recorded."""
    while (not ts.Engine.is_terminal(state)
           and state.ctx().decision_player == ts.Player.NONE
           and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
        ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))


def test_record_forced_records_every_step_so_a_replay_re_drives() -> None:
    """The property that makes a replay portable, stated as its readers implement it.

    The format does not record chance nodes -- they are reproducible from the RNG, so a reader
    regenerates them (`tests/replayer/test_replay_reproduces.py`, `web/ui`). What the loop must
    guarantee is that everything *else* is recorded, so draining plus the recorded actions
    reproduces the game exactly.
    """
    seen: list[StepRecord] = []
    loop = GameLoop(_fresh(), _both(FirstLegal()), settle=SettlePolicy.RECORD_FORCED,
                    recorder=lambda rec, st: seen.append(rec))
    res = loop.run()
    assert len(seen) == res.steps

    replay = _fresh()
    for rec in seen:
        _drain(replay)
        assert ts.Engine.try_step(replay, rec.action), (
            f"recorded action at step {rec.index} was refused on replay")
    _drain(replay)
    assert ts.Engine.is_terminal(replay) == res.terminal
    assert float(ts.Engine.get_terminal_utility(replay)) == res.utility


def test_a_chance_node_is_never_recorded_and_never_reaches_a_source() -> None:
    """A die nobody owns is not a decision.

    Recording one would desynchronise every existing reader, all of which drain; handing one to a
    source would ask a policy to choose its own luck.
    """
    seen: list[StepRecord] = []

    class Watchful:
        saw_chance = False

        def choose(self, state: ts.GameState):
            ctx = state.ctx()
            if (ctx.decision_player == ts.Player.NONE
                    and ctx.decision_type == ts.DecisionType.ROLL_DIE):
                Watchful.saw_chance = True
            return FirstLegal().choose(state)

    GameLoop(_fresh(), _both(Watchful()), settle=SettlePolicy.RECORD_FORCED,
             recorder=lambda rec, st: seen.append(rec)).run()
    assert not Watchful.saw_chance, "a chance node was offered to an action source"
    assert seen, "no steps recorded"
    assert not any(r.player == "NONE" for r in seen), "a chance node was recorded as a step"


def test_every_record_carries_the_flat_index_a_replay_needs() -> None:
    """A replay lacking flat_action_idx is *skipped* by the fidelity test, not failed."""
    seen: list[StepRecord] = []
    GameLoop(_fresh(), _both(FirstLegal()), settle=SettlePolicy.RECORD_FORCED,
             recorder=lambda rec, st: seen.append(rec)).run()
    assert seen
    assert all(r.flat is not None and 0 <= r.flat < ActionEncoder.FLAT_ACTION_SIZE for r in seen)


def test_the_snapshot_is_the_state_after_the_action() -> None:
    """The format's contract, read off its readers: the UI shows a step's resulting board."""
    seen: list[StepRecord] = []
    GameLoop(_fresh(), _both(FirstLegal()), settle=SettlePolicy.RECORD_FORCED,
             recorder=lambda rec, st: seen.append(rec),
             snapshot=lambda st: {"turn": int(st.turn), "vp": int(st.victory_points),
                                  "dt": int(st.ctx().decision_type)}).run()
    assert seen
    replay = _fresh()
    for rec in seen:
        _drain(replay)
        ts.Engine.step(replay, rec.action)
        _drain(replay)
        assert rec.state_after is not None
        assert rec.state_after["vp"] == int(replay.victory_points), (
            f"step {rec.index}: snapshot is not the post-action state")


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
        if not ts.Engine.try_step(replay, rec.action):
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
