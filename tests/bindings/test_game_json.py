"""The JSON boundary must round-trip, including the cases that make it non-trivial.

`json_to_action` is the direction that did not exist, and it is what lets a replay be re-driven
through the engine rather than merely rendered. If it is lossy anywhere, a replay silently plays
differently from the game it recorded -- which is the failure mode this whole line of work has been
chasing.
"""

from __future__ import annotations

import numpy as np
import pytest

import ts_engine as ts
from bindings.action_encoder import ActionEncoder
from tools.lib.game_json import (action_to_json, json_to_action, legal_actions_json,
                                 state_to_json)
from tools.lib.game_step import IllegalActionError, step_checked


def _fresh(seed: int = 4242) -> ts.GameState:
    s = ts.GameState()
    ts.Engine.init_game(s, seed)
    return s


def test_every_action_of_a_whole_game_round_trips() -> None:
    """flat -> json -> MicroAction must step the engine exactly as the flat action would."""
    a, b = _fresh(), _fresh()
    steps = 0
    for _ in range(2000):
        if ts.Engine.is_terminal(a):
            break
        mask = np.asarray(ActionEncoder.get_legal_mask(a))
        legal = np.flatnonzero(mask)
        if not len(legal):
            break
        flat = int(legal[steps % len(legal)])

        ma = json_to_action(a, dict(action_to_json(a, flat)))
        assert ts.Engine.try_step(b, ma), "round-tripped action was refused"
        step_checked(a, flat)
        steps += 1

        assert int(a.ctx().decision_type) == int(b.ctx().decision_type), (
            f"states diverged after {steps} actions")
        assert int(a.turn) == int(b.turn) and int(a.victory_points) == int(b.victory_points)
    assert steps > 100, f"only {steps} actions exercised"


def test_a_forced_die_survives_the_round_trip() -> None:
    """The case that cannot go through a flat index.

    The flat space has ONE slot for "roll" (211); the value lives in primary_id. Round-tripping a
    recorded roll of 3 through flat would turn it into a random roll, and a replay would quietly
    stop reproducing its own game.
    """
    s = _fresh()
    for _ in range(2000):
        if ts.Engine.is_terminal(s):
            break
        if s.ctx().decision_type == ts.DecisionType.ROLL_DIE:
            break
        legal = np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(s)))
        if not len(legal):
            break
        ts.Engine.step_flat(s, int(legal[0]))
    if s.ctx().decision_type != ts.DecisionType.ROLL_DIE:
        pytest.skip("no die roll reached with the scripted opening")

    recorded = {"decision_type": int(ts.DecisionType.ROLL_DIE), "primary_id": 3,
                "secondary_id": 0, "flags": 0}
    ma = json_to_action(s, recorded)
    assert int(ma.primary_id) == 3, "the forced value was lost"
    assert ts.Engine.try_step(s, ma), "a forced die was refused"


def test_a_malformed_action_raises_rather_than_defaulting() -> None:
    s = _fresh()
    with pytest.raises(IllegalActionError):
        json_to_action(s, {"primary_id": 4})          # no decision_type at all


def test_state_json_carries_an_observation_only_for_the_player_to_move() -> None:
    """An observation for a seat that cannot act is meaningless and invites being read as one."""
    s = _fresh()
    ts.Engine.auto_advance_step(s)
    ctx = s.ctx()
    mover = ctx.decision_player if ctx.decision_player != ts.Player.NONE else s.phasing_player
    mover_name = "US" if mover == ts.Player.US else "USSR"
    other = "USSR" if mover_name == "US" else "US"

    assert "observation_f32" in state_to_json(s, for_role=mover_name)
    assert "observation_f32" not in state_to_json(s, for_role=other)
    assert "observation_f32" not in state_to_json(s)


def test_legal_actions_json_matches_the_mask() -> None:
    s = _fresh()
    ts.Engine.auto_advance_step(s)
    info = legal_actions_json(s)
    mask = np.asarray(ActionEncoder.get_legal_mask(s))
    assert info["flat_actions"] == [int(i) for i in np.flatnonzero(mask)]
    assert info["decision_type"] == int(s.ctx().decision_type)
