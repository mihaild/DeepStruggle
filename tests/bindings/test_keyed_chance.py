"""A keyed situation retains every packed field, including when a player is USSR (-1)."""
from __future__ import annotations

import copy

import pytest
import ts_engine as ts


@pytest.mark.parametrize("player", [-1, 1])
@pytest.mark.parametrize("field,value,context", [
    ("turn", 5, False), ("action_round", 3, False),
    ("decision_type", int(ts.DecisionType.POINT_NODE), True),
    ("pending_roll", int(ts.RollType.COUP), True), ("roll_target", 24, True),
])
def test_packed_fields_distinguish_situations(player: int, field: str, value: int, context: bool) -> None:
    state = ts.GameState()
    ts.Engine.init_game(state, 1)
    saved = state.to_save_dict()
    saved.update(turn=4, action_round=1, phasing_player=player)
    saved["ctx_stack"][0].update(decision_player=player, decision_type=int(ts.DecisionType.ROLL_DIE),
                                pending_roll=0, roll_actor=player, roll_target=25)
    changed = copy.deepcopy(saved)
    (changed["ctx_stack"][0] if context else changed)[field] = value
    a = ts.state_from_save_dict(saved)
    b = ts.state_from_save_dict(changed)
    assert ts.keyed_key(a, 123).rng_state != ts.keyed_key(b, 123).rng_state
    assert ts.keyed_key(a, 123).rng_state != ts.keyed_key(a, 124).rng_state


def test_che_coups_do_not_reuse_one_die_for_different_targets() -> None:
    ids = {ts.MapData.get_country_info(i)["name"]: i for i in range(84)}
    pairs = []
    for world in range(10):
        state = ts.GameState()
        ts.Engine.init_game(state, 1)
        state.current_phase = ts.Phase.ACTION_ROUND
        state.phasing_player = ts.Player.USSR
        state.turn, state.action_round = 4, 1
        state.ctx().pending_op_card = 107
        for name in ("Colombia", "Sudan"):
            state.set_country(ids[name], 2, 0)
        ts.CardHandlers.trigger_event(state, 107, ts.Player.USSR)
        dice = []
        for name in ("Colombia", "Sudan"):
            ts.CardHandlers.handle_event_step(state, ts.MicroAction(ts.DecisionType.POINT_NODE, ids[name], 0, 0))
            state = ts.keyed_key(state, world)
            ts.CardHandlers.handle_event_step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
            dice.append(state.to_save_dict()["last_die_roll"])
        pairs.append(dice)
    assert any(a != b for a, b in pairs), pairs
