"""Forcing the action round from the log skips the bookkeeping the engine does when it ends one.

Every entry begins by forcing turn, action round and phasing player to what the log names,
because an entry the driver could not reproduce would otherwise leave them a step out. That
forcing goes around advance_after_action_round, which is where the engine clears the state
describing the round just finished.

defcon_dropped_to_2_in_ar is exactly that kind of state: NORAD asks whether DEFCON fell to 2
during *this* action round, so a flag left standing from an earlier one fires it in a round
where nothing happened at all.
"""
import glob
import gzip
import json
import os

import pytest
import ts_engine as ts

from tools.lib.ts_replayer_convert import _reconcile_turn, convert_game
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = "/workspace/data/datasets/ts_replayer"

pytestmark = pytest.mark.skipif(
    not glob.glob(os.path.join(CORPUS, "*.json.gz")),
    reason="ts-replayer corpus not downloaded")


def test_the_defcon_flag_does_not_outlive_its_action_round() -> None:
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    state.current_phase = ts.Phase.ACTION_ROUND
    state.ctx().decision_player = ts.Player.USSR
    state.ctx().decision_type = ts.DecisionType.SELECT_CARD
    state.defcon_dropped_to_2_in_ar = 1

    entry = parse_entry({"num": "5", "player": "USSR", "phase": "AR1", "card": None,
                         "text": "Turn 5, USSR AR1: Duck and Cover: ...\n"})
    _reconcile_turn(state, entry, 219)
    assert state.defcon_dropped_to_2_in_ar == 0


def test_replay_296_converts_end_to_end() -> None:
    """It used to stop at turn 5 AR1, where the USSR discards a card to escape Bear Trap and
    the US was handed a NORAD placement the log has no trace of."""
    with gzip.open(os.path.join(CORPUS, "296.json.gz"), "rt") as f:
        conv = convert_game(json.load(f))
    assert conv.failure is None, f"replay 296 stopped at {conv.failure}"


@pytest.mark.parametrize("replay_id", [219, 220])
def test_the_bear_trap_games_get_past_turn_5(replay_id: int) -> None:
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        conv = convert_game(json.load(f))
    assert conv.failure is None or conv.failure.turn > 5, (
        f"replay {replay_id} stops at {conv.failure}")
