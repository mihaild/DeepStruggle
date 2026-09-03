"""Military operations belong to the turn they were spent in.

The engine zeroes both counts when a turn ends, and the end-of-turn comparison that pays a
player for their opponent's shortfall runs on the counts as that turn left them. Reconciling
the counts from an entry of an *earlier* turn carries the old numbers across the boundary, and
a side that looks full has no deficit to pay for -- so the comparison quietly awards nothing.

That is invisible at the time. At turn 6 of replay 60 it cost the USSR the 2 VP they were owed,
and the game ran seven more entries before Central America Scoring landed on 20 and ended it.
"""
import glob
import gzip
import json
import os

import pytest
import ts_engine as ts

from tools.lib.ts_replayer_convert import _reconcile_scalars, convert_game
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = "/workspace/data/datasets/ts_replayer"

pytestmark = pytest.mark.skipif(
    not glob.glob(os.path.join(CORPUS, "*.json.gz")),
    reason="ts-replayer corpus not downloaded")


def _entry(turn: int, text: str):
    return parse_entry({"num": str(turn), "player": "USSR", "phase": "AR7",
                        "card": "Arab-Israeli War", "text": text})


def test_counts_from_the_current_turn_are_applied() -> None:
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    state.turn = 6
    state.us_mil_ops = 0
    state.ussr_mil_ops = 0
    _reconcile_scalars(state, _entry(6, "USSR Military Ops to 5\n"))
    assert int(state.ussr_mil_ops) == 5


def test_counts_from_an_earlier_turn_are_not_carried_over() -> None:
    """The engine has already zeroed them; the previous turn's figures must not come back."""
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    state.turn = 6
    state.us_mil_ops = 0
    state.ussr_mil_ops = 0
    _reconcile_scalars(state, _entry(5, "US Military Ops to 5\nUSSR Military Ops to 5\n"))
    assert int(state.us_mil_ops) == 0
    assert int(state.ussr_mil_ops) == 0


def test_replay_60_reaches_the_score_the_log_records() -> None:
    """Turn 6 ends with the US on 0 and the USSR on 5 against DEFCON 2, so the US owes 2.

    With turn 5's counts carried over, both sides looked full and nothing moved; the score then
    stood two ahead of the log all the way to turn 7 AR2, where Central America Scoring's 5 VP
    reached 20 and ended a game that ran to turn 10.
    """
    with gzip.open(os.path.join(CORPUS, "60.json.gz"), "rt") as f:
        conv = convert_game(json.load(f))
    stopped_at_the_divergence = (conv.failure is not None
                                 and conv.failure.turn == 7
                                 and conv.failure.phase == "AR2")
    assert not stopped_at_the_divergence, f"replay 60 stops at its scoring divergence: {conv.failure}"
    assert conv.entries_converted > 95
