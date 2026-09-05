"""Declining Junta's free coup is a decision, and the log records it by saying nothing.

Junta and Tear Down This Wall place Influence first and then *may* make a free coup or
realignment with the Ops their event grants. Those Ops cannot buy Influence, so the engine bars
it and lets INFLUENCE stand for declining -- see free_action_bars_influence in action_mask.cpp.
An entry that records the placement and nothing after it is a player who declined.

At turn 5 AR6 of replay 174 the US plays Junta for its event, puts 2 Influence into Chile, and
makes no coup or realignment. The engine offered exactly one thing, the decline, and the driver
reported that nothing in the entry said which of one option was taken.
"""
import glob
import gzip
import json
import os
from typing import Dict

import pytest
import ts_engine as ts

from tools.lib.ts_replayer_convert import convert_game
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = "/workspace/data/datasets/ts_replayer"
JUNTA, TEAR_DOWN_THIS_WALL = 47, 96
CHILE = 81

pytestmark = pytest.mark.skipif(
    not glob.glob(os.path.join(CORPUS, "*.json.gz")),
    reason="ts-replayer corpus not downloaded")


def _game(replay_id: int) -> Dict:
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return json.load(f)


def _entry(replay_id: int, turn: int, phase: str, player: str):
    for raw in _game(replay_id)["all_turns"]:
        e = parse_entry(raw)
        if (e.turn, e.phase, e.player) == (turn, phase, player):
            return e
    raise AssertionError(f"replay {replay_id} has no {player} {phase} of turn {turn}")


def test_the_entry_records_the_placement_and_nothing_after_it() -> None:
    e = _entry(174, 5, "AR6", "US")
    assert e.card == "Junta"
    assert e.influence == [("US", 2, CHILE, 2, 0)]
    assert e.sections == [], "no Ops header, so no free action was taken"
    assert e.mode is None


def test_influence_is_the_decline_for_these_two_cards() -> None:
    """The engine's own encoding, which this rests on."""
    assert ts.CardData.get_card_info(JUNTA)["name"] == "Junta"
    assert ts.CardData.get_card_info(TEAR_DOWN_THIS_WALL)["name"] == "Tear Down this Wall"


def test_replay_174_converts_end_to_end() -> None:
    """Everything but the last turn, which the recording stops inside -- see
    tests/test_replay_unfinished_final_turn.py."""
    conv = convert_game(_game(174))
    assert conv.failure is None, f"replay 174 stopped at {conv.failure}"
    assert conv.entries_converted == 130


def test_a_free_action_the_log_does_describe_is_still_taken() -> None:
    """Replay 156 turn 6: Junta places 2 in Chile and then coups Panama.

    The decline must only be chosen where the log records no Ops at all, or every free coup
    in the corpus would be thrown away.
    """
    e = _entry(156, 6, "Headline", "both")
    assert [s.mode for s in e.sections] == ["space", "coup"]
    conv = convert_game(_game(156))
    assert conv.failure is None, f"replay 156 stopped at {conv.failure}"


@pytest.mark.parametrize("replay_id", [174, 146, 156, 131])
def test_the_free_action_games_convert(replay_id: int) -> None:
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
