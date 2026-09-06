"""NORAD's Influence is a placement even when it lands in the country just couped.

A coup takes the opponent's Influence away and puts the couper's down; a realignment only takes
away. The *couped* player gaining Influence in that same country is therefore not a result of
the roll at all, and the only thing in the game that does it is NORAD, which lets the US add 1
Influence after an action round in which it lost some.

The driver queued a coup entry's trailing Influence lines only where they named a country that
was not the target, which is right for everything a coup itself does. At turn 6 AR3 of replay
104 the US put its NORAD Influence into Poland and it was queued without trouble; at turn 6 AR1
of replay 131 the USSR coups Argentina and the US puts it straight back into Argentina, and the
placement was dropped -- leaving the engine asking which of 32 countries the US meant, with the
log's own answer thrown away.
"""
import glob
import gzip
import json
import os
from typing import Dict

import pytest

from tools.lib.corpus_paths import corpus_dir
from tools.lib.ts_replayer_convert import convert_game, point_queue
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = str(corpus_dir())
ARGENTINA = 82
PANAMA = 70
POLAND = 15



def _game(replay_id: int) -> Dict:
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return json.load(f)


def _entry(replay_id: int, turn: int, phase: str, player: str):
    for raw in _game(replay_id)["all_turns"]:
        e = parse_entry(raw)
        if (e.turn, e.phase, e.player) == (turn, phase, player):
            return e
    raise AssertionError(f"replay {replay_id} has no {player} {phase} of turn {turn}")


def test_norad_into_the_couped_country_is_queued() -> None:
    """Replay 131 turn 6 AR1: the USSR coups Argentina, the US puts 1 back into Argentina."""
    e = _entry(131, 6, "AR1", "USSR")
    assert e.mode == "coup" and e.targets == [ARGENTINA]
    assert ("US", -1, ARGENTINA, 1, 0) in e.influence, "the coup's removal"
    assert ("US", 1, ARGENTINA, 2, 0) in e.influence, "NORAD's placement"
    assert point_queue(e) == [ARGENTINA, ARGENTINA], (
        "the target, and then the placement the log records after it")


def test_norad_into_another_country_still_works() -> None:
    """Replay 104 turn 6 AR3: couped Panama, NORAD into Poland -- the case that worked."""
    e = _entry(104, 6, "AR3", "USSR")
    assert point_queue(e) == [PANAMA, POLAND]


def test_the_coupers_own_influence_is_not_a_placement() -> None:
    """A successful coup putting the couper's Influence down is the roll's result.

    Only the couped player gaining is unexplainable by the coup, so only that is queued.
    """
    e = _entry(143, 4, "AR1", "USSR")
    assert e.targets == [PANAMA]
    gains = [(side, delta) for side, delta, cid, _u, _s in e.influence
             if cid == PANAMA and delta > 0]
    assert gains == [("US", 1)], "the USSR couped, so a US gain is NORAD's"
    assert point_queue(e) == [PANAMA, PANAMA]


@pytest.mark.parametrize("replay_id", [131, 143, 104])
def test_the_norad_games_convert_end_to_end(replay_id: int) -> None:
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
