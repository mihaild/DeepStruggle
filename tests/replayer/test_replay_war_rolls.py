"""A war's die is stated in the log, and has to reach the chance node that decides it.

A war with a fixed target -- Korean War, Arab-Israeli War -- rolls inside the event with
nothing to choose, so there is no decision to hang the outcome on. The log records the roll on
its result line ("DEFEAT: 2 (-1)  < 4"), which the parser used to discard as a line that
"carries no state change": it carries the most consequential one on the card, since a won war
replaces every point of the loser's Influence in the target.
"""
import glob
import gzip
import json
import os
from typing import Optional

import pytest

from tools.lib.ts_replayer_convert import convert_game
from tools.lib.ts_replayer_parse import country_id, parse_entry

CORPUS = "/workspace/data/datasets/ts_replayer"

pytestmark = pytest.mark.skipif(
    not glob.glob(os.path.join(CORPUS, "*.json.gz")),
    reason="ts-replayer corpus not downloaded")


def _entry(text: str):
    return parse_entry({"num": "6", "player": "US", "phase": "AR4",
                        "card": "Korean War*", "text": text})


def _convert(replay_id: int):
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return convert_game(json.load(f))


def _failed_at(conv) -> Optional[str]:
    return None if conv.failure is None else f"T{conv.failure.turn} {conv.failure.phase}"


def test_a_defeat_line_yields_its_die_and_modifier() -> None:
    e = _entry("Turn 6, US AR4: Korean War*: Event: Korean War*\n"
               "War in South Korea\n"
               "DEFEAT: 2 (-1)  < 4\n"
               "USSR Military Ops to 5\n")
    assert e.war_targets == [country_id("South Korea")]
    assert e.war_rolls == [(2, -1, False)]
    assert not e.unparsed


def test_a_victory_line_yields_its_die() -> None:
    e = _entry("Turn 9, USSR AR5: Brush War: Event: Brush War\n"
               "War in Thailand\n"
               "VICTORY: 5 >= 3\n"
               "US -6 in Thailand [0][3]\n")
    assert e.war_rolls == [(5, 0, True)]


def test_the_modifier_is_optional() -> None:
    e = _entry("War in Israel\nDEFEAT: 3 < 4\n")
    assert e.war_rolls == [(3, 0, False)]


@pytest.mark.parametrize("replay_id,was_stopping_at", [(121, "T6 AR4"), (122, "T6 AR4"),
                                                       (154, "T2 AR3"), (159, "T3 AR1")])
def test_the_war_games_get_past_their_war(replay_id: int, was_stopping_at: str) -> None:
    """Each of these lost a war in the log and won it in the reconstruction.

    The die could not be found by seeding the rng either: where a coup shares the entry it
    picks the seed first, and the guard against re-seeding then suppressed the search. Replay
    121 turn 6 AR4 is exactly that -- the US coups Tunisia, then loses the Korean War.
    """
    conv = _convert(replay_id)
    assert _failed_at(conv) != was_stopping_at, (
        f"replay {replay_id} still stops at its war: {conv.failure}")
