"""A coup's die is stated in the log, and is carried on the target choice.

The log prints it on the result line ("SUCCESS: 6 [ + 2 - 2x2 = 4 ]"). The converter used to
ignore it and search the rng for a seed that reproduced the resulting board instead, which
fails outright whenever anything else in the entry touches the same country -- the expectation
then holds that country's *final* value, which no single roll can produce.

The die rides on the target choice in secondary_id rather than being handed to the chance node
that usually follows, because Che's two free coups have no such node: they resolve inside the
event, where secondary_id is the only way in.
"""
import glob
import gzip
import json
import os
from typing import Optional

import pytest

from tools.lib.ts_replayer_convert import convert_game
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = "/workspace/data/datasets/ts_replayer"

pytestmark = pytest.mark.skipif(
    not glob.glob(os.path.join(CORPUS, "*.json.gz")),
    reason="ts-replayer corpus not downloaded")


def _entry(text: str):
    return parse_entry({"num": "4", "player": "USSR", "phase": "AR4",
                        "card": "Che", "text": text})


def _convert(replay_id: int):
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return convert_game(json.load(f))


def _failed_at(conv) -> Optional[str]:
    return None if conv.failure is None else f"T{conv.failure.turn} {conv.failure.phase}"


def test_every_coup_in_an_entry_keeps_its_own_die() -> None:
    """Che coups twice, and coup_roll held only the last."""
    e = _entry("Turn 4, USSR AR4: Che: Event: Che\n"
               "Coup (3 Ops):\nTarget: Sudan\nSUCCESS: 1 [ + 3 - 2x1 = 2 ]\n"
               "US -1 in Sudan [0][0]\nUSSR +1 in Sudan [0][1]\n"
               "Coup (3 Ops):\nTarget: Saharan States\nSUCCESS: 6 [ + 3 - 2x1 = 7 ]\n"
               "US -4 in Saharan States [0][0]\nUSSR +3 in Saharan States [0][3]\n")
    assert e.coup_rolls == [(1, True), (6, True)]
    assert e.coup_roll == 6, "the single field still holds the last, for existing callers"


def test_a_failed_coup_records_its_die_too() -> None:
    e = _entry("Coup (2 Ops):\nTarget: Panama\nFAILURE: 1 [ + 2 - 2x2 = -1 ]\n")
    assert e.coup_rolls == [(1, False)]


def test_ches_two_coups_reproduce_the_logged_boards() -> None:
    """Replay 103 turn 4 AR4: dice of 1 and 6 into Sudan and Saharan States."""
    conv = _convert(103)
    assert conv.failure is None, f"replay 103 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 76


def test_a_coup_whose_country_is_touched_again_in_the_same_entry() -> None:
    """Replay 129 turn 4 headline: Junta coups Panama, then Liberation Theology places there.

    No rng seed could ever reproduce Panama's final [1][1] from the coup alone, so this entry
    stopped the game at 39 of 114 entries.
    """
    conv = _convert(129)
    assert _failed_at(conv) != "T4 Headline", f"replay 129 stops at its coup: {conv.failure}"
    assert conv.entries_converted > 100
