"""A branch that ends the game is still the branch that was taken, if the log says so.

Choosing between an event's branches is done by trying each on a clone and seeing which reaches
what the log records. Ending the game is heavily penalised there, because a branch that ends it
is almost never the one a player took and taking it by accident loses everything after.

Wargames is the exception: at DEFCON 2 it offers 6 VP to the opponent and an immediate end, or
nothing at all, and the two differ in nothing but the score. The log states that score, which
settles it.
"""
import glob
import gzip
import json
import os

import pytest

from tools.lib.ts_replayer_convert import convert_game
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = "/workspace/data/datasets/ts_replayer"

pytestmark = pytest.mark.skipif(
    not glob.glob(os.path.join(CORPUS, "*.json.gz")),
    reason="ts-replayer corpus not downloaded")


def _convert(replay_id: int):
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return convert_game(json.load(f))


@pytest.mark.parametrize("replay_id", [113, 118, 125])
def test_the_wargames_games_convert_end_to_end(replay_id: int) -> None:
    """Turn 8 AR1: the USSR takes the ending and the log reads "US gains 6 VP. Score is USSR 7."

    Declining instead left the score six adrift and the game running on past its end.
    """
    conv = _convert(replay_id)
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"


def test_the_ending_entry_is_the_last_one_converted() -> None:
    """Wargames ends the game, so nothing after it is training data."""
    with gzip.open(os.path.join(CORPUS, "113.json.gz"), "rt") as f:
        raws = json.load(f)["all_turns"]
    wargames = [i for i, r in enumerate(raws)
                if (parse_entry(r).card or "").startswith("Wargames")]
    assert wargames, "replay 113 should contain a Wargames entry"
    conv = _convert(113)
    assert conv.entries_converted == conv.entries_total
    assert conv.entries_total == wargames[0] + 1, (
        "the game ends on the Wargames entry, so that is the last one converted")
