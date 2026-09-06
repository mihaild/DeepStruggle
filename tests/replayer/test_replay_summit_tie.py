"""Summit does not reroll ties, and the log records a tie by saying nothing.

The card reads "the player with the higher modified die roll gains 2 VP and may move the
DEFCON marker one level either direction. Do not reroll ties." A tie therefore awards nothing,
moves no DEFCON and touches no Influence, which leaves the log with nothing to narrate: it
prints "Event: Summit" and stops. Five of the corpus's twelve Summits end that way.

Reading that silence as "the log does not say" left the dice to the engine's own stream. At
turn 4's headline of replay 109 the reconstruction gave the US the 2 VP the log gave to
nobody, and the score stayed 2 adrift until the Southeast Asia Scoring at turn 4 AR2 asserted
against it -- where the scoring itself was right and only its starting score was wrong.
"""
import glob
import gzip
import json
import os
from typing import Dict, List, Optional, Tuple

import pytest

from tools.lib.ts_replayer_convert import _summit_target, convert_game
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = "/workspace/data/datasets/ts_replayer"

pytestmark = pytest.mark.skipif(
    not glob.glob(os.path.join(CORPUS, "*.json.gz")),
    reason="ts-replayer corpus not downloaded")


def _game(replay_id: int) -> Dict:
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return json.load(f)


def _raws(replay_id: int) -> List[Dict]:
    return _game(replay_id)["all_turns"]


def _summit_entry(replay_id: int) -> Tuple[int, Optional[int]]:
    for i, raw in enumerate(_raws(replay_id)):
        present, score = _summit_target(parse_entry(raw))
        if present:
            return i, score
    raise AssertionError(f"replay {replay_id} has no Summit")


def test_a_summit_that_narrates_nothing_is_read_as_a_tie() -> None:
    """Replay 109 turn 4's headline: "Event: Summit" and not another word."""
    index, score = _summit_entry(109)
    assert index == 39
    assert score is None, "no score after Event: Summit means a tie, not an unknown"


def test_a_summit_that_narrates_a_winner_is_read_as_that_score() -> None:
    """Replay 113 turn 8: "USSR gains 2 VP. Score is USSR 13." -- US-positive, so -13."""
    _index, score = _summit_entry(113)
    assert score == -13


def test_the_score_is_read_from_after_the_summit_line() -> None:
    """A headline resolves two cards, and the other one writes VP lines of its own.

    Taking the entry's last narrated score would attribute the partner card's points to
    Summit, so the search starts at the "Event: Summit" marker.
    """
    index, _score = _summit_entry(113)
    raw = dict(_raws(113)[index])
    raw["text"] = ("Event: Europe Scoring\nUS gains 4 VP. Score is US 4.\n\n"
                   + str(raw["text"]))
    present, score = _summit_target(parse_entry(raw))
    assert present
    assert score == -13, "the score stated before the Summit line is not Summit's"


def test_a_summit_with_no_event_line_is_not_claimed() -> None:
    assert _summit_target(parse_entry(_raws(113)[0])) == (False, None)


@pytest.mark.parametrize("replay_id", [109, 110, 113, 118, 125, 157])
def test_the_summit_games_convert_through_their_summit(replay_id: int) -> None:
    index, _score = _summit_entry(replay_id)
    conv = convert_game(_game(replay_id))
    assert conv.entries_converted > index, (
        f"replay {replay_id} stopped at {conv.failure} before its Summit at entry {index}")
