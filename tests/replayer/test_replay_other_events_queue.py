"""One headline event's placements must not answer for the other's.

The shared event queue holds every placement the entry hangs on an "Event:" header. While a
card is resolving that the entry hangs placements on, those rows may be its; while a card is
resolving that it does not, none of them are.

At turn 5's headline of replay 173 the USSR headlines Che and the US The Voice of America --
which *removes* USSR Influence, four of it, up to two per country. Che's two free coups are
sections and Che places nothing, so the queue held only the Voice of America's removals, and it
did two kinds of damage at once. It answered Che's first coup with Uruguay, the head of those
removals; and by being non-empty it stopped the section holding the real coup targets from
being loaded at all.

The log coups Saharan States first, so the dice went to the wrong countries too: Uruguay took
the 1 that Saharan States should have had, and 1 + 3 - 2x2 is 0 -- a failure where the log
records a success. Neither coup happened, the Voice of America then had nothing to remove, and
both countries stood at the board they had before the headline.
"""
import glob
import gzip
import json
import os
from typing import Dict

import pytest

from tools.lib.corpus_paths import corpus_dir
from tools.lib.ts_replayer_convert import (card_id, convert_game, event_point_queues,
                                           point_queue)
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = str(corpus_dir())
SAHARAN_STATES, URUGUAY, COLOMBIA = 50, 83, 74
CHE = 107



def _game(replay_id: int) -> Dict:
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return json.load(f)


def _headline(replay_id: int, turn: int):
    for raw in _game(replay_id)["all_turns"]:
        e = parse_entry(raw)
        if e.turn == turn and e.phase == "Headline":
            return e
    raise AssertionError(f"replay {replay_id} has no headline on turn {turn}")


def test_the_removals_belong_to_the_voice_of_america_alone() -> None:
    e = _headline(173, 5)
    assert e.headlines == {"USSR": "Che", "US": "The Voice of America"}
    queues = event_point_queues(e)
    assert set(queues) == {card_id("The Voice of America")}, (
        "Che places nothing, so it has no rows of its own")
    voa = queues[card_id("The Voice of America") or 0]
    assert sorted(voa) == sorted([URUGUAY, URUGUAY, SAHARAN_STATES, COLOMBIA])
    assert CHE not in queues


def test_ches_coups_are_sections_with_the_targets_in_order() -> None:
    e = _headline(173, 5)
    assert [s.mode for s in e.sections] == ["coup", "coup"]
    assert [s.targets for s in e.sections] == [[SAHARAN_STATES], [URUGUAY]]
    assert e.coup_rolls == [(1, True), (4, True)], "one die each, in that order"


def test_the_ops_queue_is_the_sections_when_there_are_several() -> None:
    """With more than one section the flat queue is cleared and the sections decide."""
    e = _headline(173, 5)
    assert len(e.sections) > 1
    assert point_queue(e)[:2] == [SAHARAN_STATES, URUGUAY]


def test_replay_173_converts_end_to_end() -> None:
    conv = convert_game(_game(173))
    assert conv.failure is None, f"replay 173 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total


@pytest.mark.parametrize("replay_id", [173, 105, 123, 129, 103])
def test_the_multi_section_event_games_still_convert(replay_id: int) -> None:
    """These are the entries the sections-are-the-authority rule was written for."""
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
