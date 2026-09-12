"""A headline resolves two cards, and neither may spend the other's placements.

The log writes each event's Influence under its own "Event:" header, which is the only thing
saying which placement belongs to which card. The driver held both in one queue and took
whichever target the engine would accept -- fine while the two events want different countries,
and wrong the moment they overlap, because a point one event could not use stayed in the queue
and the next event spent it.

At turn 8's headline of replay 150 the US's East European Unrest removes 2 USSR Influence from
each of East Germany, Poland and Yugoslavia. That is three decisions for five queued points,
because the log writes the amount and not the decision, and the two left over were still there
when the USSR's The Reformer asked where to place its four. The Reformer put one into Poland,
which the log has it never touching, and West Germany finished an Influence short.
"""
import glob
import gzip
import json
import os
from typing import Dict

import pytest

from tools.lib.corpus_paths import corpus_dir
from tools.lib.ts_replayer_convert import (card_id, convert_game,
                                           event_point_queues, event_queue)
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = str(corpus_dir())
EAST_GERMANY, POLAND, YUGOSLAVIA, WEST_GERMANY = 14, 15, 18, 7



def _game(replay_id: int) -> Dict:
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return json.load(f)


def _headline(replay_id: int, turn: int):
    for raw in _game(replay_id)["all_turns"]:
        e = parse_entry(raw)
        if e.turn == turn and e.phase == "Headline":
            return e
    raise AssertionError(f"replay {replay_id} has no headline on turn {turn}")


def test_the_log_writes_each_events_influence_under_its_own_header() -> None:
    e = _headline(150, 8)
    assert e.headlines == {"USSR": "The Reformer*", "US": "East European Unrest"}
    assert set(e.influence_by_event) == {"The Reformer*", "East European Unrest"}


def test_each_event_gets_only_its_own_points() -> None:
    queues = event_point_queues(_headline(150, 8))
    unrest_id, reformer_id = card_id("East European Unrest"), card_id("The Reformer*")
    assert unrest_id is not None and reformer_id is not None
    unrest = queues[unrest_id]
    reformer = queues[reformer_id]
    assert unrest == [EAST_GERMANY, EAST_GERMANY, POLAND, POLAND, YUGOSLAVIA]
    assert reformer == [EAST_GERMANY, EAST_GERMANY, WEST_GERMANY, WEST_GERMANY], (
        "the Reformer places two and two, and never touches Poland")
    assert POLAND not in reformer


def test_the_shared_queue_holds_the_same_points_between_them() -> None:
    """Which is why spending one in an event's own queue has to spend it there too."""
    e = _headline(150, 8)
    shared = sorted(event_queue(e))
    per_event = sorted(c for q in event_point_queues(e).values() for c in q)
    assert shared == per_event


def test_replay_150_converts_end_to_end() -> None:
    """It stopped on the headline with Poland an Influence over and West Germany one under."""
    conv = convert_game(_game(150))
    assert conv.failure is None, f"replay 150 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total


@pytest.mark.parametrize("replay_id", [150, 128, 100, 127])
def test_the_two_event_headline_games_convert(replay_id: int) -> None:
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
