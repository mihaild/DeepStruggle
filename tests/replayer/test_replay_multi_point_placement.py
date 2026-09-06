"""A decision that places more than one Influence spends more than one queued point.

The driver's queues hold one entry per point of Influence the log records, and most decisions
move exactly one. Some move the lot in a single answer -- Junta places 2 in one country, and
the engine asks once -- and the points left queued for that country were spent by that same
answer. Left in, the next decision spends them again.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Dict, List, cast

import pytest

from tools.lib.corpus_paths import corpus_dir
from tools.lib.ts_replayer_convert import convert_game, point_queue
from tools.lib.ts_replayer_parse import parse_entry

_CORPUS = str(corpus_dir())
_ARGENTINA = 82
_UNITED_KINGDOM = 1


def _game(replay_id: int) -> Dict[str, object]:
    path = os.path.join(_CORPUS, f"{replay_id}.json.gz")
    if not os.path.exists(path):
        pytest.skip("corpus not downloaded")
    with gzip.open(path, "rt") as f:
        return json.load(f)


def test_replay_260_puts_norads_influence_where_the_log_does() -> None:
    """Turn 9 AR3: Junta's 2 into Argentina, a coup on Argentina that drops DEFCON to 2, and
    NORAD's 1 into the United Kingdom -- printed last, under no header of its own."""
    game = _game(260)
    raws = cast(List[Dict[str, object]], game["all_turns"])
    entry = next(parse_entry(r) for r in raws
                 if parse_entry(r).turn == 9 and parse_entry(r).phase == "AR3"
                 and parse_entry(r).player == "US")
    queue = point_queue(entry)
    assert queue.count(_ARGENTINA) == 3 and _UNITED_KINGDOM in queue, (
        "the premise: three points in Argentina and one in the United Kingdom")

    conv = convert_game(game)
    assert conv.failure is None, f"replay 260 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 144


@pytest.mark.parametrize("replay_id,entries", [(131, 144), (123, 144), (184, 144)])
def test_the_other_games_this_freed_convert_in_full(replay_id: int, entries: int) -> None:
    """All of them lose an Influence to a queue point that had already been spent."""
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == entries
