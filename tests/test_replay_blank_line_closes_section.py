"""A blank line closes the Ops section above it, not just the event's placements.

A coup prints its removal and its placement directly under its own header. A placement below a
blank line is something else -- NORAD's, in every case in the corpus -- and reading it as part
of the coup relied on the victim rule, which only catches a placement by the side that just
lost Influence there.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Dict, List, cast

import pytest

from tools.lib.ts_replayer_convert import convert_game, point_queue
from tools.lib.ts_replayer_parse import parse_entry

_CORPUS = "/workspace/data/datasets/ts_replayer"
_ARGENTINA = 82
_UNITED_KINGDOM = 1


def _game(replay_id: int) -> Dict[str, object]:
    path = os.path.join(_CORPUS, f"{replay_id}.json.gz")
    if not os.path.exists(path):
        pytest.skip("corpus not downloaded")
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _entry(replay_id: int, turn: int, phase: str, player: str):
    raws = cast(List[Dict[str, object]], _game(replay_id)["all_turns"])
    for raw in raws:
        e = parse_entry(raw)
        if (e.turn, e.phase, e.player) == (turn, phase, player):
            return e
    raise AssertionError(f"no entry at replay {replay_id} T{turn} {phase} {player}")


def test_a_placement_after_the_gap_is_not_the_coups() -> None:
    """Replay 281 turn 6 AR4: ABM Treaty's coup on Argentina has a margin of 3, which removes
    the 3 USSR Influence there and leaves nothing over to place. The "US +1 in Argentina" after
    the blank line is NORAD's, and it is the same side and the same country as a coup's own
    placement would be."""
    e = _entry(281, 6, "AR4", "US")
    coup = [s for s in e.sections if s.mode == "coup"]
    assert len(coup) == 1
    assert [row[1] for row in coup[0].influence] == [-3], (
        "the removal, and nothing after the gap")
    assert point_queue(e).count(_ARGENTINA) == 2, "the coup's target, and NORAD's placement"


def test_replay_281_converts_in_full() -> None:
    conv = convert_game(_game(281))
    assert conv.failure is None, f"replay 281 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 145


def test_the_placement_is_queued_where_the_log_prints_it() -> None:
    """Replay 260 turn 9 AR3: Junta's 2 into Argentina, a coup on Argentina, then NORAD's 1
    into the United Kingdom -- last in the log and last in the queue."""
    e = _entry(260, 9, "AR3", "US")
    queue = point_queue(e)
    assert queue == [_ARGENTINA, _ARGENTINA, _ARGENTINA, _UNITED_KINGDOM]

    conv = convert_game(_game(260))
    assert conv.failure is None, f"replay 260 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 144


@pytest.mark.parametrize("replay_id", [131, 104, 150, 133, 277, 203])
def test_the_other_norad_games_are_unaffected(replay_id: int) -> None:
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
