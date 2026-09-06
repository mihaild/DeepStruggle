"""Influence the engine moves without asking is Influence the queues have spent.

The driver's queues hold one point per Influence the log records. The engine does not always
ask once per point: Junta places 2 in one country and asks once, and Tear Down This Wall puts
its 3 into East Germany without asking at all. What it moves is spent either way, so the board
before and after each step is what says how many points a queue still owes.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Dict, List, cast

import pytest

from tools.lib.ts_replayer_convert import (
    convert_game,
    event_queue,
    point_queue,
    rows_queued_twice,
)
from tools.lib.ts_replayer_parse import parse_entry

_CORPUS = "/workspace/data/datasets/ts_replayer"
_EAST_GERMANY = 14
_PANAMA = 70


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


def test_a_row_under_an_event_header_is_queued_in_both_queues() -> None:
    """Replay 277 turn 9 AR7: Tear Down This Wall's 3 into East Germany, and 3 Ops of free
    realignment after them. The placement is in the event queue and, since no Ops section
    claims it, in the Ops queue too."""
    e = _entry(277, 9, "AR7", "US")
    assert point_queue(e).count(_EAST_GERMANY) == 3
    assert event_queue(e).count(_EAST_GERMANY) == 3
    assert rows_queued_twice(e) == {_EAST_GERMANY: 3}


def test_replay_277_leaves_the_free_realignments_declined() -> None:
    """The US realigns France once and stops. Spending the two Ops they declined took the
    Influence the card had just placed straight back off the board."""
    conv = convert_game(_game(277))
    assert conv.failure is None, f"replay 277 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 144


def test_a_target_is_not_a_duplicate_of_a_placement_in_the_same_country() -> None:
    """Replay 177 turn 6 AR4: Junta places 2 in Panama and then realigns Panama itself.

    Panama is in the Ops queue three times over -- as the realignment's target and as the
    event's two placements -- and only the placements are the duplicates.
    """
    e = _entry(177, 6, "AR4", "US")
    queue = point_queue(e)
    assert queue.count(_PANAMA) == 3
    assert rows_queued_twice(e) == {_PANAMA: 2}
    assert queue[0] == _PANAMA, "the target comes first, the duplicates after"

    conv = convert_game(_game(177))
    assert conv.failure is None, f"replay 177 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 144


def test_an_event_on_the_far_side_of_a_die_spends_what_it_moves() -> None:
    """Replay 283 turn 3 AR5: the US coups South Africa with Nasser and the card's own event
    follows the roll -- 2 USSR Influence into Egypt, half the US Influence out of it.

    Those are the engine's to move, and both are queued. NORAD's placement, which the log puts
    in Pakistan, took Egypt from the queue instead and put an Influence back where the event
    had just removed one. The country the die itself settles is the only one exempt.
    """
    e = _entry(283, 3, "AR5", "US")
    assert e.targets == [63], "the coup is on South Africa"
    conv = convert_game(_game(283))
    assert conv.failure is None, f"replay 283 stopped at {conv.failure}"
    assert conv.entries_converted == 98


def test_an_entry_with_no_unclaimed_rows_shares_nothing() -> None:
    e = _entry(277, 9, "AR6", "US")
    assert rows_queued_twice(e) == {}
