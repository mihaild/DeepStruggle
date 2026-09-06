"""An event that may stop asks only for what its own lines record.

Left to fall through to the Ops queue, an event with countries still queued under its header
takes one the entry's coup or realignment was going to use, and the count it owes goes with it.
Only where the event's own queue is empty does the Ops queue answer -- which is how Che's free
coups, recorded under Che's header but spent as Operations, still find their targets.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Dict, List, cast

import pytest

from tools.lib.ts_replayer_convert import convert_game, event_point_queues
from tools.lib.ts_replayer_parse import parse_entry

_CORPUS = "/workspace/data/datasets/ts_replayer"
_DE_STALINIZATION = 33
_FINLAND = 5
_CHILE = 81


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


def test_the_event_queue_holds_the_relocation_and_the_coup_is_the_ops_queues() -> None:
    """Replay 80 turn 1 AR1: the US plays De-Stalinization for Operations, its event relocates
    one USSR Influence from Finland to Chile, and the Operations coup Egypt."""
    e = _entry(80, 1, "AR1", "US")
    assert event_point_queues(e)[_DE_STALINIZATION] == [_FINLAND, _CHILE]
    assert e.targets == [29], "Egypt is the coup's, not the event's"


def test_replay_80_converts_in_full() -> None:
    """The event may relocate up to four and relocated one. Falling through to the Ops queue
    took a second out of Egypt -- the country the coup was about to hit -- and two removed
    means two to place, so the entry ended with the engine still asking."""
    conv = convert_game(_game(80))
    assert conv.failure is None, f"replay 80 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 114


@pytest.mark.parametrize("replay_id", [131, 105, 150, 123])
def test_a_free_coup_recorded_under_its_event_still_finds_its_target(replay_id: int) -> None:
    """Che's coups are printed under Che's own header and spent as Operations."""
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
