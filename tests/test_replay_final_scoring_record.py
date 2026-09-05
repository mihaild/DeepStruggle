"""The log writes final scoring on a record of its own, uncapped.

ts-replayer hangs it on a header with no card and no play -- "Turn 10, US AR8: :" -- and states
the whole swing in one line. The engine has already played it out and stopped at the cap, so
the record is something to recognise rather than to drive.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Dict, List, cast

import pytest

from tools.lib.ts_replayer_convert import convert_game
from tools.lib.ts_replayer_parse import parse_entry

_CORPUS = "/workspace/data/datasets/ts_replayer"


def _game(replay_id: int) -> Dict[str, object]:
    path = os.path.join(_CORPUS, f"{replay_id}.json.gz")
    if not os.path.exists(path):
        pytest.skip("corpus not downloaded")
    with gzip.open(path, "rt") as f:
        return json.load(f)


def test_the_record_states_a_score_past_the_cap() -> None:
    """Replay 69: "US gains 28 VP. Score is US 34.", where the game is won at 20."""
    raws = cast(List[Dict[str, object]], _game(69)["all_turns"])
    last = parse_entry(raws[-1])
    assert (last.turn, last.phase) == (10, "AR8")
    assert not last.card
    assert last.vp_gains == [("US", 28, "US", 34)]


def test_replay_69_converts_in_full_and_ends_where_the_log_ends() -> None:
    """Clamped, the log's 34 and the engine's 20 are the same ending -- so an entry the engine
    has already played out is not one it ended early."""
    conv = convert_game(_game(69))
    assert conv.failure is None, f"replay 69 stopped at {conv.failure}"
    assert conv.game_ended is True
    assert conv.entries_converted == conv.entries_total == 154


@pytest.mark.parametrize("replay_id", [154, 105, 109])
def test_the_other_games_that_end_at_the_cap_are_unaffected(replay_id: int) -> None:
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
    assert conv.game_ended is True
