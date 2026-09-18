"""A play the rules do not allow is not a decision worth learning.

The engine is right to refuse it, and the humans did it anyway, so the entry is driven with
whatever the engine accepts and emits nothing. The board it reaches is put back from the log at
the next entry, as every entry's is. Listed, and only for plays the rules settle outright.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Dict, List, cast

import pytest

from tools.lib.corpus_paths import corpus_dir
from tools.lib.ts_replayer_convert import _INVALID_PLAYS, _LOG_MISCOUNTED, convert_game
from tools.lib.ts_replayer_parse import parse_entry

_CORPUS = str(corpus_dir())


def _game(replay_id: int) -> Dict[str, object]:
    path = os.path.join(_CORPUS, f"{replay_id}.json.gz")
    if not os.path.exists(path):
        pytest.skip("corpus not downloaded")
    with gzip.open(path, "rt") as f:
        return json.load(f)


def test_a_missile_envy_card_may_not_go_to_the_space_race() -> None:
    """Replay 59 turn 8: the USSR headlines Missile Envy and takes Five Year Plan, a US card,
    so they get its Operations rather than its event -- and a space race attempt is not
    Operations. The attempt failed in the log, so nothing came of it there either."""
    raws = cast(List[Dict[str, object]], _game(59)["all_turns"])
    entry = next(parse_entry(r) for r in raws
                 if (parse_entry(r).turn, parse_entry(r).phase) == (8, "Headline"))
    assert entry.mode == "space"
    assert "Missile Envy" in (entry.headlines or {}).values()
    assert (8, "Headline", "both") in _INVALID_PLAYS[59]


def test_the_entry_converts_and_teaches_nothing() -> None:
    conv = convert_game(_game(59))
    assert conv.failure is None, f"replay 59 stopped at {conv.failure}"
    assert conv.invalid_decisions == 1, "the one decision inside it, emitted as nothing"
    assert conv.entries_converted == 122


def test_the_voice_of_america_entry_converts_with_the_board_taken_from_the_log() -> None:
    """Replay 260 turn 10 AR7: the log removes 3 of the card's 4 with a fourth still available.

    P17 6a took away the early stop that placement never should have had, so the engine now
    completes the removal and the boards differ by construction -- the engine's is right and the
    record's is not. The entry still has to convert: it is what carries the game to its ending,
    and without it the whole of turn 10 rewinds as an unfinished fragment.
    """
    conv = convert_game(_game(260))
    assert conv.failure is None, f"replay 260 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 144
    assert conv.game_ended, "the entry is what carries this game to its ending"
    assert conv.invalid_board_resyncs == 1


def test_no_other_game_carries_an_invalid_play() -> None:
    """The list is for what the rules settle outright, not for a decision the log fails to
    determine -- that is a failure, and stays one.

    Two entries, and both are rules calls rather than gaps: replay 59's Missile Envy card put on
    the space race, and replay 260's Voice of America stopped one short of its mandatory four.
    """
    assert sum(len(v) for v in _INVALID_PLAYS.values()) == 2
    for replay_id, entries in _INVALID_PLAYS.items():
        for key in entries:
            assert key not in _LOG_MISCOUNTED.get(replay_id, {})
