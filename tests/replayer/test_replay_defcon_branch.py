"""How I Learned To Stop Worrying names a DEFCON the entry never states.

The card sets DEFCON to whatever its player chooses, and the log narrates only the 5 Military
Operations that come with it. An entry's own DEFCON field is the level it began at, so the
level chosen shows up in the entry that follows.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Dict, List, cast

import pytest

from tools.lib.ts_replayer_convert import _defcon_after, convert_game
from tools.lib.ts_replayer_parse import parse_entry

_CORPUS = "/workspace/data/datasets/ts_replayer"


def _game(replay_id: int) -> Dict[str, object]:
    path = os.path.join(_CORPUS, f"{replay_id}.json.gz")
    if not os.path.exists(path):
        pytest.skip("corpus not downloaded")
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _raws(replay_id: int) -> List[Dict[str, object]]:
    return cast(List[Dict[str, object]], _game(replay_id)["all_turns"])


def test_the_level_comes_from_the_entry_after() -> None:
    """Replay 260 turn 5 AR7: the USSR sets DEFCON to 3, and only the US's AR7 says so."""
    raws = _raws(260)
    index = next(i for i, r in enumerate(raws)
                 if parse_entry(r).turn == 5 and parse_entry(r).phase == "AR7"
                 and parse_entry(r).player == "USSR")
    entry = parse_entry(raws[index])
    assert entry.defcon is not None and int(entry.defcon) == 2, (
        "the entry's own field is the level it began at")
    assert _defcon_after(raws, index, entry) == 3


def test_a_turns_last_entry_reads_no_level_from_the_next_turn() -> None:
    """DEFCON improves by one at every turn end, so the next turn's field is not this one's."""
    raws = _raws(260)
    last = max(i for i, r in enumerate(raws) if parse_entry(r).turn == 5)
    assert _defcon_after(raws, last, parse_entry(raws[last])) is None


def test_replay_260_gets_past_the_choice() -> None:
    """Left to the board and the score -- neither of which the card moves -- the search settled
    on DEFCON 2, and NORAD then asked the US for a placement that never happened."""
    conv = convert_game(_game(260))
    assert conv.entries_converted > 100, (
        f"stopped at {conv.failure}")
    assert not (conv.failure is not None and conv.failure.turn == 5
                and conv.failure.phase == "AR7")
