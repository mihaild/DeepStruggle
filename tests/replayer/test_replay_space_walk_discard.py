"""A player at box 6 of the space race may discard a held card as the turn ends.

ts-replayer records that where it records the rest of the turn's bookkeeping: on a second copy
of the last entry's header, with a body holding nothing but the discard. The engine asks for it
inside the entry above, as that entry's last action round ends.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Dict, List, cast

import pytest

from tools.lib.corpus_paths import corpus_dir
from tools.lib.ts_replayer_convert import _space_walk_discard, convert_game
from tools.lib.ts_replayer_parse import parse_entry

_CORPUS = str(corpus_dir())
_COLONIAL_REAR_GUARDS = 63


def _game(replay_id: int) -> Dict[str, object]:
    path = os.path.join(_CORPUS, f"{replay_id}.json.gz")
    if not os.path.exists(path):
        pytest.skip("corpus not downloaded")
    with gzip.open(path, "rt") as f:
        return json.load(f)


def test_the_discard_is_read_off_the_cleanup_copy() -> None:
    """Replay 56 turn 9 AR7: the USSR is at box 6 with the US at 5, and discards Colonial Rear
    Guards. The card is named on the copy of the entry, not in the entry itself."""
    raws = cast(List[Dict[str, object]], _game(56)["all_turns"])
    index = next(i for i, r in enumerate(raws)
                 if (parse_entry(r).turn, parse_entry(r).phase, parse_entry(r).player)
                 == (9, "AR7", "US"))
    entry = parse_entry(raws[index])
    assert not entry.discards, "the entry itself records the play and nothing else"
    assert parse_entry(raws[index + 1]).discards == [("USSR", "Colonial Rear Guards")]
    assert _space_walk_discard(raws, index, entry) == _COLONIAL_REAR_GUARDS


def test_an_ordinary_entry_names_no_such_discard() -> None:
    raws = cast(List[Dict[str, object]], _game(56)["all_turns"])
    index = next(i for i, r in enumerate(raws)
                 if (parse_entry(r).turn, parse_entry(r).phase, parse_entry(r).player)
                 == (9, "AR6", "US"))
    assert _space_walk_discard(raws, index, parse_entry(raws[index])) is None


def test_replay_56_converts_in_full() -> None:
    """Declined for want of a card to name, the turn ended with Colonial Rear Guards still in
    hand -- and the cleanup copy was then driven as a play into an engine already dealing turn
    10, which is where the conversion stopped."""
    conv = convert_game(_game(56))
    assert conv.failure is None, f"replay 56 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 153
