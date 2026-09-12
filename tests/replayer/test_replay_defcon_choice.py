"""How I Learned To Stop Worrying is read off the level the log prints, where it prints one.

The card's branches are the five DEFCON levels. Stepping one runs the rest of the action round
with it -- including the turn end, which improves DEFCON by one -- so the board a branch
reaches cannot tell 4 from 5. The branch itself can, and the log names the level outright.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Dict, List, cast

import pytest

from tools.lib.corpus_paths import corpus_dir
from tools.lib.ts_replayer_convert import _defcon_after, _defcon_set_under, convert_game
from tools.lib.ts_replayer_parse import parse_entry

_CORPUS = str(corpus_dir())
_MARKER = "Event: How I Learned To Stop Worrying*"


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


def test_the_level_is_read_from_under_the_cards_own_header() -> None:
    """Replay 32 turn 8 AR7: "DEFCON improves to 5", printed by the card itself."""
    e = _entry(32, 8, "AR7", "US")
    assert _defcon_set_under(e, _MARKER) == 5


def test_an_entry_that_does_not_print_one_falls_back_to_the_next() -> None:
    """Replay 260 turn 5 AR7 narrates only the 5 Military Operations, so the level chosen
    shows up in the entry that follows."""
    e = _entry(260, 5, "AR7", "USSR")
    assert _defcon_set_under(e, _MARKER) is None
    raws = cast(List[Dict[str, object]], _game(260)["all_turns"])
    index = next(i for i, r in enumerate(raws)
                 if (parse_entry(r).turn, parse_entry(r).phase, parse_entry(r).player)
                 == (5, "AR7", "USSR"))
    assert _defcon_after(raws, index, e) == 3


def test_replay_32_pays_the_military_operations_the_level_it_set_is_worth() -> None:
    """The US sets DEFCON to 5 with the USSR two short of it and the turn ends immediately.
    Taking 4 instead -- which the turn end also improves to 5 -- paid the US 1 VP where the log
    pays 2, and the game was a VP short from there on."""
    conv = convert_game(_game(32))
    assert conv.failure is None, f"replay 32 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 154


@pytest.mark.parametrize("replay_id", [260, 101, 104])
def test_the_other_games_with_this_card_are_unaffected(replay_id: int) -> None:
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
