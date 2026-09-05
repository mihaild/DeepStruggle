"""Grain Sales' play-or-return choice is stated in the log, not inferred from the board.

The entry says "US plays Panama Canal Returned*" or "US returns Brezhnev Doctrine* to USSR".
The return was already read; the play was not, so that half of the choice was decided by how
close the resulting board came out -- which is nothing at all where the card's play barely
moves it.
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


def _entry(replay_id: int, turn: int, phase: str):
    raws = cast(List[Dict[str, object]], _game(replay_id)["all_turns"])
    for raw in raws:
        e = parse_entry(raw)
        if (e.turn, e.phase) == (turn, phase):
            return e
    raise AssertionError(f"no entry at replay {replay_id} T{turn} {phase}")


def test_the_log_says_which_of_the_two_happened() -> None:
    """Replay 105 turn 6 hands Brezhnev Doctrine back; replay 73 turn 5 plays Junta."""
    returned = _entry(105, 6, "Headline")
    assert returned.returned_card == "Brezhnev Doctrine*"
    assert returned.played_card is None

    played = _entry(73, 5, "Headline")
    assert played.played_card == "Junta"
    assert played.returned_card is None


@pytest.mark.parametrize("replay_id", [105, 73, 119, 304, 270, 14, 96])
def test_the_games_either_side_of_the_choice_convert(replay_id: int) -> None:
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"


def test_the_hand_miss_counter_counts_only_real_misses() -> None:
    """It is diagnostic, and two things that are not misses were drowning out the ones that
    are: the China Card, which no hand list carries, and an entry's own card named again by its
    own discard line -- which is how a Quagmire or Bear Trap escape is recorded."""
    conv = convert_game(_game(100))
    assert conv.entries_converted == conv.entries_total == 144
    assert conv.hand_misses == 0, "every card this game plays was in the hand we tracked"

    # Replay 105 turn 6: Grain Sales draws Brezhnev Doctrine and the US hands it straight back,
    # so the USSR still holds it -- the one miss here is that card, marked spent when taken.
    conv = convert_game(_game(105))
    assert conv.hand_misses == 1
