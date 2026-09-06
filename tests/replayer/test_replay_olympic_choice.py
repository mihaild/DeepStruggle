"""Olympic Games is the opponent's choice, and the log states it in words.

Participating rolls dice for 2 VP; boycotting degrades DEFCON and hands the player the card's
Operations. Nothing else in an entry distinguishes the two before they resolve, so the branch
search -- which judges a branch by the board and score it reaches -- had nothing to go on.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Dict, List, cast

import pytest

from tools.lib.corpus_paths import corpus_dir
from tools.lib.ts_replayer_convert import convert_game
from tools.lib.ts_replayer_parse import parse_entry

_CORPUS = str(corpus_dir())
_SOUTH_KOREA = 38


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


def test_a_boycott_is_read_from_the_entry() -> None:
    e = _entry(28, 3, "Headline", "both")
    assert e.olympics_boycotted is True


def test_participating_is_read_the_same_way() -> None:
    e = _entry(260, 3, "Headline", "both")
    assert e.olympics_boycotted is False


def test_an_entry_without_the_card_says_nothing() -> None:
    e = _entry(28, 3, "AR1", "USSR")
    assert e.olympics_boycotted is None


def test_replay_28_spends_the_four_operations_the_boycott_grants() -> None:
    """Turn 3's headline: 2 into South Korea, 1 into Angola, 1 into Indonesia. Taking the
    participation roll instead lost all four."""
    conv = convert_game(_game(28))
    assert conv.failure is None, f"replay 28 stopped at {conv.failure}"
    assert conv.entries_converted == 122


def test_the_games_that_participate_are_unaffected() -> None:
    for replay_id in (260, 161, 101):
        conv = convert_game(_game(replay_id))
        assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
