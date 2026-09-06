"""Which card's reveal is which, when two cards in one entry both reveal.

The log prints every reveal as "USSR reveals X", under the header of the card that caused it.
SALT Negotiations reclaims a card from the discard pile, Grain Sales To Soviets draws one at
random from the opponent's hand, and Missile Envy takes their highest -- and a headline can
put two of them in the same entry.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Dict, List, cast

import pytest

from tools.lib.ts_replayer_convert import _mid_turn_acquisitions, _revealed_under, convert_game
from tools.lib.ts_replayer_parse import parse_entry

_CORPUS = "/workspace/data/datasets/ts_replayer"
_OPEC = 61
_SHUTTLE_DIPLOMACY = 73
_RED_SCARE_PURGE = 31
_SUMMIT = 45


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


def test_salt_reclaims_only_the_card_under_its_own_header() -> None:
    """Replay 288 turn 5: SALT Negotiations takes OPEC back out of the discard pile, and the
    US's Missile Envy then reveals Shuttle Diplomacy out of the USSR hand."""
    e = _entry(288, 5, "Headline", "both")
    assert [nm for _side, nm in e.revealed] == ["OPEC", "Shuttle Diplomacy"]
    assert _revealed_under(e, "Event: SALT Negotiations*") == ("USSR", _OPEC)

    raws = cast(List[Dict[str, object]], _game(288)["all_turns"])
    held = [c for c in (_OPEC, _SHUTTLE_DIPLOMACY, _RED_SCARE_PURGE)]
    acquired = _mid_turn_acquisitions(raws, 5, "USSR", held)
    assert _OPEC in acquired, "reclaimed from the discard pile, so not dealt"
    assert _SHUTTLE_DIPLOMACY not in acquired, "in the hand all along, and given to Missile Envy"


def test_replay_288_leaves_missile_envy_a_tie_to_break() -> None:
    """Both cards are 3 Ops, so the USSR chooses which to hand over -- and they chose Shuttle
    Diplomacy. Held out of the dealt hand, it was not there to choose."""
    conv = convert_game(_game(288))
    assert conv.failure is None, f"replay 288 stopped at {conv.failure}"
    assert conv.entries_converted == 69


def test_grain_sales_draws_the_card_under_its_own_header() -> None:
    """Replay 304 turn 5: SALT reclaims Red Scare/Purge and Grain Sales then draws Summit.

    Correcting the draw to the entry's first reveal handed the US Red Scare/Purge -- 4 Ops
    where Summit has 1 -- and the coup on Angola that the log records failing succeeded.
    """
    e = _entry(304, 5, "Headline", "both")
    assert _revealed_under(e, "Event: SALT Negotiations*") == ("USSR", _RED_SCARE_PURGE)
    assert _revealed_under(e, "Event: Grain Sales To Soviets") == ("USSR", _SUMMIT)

    conv = convert_game(_game(304))
    assert conv.failure is None, f"replay 304 stopped at {conv.failure}"
    assert conv.entries_converted == 114


@pytest.mark.parametrize("replay_id", [119, 14, 262, 64])
def test_the_ordinary_grain_sales_games_are_unaffected(replay_id: int) -> None:
    """One reveal in the entry, so the first is the only one."""
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
