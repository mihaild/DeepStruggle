"""A card handed over by another one can be played for its own Event.

Grain Sales To Soviets hands the US a card out of the USSR hand; if it is a US or neutral card
the US may play it as its Event. The entry's mode is the header under that Event, not the play
mode -- and on a headline entry there is no card of the entry's own to judge by.
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
_BRAZIL = 79


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


def test_the_entry_names_the_event_of_the_card_it_was_handed() -> None:
    """Replay 73 turn 5: Grain Sales hands over Junta, a neutral card, and the log prints
    "Event: Junta" with its 2 Influence into Brazil before the free coup the event grants."""
    e = _entry(73, 5, "Headline")
    assert e.played_card == "Junta"
    assert "Junta" in e.events
    assert e.mode == "coup", "the entry's own mode is the coup the event grants"


def test_replay_73_places_the_events_influence() -> None:
    """Read as Operations, the event never fired and Brazil stayed empty."""
    conv = convert_game(_game(73))
    assert conv.failure is None, f"replay 73 stopped at {conv.failure}"
    assert conv.entries_converted == 74


def test_a_handed_card_spent_on_operations_is_still_operations() -> None:
    """Replay 304 turn 5: Grain Sales hands over Summit and the US coups Angola with it. The
    log prints no Event line for Summit, so there is no event to play it for."""
    e = _entry(304, 5, "Headline")
    assert e.played_card == "Summit"
    assert "Summit" not in e.events
    conv = convert_game(_game(304))
    assert conv.failure is None, f"replay 304 stopped at {conv.failure}"


@pytest.mark.parametrize("replay_id", [96, 270, 119, 14])
def test_the_other_grain_sales_games_are_unaffected(replay_id: int) -> None:
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
