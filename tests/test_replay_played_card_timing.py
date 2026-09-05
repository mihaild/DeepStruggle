"""Whose timing the log is describing when one card hands over another.

Playing an opponent's card for Operations asks which of the Event and the Operations resolves
first, and the log answers by line order. When the card was handed over by another one -- Grain
Sales To Soviets, Missile Envy -- the "Event:" header above that exchange belongs to the card
that did the handing, so the ordering has to be read from the lines after the handover.
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


def _entry(replay_id: int, turn: int, phase: str, player: str):
    raws = cast(List[Dict[str, object]], _game(replay_id)["all_turns"])
    for raw in raws:
        e = parse_entry(raw)
        if (e.turn, e.phase, e.player) == (turn, phase, player):
            return e
    raise AssertionError(f"no entry at replay {replay_id} T{turn} {phase} {player}")


def test_the_handing_cards_own_header_is_not_the_answer() -> None:
    """Replay 270 turn 5 AR1: Grain Sales' "Event:" opens the entry, and OPEC's comes last."""
    e = _entry(270, 5, "AR1", "US")
    assert e.played_card == "OPEC"
    assert e.event_first is True, "read from the top, the entry looks event-first"
    assert e.played_event_first is False, "OPEC's own Event is printed after the coup"


def test_an_entry_with_no_handover_says_nothing_about_a_played_card() -> None:
    e = _entry(270, 5, "AR6", "USSR")
    assert e.played_card is None
    assert e.played_event_first is None


def test_replay_270_scores_opec_after_the_coup() -> None:
    """OPEC pays the USSR 1 VP for each oil country they control, and the coup on Venezuela
    takes it out of USSR hands before OPEC counts: 2 VP, not 3."""
    conv = convert_game(_game(270))
    assert conv.failure is None, f"replay 270 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 144


@pytest.mark.parametrize("replay_id,entries", [(137, 58), (123, 144), (124, 144)])
def test_the_headline_games_this_needed_the_engine_for(replay_id: int, entries: int) -> None:
    """Ops-first in a headline used to lose the Event outright -- see advance_after_ops."""
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == entries
