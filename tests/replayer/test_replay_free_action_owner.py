"""Whose Ops header is whose, when an entry holds more than one card.

Junta and Tear Down This Wall place Influence and then *may* coup or realign, so a log that
records the placement and nothing after it is a player who declined. A headline resolves two
cards and prints each one's Ops under its own "Event:" header, so a header still queued is not
necessarily this card's to spend.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Dict, List, cast

import pytest

from tools.lib.ts_replayer_convert import _free_action_declined, convert_game
from tools.lib.ts_replayer_parse import parse_entry

_CORPUS = "/workspace/data/datasets/ts_replayer"
_JUNTA = 47
_TEAR_DOWN_THIS_WALL = 96


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


def test_the_other_headlines_coup_is_not_juntas_to_take() -> None:
    """Replay 279 turn 7: the US headlines Grain Sales To Soviets and coups Angola with its
    Ops; the USSR's Junta places 2 in Mexico and takes no free action at all."""
    e = _entry(279, 7, "Headline", "both")
    coup = [s for s in e.sections if s.mode == "coup"]
    assert [s.under_event for s in coup] == ["Grain Sales To Soviets"]
    assert _free_action_declined(e, list(e.sections), _JUNTA) is True


def test_a_card_played_through_another_keeps_its_ops() -> None:
    """Replay 141 turn 9 AR1: UN Intervention names Tear Down This Wall and the USSR coups
    Libya with its 3 Ops -- printed under UN Intervention's header, because that is the card
    that named it. Its own event never fires, so there is no free action to decline."""
    e = _entry(141, 9, "AR1", "USSR")
    assert e.played_card == "Tear Down This Wall*"
    assert [s.under_event for s in e.sections] == ["UN Intervention"]
    assert _free_action_declined(e, list(e.sections), _TEAR_DOWN_THIS_WALL) is False


def test_a_placement_with_nothing_after_it_is_still_a_decline() -> None:
    """Replay 174 turn 5 AR6: Junta for its event, 2 Influence into Chile, and no coup."""
    e = _entry(174, 5, "AR6", "US")
    assert _free_action_declined(e, [], _JUNTA) is True


@pytest.mark.parametrize("replay_id,entries", [(141, 144), (275, 145), (277, 144), (177, 144)])
def test_these_games_convert_in_full(replay_id: int, entries: int) -> None:
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == entries
