"""Missile Envy takes one card, and only that card leaves the hand.

The card it takes is gone from the opponent's hand for the rest of the turn, while the turn's
hand list still names it -- it was theirs when the turn began and became visible while they
held it. Leaving it in put a card back that had changed hands: at turn 5 AR1 of replay 114 the
US takes Nuclear Test Ban, and the USSR, caught by Bear Trap with nothing left to discard and
only a scoring card in hand, was still being offered it two action rounds later.

Which card it took is read from the line after "Event: Missile Envy" rather than from the
entry's reveals as a whole, because other cards reveal too and one of them can share the entry.
At turn 5 of replay 141 the USSR headlines "Lone Gunman", which reveals the entire US hand,
against the US's Missile Envy. Taking every reveal as exchanged emptied the US hand outright,
and the UN Intervention they played two action rounds later was not there to play -- the coup
on Nicaragua never happened and the country stayed [0][1] where the log has [3][0].
"""
import glob
import gzip
import json
import os
from typing import Dict

import pytest

from tools.lib.corpus_paths import corpus_dir
from tools.lib.ts_replayer_convert import _missile_envy_took, card_id, convert_game
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = str(corpus_dir())
NICARAGUA = 69



def _game(replay_id: int) -> Dict:
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return json.load(f)


def _entry(replay_id: int, turn: int, phase: str, player: str):
    for raw in _game(replay_id)["all_turns"]:
        e = parse_entry(raw)
        if (e.turn, e.phase, e.player) == (turn, phase, player):
            return e
    raise AssertionError(f"replay {replay_id} has no {player} {phase} of turn {turn}")


def test_the_exchanged_card_is_the_one_named_under_the_event() -> None:
    """Replay 114 turn 5 AR1: "Event: Missile Envy / USSR reveals Nuclear Test Ban"."""
    e = _entry(114, 5, "AR1", "US")
    assert e.card == "Missile Envy"
    assert _missile_envy_took(e) == ("USSR", card_id("Nuclear Test Ban"))


def test_a_hand_revealed_by_something_else_is_not_an_exchange() -> None:
    """Replay 141 turn 5 headline: "Lone Gunman" reveals the whole US hand.

    Only the card under "Event: Missile Envy" changed sides; the rest were merely seen.
    """
    e = _entry(141, 5, "Headline", "both")
    assert len(e.revealed) > 1, "Lone Gunman reveals a whole hand"
    taken = _missile_envy_took(e)
    assert taken is not None
    _side, cid = taken
    assert sum(1 for _s, nm in e.revealed if card_id(nm) == cid) >= 1
    assert len([nm for _s, nm in e.revealed]) > 1, (
        "and the other reveals must not be treated as exchanges")


def test_an_entry_without_missile_envy_exchanges_nothing() -> None:
    assert _missile_envy_took(_entry(141, 5, "AR1", "US")) is None


def test_replay_141_converts_end_to_end() -> None:
    """It stopped at turn 5 AR3 with Nicaragua [0][1] against the log's [3][0]."""
    conv = convert_game(_game(141))
    assert conv.failure is None, f"replay 141 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 144


def test_the_coup_the_empty_hand_prevented() -> None:
    """Turn 5 AR3: UN Intervention on Liberation Theology, 2 Ops, roll 4 into Nicaragua.

    4 + 2 - 2x1 = 4, so the USSR loses the 1 Influence it has and the US places 3.
    """
    e = _entry(141, 5, "AR3", "US")
    assert e.card == "UN Intervention" and e.played_card == "Liberation Theology"
    assert e.mode == "coup" and e.ops == 2 and e.targets == [NICARAGUA]
    assert e.coup_rolls == [(4, True)]
    assert ("USSR", -1, NICARAGUA, 0, 0) in e.influence
    assert ("US", 3, NICARAGUA, 3, 0) in e.influence


@pytest.mark.parametrize("replay_id", [141, 114, 115])
def test_the_missile_envy_games_convert(replay_id: int) -> None:
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
