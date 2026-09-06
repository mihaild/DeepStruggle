"""Judging a branch by where it can still go, not by how far it has already run.

A two-sided event is chosen by trying each branch on a clone and seeing which one leads where
the log went. Two things flattered the branch that finishes the event outright.

`primary_id` means whatever the decision it belongs to means, and a card selection's is a card
id -- which shares its numeric range with the country ids the log names as targets. A branch
that finished and left the engine asking for the next card scored for every country id that
happened to match a card in hand.

DEFCON improves at a turn end, so a branch that finishes sails on to it and matches the DEFCON
the entry records, while the branch that opens further decisions stops short of it. That is the
same trap the score comparison already guards against by asking only where the log names no
targets, and the DEFCON comparison now does the same.

At turn 8 AR7 of replay 234 -- the last action round of the turn -- the US plays South African
Unrest for Operations and the USSR's event places 1 Influence in South Africa and 2 in Angola.
The other branch, 2 Influence in South Africa, collected both bonuses: 1582 against 1083.
"""
import glob
import gzip
import json
import os
from typing import Dict

import pytest

from tools.lib.ts_replayer_convert import convert_game, event_queue
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = "/workspace/data/datasets/ts_replayer"
SOUTH_AFRICA, ANGOLA = 63, 59

pytestmark = pytest.mark.skipif(
    not glob.glob(os.path.join(CORPUS, "*.json.gz")),
    reason="ts-replayer corpus not downloaded")


def _game(replay_id: int) -> Dict:
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return json.load(f)


def _entry(replay_id: int, turn: int, phase: str, player: str):
    for raw in _game(replay_id)["all_turns"]:
        e = parse_entry(raw)
        if (e.turn, e.phase, e.player) == (turn, phase, player):
            return e
    raise AssertionError(f"replay {replay_id} has no {player} {phase} of turn {turn}")


def test_the_log_takes_the_split_branch() -> None:
    """1 Influence in South Africa and 2 in Angola, not 2 in South Africa."""
    e = _entry(234, 8, "AR7", "US")
    assert e.card == "South African Unrest"
    assert ("USSR", 1, SOUTH_AFRICA, 3, 1) in e.influence
    assert ("USSR", 2, ANGOLA, 5, 4) in e.influence
    assert event_queue(e) == [SOUTH_AFRICA, ANGOLA, ANGOLA]


def test_replay_234_converts_end_to_end() -> None:
    conv = convert_game(_game(234))
    assert conv.failure is None, f"replay 234 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 143


@pytest.mark.parametrize("replay_id", [302, 113, 119, 101, 104, 164])
def test_the_branch_choices_these_games_turn_on_are_unchanged(replay_id: int) -> None:
    """Warsaw Pact's two branches, Wargames' ending, and How I Learned's DEFCON.

    Each of these is a branch the search gets right for a different reason -- named targets,
    the score the entry ends on, and a numeric setting with no targets at all -- so between
    them they cover what the two bonuses were there for.
    """
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
