"""Where the hand lists and the entries disagree about whose card it is, the entries win.

A turn's hand list is a summary the recording assembled; an entry is the play itself, narrated
as it happened. At turn 6 of replay 111 the lists put Asia Scoring in the USSR's hand and ABM
Treaty in the US's, while the headline reads "US Headlines Asia Scoring / USSR Headlines ABM
Treaty" -- the two are swapped. Asia Scoring then scored for the wrong side, five VP the wrong
way, and the reconstruction ran on to a US win at 20 VP where the log has the USSR ahead by 8.

236 cards across 48 of the 287 games are listed under the wrong side.
"""
import glob
import gzip
import json
import os
from typing import Dict, List

import pytest

from tools.lib.ts_replayer_convert import (_reattribute_hands, card_id,
                                           convert_game)
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = "/workspace/data/datasets/ts_replayer"
ASIA_SCORING = card_id("Asia Scoring")
ABM_TREATY = card_id("ABM Treaty")

pytestmark = pytest.mark.skipif(
    not glob.glob(os.path.join(CORPUS, "*.json.gz")),
    reason="ts-replayer corpus not downloaded")


def _game(replay_id: int) -> Dict:
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return json.load(f)


def test_the_headline_says_who_played_which_card() -> None:
    entry = next(parse_entry(r) for r in _game(111)["all_turns"]
                 if parse_entry(r).turn == 6 and parse_entry(r).phase == "Headline")
    assert entry.headlines == {"US": "Asia Scoring", "USSR": "ABM Treaty"}


def test_the_hand_lists_have_those_two_the_other_way_round() -> None:
    hands = _game(111)["hands"]["6"]
    assert "Asia Scoring" in hands["ussr"], "the list gives the USSR the US's card"
    assert "ABM Treaty" in hands["us"], "and the US the USSR's"


def test_the_cards_are_moved_to_the_side_that_played_them() -> None:
    game = _game(111)
    hands = game["hands"]["6"]
    turn_hands = {
        "US": [c for c in (card_id(n) for n in hands["us"]) if c],
        "USSR": [c for c in (card_id(n) for n in hands["ussr"]) if c],
    }
    moved = _reattribute_hands(game["all_turns"], 6, turn_hands)
    assert moved >= 2
    assert ASIA_SCORING in turn_hands["US"]
    assert ASIA_SCORING not in turn_hands["USSR"]
    assert ABM_TREATY in turn_hands["USSR"]
    assert ABM_TREATY not in turn_hands["US"]


def test_a_card_already_on_the_right_side_is_left_alone() -> None:
    game = _game(100)
    hands = game["hands"]["1"]
    turn_hands = {
        "US": [c for c in (card_id(n) for n in hands["us"]) if c],
        "USSR": [c for c in (card_id(n) for n in hands["ussr"]) if c],
    }
    before = {k: list(v) for k, v in turn_hands.items()}
    _reattribute_hands(game["all_turns"], 1, turn_hands)
    assert {k: sorted(v) for k, v in turn_hands.items()} == \
           {k: sorted(v) for k, v in before.items()}


def test_replay_111_converts_end_to_end() -> None:
    """It stopped at turn 6 AR7, the engine at +20 VP where the log says -8."""
    conv = convert_game(_game(111))
    assert conv.failure is None, f"replay 111 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total
    assert conv.hand_reattributions > 0


@pytest.mark.parametrize("replay_id", [111, 105, 114, 152])
def test_the_reattributed_games_convert(replay_id: int) -> None:
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
