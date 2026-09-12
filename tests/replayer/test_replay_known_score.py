"""A score the engine and the log reach differently because of a choice neither gets wrong.

Shuttle Diplomacy subtracts one USSR battleground country from the USSR's total at the next
Asia or Middle East scoring, and *which* battleground is the US player's choice. Taking Japan
costs the USSR the superpower-adjacent Influence there as well, which is why it is the choice
to make -- but taking another one is legal, and a player may simply not see it.

At turn 5 AR3 of replay 127 the USSR holds Japan [4][8] and dominates Asia: 7 for Domination,
4 battlegrounds and Japan's adjacency, 12 against the US's 5. Shuttle Diplomacy takes one
battleground and one country off the USSR either way, leaving Domination and 3 battlegrounds.
Take Japan and the adjacency goes too, so the USSR scores 10 - 5 = 5, which is what the engine
assumes. This US took a different battleground and the USSR scored 11 - 5 = 6, which is what
the log records.

The engine cannot reconstruct it until Shuttle Diplomacy's target is a decision it asks for,
so the score is listed in _KNOWN_SCORE, the log's value stands, and the game carries on. The
list is deliberately small: a disagreement that is not understood belongs in a failure.
"""
import glob
import gzip
import json
import os
from typing import Dict

import pytest
import ts_engine as ts

from tools.lib.corpus_paths import corpus_dir
from tools.lib.ts_replayer_convert import _KNOWN_SCORE, convert_game, country_id
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = str(corpus_dir())
ASIA = ts.Region.ASIA



def _game(replay_id: int) -> Dict:
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return json.load(f)


def test_the_log_scores_the_ussr_six_where_best_play_allows_five() -> None:
    raws = _game(127)["all_turns"]
    scoring = parse_entry(raws[59])
    assert (scoring.turn, scoring.phase, scoring.player) == (5, "AR3", "USSR")
    assert scoring.card == "Asia Scoring"
    assert scoring.vp_gains == [("USSR", 6, "USSR", 5)]
    assert "Shuttle Diplomacy is no longer in play." in (scoring.text or "")
    assert parse_entry(raws[58]).vp_gains == [("USSR", 1, "US", 1)], (
        "the score going in is US 1, so a gain of 6 leaves USSR 5")


def test_japan_is_the_battleground_that_carries_the_adjacency() -> None:
    """It is worth two to the USSR: the battleground and the adjacency both."""
    japan = country_id("Japan")
    assert japan is not None
    info = ts.MapData.get_country_info(japan)
    assert info["battleground"] is True
    assert info["superpower_adjacent"] == "US"

    board = _game(127)["all_turns"][58]["countries"]
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    for key, c in board.items():
        cid = country_id(key)
        if cid is not None:
            state.set_country(cid, int(c["inflUS"]), int(c["inflUSSR"]))
    held = state.get_country(japan)
    assert (int(held.us_influence), int(held.ussr_influence)) == (4, 8), (
        "the USSR controls Japan going into the scoring")

    asia = ts.Scoring.evaluate_region(state, ASIA)
    assert asia.ussr_status == ts.RegionalStatus.DOMINATION
    assert int(asia.ussr_battlegrounds) == 4
    assert int(asia.ussr_superpower_adjacent) == 1, "Japan, and nothing else"
    assert int(asia.us_battlegrounds) == 2
    assert int(asia.us_superpower_adjacent) == 0
    # 7 + 4 + 1 = 12 against 3 + 2 + 0 = 5. Shuttle Diplomacy takes one battleground and one
    # country from the USSR, leaving Domination and 3 battlegrounds either way: 10 - 5 = 5 if
    # the one taken is Japan, 11 - 5 = 6 if it is not.


def test_the_disagreement_is_listed_and_the_logs_score_stands() -> None:
    assert _KNOWN_SCORE[127][(5, "AR3", "USSR")] == -5, "US positive, so USSR 5"
    conv = convert_game(_game(127))
    assert conv.failure is None, f"replay 127 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total
    assert conv.scores_forced == 1, "exactly the one entry, and no other"


def test_no_other_game_needs_a_forced_score() -> None:
    """The list stays a list of understood cases, not a way past unexplained drift."""
    assert set(_KNOWN_SCORE) == {127}
    assert sum(len(v) for v in _KNOWN_SCORE.values()) == 1
