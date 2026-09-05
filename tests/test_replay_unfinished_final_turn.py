"""A turn the recording stops inside is a fragment, whether or not anything in it fails.

Every turn plays all of its action rounds -- six in turns 1 to 3, seven after -- unless the
game ends. A last turn that stops short of its own last action round, with no ending recorded,
is a recording that stopped rather than a game that finished. 156 of the corpus's 278 games end
that way, and almost all of them simply stop.

What makes the turn unusable is not the entries never written but the ones that were. The
turn's hand list is assembled from the cards that became visible during it, so a turn cut short
lists only the few played before the recording stopped: replay 246's turn 9 reaches AR3 and
credits the US with three cards and the USSR with six, where a turn 9 hand holds nine. Every
decision converted in that turn was driven from a hand the player never held, and the board it
reaches is one nobody played to.

The final-scoring case is the other half of this. Final scoring runs when the last action round
of turn 10 finishes, and moves the score by whatever the board is worth without the turn number
changing -- so the score check, which skips an entry whose play carried the game into a new
turn, did not skip it. At turn 10 AR7 of replay 154 the US plays the China Card for Influence,
the turn ends, final scoring hands the USSR enough to reach the cap, and the engine stands at
-20 against the -16 the log states. That -16 is the score before final scoring, which the log
never gets to record.
"""
import glob
import gzip
import json
import os
from typing import Dict

import pytest

from tools.lib.ts_replayer_convert import (_log_agrees_the_game_ended, convert_game,
                                          unfinished_final_turn)
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = "/workspace/data/datasets/ts_replayer"

pytestmark = pytest.mark.skipif(
    not glob.glob(os.path.join(CORPUS, "*.json.gz")),
    reason="ts-replayer corpus not downloaded")


def _game(replay_id: int) -> Dict:
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return json.load(f)


def test_a_turn_that_stops_short_is_named() -> None:
    """Replay 246 reaches turn 9 AR3 of seven."""
    raws = _game(246)["all_turns"]
    assert unfinished_final_turn(raws) == 9
    last = parse_entry(raws[-1])
    assert (last.turn, last.phase) == (9, "AR3")


def test_the_hand_list_of_such_a_turn_is_a_fragment() -> None:
    hands = _game(246)["hands"]["9"]
    assert len(hands["us"]) == 3
    assert len(hands["ussr"]) == 6, "a turn 9 hand holds nine"


def test_that_turn_is_dropped_even_though_it_fails_inside() -> None:
    """Replay 246 failed on a board mismatch there; the turn goes either way."""
    conv = convert_game(_game(246))
    assert conv.failure is None, f"replay 246 reported {conv.failure}"
    assert conv.truncated_at is not None
    raws = _game(246)["all_turns"]
    kept = [parse_entry(r) for r in raws[:conv.entries_converted]]
    assert kept[-1].turn == 8
    assert all(e.turn <= 8 for e in kept)


def test_a_turn_that_runs_to_its_last_action_round_is_kept() -> None:
    """Replay 154's turn 10 reaches AR7, so nothing is given back."""
    raws = _game(154)["all_turns"]
    assert unfinished_final_turn(raws) is None
    conv = convert_game(_game(154))
    assert conv.failure is None, f"replay 154 stopped at {conv.failure}"
    assert conv.truncated_at is None
    assert conv.entries_converted == conv.entries_total


def test_the_game_that_ends_in_final_scoring_is_not_checked_against_the_score_before_it() -> None:
    """Replay 154 turn 10 AR7: the log says -16 and final scoring takes the engine to -20."""
    raws = _game(154)["all_turns"]
    last = parse_entry(raws[-1])
    assert (last.turn, last.phase, last.player) == (10, "AR7", "US")
    assert last.score == -16
    conv = convert_game(_game(154))
    assert conv.failure is None
    assert conv.game_ended, "the engine ran final scoring and the game is over"


def test_the_ussr_skipped_the_last_action_round() -> None:
    """Turn 10 AR7 has a US entry and no USSR one: they had too few cards."""
    raws = _game(154)["all_turns"]
    ar7 = [parse_entry(r) for r in raws if parse_entry(r).turn == 10
           and parse_entry(r).phase == "AR7"]
    assert [e.player for e in ar7] == ["US"]


@pytest.mark.parametrize("replay_id", [100, 109, 114])
def test_the_complete_games_are_untouched(replay_id: int) -> None:
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
    assert conv.truncated_at is None
    assert conv.entries_converted == conv.entries_total


def test_an_ending_only_the_engine_believes_in_does_not_save_the_turn() -> None:
    """A fragment turn is driven from hands the players never held, so what happens in it is
    not evidence of anything -- least of all that the game ended there.

    Replay 246 stops mid-line at turn 9 AR3 with the US on 13. Played out from a hand three
    cards short of a real one, it runs on to a US win at 20, and being "ended" used to exempt
    it from the fragment rule and keep the turn.
    """
    raws = _game(246)["all_turns"]
    assert unfinished_final_turn(raws) is not None
    assert not _log_agrees_the_game_ended(raws, 20), "the log's last score is 13"
    conv = convert_game(_game(246))
    assert conv.failure is None, f"replay 246 stopped at {conv.failure}"
    assert conv.truncated_at is not None
    assert conv.game_ended is False
    kept = [parse_entry(r) for r in raws[:conv.entries_converted]]
    assert all(e.turn < conv.truncated_at.turn for e in kept)


@pytest.mark.parametrize("replay_id,score", [(111, -5), (113, -7)])
def test_an_ending_the_logs_own_score_bears_out_keeps_its_turn(replay_id: int, score: int) -> None:
    """Replays 111 and 113 end on Wargames, which ends the game where it stands -- at DEFCON 2,
    with a score neither side won on and two action rounds of turn 8 never played. That is an
    ending the log counted, so its last stated score is the one the engine finishes at."""
    raws = _game(replay_id)["all_turns"]
    assert unfinished_final_turn(raws) is not None, "and both stop short of AR7"
    assert _log_agrees_the_game_ended(raws, score)
    conv = convert_game(_game(replay_id))
    assert conv.game_ended is True
    assert conv.truncated_at is None
    assert conv.entries_converted == conv.entries_total
