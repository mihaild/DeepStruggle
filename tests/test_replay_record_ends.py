"""The log running out is not the reconstruction going wrong.

An unanswered decision on the file's last entry means the recording stopped there. On any
earlier entry it means the answer is somewhere we are not reading, which is a defect. Nine of
the corpus's games end mid-entry: at turn 3 AR3 of replay 153 the final entry is "Event:
Arab-Israeli War" and not one word more, and at turn 6 AR1 of replay 251 it is "Coup (3 Ops):"
with no target, roll or result.

A disagreement is never read this way, wherever it lands. Fifteen games fail their last entry
on a pass the log records and the engine will not allow, and three more on a board or a score.
Those are the reconstruction being wrong about something the log does state, and they stay
failures. The distinction is between the log saying nothing and the log saying something else.
"""
import glob
import gzip
import json
import os
from typing import Dict

import pytest

from tools.lib.ts_replayer_convert import (Mismatch, _is_the_record_ending,
                                           convert_game)
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = "/workspace/data/datasets/ts_replayer"

pytestmark = pytest.mark.skipif(
    not glob.glob(os.path.join(CORPUS, "*.json.gz")),
    reason="ts-replayer corpus not downloaded")


def _game(replay_id: int) -> Dict:
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return json.load(f)


@pytest.mark.parametrize("replay_id,tail", [
    (153, "Event: Arab-Israeli War"),
    (251, "Coup (3 Ops):"),
])
def test_the_record_stops_mid_entry(replay_id: int, tail: str) -> None:
    raws = _game(replay_id)["all_turns"]
    assert (raws[-1].get("text") or "").strip().endswith(tail)
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} reported {conv.failure}"
    assert conv.truncated_at is not None
    # The whole of the turn it stops in is given back, so what remains ends on a turn boundary.
    kept = [parse_entry(r) for r in raws[:conv.entries_converted]]
    stopped_in = conv.truncated_at.turn
    assert all(e.turn < stopped_in for e in kept)
    assert kept[-1].turn == stopped_in - 1


@pytest.mark.parametrize("replay_id", [153, 251, 112, 158, 37, 38, 44])
def test_these_games_convert_up_to_where_the_log_ends(replay_id: int) -> None:
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} reported {conv.failure}"
    assert conv.truncated_at is not None


def test_replay_182_is_whole_and_no_longer_stops_short() -> None:
    """Its last entry is turn 10 AR8, the US racing with the China Card.

    It sat in the list above while the engine refused to play the China Card for the Space
    Race, which made the last turn unconvertible and read as a record that stops early.
    """
    conv = convert_game(_game(182))
    assert conv.failure is None, f"replay 182 reported {conv.failure}"
    assert conv.truncated_at is None
    assert conv.entries_converted == conv.entries_total == 143


def test_a_disagreement_in_a_turn_the_log_completes_is_still_a_failure() -> None:
    """Replay 323 turn 10 AR1: the log states a score and the engine reaches another.

    A disagreement inside a turn is ours to fix rather than the recording's to excuse -- it
    stops the conversion where it stands, and nothing after it is converted.
    """
    conv = convert_game(_game(323))
    assert conv.failure is not None
    assert conv.failure.kind == "score mismatch after replay"
    assert (conv.failure.turn, conv.failure.phase) == (10, "AR1")
    assert conv.truncated_at is None


def test_a_skipped_round_at_the_end_of_the_file_is_the_record_ending() -> None:
    """A bare "Turn N, SIDE ARk" header means a skip when entries follow it, and not at EOF.

    Every one of the corpus's pass failures is the second shape -- the header last in the file
    with the player still holding cards -- and not one is the first, where a genuine skip is
    always followed by the other player's entries.
    """
    for replay_id in (191, 274, 76):
        conv = convert_game(_game(replay_id))
        assert conv.failure is None, f"replay {replay_id} reported {conv.failure}"
        assert conv.truncated_at is not None
        assert conv.truncated_at.kind == "logged pass is not legal"


def test_a_skipped_round_with_entries_after_it_is_still_driven() -> None:
    """Replay 114 turn 5: the USSR skips AR4 through AR7 and the US plays on."""
    conv = convert_game(_game(114))
    assert conv.failure is None, f"replay 114 stopped at {conv.failure}"
    assert conv.truncated_at is None
    assert conv.entries_converted == conv.entries_total


def test_a_decision_unanswered_in_an_earlier_turn_is_a_defect() -> None:
    """The record can only end where it ends: in the last turn, at the last entry.

    Anywhere before that, an unanswered decision means the answer is somewhere we are not
    reading.
    """
    raws = _game(153)["all_turns"]
    last = parse_entry(raws[-1])
    conv = convert_game(_game(153))
    assert conv.truncated_at is not None
    earlier_turn = type(conv.truncated_at)(
        conv.truncated_at.replay_id, last.turn - 1, "AR1", last.player,
        last.card, conv.truncated_at.kind, conv.truncated_at.detail)
    assert not _is_the_record_ending(earlier_turn, raws)


def test_a_disagreement_in_a_completed_turn_is_never_read_as_the_record_ending() -> None:
    """Replay 154 plays its last turn out, so nothing in it is excused.

    Inside a turn the recording stops in, a disagreement *is* read that way -- the turn is a
    fragment and is dropped whatever the reason, so whether we could have reproduced it is
    moot. That is the case the test above covers.
    """
    raws = _game(154)["all_turns"]
    last = parse_entry(raws[-1])
    conv = convert_game(_game(154))
    assert conv.truncated_at is None, "its last turn runs to AR7"
    for kind in ("board mismatch after replay", "score mismatch after replay",
                 "logged pass is not legal", "target not legal"):
        disagreement = Mismatch(154, last.turn, last.phase, last.player,
                                last.card, kind, "")
        assert not _is_the_record_ending(disagreement, raws), kind
