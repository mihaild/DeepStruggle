"""The repeated last entry of each turn is the cleanup record, not another action round.

ts-replayer hangs a turn's end-of-turn bookkeeping on the header of the entry above it, so the
last action round of every turn appears twice: once with the play, and once with the same turn,
phase, player and card but a body holding only the effects that expired, the same board, and
DEFCON improved by one. Driven as a play the second copy asks the engine for a card it has
already ended the turn to give.
"""
import glob
import gzip
import json
import os

import pytest

from tools.lib.ts_replayer_convert import _is_turn_end_record, convert_game
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = "/workspace/data/datasets/ts_replayer"

pytestmark = pytest.mark.skipif(
    not glob.glob(os.path.join(CORPUS, "*.json.gz")),
    reason="ts-replayer corpus not downloaded")


def _raws(replay_id: int):
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return json.load(f)["all_turns"]


def test_the_second_copy_of_turn_1s_last_action_round_is_recognised() -> None:
    """Replay 14: entries 12 and 13 share a header; only the first records a play."""
    raws = _raws(14)
    played, cleanup = parse_entry(raws[12]), parse_entry(raws[13])
    assert (played.turn, played.phase, played.player, played.card) == (1, "AR6", "US",
                                                                      "East European Unrest")
    assert not _is_turn_end_record(played, parse_entry(raws[11])), (
        "the entry carrying the play must still be driven")
    assert _is_turn_end_record(cleanup, played)
    assert cleanup.defcon is not None and played.defcon is not None
    assert int(cleanup.defcon) == int(played.defcon) + 1, (
        "the cleanup record carries the turn-end DEFCON improvement")


def test_a_cleanup_record_naming_expired_effects_is_still_one() -> None:
    """Replay 14 entry 27: 'Red Scare/Purge is no longer in play.' and nothing else."""
    raws = _raws(14)
    cleanup = parse_entry(raws[27])
    assert _is_turn_end_record(cleanup, parse_entry(raws[26]))
    assert cleanup.out_of_play, "its expiring effects are what makes it worth reconciling"


def test_an_ordinary_entry_is_never_mistaken_for_one() -> None:
    """Two action rounds in a row by the same player would have to differ in content."""
    raws = _raws(14)
    for i in range(1, len(raws)):
        prev, cur = parse_entry(raws[i - 1]), parse_entry(raws[i])
        if _is_turn_end_record(cur, prev):
            assert not (cur.influence or cur.sections or cur.targets or cur.events), (
                f"entry {i} records a play and must be driven")


def test_replay_14_now_runs_past_turn_1() -> None:
    with gzip.open(os.path.join(CORPUS, "14.json.gz"), "rt") as f:
        conv = convert_game(json.load(f))
    assert conv.failure is None or conv.failure.turn > 1, (
        f"replay 14 used to stop at turn 1 AR6: {conv.failure}")
