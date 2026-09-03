"""Some recordings simply stop in the middle of an entry.

Nothing about the engine or the driver can recover what the log does not contain, so those
entries are listed rather than diagnosed again each time one is met. The game converts up to
the entry before, and that is where its training data ends.
"""
import glob
import gzip
import json
import os

import pytest

from tools.lib.ts_replayer_convert import _KNOWN_INCOMPLETE, convert_game
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = "/workspace/data/datasets/ts_replayer"

pytestmark = pytest.mark.skipif(
    not glob.glob(os.path.join(CORPUS, "*.json.gz")),
    reason="ts-replayer corpus not downloaded")


def test_replay_60_stops_where_its_log_stops() -> None:
    """Turn 7 AR6 reads "Coup (4 Ops):" and ends -- no target, no roll, no result."""
    with gzip.open(os.path.join(CORPUS, "60.json.gz"), "rt") as f:
        conv = convert_game(json.load(f))
    assert conv.failure is None, f"a truncated log is not a failure: {conv.failure}"
    assert conv.truncated_at is not None
    assert (conv.truncated_at.turn, conv.truncated_at.phase) == (7, "AR6")
    assert conv.entries_converted == conv.entries_total, (
        "everything the log does contain must still convert")


def test_every_listed_entry_really_is_unfinished() -> None:
    """The registry is for logs that stop, not for entries that are merely hard."""
    for replay_id, entries in _KNOWN_INCOMPLETE.items():
        path = os.path.join(CORPUS, f"{replay_id}.json.gz")
        if not os.path.exists(path):
            continue
        with gzip.open(path, "rt") as f:
            raws = json.load(f)["all_turns"]
        for turn, phase in entries:
            matching = [r for r in raws
                        if (lambda e: (e.turn, e.phase) == (turn, phase))(parse_entry(r))]
            assert matching, f"replay {replay_id} has no entry at T{turn} {phase}"
            last = matching[-1]
            assert raws.index(last) == len(raws) - 1, (
                f"replay {replay_id} T{turn} {phase} is not the last entry in the file")


def test_an_entry_that_states_the_score_twice_is_read_at_its_end() -> None:
    """A headline states one score per card, and it is the value it leaves that must match.

    At turn 2 of replay 100 the US takes 2 VP from Captured Nazi Scientist and the USSR then
    takes 1 from Europe Scoring, leaving the score at 1. Reading the first statement instead
    asserted against 2 and called the engine wrong where it was right.
    """
    from tools.lib.ts_replayer_convert import _narrated_score

    with gzip.open(os.path.join(CORPUS, "100.json.gz"), "rt") as f:
        raws = json.load(f)["all_turns"]
    headline = next(r for r in raws
                    if (lambda e: e.turn == 2 and e.phase == "Headline")(parse_entry(r)))
    entry = parse_entry(headline)
    assert len(entry.vp_gains) == 2, "this headline states the score twice"
    assert _narrated_score(entry) == 1
