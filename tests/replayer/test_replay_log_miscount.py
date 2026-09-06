"""Where the log's own arithmetic is wrong, the engine's score stands.

Almost every disagreement between the two is ours: the engine is what has to be fixed, and a
score mismatch stops the conversion. A handful are the recording's, and adopting its number
would write a score into the training data that the board it comes with does not pay -- so
those entries are listed, the engine's score is kept, and every score the log states from
there on is compared against its own number plus the offset.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Dict, List, Tuple, cast

import pytest

from tools.lib.corpus_paths import corpus_dir
from tools.lib.ts_replayer_convert import _KNOWN_SCORE, _LOG_MISCOUNTED, convert_game
from tools.lib.ts_replayer_parse import parse_entry

_CORPUS = str(corpus_dir())


def _game(replay_id: int) -> Dict[str, object]:
    path = os.path.join(_CORPUS, f"{replay_id}.json.gz")
    if not os.path.exists(path):
        pytest.skip("corpus not downloaded")
    with gzip.open(path, "rt") as f:
        return json.load(f)


def test_replay_259_converts_past_the_miscounted_asia_scoring() -> None:
    """Turn 7 AR3: Asia Scoring under Shuttle Diplomacy, with the USSR holding Japan.

    The card takes a USSR battleground country out of the region, and Japan borders the United
    States, so taking it takes the USSR's superpower-adjacent Influence too. The log scores the
    USSR 2 and keeps the adjacency; it is 1, which is what the engine scores.
    """
    conv = convert_game(_game(259))
    assert conv.failure is None, f"replay 259 stopped at {conv.failure}"
    assert conv.log_miscounts == 1
    assert conv.entries_converted == 128


def test_the_offset_is_carried_to_every_later_score() -> None:
    """Every score the log states after a miscount is out by the same amount.

    The log's score field reads 17 all through the rest of turn 7 where the truth is 18, and
    the engine is reconciled to the logged score on every entry -- so without carrying the
    offset the correction would be undone on the very next one.
    """
    raws = cast(List[Dict[str, object]], _game(259)["all_turns"])
    after: List[Tuple[int, str, int]] = []
    seen = False
    for raw in raws:
        e = parse_entry(raw)
        if (e.turn, e.phase, e.player) == (7, "AR3", "USSR"):
            seen = True
            continue
        if seen and e.turn == 7 and e.score is not None:
            after.append((e.turn, e.phase, int(e.score)))
    assert after, "turn 7 continues past the miscounted entry"
    assert {score for _t, _p, score in after} == {17}, (
        "the log stays on its own number for the rest of the turn")

    conv = convert_game(_game(259))
    assert conv.failure is None


def test_the_two_lists_are_kept_apart_and_small() -> None:
    """_KNOWN_SCORE is a legal choice the engine does not offer, where the log is the record of
    what happened; _LOG_MISCOUNTED is the log being wrong. Nothing belongs in both, and a
    disagreement that is not understood belongs in a failure rather than in either."""
    for replay_id, entries in _LOG_MISCOUNTED.items():
        for key in entries:
            assert key not in _KNOWN_SCORE.get(replay_id, {})
    assert sum(len(v) for v in _LOG_MISCOUNTED.values()) <= 3
    assert sum(len(v) for v in _KNOWN_SCORE.values()) <= 3
