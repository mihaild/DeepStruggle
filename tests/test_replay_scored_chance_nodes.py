"""Some chance nodes leave no trace on the board, only on the score.

A die is steered by seeding the engine's stream until the result matches what the log recorded,
and that match was made against country influence. Olympic Games and Summit move no Influence
at all, so nothing matched and the winner was a coin flip -- which puts the points on the wrong
side half the time, a four VP swing from one roll.

The log records who gained them, so the score is what the seed is chosen against instead.
"""
import glob
import gzip
import json
import os

import pytest

from tools.lib.ts_replayer_convert import convert_game
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = "/workspace/data/datasets/ts_replayer"

pytestmark = pytest.mark.skipif(
    not glob.glob(os.path.join(CORPUS, "*.json.gz")),
    reason="ts-replayer corpus not downloaded")


def _convert(replay_id: int):
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return convert_game(json.load(f))


def _failed_at(conv):
    return None if conv.failure is None else (conv.failure.turn, conv.failure.phase)


def test_the_olympics_go_to_the_side_the_log_says_won() -> None:
    """Replay 245 turn 6 headline: the US wins and takes 2 VP, reaching 6.

    The reconstruction gave them to the USSR and the score went to 2 instead.
    """
    conv = _convert(245)
    assert _failed_at(conv) != (6, "Headline"), (
        f"replay 245 stops at its Olympics: {conv.failure}")


def test_the_same_holds_when_the_ussr_wins_them() -> None:
    """Replay 165 turn 3 headline: the USSR wins and takes 2 VP, reaching USSR 8."""
    conv = _convert(165)
    assert _failed_at(conv) != (3, "Headline"), (
        f"replay 165 stops at its Olympics: {conv.failure}")


def test_the_log_records_the_winner_but_never_the_dice() -> None:
    """Which is why the board cannot settle it and the score has to."""
    with gzip.open(os.path.join(CORPUS, "245.json.gz"), "rt") as f:
        raws = json.load(f)["all_turns"]
    headline = next(r for r in raws
                    if (lambda e: e.turn == 6 and e.phase == "Headline")(parse_entry(r)))
    entry = parse_entry(headline)
    assert "Olympic Games" in (entry.card or "")
    assert not entry.die_rolls and not entry.bare_rolls, "no dice are printed"
    assert entry.vp_gains, "only the outcome is"
