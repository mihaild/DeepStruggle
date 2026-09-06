"""A headline holds two cards, and their events must not borrow each other's targets.

Both headlines resolve in one entry, so the driver's target queues hold the placements and
targets of two different cards at once. Nothing in a flat queue says which card a target
belongs to, and offering one card's target to the other's event is not a near miss: it is
usually not even legal, and the driver would then decline a decision the human made.
"""
import glob
import gzip
import json
import os
from typing import Optional

import pytest

from tools.lib.corpus_paths import corpus_dir
from tools.lib.ts_replayer_convert import convert_game

CORPUS = str(corpus_dir())



def _convert(replay_id: int):
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return convert_game(json.load(f))


def _failed_at(conv) -> Optional[str]:
    return None if conv.failure is None else f"T{conv.failure.turn} {conv.failure.phase}"


def test_che_and_junta_headline_keeps_each_events_targets_to_itself() -> None:
    """Replay 128 turn 7: the USSR headlines Che, the US headlines Junta.

    Che's first coup was offered while the event queue still held Junta's two placements into
    Venezuela -- a battleground, which Che may not touch at all. With neither queue able to
    answer, the driver took the decline, and both of Che's coups were lost: Saharan States and
    Colombia stayed exactly as they were and the USSR's military operations never moved.
    """
    conv = _convert(128)
    assert _failed_at(conv) != "T7 Headline", (
        f"Che's coups must not be declined for Junta's targets: {conv.failure}")


@pytest.mark.parametrize("replay_id,turn", [(128, 7), (190, 6)])
def test_the_che_headline_games_get_past_it(replay_id: int, turn: int) -> None:
    """Only that the Che headline itself is passed; these games stop later for other reasons."""
    conv = _convert(replay_id)
    assert conv.failure is None or (conv.failure.turn, conv.failure.phase) != (turn, "Headline"), (
        f"replay {replay_id} stops at its Che headline: {conv.failure}")
