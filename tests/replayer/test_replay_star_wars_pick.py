"""Star Wars takes the card the log says it fired, not one another card revealed.

Star Wars reaches into the discard pile and plays any non-scoring card there as its event, and
the log names that card on an "Event:" line of its own. A card revealed by something else in
the same entry can be sitting in that pile too.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Dict, List, cast

import pytest

from tools.lib.ts_replayer_convert import convert_game
from tools.lib.ts_replayer_parse import parse_entry

_CORPUS = "/workspace/data/datasets/ts_replayer"
_BEAR_TRAP = 44
_SALT_NEGOTIATIONS = 43


def _game(replay_id: int) -> Dict[str, object]:
    path = os.path.join(_CORPUS, f"{replay_id}.json.gz")
    if not os.path.exists(path):
        pytest.skip("corpus not downloaded")
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _entry(replay_id: int, turn: int, phase: str, player: str):
    raws = cast(List[Dict[str, object]], _game(replay_id)["all_turns"])
    for raw in raws:
        e = parse_entry(raw)
        if (e.turn, e.phase, e.player) == (turn, phase, player):
            return e
    raise AssertionError(f"no entry at replay {replay_id} T{turn} {phase} {player}")


def test_the_entry_names_both_the_reveal_and_the_pick() -> None:
    """Replay 291 turn 10: Aldrich Ames Remix reads the US hand and discards SALT Negotiations
    out of it; Star Wars then plays Bear Trap out of the pile."""
    e = _entry(291, 10, "Headline", "both")
    assert "SALT Negotiations*" in [nm for _side, nm in e.revealed]
    assert [nm for _side, nm in e.discards] == ["SALT Negotiations*"]
    assert "Bear Trap*" in e.events


def test_replay_291_traps_the_ussr_for_turn_10() -> None:
    """SALT, freshly discarded and freshly revealed, was offered to Star Wars first. With Bear
    Trap never in play, the USSR spent turn 10 playing cards where the log has them discarding
    Nuclear Subs to the trap and failing to escape."""
    conv = convert_game(_game(291))
    assert conv.failure is None, f"replay 291 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 144


@pytest.mark.parametrize("replay_id", [104, 158, 234])
def test_the_other_star_wars_games_are_unaffected(replay_id: int) -> None:
    """104 takes How I Learned To Stop Worrying, 158 takes Grain Sales To Soviets, and 234's
    entry answers a peek from a queue Star Wars must not touch."""
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
