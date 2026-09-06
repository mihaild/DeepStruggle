"""A space race is settled at the play mode, so its section is not an Ops answer.

An entry can hold several Ops sections and the driver takes them in the order the log prints
them, one per Ops decision. A "Space Race (N Ops):" header is a section like any other, but the
decision that answers it is SELECT_PLAY_MODE, not SELECT_OP_MODE -- so it is never taken off
the queue, and it is still at the head when the next Ops decision arrives.

At turn 6's headline of replay 156 the US headlines Grain Sales To Soviets and the USSR
headlines Junta. Both are 2 Ops and the US wins ties, so Grain Sales resolves first: it draws
Brezhnev Doctrine out of the USSR hand and the US puts it on the space race. Junta then places
2 Influence in Chile and asks the USSR how to spend its free action -- and the space section
answered for it. "space" is not an Ops mode, so nothing was chosen, and the coup on Panama the
log records never happened.
"""
import glob
import gzip
import json
import os
from typing import Dict

import pytest

from tools.lib.ts_replayer_convert import convert_game
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = "/workspace/data/datasets/ts_replayer"
PANAMA = 70
CHILE = 81

pytestmark = pytest.mark.skipif(
    not glob.glob(os.path.join(CORPUS, "*.json.gz")),
    reason="ts-replayer corpus not downloaded")


def _game(replay_id: int) -> Dict:
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return json.load(f)


def _headline(replay_id: int, turn: int):
    for raw in _game(replay_id)["all_turns"]:
        e = parse_entry(raw)
        if e.turn == turn and e.phase == "Headline":
            return e
    raise AssertionError(f"replay {replay_id} has no headline on turn {turn}")


def test_the_entry_holds_a_space_section_ahead_of_the_coup() -> None:
    e = _headline(156, 6)
    assert e.headlines == {"USSR": "Junta", "US": "Grain Sales To Soviets"}
    modes = [s.mode for s in e.sections]
    assert modes == ["space", "coup"], (
        "the space race is printed first and the free coup second")
    assert e.sections[1].targets == [PANAMA]


def test_the_free_action_is_the_coup_the_log_records() -> None:
    """Junta places 2 in Chile, then coups Panama."""
    e = _headline(156, 6)
    placed = [(cid, delta) for _s, delta, cid, _u, _ss in e.influence if cid == CHILE]
    assert placed == [(CHILE, 2)]
    assert e.event_mode == "coup"


def test_replay_156_converts_end_to_end() -> None:
    """Everything but the last turn, which the recording stops inside -- see
    tests/replayer/test_replay_unfinished_final_turn.py."""
    conv = convert_game(_game(156))
    assert conv.failure is None, f"replay 156 stopped at {conv.failure}"
    assert conv.entries_converted == 69


def test_an_entry_whose_sections_are_all_ops_is_unaffected() -> None:
    """Replay 103 turn 4 AR4: two coups under two headers, and both must still be taken."""
    conv = convert_game(_game(103))
    assert conv.failure is None, f"replay 103 stopped at {conv.failure}"


@pytest.mark.parametrize("replay_id", [156, 103, 100, 150])
def test_the_multi_section_games_convert(replay_id: int) -> None:
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
