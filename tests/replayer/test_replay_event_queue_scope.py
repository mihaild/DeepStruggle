"""Which queue may answer a decision, and when the shared one may not.

The shared event queue holds every placement the entry hangs on an "Event:" header. While a
card is resolving that the entry hangs placements on, those rows may be its; while a card is
resolving that it does not, none of them are -- provided the card has somewhere else to look.

Three things decide that, and each of them cost a game to get wrong.

Sections already consumed still count as somewhere else. At turn 7's headline of replay 239 the
USSR headlines Che and coups Nicaragua and then Guatemala; by the second coup both sections
have been taken and the target sits in the Ops queue. Testing for sections *remaining* let the
US's Puppet Governments answer it with El Salvador, the head of its own three placements, so
the USSR's 4 Influence went there instead of Guatemala.

A war's target is in the shared queue too -- event_queue starts with them -- and it is the
war's own whatever else the entry hangs on an event. At turn 8's headline of replay 112 the US
headlines Indo-Pakistani War against the USSR's Junta, and blanking the queue outright left the
war with no country to be fought in.

An event the log never names by header is never a key of the per-event queues, so its absence
there says nothing about it. NORAD prints the Influence it places and no header at all. At turn
9 AR1 of replay 133 Independent Reds places in Czechoslovakia and NORAD in Venezuela, and
blanking the queue for NORAD sent it to the Czechoslovakia the Ops queue still had a copy of.
"""
import glob
import gzip
import json
import os
from typing import Dict

import pytest

from tools.lib.ts_replayer_convert import (_UNNAMED_EVENTS, convert_game,
                                           event_point_queues, event_queue)
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = "/workspace/data/datasets/ts_replayer"
NICARAGUA, GUATEMALA, EL_SALVADOR = 69, 65, 66
CZECHOSLOVAKIA, VENEZUELA = 16, 77
NORAD = 106

pytestmark = pytest.mark.skipif(
    not glob.glob(os.path.join(CORPUS, "*.json.gz")),
    reason="ts-replayer corpus not downloaded")


def _game(replay_id: int) -> Dict:
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return json.load(f)


def _entry(replay_id: int, turn: int, phase: str, player: str):
    for raw in _game(replay_id)["all_turns"]:
        e = parse_entry(raw)
        if (e.turn, e.phase, e.player) == (turn, phase, player):
            return e
    raise AssertionError(f"replay {replay_id} has no {player} {phase} of turn {turn}")


def test_ches_second_coup_takes_its_own_target() -> None:
    """Replay 239 turn 7: Nicaragua then Guatemala, not the other event's El Salvador."""
    e = _entry(239, 7, "Headline", "both")
    assert [s.targets for s in e.sections] == [[NICARAGUA], [GUATEMALA]]
    queues = event_point_queues(e)
    assert EL_SALVADOR in next(iter(queues.values())), "the US's placement, not Che's target"
    conv = convert_game(_game(239))
    assert conv.failure is None, f"replay 239 stopped at {conv.failure}"


def test_a_wars_target_survives_the_blanking() -> None:
    """Replay 112 turn 8: the war needs the country it is fought in."""
    e = _entry(112, 8, "Headline", "both")
    assert e.war_targets, "the log names where the war is fought"
    assert set(e.war_targets) <= set(event_queue(e)), (
        "war targets live in the shared queue, so blanking it must spare them")
    conv = convert_game(_game(112))
    assert conv.failure is None, f"replay 112 stopped at {conv.failure}"


def test_a_blank_line_ends_an_events_own_placements() -> None:
    """Replay 133 turn 9 AR1: Czechoslovakia is the event's, Venezuela is NORAD's."""
    e = _entry(133, 9, "AR1", "USSR")
    queues = event_point_queues(e)
    assert list(queues.values()) == [[CZECHOSLOVAKIA]], (
        "what follows the blank line is not the event's")
    assert VENEZUELA in event_queue(e), "but it is still a placement the entry records"


def test_norad_is_never_blanked_out_of_the_shared_queue() -> None:
    assert NORAD in _UNNAMED_EVENTS


def test_replay_133_converts_to_where_its_log_stops() -> None:
    conv = convert_game(_game(133))
    assert conv.failure is None, f"replay 133 stopped at {conv.failure}"
    assert conv.truncated_at is not None


@pytest.mark.parametrize("replay_id", [239, 112, 133, 173, 150, 184, 131, 100, 105, 123])
def test_the_two_event_entries_convert(replay_id: int) -> None:
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
