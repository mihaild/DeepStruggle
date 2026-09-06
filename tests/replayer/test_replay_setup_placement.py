"""The opening placement is spread a country at a time, because the handicap follows it.

The log states the setup as a total per country -- "US +4 in West Germany", "+3 in France",
"+2 in Italy" -- and says nothing about the order, because in the game there is none: the
placement is simultaneous. The order still matters to the reconstruction, because the engine
places it in two stages: 7 Influence anywhere in Western Europe, and then the handicap, which
may only go where the US already has Influence.

Placing each country's total in one run spends the base allotment before the last country is
reached. At turn 1 of replay 152 four into West Germany and three into France used all seven,
and Italy -- still empty -- was not a legal target for the handicap. The reconstruction put
those two into West Germany and France instead, opening the game two Influence out in three
countries, and the Socialist Governments headline then removed Influence from a board that had
never held it. The game converted no entries at all.

Dealing one at a time round the named countries gives every one of them Influence inside the
base allotment, so the handicap has somewhere to go, and both spreads reach the same board.
"""
import collections
import glob
import gzip
import json
import os
from typing import Dict, List

import pytest
import ts_engine as ts

from tools.lib.ts_replayer_convert import convert_game, point_queue
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = "/workspace/data/datasets/ts_replayer"
WEST_GERMANY, FRANCE, ITALY = 7, 8, 10

pytestmark = pytest.mark.skipif(
    not glob.glob(os.path.join(CORPUS, "*.json.gz")),
    reason="ts-replayer corpus not downloaded")


def _game(replay_id: int) -> Dict:
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return json.load(f)


def _first(replay_id: int):
    return parse_entry(_game(replay_id)["all_turns"][0])


def test_the_first_entry_is_marked_as_carrying_the_setup() -> None:
    assert _first(152).setup is True
    assert parse_entry(_game(152)["all_turns"][5]).setup is False


def test_the_setup_is_dealt_one_at_a_time_round_the_named_countries() -> None:
    """Replay 152's US opens 4 West Germany, 3 France, 2 Italy."""
    queue = point_queue(_first(152))
    us = [c for c in queue if c in (WEST_GERMANY, FRANCE, ITALY)]
    assert collections.Counter(us) == {WEST_GERMANY: 4, FRANCE: 3, ITALY: 2}, (
        "the totals are the log's and must not change")
    assert us[:3] == [WEST_GERMANY, FRANCE, ITALY], (
        "every named country needs Influence inside the base seven")
    assert ITALY in us[:7], "Italy must be reached before the handicap opens"


def test_the_totals_survive_the_reordering() -> None:
    """Reordering may not add, drop or move a single point."""
    for replay_id in (100, 107, 109, 136, 152):
        entry = _first(replay_id)
        want: collections.Counter = collections.Counter()
        for _side, delta, cid, _u, _s in entry.ops_influence:
            want[cid] += abs(int(delta))
        assert collections.Counter(point_queue(entry)) == want, replay_id


def test_replay_152_converts_past_the_setup() -> None:
    """It converted nothing at all before: the very first entry's board was wrong."""
    conv = convert_game(_game(152))
    assert conv.entries_converted > 100, f"replay 152 stopped at {conv.failure}"


@pytest.mark.parametrize("replay_id", [100, 107, 109, 113, 136])
def test_the_ordinary_setups_still_convert(replay_id: int) -> None:
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"


def test_the_engine_gives_the_us_seven_plus_a_two_influence_handicap() -> None:
    """The stages the ordering has to satisfy, asserted so a change to them is noticed.

    A game whose log records a different handicap cannot be reconstructed: replay 196 opens
    "Handicap influence: US +5" and the engine has no way to grant more than two.
    """
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    assert state.ctx().decision_player == ts.Player.USSR
    assert int(state.ctx().remaining_steps) == 6, "the USSR spreads 6 in Eastern Europe"

    def place(cid: int) -> None:
        ts.Engine.step(state, ts.MicroAction(ts.DecisionType.POINT_NODE, cid, 0, 0))

    for _ in range(6):
        place(14)                              # East Germany
    assert state.ctx().decision_player == ts.Player.US
    assert int(state.ctx().remaining_steps) == 7, "the US spreads 7 in Western Europe"
    for _ in range(7):
        place(WEST_GERMANY)
    assert int(state.ctx().remaining_steps) == 2, "then a handicap of exactly 2"
    assert int(state.get_country(ITALY).us_influence) == 0
