"""An entry the log stops inside is converted up to, and no further.

The registry is keyed by turn, round *and* player, because an action round holds an entry for
each side and only one of them need be cut short.

Replay 55's last entry is turn 9 AR7 for the US. It stops one line into Tear Down This Wall:
the 3 US Influence the card puts into East Germany is recorded, then "Coup (3 Ops):" and
nothing -- no target, no roll, no result. The free coup in Europe the card grants is not in the
record at all, and the engine taking one of its own removed 2 USSR Influence from East Germany.
The log has East Germany standing at 5 USSR Influence since turn 3 AR2 and never moving again,
which is what made the entry look like a board mismatch rather than a truncated record.

The USSR's half of that same action round is complete and still converts.
"""
import glob
import gzip
import json
import os
from typing import Dict, Optional, Tuple

import pytest

from tools.lib.ts_replayer_convert import _KNOWN_INCOMPLETE, convert_game, country_id
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = "/workspace/data/datasets/ts_replayer"
EAST_GERMANY = 14

pytestmark = pytest.mark.skipif(
    not glob.glob(os.path.join(CORPUS, "*.json.gz")),
    reason="ts-replayer corpus not downloaded")


def _game(replay_id: int) -> Dict:
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return json.load(f)


def _east_germany(raw: Dict) -> Optional[Tuple[int, int]]:
    for key, c in (raw.get("countries") or {}).items():
        if country_id(key) == EAST_GERMANY:
            return int(c["inflUS"]), int(c["inflUSSR"])
    return None


def test_the_last_entry_stops_mid_coup() -> None:
    raws = _game(55)["all_turns"]
    last = parse_entry(raws[-1])
    assert (last.turn, last.phase, last.player) == (9, "AR7", "US")
    assert last.card == "Tear Down This Wall*"
    assert (last.text or "").rstrip().endswith("Coup (3 Ops):"), (
        "the record ends on the header, with no target, roll or result")
    assert last.targets == [], "and so there is nothing to drive it with"


def test_the_ussr_influence_the_missing_coup_would_have_taken() -> None:
    """East Germany settles at 5 at turn 3 AR2 and the log never moves it again.

    It passes through 5 earlier -- Warsaw Pact Formed puts it there at turn 1 AR2 and COMECON
    takes it to 6 at turn 2 AR6 -- so what matters is where it comes to rest: East European
    Unrest at turn 3 AR2, after which nothing in the record touches the USSR's Influence there.
    """
    raws = _game(55)["all_turns"]
    settles_at = next(i for i, r in enumerate(raws)
                      if (parse_entry(r).turn, parse_entry(r).phase) == (3, "AR2"))
    assert _east_germany(raws[settles_at]) == (0, 5)
    for raw in raws[settles_at:]:
        cell = _east_germany(raw)
        if cell is not None:
            assert cell[1] == 5, "the USSR's 5 stands to the end of the record"
    assert _east_germany(raws[-1]) == (3, 5), "Tear Down This Wall only adds the US's 3"


def test_the_game_converts_up_to_the_turn_the_record_stops_in() -> None:
    """Turn 9 goes entire, not just the entry that was cut short.

    A turn the recording stops inside lists only the cards played before it stopped, so every
    decision already converted in it was driven from a hand the player never held.
    """
    raws = _game(55)["all_turns"]
    conv = convert_game(_game(55))
    assert conv.failure is None, f"replay 55 stopped at {conv.failure}"
    assert conv.truncated_at is not None
    kept = [parse_entry(r) for r in raws[:conv.entries_converted]]
    assert kept[-1].turn == 8, "the last complete turn"
    assert all(e.turn <= 8 for e in kept)


def test_the_other_half_of_that_action_round_goes_with_it() -> None:
    """Turn 9 AR7 holds a complete USSR entry, and it is dropped too.

    Not because anything is wrong with it -- it replays to the log's own board -- but because
    it belongs to a turn whose hand list is a fragment. Which half of the round the registry
    names no longer changes how much is given back; it records where the log stopped.
    """
    raws = _game(55)["all_turns"]
    ussr = parse_entry(raws[-2])
    assert (ussr.turn, ussr.phase, ussr.player) == (9, "AR7", "USSR")
    assert ussr.card == "Latin American Debt Crisis"
    conv = convert_game(_game(55))
    assert conv.entries_converted < len(raws) - 1
    assert parse_entry(raws[conv.entries_converted - 1]).turn == 8


def test_the_registry_names_the_player() -> None:
    assert (9, "AR7", "US") in _KNOWN_INCOMPLETE[55]
    assert (9, "AR7", "USSR") not in _KNOWN_INCOMPLETE[55]


@pytest.mark.parametrize("replay_id,last_complete_turn", [(60, 6), (133, 9)])
def test_the_other_listed_games_stop_at_a_turn_boundary(replay_id: int,
                                                        last_complete_turn: int) -> None:
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
    assert conv.truncated_at is not None
    raws = _game(replay_id)["all_turns"]
    kept = [parse_entry(r) for r in raws[:conv.entries_converted]]
    assert kept[-1].turn == last_complete_turn
    assert all(e.turn <= last_complete_turn for e in kept)
