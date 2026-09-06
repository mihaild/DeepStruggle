"""Chernobyl names its region in the log, and nothing else can tell you which it was.

The card designates a region and bars the USSR from adding Influence there for the rest of the
turn. Which region is a choice, and the log records it outright -- "US chooses South America".

Left to the branch search, which judges a branch by the board and the score it reaches, the
choice was arbitrary: Chernobyl moves neither. At turn 8 AR2 of replay 161 the US chose South
America and the engine took Europe, and the consequence surfaced three action rounds later,
somewhere else entirely -- the USSR played North Sea Oil for 3 Ops into East Germany and
Spain/Portugal, both in Europe, and placed nothing at all. The entry was reported as two
countries disagreeing about Influence that had never been barred.
"""
import glob
import gzip
import json
import os
from typing import Dict

import pytest

from tools.lib.corpus_paths import corpus_dir
from tools.lib.ts_replayer_convert import convert_game
from tools.lib.ts_replayer_parse import REGIONS, parse_entry

CORPUS = str(corpus_dir())
EAST_GERMANY, SPAIN_PORTUGAL = 14, 9



def _game(replay_id: int) -> Dict:
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return json.load(f)


def _entry(replay_id: int, turn: int, phase: str, player: str):
    for raw in _game(replay_id)["all_turns"]:
        e = parse_entry(raw)
        if (e.turn, e.phase, e.player) == (turn, phase, player):
            return e
    raise AssertionError(f"replay {replay_id} has no {player} {phase} of turn {turn}")


def test_the_log_names_the_region() -> None:
    e = _entry(161, 8, "AR2", "US")
    assert e.card == "Chernobyl*"
    assert "US chooses South America" in (e.text or "")
    assert e.region_choice == REGIONS["South America"] == 5


def test_the_region_numbers_are_the_engines() -> None:
    """The branch is answered with this number, so it has to be the engine's ordering."""
    import ts_engine as ts
    for name, index in REGIONS.items():
        assert str(ts.Region(index)).split(".")[-1].replace("_", " ").title() == name.title()


def test_an_entry_without_a_region_line_names_none() -> None:
    assert _entry(161, 8, "AR4", "USSR").region_choice is None


def test_the_barred_region_lets_north_sea_oil_place_where_the_log_places_it() -> None:
    """Turn 8 AR4: 3 Ops into East Germany and Spain/Portugal, both in Europe."""
    e = _entry(161, 8, "AR4", "USSR")
    assert e.card == "North Sea Oil*"
    touched = {cid for _s, _d, cid, _u, _ss in e.influence}
    assert {EAST_GERMANY, SPAIN_PORTUGAL} <= touched
    conv = convert_game(_game(161))
    assert conv.failure is None, f"replay 161 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total


def test_every_region_name_the_corpus_uses_is_understood() -> None:
    """55 region choices across the corpus, in five of the six regions."""
    seen = set()
    for path in sorted(glob.glob(os.path.join(CORPUS, "*.json.gz"))):
        with gzip.open(path, "rt") as f:
            for raw in json.load(f).get("all_turns") or []:
                choice = parse_entry(raw).region_choice
                if choice is not None:
                    seen.add(choice)
    assert seen, "no region choice parsed at all"
    assert seen <= set(REGIONS.values())
