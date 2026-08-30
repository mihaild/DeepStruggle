"""
Struggler <-> ts_ai Bidirectional Adapter & Translation Layer
============================================================
Provides mapping of board state, country names, card IDs, action kinds,
and decision contexts between the C++ engine (ts_engine) and Python engine (struggler).
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence
import ts_engine
from struggler.engine.core import Engine as SEngine
from struggler.engine.board import Board as SBoard
from struggler.engine.types import (
    Side as SSide,
    Region as SRegion,
    CardSide as SCardSide,
    DecisionKind as SDecisionKind,
    Action as SAction,
    Decision as SDecision,
)
from struggler.engine.cards import load_cards as s_load_cards

# Load struggler card dictionary
_STRUGGLER_CARDS = s_load_cards()

# Mapping table between ts_ai country name and struggler country name
_TS_TO_STRUGGLER_COUNTRY: dict[str, str] = {
    "United Kingdom": "UK",
    "Dominican Rep": "Dominican_Republic",
    "Spain/Portugal": "Spain_Portugal",
    "Laos/Cambodia": "Laos_Cambodia",
}

_STRUGGLER_TO_TS_COUNTRY: dict[str, str] = {
    "UK": "United Kingdom",
    "Dominican_Republic": "Dominican Rep",
    "Spain_Portugal": "Spain/Portugal",
    "Laos_Cambodia": "Laos/Cambodia",
}

# Pre-populate bidirectional country dictionaries
for cid in range(84):
    ts_name = ts_engine.MapData.get_country_name(cid)
    if ts_name not in _TS_TO_STRUGGLER_COUNTRY:
        _TS_TO_STRUGGLER_COUNTRY[ts_name] = ts_name.replace(" ", "_")
    s_name = _TS_TO_STRUGGLER_COUNTRY[ts_name]
    _STRUGGLER_TO_TS_COUNTRY[s_name] = ts_name


def country_ts_to_struggler(ts_name: str) -> str:
    """Convert ts_ai country name to struggler country identifier."""
    if ts_name in _TS_TO_STRUGGLER_COUNTRY:
        return _TS_TO_STRUGGLER_COUNTRY[ts_name]
    return ts_name.replace(" ", "_")


def country_struggler_to_ts(s_name: str) -> str:
    """Convert struggler country identifier to ts_ai country name."""
    if s_name in _STRUGGLER_TO_TS_COUNTRY:
        return _STRUGGLER_TO_TS_COUNTRY[s_name]
    return s_name.replace("_", " ")


def country_id_to_struggler(cid: int) -> str:
    """Convert country ID (0..83) to struggler country name."""
    ts_name = ts_engine.MapData.get_country_name(cid)
    return country_ts_to_struggler(ts_name)


def country_struggler_to_id(s_name: str) -> int:
    """Convert struggler country name to ts_ai country ID (0..83)."""
    ts_name = country_struggler_to_ts(s_name)
    return ts_engine.MapData.get_country_by_name(ts_name)


# Card ID mappings (1..110)
_TS_TO_STRUGGLER_CARD: dict[int, str] = {}
_STRUGGLER_TO_TS_CARD: dict[str, int] = {}

for s_id, card in _STRUGGLER_CARDS.items():
    _TS_TO_STRUGGLER_CARD[card.number] = s_id
    _STRUGGLER_TO_TS_CARD[s_id] = card.number


def card_id_ts_to_struggler(card_num: int) -> str:
    """Convert ts_ai card number (1..110) to struggler card string ID."""
    return _TS_TO_STRUGGLER_CARD[card_num]


def card_id_struggler_to_ts(s_id: str) -> int:
    """Convert struggler card string ID to ts_ai card number (1..110)."""
    return _STRUGGLER_TO_TS_CARD[s_id]


# Region mapping
_TS_TO_STRUGGLER_REGION: dict[ts_engine.Region, SRegion] = {
    ts_engine.Region.EUROPE: SRegion.EUROPE,
    ts_engine.Region.ASIA: SRegion.ASIA,
    ts_engine.Region.MIDDLE_EAST: SRegion.MIDDLE_EAST,
    ts_engine.Region.AFRICA: SRegion.AFRICA,
    ts_engine.Region.CENTRAL_AMERICA: SRegion.CENTRAL_AMERICA,
    ts_engine.Region.SOUTH_AMERICA: SRegion.SOUTH_AMERICA,
}

_STRUGGLER_TO_TS_REGION: dict[SRegion, ts_engine.Region] = {v: k for k, v in _TS_TO_STRUGGLER_REGION.items()}


def region_ts_to_struggler(r: ts_engine.Region) -> SRegion:
    return _TS_TO_STRUGGLER_REGION[r]


def region_struggler_to_ts(r: SRegion) -> ts_engine.Region:
    return _STRUGGLER_TO_TS_REGION[r]


def sync_ts_to_struggler(ts_state: ts_engine.GameState, s_eng: SEngine) -> None:
    """Copy all board state, tracks, and status from ts_state into s_eng."""
    # 1. Countries influence
    for cid in range(84):
        c = ts_state.get_country(cid)
        s_name = country_id_to_struggler(cid)
        s_eng.board.influence[s_name]["US"] = c.us_influence
        s_eng.board.influence[s_name]["USSR"] = c.ussr_influence

    # 2. Global Tracks
    s_eng.defcon = ts_state.defcon
    s_eng.vp = ts_state.victory_points
    s_eng.military_ops["US"] = ts_state.us_mil_ops
    s_eng.military_ops["USSR"] = ts_state.ussr_mil_ops
    s_eng.space_race["US"] = ts_state.us_space_track
    s_eng.space_race["USSR"] = ts_state.ussr_space_track
    s_eng.turn = ts_state.turn
    s_eng.action_round = ts_state.action_round

    # 3. China Card
    s_eng.china_card_owner = "US" if ts_state.china_card_holder == ts_engine.Player.US else "USSR"
    s_eng.china_card_available = bool(ts_state.china_card_playable)


def sync_struggler_to_ts(s_eng: SEngine, ts_state: ts_engine.GameState) -> None:
    """Copy all board state, tracks, and status from s_eng into ts_state."""
    # 1. Countries influence
    for s_name, inf in s_eng.board.influence.items():
        if s_name == "Chinese_Civil_War":
            continue
        cid = country_struggler_to_id(s_name)
        ts_state.set_country(cid, inf["US"], inf["USSR"])

    # 2. Global Tracks
    ts_state.defcon = s_eng.defcon
    ts_state.victory_points = s_eng.vp
    ts_state.us_mil_ops = s_eng.military_ops["US"]
    ts_state.ussr_mil_ops = s_eng.military_ops["USSR"]
    ts_state.us_space_track = s_eng.space_race["US"]
    ts_state.ussr_space_track = s_eng.space_race["USSR"]
    ts_state.turn = s_eng.turn
    ts_state.action_round = s_eng.action_round

    # 3. China Card
    ts_state.china_card_holder = ts_engine.Player.US if s_eng.china_card_owner == "US" else ts_engine.Player.USSR
    ts_state.china_card_playable = 1 if s_eng.china_card_available else 0


def get_ts_legal_country_names(ts_state: ts_engine.GameState) -> set[str]:
    """Get legal action country names for current POINT_NODE decision in ts_ai."""
    indices = ts_engine.Engine.get_legal_action_indices(ts_state)
    return {ts_engine.MapData.get_country_name(idx) for idx in indices}


def get_struggler_legal_country_names(decision: SDecision | None) -> set[str]:
    """Extract target country names from a struggler Decision."""
    if decision is None:
        return set()
    names = set()
    for opt in decision.options:
        if "country" in opt.payload:
            c_name = country_struggler_to_ts(opt.payload["country"])
            if c_name not in ("Chinese Civil War", "Chinese_Civil_War", "US", "USSR"):
                names.add(c_name)
        elif "choice" in opt.payload:
            val = opt.payload["choice"]
            if val not in ("none", "refuse", "discard", "remove", "add", "south_africa_only", "and_adjacent", "participate", "boycott", "1", "2", "3", "4", "5", "end_game", "decline"):
                c_name = country_struggler_to_ts(val)
                if c_name not in ("Chinese Civil War", "Chinese_Civil_War", "US", "USSR"):
                    names.add(c_name)
    return names


def assert_board_equal(ts_state: ts_engine.GameState, s_eng: SEngine) -> None:
    """Assert all 84 countries have identical US and USSR influence."""
    for cid in range(84):
        c = ts_state.get_country(cid)
        s_name = country_id_to_struggler(cid)
        s_inf = s_eng.board.influence[s_name]
        assert c.us_influence == s_inf["US"], f"US influence mismatch in {s_name} (ID {cid}): ts_ai={c.us_influence} vs struggler={s_inf['US']}"
        assert c.ussr_influence == s_inf["USSR"], f"USSR influence mismatch in {s_name} (ID {cid}): ts_ai={c.ussr_influence} vs struggler={s_inf['USSR']}"


def assert_tracks_equal(ts_state: ts_engine.GameState, s_eng: SEngine) -> None:
    """Assert all global tracks (DEFCON, VP, mil ops, space race, turn, AR) match."""
    assert ts_state.defcon == s_eng.defcon, f"DEFCON mismatch: ts_ai={ts_state.defcon} vs struggler={s_eng.defcon}"
    assert ts_state.victory_points == s_eng.vp, f"VP mismatch: ts_ai={ts_state.victory_points} vs struggler={s_eng.vp}"
    assert ts_state.us_mil_ops == s_eng.military_ops["US"], f"US Mil Ops mismatch: ts_ai={ts_state.us_mil_ops} vs struggler={s_eng.military_ops['US']}"
    assert ts_state.ussr_mil_ops == s_eng.military_ops["USSR"], f"USSR Mil Ops mismatch: ts_ai={ts_state.ussr_mil_ops} vs struggler={s_eng.military_ops['USSR']}"
    assert ts_state.us_space_track == s_eng.space_race["US"], f"US Space Track mismatch: ts_ai={ts_state.us_space_track} vs struggler={s_eng.space_race['US']}"
    assert ts_state.ussr_space_track == s_eng.space_race["USSR"], f"USSR Space Track mismatch: ts_ai={ts_state.ussr_space_track} vs struggler={s_eng.space_race['USSR']}"
    assert ts_state.turn == s_eng.turn, f"Turn mismatch: ts_ai={ts_state.turn} vs struggler={s_eng.turn}"


def assert_full_state_equal(ts_state: ts_engine.GameState, s_eng: SEngine) -> None:
    """Assert board influence and all global tracks are completely identical."""
    assert_board_equal(ts_state, s_eng)
    assert_tracks_equal(ts_state, s_eng)


def create_paired_state(seed: int = 42, defcon: int = 5, turn: int = 1, vp: int = 0) -> tuple[ts_engine.GameState, SEngine]:
    """Create a synchronized pair of (ts_state, s_eng) with identical baseline configuration."""
    s_eng = SEngine.new_game(seed=seed)
    ts_state = ts_engine.GameState()
    ts_engine.Engine.init_game(ts_state, seed)

    s_eng.defcon = defcon
    s_eng.turn = turn
    s_eng.vp = vp

    ts_state.defcon = defcon
    ts_state.turn = turn
    ts_state.victory_points = vp
    ts_state.current_phase = ts_engine.Phase.ACTION_ROUND

    sync_ts_to_struggler(ts_state, s_eng)
    return ts_state, s_eng
