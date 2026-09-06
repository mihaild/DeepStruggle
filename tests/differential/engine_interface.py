"""
Twilight Struggle Common Engine Interface & Protocols
=====================================================
Pure Python interface specifications matching ts_engine pybinding signatures.
Does NOT import or depend on C++ ts_engine.
"""

from __future__ import annotations

import enum
import json
import os
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class Player(enum.IntEnum):
    NONE = 0
    US = 1
    USSR = -1


class Region(enum.IntEnum):
    EUROPE = 0
    ASIA = 1
    MIDDLE_EAST = 2
    AFRICA = 3
    CENTRAL_AMERICA = 4
    SOUTH_AMERICA = 5
    SOUTHEAST_ASIA = 6
    NONE_REGION = 7


REGION_NAMES: dict[Region, str] = {
    Region.EUROPE: "Europe",
    Region.ASIA: "Asia",
    Region.MIDDLE_EAST: "Middle East",
    Region.AFRICA: "Africa",
    Region.CENTRAL_AMERICA: "Central America",
    Region.SOUTH_AMERICA: "South America",
    Region.SOUTHEAST_ASIA: "Southeast Asia",
}


class Phase(enum.IntEnum):
    SETUP = 0
    HEADLINE = 1
    ACTION_ROUND = 2
    INTERRUPT = 3
    DISCARD = 4
    END_TURN = 5
    GAME_OVER = 6


class DecisionType(enum.IntEnum):
    NONE = 0
    SELECT_CARD = 1
    SELECT_PLAY_MODE = 2
    CHOOSE_TIMING_BRANCH = 3
    SELECT_OP_MODE = 4
    POINT_NODE = 5
    CHOOSE_BRANCH = 6
    ROLL_DIE = 7


class PlayMode(enum.IntEnum):
    EVENT = 0
    OPS = 1
    SPACE = 2


class OpMode(enum.IntEnum):
    NONE = 0
    COUP = 1
    INFLUENCE = 2
    REALIGN = 3


class CardLocation(enum.IntEnum):
    UNAVAILABLE = 0
    DRAW_DECK = 1
    HAND_US = 2
    HAND_USSR = 3
    DISCARD_PILE = 4
    REMOVED_FROM_GAME = 5
    ONGOING_EVENT = 6
    PEEKED_TEMP = 7


@dataclass
class MicroAction:
    """Action representation matching ts_engine.MicroAction."""
    decision_type: DecisionType = DecisionType.NONE
    primary_id: int = 0
    secondary_id: int = 0
    flags: int = 0


@runtime_checkable
class CountryStateProtocol(Protocol):
    """Protocol for country influence state."""
    @property
    def us_influence(self) -> int: ...
    @property
    def ussr_influence(self) -> int: ...


class SimpleCountryState:
    """Lightweight country state holder for external engine adapters."""
    __slots__ = ("_us", "_ussr")

    def __init__(self, us: int, ussr: int) -> None:
        self._us = us
        self._ussr = ussr

    @property
    def us_influence(self) -> int:
        return self._us

    @property
    def ussr_influence(self) -> int:
        return self._ussr

    def __repr__(self) -> str:
        return f"CountryState(us={self._us}, ussr={self._ussr})"


@runtime_checkable
class GameStateProtocol(Protocol):
    """Protocol matching ts_engine.GameState properties and methods."""

    @property
    def defcon(self) -> int: ...
    @defcon.setter
    def defcon(self, val: int, /) -> None: ...

    @property
    def victory_points(self) -> int: ...
    @victory_points.setter
    def victory_points(self, val: int, /) -> None: ...

    @property
    def turn(self) -> int: ...
    @turn.setter
    def turn(self, val: int, /) -> None: ...

    @property
    def action_round(self) -> int: ...
    @action_round.setter
    def action_round(self, val: int, /) -> None: ...

    @property
    def phase(self) -> str: ...
    @phase.setter
    def phase(self, val: str, /) -> None: ...

    @property
    def us_mil_ops(self) -> int: ...
    @us_mil_ops.setter
    def us_mil_ops(self, val: int, /) -> None: ...

    @property
    def ussr_mil_ops(self) -> int: ...
    @ussr_mil_ops.setter
    def ussr_mil_ops(self, val: int, /) -> None: ...

    @property
    def us_space_track(self) -> int: ...
    @us_space_track.setter
    def us_space_track(self, val: int, /) -> None: ...

    @property
    def ussr_space_track(self) -> int: ...
    @ussr_space_track.setter
    def ussr_space_track(self, val: int, /) -> None: ...

    def get_country(self, cid: int, /) -> CountryStateProtocol: ...
    def set_country(self, cid: int, us: int, ussr: int, /) -> None: ...

    def has_flag(self, flag: int | str, /) -> bool: ...
    def set_flag(self, flag: int | str, /) -> None: ...
    def clear_flag(self, flag: int | str, /) -> None: ...

    def get_hand(self, player: Player, /) -> list[int]: ...
    def set_hand(self, player: Player, cards: list[int], /) -> None: ...
    def get_discard_pile(self) -> list[int]: ...

    def to_dict(self) -> dict[str, Any]: ...


@runtime_checkable
class EngineProtocol(Protocol):
    """Protocol matching common engine operations."""

    def init_game(self, state: GameStateProtocol, seed: int = 42, /) -> None: ...
    def step(self, state: GameStateProtocol, action: MicroAction, /) -> bool: ...
    def is_terminal(self, state: GameStateProtocol, /) -> bool: ...
    def get_legal_placements(self, state: GameStateProtocol, player: Player, /) -> set[str]: ...
    def get_legal_coups(self, state: GameStateProtocol, player: Player, /, ops: int = 1) -> set[str]: ...
    def get_legal_realignments(self, state: GameStateProtocol, player: Player, /) -> set[str]: ...
    def score_region(self, state: GameStateProtocol, region: Region, /) -> int: ...
    def play_event(self, state: GameStateProtocol, player: Player, card_id: int, /) -> None: ...
    def lower_defcon(self, state: GameStateProtocol, player: Player, target_defcon: int = 2, /) -> None: ...

    def has_pending_decision(self, state: GameStateProtocol, /) -> bool: ...
    def get_pending_decision_type(self, state: GameStateProtocol, /) -> str: ...
    def get_decision_targets(self, state: GameStateProtocol, /) -> set[str]: ...
    def resolve_decision(self, state: GameStateProtocol, target_or_choice: str | int, /) -> bool: ...


# Load country and card databases directly from repo json without importing ts_engine
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MAP_JSON_PATH = os.path.join(_BASE_DIR, "rules", "map.json")
_CARDS_JSON_PATH = os.path.join(_BASE_DIR, "rules", "cards.json")

with open(_MAP_JSON_PATH, "r", encoding="utf-8") as _f:
    _map_data = json.load(_f)
    COUNTRY_NAMES: list[str] = [c["name"] for c in _map_data["countries"]]
    COUNTRY_IDS: dict[str, int] = {c["name"]: c["id"] for c in _map_data["countries"]}

with open(_CARDS_JSON_PATH, "r", encoding="utf-8") as _f:
    _cards_data = json.load(_f)
    CARD_NAMES: dict[int, str] = {c["id"]: c["name"] for c in _cards_data}
    CARD_IDS: dict[str, int] = {c["name"]: c["id"] for c in _cards_data}


def get_country_name(cid: int) -> str:
    return COUNTRY_NAMES[cid]


def get_country_id(name: str) -> int:
    return COUNTRY_IDS[name]


def get_card_name(cid: int) -> str:
    return CARD_NAMES[cid]


def get_card_id(name: str) -> int:
    return CARD_IDS[name]


def get_country_info(cid_or_name: int | str) -> dict[str, Any]:
    if isinstance(cid_or_name, str):
        cid = COUNTRY_IDS[cid_or_name]
    else:
        cid = cid_or_name
    return _map_data["countries"][cid]


def get_card_info(card_id: int) -> dict[str, Any]:
    for c in _cards_data:
        if c["id"] == card_id:
            return c
    raise KeyError(f"Unknown card id {card_id}")


# =============================================================================
# UNIFIED STATE ASSERTIONS
# =============================================================================

def assert_tracks_equal(state_a: GameStateProtocol, state_b: GameStateProtocol) -> None:
    """Asserts that all status tracks (DEFCON, VP, Turn, AR, MilOps, Space) match."""
    assert state_a.defcon == state_b.defcon, f"DEFCON: {state_a.defcon} vs {state_b.defcon}"
    assert state_a.victory_points == state_b.victory_points, f"VP: {state_a.victory_points} vs {state_b.victory_points}"
    assert state_a.turn == state_b.turn, f"Turn: {state_a.turn} vs {state_b.turn}"
    assert state_a.action_round == state_b.action_round, f"AR: {state_a.action_round} vs {state_b.action_round}"
    assert state_a.us_mil_ops == state_b.us_mil_ops, f"US MilOps: {state_a.us_mil_ops} vs {state_b.us_mil_ops}"
    assert state_a.ussr_mil_ops == state_b.ussr_mil_ops, f"USSR MilOps: {state_a.ussr_mil_ops} vs {state_b.ussr_mil_ops}"
    assert state_a.us_space_track == state_b.us_space_track, f"US Space: {state_a.us_space_track} vs {state_b.us_space_track}"
    assert state_a.ussr_space_track == state_b.ussr_space_track, f"USSR Space: {state_a.ussr_space_track} vs {state_b.ussr_space_track}"


def assert_boards_equal(state_a: GameStateProtocol, state_b: GameStateProtocol) -> None:
    """Asserts that influence across all 84 countries is identical."""
    mismatches = []
    for cid in range(84):
        c_name = get_country_name(cid)
        ca = state_a.get_country(cid)
        cb = state_b.get_country(cid)
        if ca.us_influence != cb.us_influence or ca.ussr_influence != cb.ussr_influence:
            mismatches.append(f"{c_name}: a=({ca.us_influence}/{ca.ussr_influence}) vs b=({cb.us_influence}/{cb.ussr_influence})")
    assert not mismatches, f"Board influence mismatches ({len(mismatches)}):\n" + "\n".join(mismatches)


def assert_states_equal(state_a: GameStateProtocol, state_b: GameStateProtocol) -> None:
    """Asserts complete parity across tracks and board influence."""
    assert_tracks_equal(state_a, state_b)
    assert_boards_equal(state_a, state_b)


# =============================================================================
# PERSISTENT EFFECT FLAG MAPPINGS
# =============================================================================

EFFECT_FLAG_BITS: dict[str, int] = {
    "nato": 1,
    "nato_canceled_france": 2,
    "nato_canceled_west_germany": 4,
    "marshall_plan": 8,
    "warsaw_pact": 16,
    "us_japan": 32,
    "us_japan_pact": 32,
    "containment": 64,
    "purge_us": 128,
    "purge_ussr": 256,
    "vietnam_revolts": 512,
    "formosan_resolution": 1024,
    "quagmire": 16384,
    "bear_trap": 32768,
    "we_will_bury_you": 131072,
    "brezhnev": 262144,
    "brezhnev_doctrine": 262144,
    "flower_power": 524288,
    "shuttle_diplomacy": 2097152,
    "camp_david": 16777216,
    "iron_lady": 33554432,
    "north_sea_oil": 67108864,
    "reformer": 268435456,
    "evil_empire": 1073741824,
    "aldrich_ames": 2147483648,
    "norad": 8589934592,
    "yuri_samantha": 17179869184,
    "yuri_and_samantha": 17179869184,
    "awacs": 34359738368,
    "iranian_hostage": 68719476736,
    "willy_brandt": 137438953472,
    "tear_down_this_wall": 274877906944,
}

EFFECT_BIT_TO_NAME: dict[int, str] = {v: k for k, v in EFFECT_FLAG_BITS.items()}
