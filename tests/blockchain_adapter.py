"""
ts-blockchain Adapter & Common Interface Implementation
=======================================================
Provides translation between the canonical engine interface and the
headless JavaScript ts-blockchain engine (external/ts-blockchain via Node.js bridge).

Does NOT import or refer to C++ ts_engine.
All game rules, operations, and event resolution are handled by ts-blockchain itself.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from tests.engine_interface import (
    Region,
    Player,
    DecisionType,
    MicroAction,
    GameStateProtocol,
    EngineProtocol,
    SimpleCountryState,
    get_country_name,
    get_country_id,
    get_card_name,
    get_card_id,
)

# Mapping table between standard country name and ts-blockchain country identifier
_TS_TO_BC_COUNTRY: dict[str, str] = {
    "United Kingdom": "uk",
    "Spain/Portugal": "spain",
    "Laos/Cambodia": "laos",
    "SE African States": "seafricanstates",
    "Dominican Rep": "dominicanrepublic",
}

_BC_TO_TS_COUNTRY: dict[str, str] = {
    "uk": "United Kingdom",
    "spain": "Spain/Portugal",
    "laos": "Laos/Cambodia",
    "seafricanstates": "SE African States",
    "dominicanrepublic": "Dominican Rep",
}

for cid in range(84):
    ts_name = get_country_name(cid)
    if ts_name not in _TS_TO_BC_COUNTRY:
        _TS_TO_BC_COUNTRY[ts_name] = ts_name.lower().replace(" ", "")
    bc_key = _TS_TO_BC_COUNTRY[ts_name]
    _BC_TO_TS_COUNTRY[bc_key] = ts_name


def country_ts_to_bc(ts_name: str) -> str:
    if ts_name in _TS_TO_BC_COUNTRY:
        return _TS_TO_BC_COUNTRY[ts_name]
    return ts_name.lower().replace(" ", "")


def country_bc_to_ts(bc_key: str) -> str:
    if bc_key in _BC_TO_TS_COUNTRY:
        return _BC_TO_TS_COUNTRY[bc_key]
    raise KeyError(f"Unknown ts-blockchain country identifier: {bc_key}")


def country_id_to_bc(cid: int) -> str:
    return country_ts_to_bc(get_country_name(cid))


def country_bc_to_id(bc_key: str) -> int:
    return get_country_id(country_bc_to_ts(bc_key))


_TS_TO_BC_REGION: dict[Region, str] = {
    Region.EUROPE: "europe",
    Region.ASIA: "asia",
    Region.MIDDLE_EAST: "mideast",
    Region.AFRICA: "africa",
    Region.CENTRAL_AMERICA: "centralamerica",
    Region.SOUTH_AMERICA: "southamerica",
    Region.SOUTHEAST_ASIA: "seasia",
}

_TS_TO_BC_CARD: dict[int, str] = {
    1: "asia",
    2: "europe",
    3: "mideast",
    4: "duckandcover",
    5: "fiveyearplan",
    6: "china",
    7: "socgov",
    8: "fidel",
    9: "vietnamrevolts",
    10: "blockade",
    11: "koreanwar",
    12: "romanianab",
    13: "arabisraeli",
    14: "comecon",
    15: "nasser",
    16: "warsawpact",
    17: "degaulle",
    18: "naziscientist",
    19: "truman",
    20: "olympic",
    21: "nato",
    22: "indreds",
    23: "marshall",
    24: "indopaki",
    25: "containment",
    26: "cia",
    27: "usjapan",
    28: "suezcrisis",
    29: "easteuropean",
    30: "decolonization",
    31: "redscare",
    32: "unintervention",
    33: "destalinization",
    34: "nucleartestban",
    35: "formosan",
    36: "brushwar",
    37: "centralamerica",
    38: "seasia",
    39: "armsrace",
    40: "cubanmissile",
    41: "nuclearsubs",
    42: "quagmire",
    43: "saltnegotiations",
    44: "beartrap",
    45: "summit",
    46: "howilearned",
    47: "junta",
    48: "kitchendebates",
    49: "missileenvy",
    50: "wwby",
    51: "brezhnev",
    52: "portuguese",
    53: "southafrican",
    54: "allende",
    55: "willybrandt",
    56: "muslimrevolution",
    57: "abmtreaty",
    58: "culturalrev",
    59: "flowerpower",
    60: "u2",
    61: "opec",
    62: "lonegunman",
    63: "colonial",
    64: "panamacanal",
    65: "campdavid",
    66: "puppet",
    67: "grainsales",
    68: "johnpaul",
    69: "deathsquads",
    70: "oas",
    71: "nixon",
    72: "sadat",
    73: "shuttle",
    74: "voiceofamerica",
    75: "liberation",
    76: "ussuri",
    77: "asknot",
    78: "alliance",
    79: "africa",
    80: "onesmallstep",
    81: "southamerica",
    82: "iranianhostage",
    83: "ironlady",
    84: "reagan",
    85: "starwars",
    86: "northseaoil",
    87: "reformer",
    88: "marine",
    89: "KAL007",
    90: "glasnost",
    91: "ortega",
    92: "terrorism",
    93: "irancontra",
    94: "chernobyl",
    95: "debtcrisis",
    96: "teardown",
    97: "evilempire",
    98: "aldrichames",
    99: "pershing",
    100: "wargames",
    101: "solidarity",
    102: "iraniraq",
    103: "defectors",
    104: "cambridge",
    105: "specialrelation",
    106: "norad",
    107: "che",
    108: "tehran",
    109: "yuri",
    110: "awacs",
}

_BC_TO_TS_CARD: dict[str, int] = {v: k for k, v in _TS_TO_BC_CARD.items()}


def card_id_ts_to_bc(cid: int) -> str:
    return _TS_TO_BC_CARD[cid]


def card_bc_to_ts_id(bc_key: str) -> int:
    return _BC_TO_TS_CARD[bc_key]


# =============================================================================
# BLOCKCHAIN CLIENT BRIDGE (STDIO JSON-RPC TO NODE.JS)
# =============================================================================

class BlockchainClient:
    """Subprocess manager for the headless Node.js bridge."""

    def __init__(self) -> None:
        bridge_path = os.path.join(os.path.dirname(__file__), "blockchain_bridge.js")
        self._proc = subprocess.Popen(
            ["node", bridge_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._closed = False

    def send_command(self, cmd: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("BlockchainClient is closed.")
        payload = json.dumps({"cmd": cmd, "args": args or {}})
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        self._proc.stdin.write(payload + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            err = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"Blockchain bridge exited unexpectedly: {err}")
        res = json.loads(line)
        if res.get("status") == "error":
            raise RuntimeError(f"Bridge error: {res.get('message')}")
        return res

    def init_game(self, defcon: int = 5, turn: int = 1, vp: int = 0) -> None:
        self.send_command("init_game", {"defcon": defcon, "turn": turn, "vp": vp})

    def get_state(self) -> dict[str, Any]:
        return self.send_command("get_state")["state"]

    def sync_state(self, **kwargs: Any) -> None:
        self.send_command("sync_state", kwargs)

    def place_influence(self, country: str, inf: int, player: str) -> None:
        bc_key = country_ts_to_bc(country)
        self.send_command("place_influence", {"country": bc_key, "inf": inf, "player": player.lower()})

    def resolve_coup(self, player: str, country: str, ops: int, roll: int) -> None:
        bc_key = country_ts_to_bc(country)
        self.send_command("resolve_coup", {"player": player.lower(), "country": bc_key, "ops": ops, "roll": roll})

    def resolve_realignment(self, country: str, us_roll: int, ussr_roll: int) -> None:
        bc_key = country_ts_to_bc(country)
        self.send_command("resolve_realignment", {"country": bc_key, "us_roll": us_roll, "ussr_roll": ussr_roll})

    def play_event(self, player: str, card_key: str) -> None:
        self.send_command("play_event", {"player": player.lower(), "card": card_key})

    def score_region(self, region: str) -> int:
        return self.send_command("score_region", {"region": region})["vp_delta"]

    def calculate_scoring(self, region: str) -> dict[str, Any]:
        return self.send_command("calculate_scoring", {"region": region})["breakdown"]

    def get_legal_placements(self, player: str) -> set[str]:
        res = self.send_command("get_legal_placements", {"player": player.lower()})
        return {country_bc_to_ts(k) for k in res["countries"]}

    def get_legal_coups(self, player: str, ops: int = 1, card: str = "") -> set[str]:
        res = self.send_command("get_legal_coups", {"player": player.lower(), "ops": ops, "card": card})
        return {country_bc_to_ts(k) for k in res["countries"]}

    def get_legal_realignments(self, player: str, card: str = "") -> set[str]:
        res = self.send_command("get_legal_realignments", {"player": player.lower(), "card": card})
        return {country_bc_to_ts(k) for k in res["countries"]}

    def get_legal_setup(self, player: str) -> set[str]:
        res = self.send_command("get_legal_setup", {"player": player.lower()})
        return {country_bc_to_ts(k) for k in res["countries"]}

    def remove_influence(self, country: str, inf: int, player: str) -> None:
        bc_key = country_ts_to_bc(country)
        self.send_command("remove_influence", {"country": bc_key, "inf": inf, "player": player.lower()})

    def lower_defcon(self) -> int:
        res = self.send_command("lower_defcon")
        return int(res["defcon"])

    def advance_space_race(self, player: str, roll: int = 1) -> dict[str, Any]:
        return self.send_command("advance_space_race", {"player": player.lower(), "roll": roll})

    def get_card_info(self) -> dict[str, Any]:
        res = self.send_command("get_card_info")
        return res["cards"]

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._proc.stdin:
                try:
                    self._proc.stdin.close()
                except Exception:
                    pass
            self._proc.terminate()
            self._proc.wait(timeout=2.0)


# =============================================================================
# BLOCKCHAIN STATE WRAPPER
# =============================================================================

from tests.engine_interface import EFFECT_BIT_TO_NAME

_TS_TO_BC_FLAGS: dict[str, str] = {
    "nato": "nato",
    "marshall_plan": "marshall",
    "warsaw_pact": "warsawpact",
    "us_japan": "usjapan",
    "vietnam_revolts": "vietnam_revolts",
    "containment": "containment",
    "willy_brandt": "willybrandt",
    "quagmire": "quagmire",
    "bear_trap": "beartrap",
    "camp_david": "campdavid",
    "flower_power": "flowerpower",
    "shuttle_diplomacy": "shuttlediplomacy",
    "north_sea_oil": "northseaoil",
    "brezhnev": "brezhnev",
    "norad": "norad",
    "formosan_resolution": "formosan",
    "awacs": "awacs",
    "tear_down_this_wall": "teardown",
    "iranian_hostage": "iranianhostage",
    "iron_lady": "ironlady",
    "reformer": "reformer",
    "evil_empire": "evilempire",
    "yuri_samantha": "yuri",
    "aldrich_ames": "aldrich",
}


def _resolve_bc_flag_name(flag: int | str) -> tuple[str, str]:
    raw_name = EFFECT_BIT_TO_NAME.get(flag) if isinstance(flag, int) else str(flag).lower()
    name = raw_name if raw_name is not None else str(flag).lower()
    bc_key = _TS_TO_BC_FLAGS.get(name, name.replace("_", ""))
    return name, bc_key

class BlockchainGameState(GameStateProtocol):
    """Exposes a ts-blockchain game state conforming to GameStateProtocol."""

    def __init__(self, client: BlockchainClient | None = None) -> None:
        self.raw_client: BlockchainClient = client if client is not None else BlockchainClient()
        self._owned_client = client is None
        self._ar_override: int | None = None
        self.setup_count: int = 0

    @property
    def defcon(self) -> int:
        return self.raw_client.get_state()["defcon"]

    @defcon.setter
    def defcon(self, val: int, /) -> None:
        self.raw_client.sync_state(defcon=val)

    @property
    def victory_points(self) -> int:
        return self.raw_client.get_state()["vp"]

    @victory_points.setter
    def victory_points(self, val: int, /) -> None:
        self.raw_client.sync_state(vp=val)

    @property
    def turn(self) -> int:
        return self.raw_client.get_state()["turn"]

    @turn.setter
    def turn(self, val: int, /) -> None:
        self.raw_client.sync_state(turn=val)

    @property
    def action_round(self) -> int:
        if self._ar_override is not None:
            return self._ar_override
        return self.raw_client.get_state()["round"]

    @action_round.setter
    def action_round(self, val: int, /) -> None:
        self._ar_override = val
        self.raw_client.sync_state(round=val)

    @property
    def phase(self) -> str:
        st = self.raw_client.get_state()
        return "headline" if st.get("round", 1) == 0 else "action_round"

    @phase.setter
    def phase(self, val: str, /) -> None:
        is_hl = (val == "headline")
        rnd = 0 if is_hl else 1
        self._ar_override = rnd
        self.raw_client.sync_state(round=rnd, headline=1 if is_hl else 0)

    @property
    def us_mil_ops(self) -> int:
        return self.raw_client.get_state()["milops_us"]

    @us_mil_ops.setter
    def us_mil_ops(self, val: int, /) -> None:
        self.raw_client.sync_state(milops_us=val)

    @property
    def ussr_mil_ops(self) -> int:
        return self.raw_client.get_state()["milops_ussr"]

    @ussr_mil_ops.setter
    def ussr_mil_ops(self, val: int, /) -> None:
        self.raw_client.sync_state(milops_ussr=val)

    @property
    def us_space_track(self) -> int:
        return self.raw_client.get_state()["space_race_us"]

    @us_space_track.setter
    def us_space_track(self, val: int, /) -> None:
        self.raw_client.sync_state(space_race_us=val)

    @property
    def ussr_space_track(self) -> int:
        return self.raw_client.get_state()["space_race_ussr"]

    @ussr_space_track.setter
    def ussr_space_track(self, val: int, /) -> None:
        self.raw_client.sync_state(space_race_ussr=val)

    def get_country(self, cid: int, /) -> SimpleCountryState:
        bc_key = country_id_to_bc(cid)
        c = self.raw_client.get_state()["countries"][bc_key]
        return SimpleCountryState(c["us"], c["ussr"])

    def set_country(self, cid: int, us: int, ussr: int, /) -> None:
        bc_key = country_id_to_bc(cid)
        self.raw_client.sync_state(countries={bc_key: {"us": us, "ussr": ussr}})

    def has_flag(self, flag: int | str, /) -> bool:
        name, bc_key = _resolve_bc_flag_name(flag)
        events = self.raw_client.get_state()["events"]
        return bool(events.get(name, 0) or events.get(bc_key, 0))

    def set_flag(self, flag: int | str, /) -> None:
        name, bc_key = _resolve_bc_flag_name(flag)
        self.raw_client.sync_state(events={name: 1, bc_key: 1})

    def clear_flag(self, flag: int | str, /) -> None:
        name, bc_key = _resolve_bc_flag_name(flag)
        self.raw_client.sync_state(events={name: 0, bc_key: 0})

    def get_hand(self, player: Player, /) -> list[int]:
        p = "us" if player == Player.US else "ussr"
        st = self.raw_client.get_state()
        hands = st.get("hands", {})
        return [card_bc_to_ts_id(card) for card in hands.get(p, []) if card in _BC_TO_TS_CARD]

    def set_hand(self, player: Player, cards: list[int], /) -> None:
        p = "us" if player == Player.US else "ussr"
        bc_cards = [card_id_ts_to_bc(cid) for cid in cards]
        self.raw_client.sync_state(hands={p: bc_cards})

    def get_discard_pile(self) -> list[int]:
        st = self.raw_client.get_state()
        return [card_bc_to_ts_id(card) for card in st.get("discards", []) if card in _BC_TO_TS_CARD]

    def to_dict(self) -> dict[str, Any]:
        st = self.raw_client.get_state()
        return {
            "defcon": st["defcon"],
            "vp": st["vp"],
            "turn": st["turn"],
            "action_round": st["round"],
            "us_mil_ops": st["milops_us"],
            "ussr_mil_ops": st["milops_ussr"],
            "us_space_track": st["space_race_us"],
            "ussr_space_track": st["space_race_ussr"],
        }

    def close(self) -> None:
        if self._owned_client:
            self.raw_client.close()


# =============================================================================
# BLOCKCHAIN ENGINE WRAPPER
# =============================================================================

# Alias for backwards compatibility with tests directly using BlockchainClient
BlockchainEngine = BlockchainClient

class BlockchainEngineAdapter(EngineProtocol):
    """Engine adapter for ts-blockchain executing all actions via Node.js bridge."""

    def init_game(self, state: GameStateProtocol, seed: int = 42, /) -> None:
        if isinstance(state, BlockchainGameState):
            state.setup_count = 0
            state._ar_override = None
            state.raw_client.init_game(defcon=5, turn=1, vp=0)

    def step(self, state: GameStateProtocol, action: MicroAction, /) -> bool:
        """Converts MicroAction representation and delegates execution to ts-blockchain bridge."""
        if not isinstance(state, BlockchainGameState):
            return False

        # 1. POINT_NODE action (country selection)
        if action.decision_type == DecisionType.POINT_NODE:
            country_name = get_country_name(action.primary_id)
            player = "ussr" if state.setup_count < 6 else "us"
            state.setup_count += 1
            state.raw_client.place_influence(country_name, 1, player)
            return True

        # 2. SELECT_CARD action (card event trigger)
        elif action.decision_type == DecisionType.SELECT_CARD:
            card_key = card_id_ts_to_bc(action.primary_id)
            player_str = "us" if action.primary_id % 2 == 0 else "ussr"
            state.raw_client.play_event(player_str, card_key)
            return True

        return False

    def is_terminal(self, state: GameStateProtocol, /) -> bool:
        if isinstance(state, BlockchainGameState):
            st = state.raw_client.get_state()
            return bool(st.get("game_over", 0) != 0 or st.get("defcon", 5) <= 1)
        return False

    def get_legal_placements(self, state: GameStateProtocol, player: Player, /) -> set[str]:
        if not isinstance(state, BlockchainGameState):
            return set()
        p_str = "us" if player == Player.US else "ussr"
        return state.raw_client.get_legal_placements(p_str)

    def get_legal_coups(self, state: GameStateProtocol, player: Player, /, ops: int = 1) -> set[str]:
        if not isinstance(state, BlockchainGameState):
            return set()
        p_str = "us" if player == Player.US else "ussr"
        return state.raw_client.get_legal_coups(p_str, ops=ops)

    def get_legal_realignments(self, state: GameStateProtocol, player: Player, /) -> set[str]:
        if not isinstance(state, BlockchainGameState):
            return set()
        p_str = "us" if player == Player.US else "ussr"
        return state.raw_client.get_legal_realignments(p_str)

    def score_region(self, state: GameStateProtocol, region: Region, /) -> int:
        if not isinstance(state, BlockchainGameState):
            return 0
        if region == Region.SOUTHEAST_ASIA:
            breakdown = state.raw_client.calculate_scoring("seasia")
            return breakdown["us"]["vp"] - breakdown["ussr"]["vp"]
        bc_r = _TS_TO_BC_REGION.get(region, region.name.lower())
        return state.raw_client.score_region(bc_r)

    def play_event(self, state: GameStateProtocol, player: Player, card_id: int, /) -> None:
        if isinstance(state, BlockchainGameState):
            p_str = "us" if player == Player.US else "ussr"
            bc_key = card_id_ts_to_bc(card_id)
            state.raw_client.play_event(p_str, bc_key)

    def lower_defcon(self, state: GameStateProtocol, player: Player, target_defcon: int = 2, /) -> None:
        if isinstance(state, BlockchainGameState):
            diff = state.defcon - target_defcon
            if diff > 0:
                for _ in range(diff):
                    state.raw_client.send_command("lower_defcon", {})
            else:
                state.raw_client.sync_state(defcon=target_defcon)

    def has_pending_decision(self, state: GameStateProtocol, /) -> bool:
        if isinstance(state, BlockchainGameState):
            res = state.raw_client.send_command("get_pending_decision", {})
            return bool(res.get("pending", False))
        return False

    def get_pending_decision_type(self, state: GameStateProtocol, /) -> str:
        if isinstance(state, BlockchainGameState):
            res = state.raw_client.send_command("get_pending_decision", {})
            return str(res.get("type", "none"))
        return "none"

    def get_decision_targets(self, state: GameStateProtocol, /) -> set[str]:
        if isinstance(state, BlockchainGameState):
            res = state.raw_client.send_command("get_pending_decision", {})
            targets = res.get("targets", [])
            out = set()
            for t in targets:
                if t in _BC_TO_TS_COUNTRY:
                    out.add(_BC_TO_TS_COUNTRY[t])
                else:
                    out.add(str(t))
            return out
        return set()

    def resolve_decision(self, state: GameStateProtocol, target_or_choice: str | int, /) -> bool:
        if isinstance(state, BlockchainGameState):
            bc_target = target_or_choice
            if isinstance(target_or_choice, str) and not target_or_choice.isdigit():
                bc_target = country_ts_to_bc(target_or_choice)
            res = state.raw_client.send_command("resolve_decision", {"choice": bc_target})
            return res.get("status") == "ok"
        return False

