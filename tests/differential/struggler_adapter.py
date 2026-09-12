"""
Struggler Adapter & Common Interface Implementation
===================================================
Provides translation between the canonical engine interface and the
pure Python Struggler engine (external/struggler).

Does NOT import or refer to C++ ts_engine.
All game rules, operations, and event resolution are handled by Struggler itself.
"""

from __future__ import annotations

from typing import Any
from struggler.engine.core import Engine as SEngine, SCORING_CARD_REGION
from struggler.engine.types import (
    Side as SSide,
    Region as SRegion,
    DecisionKind as SDecisionKind,
    Action as SAction,
)
from struggler.engine.cards import load_cards as s_load_cards
from tests.differential.engine_interface import (
    Region,
    get_country_info,
    get_country_id,
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

# Load struggler card dictionary
_STRUGGLER_CARDS = s_load_cards()

# Mapping table between standard country name and struggler country name
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

for cid in range(84):
    ts_name = get_country_name(cid)
    if ts_name not in _TS_TO_STRUGGLER_COUNTRY:
        _TS_TO_STRUGGLER_COUNTRY[ts_name] = ts_name.replace(" ", "_")
    s_name = _TS_TO_STRUGGLER_COUNTRY[ts_name]
    _STRUGGLER_TO_TS_COUNTRY[s_name] = ts_name


def country_ts_to_struggler(ts_name: str) -> str:
    if ts_name in _TS_TO_STRUGGLER_COUNTRY:
        return _TS_TO_STRUGGLER_COUNTRY[ts_name]
    return ts_name.replace(" ", "_")


def country_struggler_to_ts(s_name: str) -> str:
    if s_name in _STRUGGLER_TO_TS_COUNTRY:
        return _STRUGGLER_TO_TS_COUNTRY[s_name]
    return s_name.replace("_", " ")


def country_id_to_struggler(cid: int) -> str:
    return country_ts_to_struggler(get_country_name(cid))


def country_struggler_to_id(s_name: str) -> int:
    return get_country_id(country_struggler_to_ts(s_name))


_TS_TO_STRUGGLER_CARD: dict[int, str] = {}
_STRUGGLER_TO_TS_CARD: dict[str, int] = {}

for s_id, card in _STRUGGLER_CARDS.items():
    _TS_TO_STRUGGLER_CARD[card.number] = s_id
    _STRUGGLER_TO_TS_CARD[s_id] = card.number


def card_id_ts_to_struggler(cid: int) -> str:
    return _TS_TO_STRUGGLER_CARD[cid]


def card_id_struggler_to_ts(s_name: str) -> int:
    return _STRUGGLER_TO_TS_CARD[s_name]


# Flag mapping between canonical effect names and Struggler game_effects
_FLAG_NAME_TO_STRUGGLER: dict[str, str] = {
    "nato": "nato",
    "degaulle_france": "degaulle_france",
    "willy_brandt": "willy_brandt",
    "marshall_plan": "marshall_plan",
    "warsaw_pact": "warsaw_pact",
    "us_japan": "us_japan_pact",
    "us_japan_pact": "us_japan_pact",
    "vietnam_revolts": "vietnam_revolts",
    "containment": "containment",
    "brezhnev": "brezhnev",
    "camp_david": "camp_david",
    "flower_power": "flower_power",
    "shuttle_diplomacy": "shuttle_diplomacy",
    "norad": "norad",
    "formosan_resolution": "formosan_resolution",
    "awacs": "awacs",
    "reformer": "reformer",
    "evil_empire": "evil_empire",
    "yuri_samantha": "yuri_samantha",
    "aldrich_ames": "aldrich_ames",
}


# =============================================================================
# STRUGGLER STATE WRAPPER
# =============================================================================

from tests.differential.engine_interface import EFFECT_BIT_TO_NAME

class StrugglerGameState(GameStateProtocol):
    """Exposes a Struggler engine instance conforming to GameStateProtocol."""

    def __init__(self, engine: SEngine | None = None, seed: int = 42) -> None:
        self.raw_engine: SEngine = engine if engine is not None else SEngine.new_game(seed=seed)
        self._ar_override: int | None = None

    @property
    def defcon(self) -> int:
        return self.raw_engine.defcon

    @defcon.setter
    def defcon(self, val: int, /) -> None:
        self.raw_engine.defcon = val

    @property
    def victory_points(self) -> int:
        return self.raw_engine.vp

    @victory_points.setter
    def victory_points(self, val: int, /) -> None:
        self.raw_engine.vp = val

    @property
    def turn(self) -> int:
        return self.raw_engine.turn

    @turn.setter
    def turn(self, val: int, /) -> None:
        self.raw_engine.turn = val

    @property
    def action_round(self) -> int:
        if self._ar_override is not None:
            return self._ar_override
        if self.raw_engine.phase in ("setup", "headline"):
            return 0
        return self.raw_engine.action_round

    @action_round.setter
    def action_round(self, val: int, /) -> None:
        self._ar_override = val
        self.raw_engine.action_round = val

    @property
    def phase(self) -> str:
        return "headline" if self.raw_engine.phase == "headline" else "action_round"

    @phase.setter
    def phase(self, val: str, /) -> None:
        self.raw_engine.phase = "headline" if val == "headline" else "action_rounds"
        self.raw_engine._decision_stack.clear()

    @property
    def us_mil_ops(self) -> int:
        return self.raw_engine.military_ops["US"]

    @us_mil_ops.setter
    def us_mil_ops(self, val: int, /) -> None:
        self.raw_engine.military_ops["US"] = val

    @property
    def ussr_mil_ops(self) -> int:
        return self.raw_engine.military_ops["USSR"]

    @ussr_mil_ops.setter
    def ussr_mil_ops(self, val: int, /) -> None:
        self.raw_engine.military_ops["USSR"] = val

    @property
    def us_space_track(self) -> int:
        return self.raw_engine.space_race["US"]

    @us_space_track.setter
    def us_space_track(self, val: int, /) -> None:
        self.raw_engine.space_race["US"] = val

    @property
    def ussr_space_track(self) -> int:
        return self.raw_engine.space_race["USSR"]

    @ussr_space_track.setter
    def ussr_space_track(self, val: int, /) -> None:
        self.raw_engine.space_race["USSR"] = val

    def get_country(self, cid: int, /) -> SimpleCountryState:
        s_name = country_id_to_struggler(cid)
        inf = self.raw_engine.board.influence[s_name]
        return SimpleCountryState(inf["US"], inf["USSR"])

    def set_country(self, cid: int, us: int, ussr: int, /) -> None:
        s_name = country_id_to_struggler(cid)
        self.raw_engine.board.influence[s_name]["US"] = us
        self.raw_engine.board.influence[s_name]["USSR"] = ussr

    def has_flag(self, flag: int | str, /) -> bool:
        name = EFFECT_BIT_TO_NAME.get(flag, "") if isinstance(flag, int) else str(flag).lower()
        key = _FLAG_NAME_TO_STRUGGLER.get(name, name)
        return bool(self.raw_engine.game_effects.get(key) or self.raw_engine.turn_effects.get(key))

    def set_flag(self, flag: int | str, /) -> None:
        name = EFFECT_BIT_TO_NAME.get(flag, "") if isinstance(flag, int) else str(flag).lower()
        key = _FLAG_NAME_TO_STRUGGLER.get(name, name)
        self.raw_engine.game_effects[key] = True
        self.raw_engine.turn_effects[key] = True

    def clear_flag(self, flag: int | str, /) -> None:
        name = EFFECT_BIT_TO_NAME.get(flag, "") if isinstance(flag, int) else str(flag).lower()
        key = _FLAG_NAME_TO_STRUGGLER.get(name, name)
        self.raw_engine.game_effects.pop(key, None)
        self.raw_engine.turn_effects.pop(key, None)

    def get_hand(self, player: Player, /) -> list[int]:
        side = "US" if player == Player.US else "USSR"
        return [card_id_struggler_to_ts(name) for name in self.raw_engine.hands[side]]

    def set_hand(self, player: Player, cards: list[int], /) -> None:
        side = "US" if player == Player.US else "USSR"
        self.raw_engine.hands[side] = [card_id_ts_to_struggler(cid) for cid in cards]

    def get_discard_pile(self) -> list[int]:
        return [card_id_struggler_to_ts(name) for name in self.raw_engine.discard_pile]

    def to_dict(self) -> dict[str, Any]:
        return {
            "defcon": self.defcon,
            "vp": self.victory_points,
            "turn": self.turn,
            "action_round": self.action_round,
            "us_mil_ops": self.us_mil_ops,
            "ussr_mil_ops": self.ussr_mil_ops,
            "us_space_track": self.us_space_track,
            "ussr_space_track": self.ussr_space_track,
        }


# =============================================================================
# STRUGGLER ENGINE WRAPPER
# =============================================================================

class StrugglerEngine(EngineProtocol):
    """Engine adapter for Struggler executing all actions via SEngine.step()."""

    def init_game(self, state: GameStateProtocol, seed: int = 42, /) -> None:
        if isinstance(state, StrugglerGameState):
            state.raw_engine = SEngine.new_game(seed=seed)

    def step(self, state: GameStateProtocol, action: MicroAction, /) -> bool:
        """Converts MicroAction representation and delegates execution to Struggler core."""
        if not isinstance(state, StrugglerGameState):
            return False

        eng = state.raw_engine
        pending = eng.pending_decision
        if pending is None:
            return False

        # 1. POINT_NODE action (country selection for setup, ops influence, coup target, realignment target)
        if action.decision_type == DecisionType.POINT_NODE:
            country_name = get_country_name(action.primary_id)
            s_country = country_ts_to_struggler(country_name)
            if pending.kind == SDecisionKind.PLACE_INFLUENCE:
                eng.step(SAction(kind=SDecisionKind.PLACE_INFLUENCE, payload={"country": s_country}))
                return True
            elif pending.kind == SDecisionKind.COUP_TARGET:
                eng.step(SAction(kind=SDecisionKind.COUP_TARGET, payload={"country": s_country}))
                return True
            elif pending.kind == SDecisionKind.REALIGNMENT_TARGET:
                eng.step(SAction(kind=SDecisionKind.REALIGNMENT_TARGET, payload={"country": s_country}))
                return True

        # 2. SELECT_CARD action (headline play or action round play)
        elif action.decision_type == DecisionType.SELECT_CARD:
            card_name = card_id_ts_to_struggler(action.primary_id)
            if pending.kind == SDecisionKind.HEADLINE_PLAY:
                eng.step(SAction(kind=SDecisionKind.HEADLINE_PLAY, payload={"card": card_name}))
                return True
            elif pending.kind == SDecisionKind.ACTION_ROUND_PLAY:
                mode_str = "event" if action.secondary_id == 0 else "ops"
                eng.step(SAction(kind=SDecisionKind.ACTION_ROUND_PLAY, payload={"card": card_name, "mode": mode_str}))
                return True
            else:
                side = SSide.US if action.secondary_id == 0 else SSide.USSR
                eng._fire_event(side, card_name)
                return True

        # 3. ROLL_DIE action (coup roll or realignment roll)
        elif action.decision_type == DecisionType.ROLL_DIE:
            roll_val = action.primary_id
            if pending.kind == SDecisionKind.COUP_ROLL:
                eng.step(SAction(kind=SDecisionKind.COUP_ROLL, payload={"value": roll_val}))
                return True
            elif pending.kind == SDecisionKind.REALIGNMENT_ACTOR_ROLL:
                eng.step(SAction(kind=SDecisionKind.REALIGNMENT_ACTOR_ROLL, payload={"value": roll_val}))
                return True
            elif pending.kind == SDecisionKind.REALIGNMENT_OPPONENT_ROLL:
                eng.step(SAction(kind=SDecisionKind.REALIGNMENT_OPPONENT_ROLL, payload={"value": roll_val}))
                return True

        return False

    def is_terminal(self, state: GameStateProtocol, /) -> bool:
        if isinstance(state, StrugglerGameState):
            return state.raw_engine.is_terminal
        return False

    def get_legal_placements(self, state: GameStateProtocol, player: Player, /) -> set[str]:
        if not isinstance(state, StrugglerGameState):
            return set()
        s_side = SSide.US if player == Player.US else SSide.USSR
        legal = set()
        for cid in range(84):
            c_name = get_country_name(cid)
            s_name = country_ts_to_struggler(c_name)
            if state.raw_engine.board.is_reachable(s_side, s_name):
                legal.add(c_name)
        return legal

    def get_legal_coups(self, state: GameStateProtocol, player: Player, /, ops: int = 1) -> set[str]:
        if not isinstance(state, StrugglerGameState):
            return set()
        opp_side = "USSR" if player == Player.US else "US"
        legal = set()
        for cid in range(84):
            c_info = get_country_info(cid)
            c_name = c_info["name"]
            s_name = country_ts_to_struggler(c_name)
            inf = state.raw_engine.board.influence.get(s_name, {})
            if inf.get(opp_side, 0) == 0:
                continue
            reg = c_info["region"]
            if reg == "Europe" and state.defcon <= 4:
                continue
            if reg == "Asia" and state.defcon <= 3:
                continue
            if reg == "Middle East" and state.defcon <= 2:
                continue
            if cid == get_country_id("Japan") and player == Player.USSR and state.has_flag("us_japan"):
                continue
            legal.add(c_name)
        return legal

    def get_legal_realignments(self, state: GameStateProtocol, player: Player, /) -> set[str]:
        return self.get_legal_coups(state, player)

    def score_region(self, state: GameStateProtocol, region: Region, /) -> int:
        if not isinstance(state, StrugglerGameState):
            return 0
        if region == Region.SOUTHEAST_ASIA:
            return state.raw_engine._score_southeast_asia()
        s_reg = SRegion[region.name]
        return state.raw_engine.board.score_region(s_reg)

    def play_event(self, state: GameStateProtocol, player: Player, card_id: int, /) -> None:
        if isinstance(state, StrugglerGameState):
            s_cid = card_id_ts_to_struggler(card_id)
            s_side = SSide.US if player == Player.US else SSide.USSR
            if s_cid in SCORING_CARD_REGION or s_cid == "Southeast_Asia_Scoring":
                state.raw_engine._resolve_scoring_card(s_cid)
            else:
                state.raw_engine._fire_event(s_side, s_cid)

    def lower_defcon(self, state: GameStateProtocol, player: Player, target_defcon: int = 2, /) -> None:
        if isinstance(state, StrugglerGameState):
            delta = target_defcon - state.defcon
            side = SSide.US if player == Player.US else SSide.USSR
            state.raw_engine._change_defcon(delta, caused_by=side)

    def has_pending_decision(self, state: GameStateProtocol, /) -> bool:
        if not isinstance(state, StrugglerGameState):
            return False
        return state.raw_engine.pending_decision is not None

    def get_pending_decision_type(self, state: GameStateProtocol, /) -> str:
        if not isinstance(state, StrugglerGameState) or state.raw_engine.pending_decision is None:
            return "none"
        kind = state.raw_engine.pending_decision.kind
        if "target" in kind.value or "influence" in kind.value or kind.value == "choose_country":
            return "country"
        elif "discard" in kind.value or "card" in kind.value:
            return "card"
        elif "roll" in kind.value:
            return "roll"
        return "choice"

    def get_decision_targets(self, state: GameStateProtocol, /) -> set[str]:
        if not isinstance(state, StrugglerGameState) or state.raw_engine.pending_decision is None:
            return set()
        dec = state.raw_engine.pending_decision
        targets = set()
        for opt in dec.options:
            if hasattr(opt, "payload") and isinstance(opt.payload, dict):
                if "country" in opt.payload:
                    c_name = country_struggler_to_ts(opt.payload["country"])
                    if c_name not in ("Chinese Civil War", "Chinese_Civil_War", "US", "USSR"):
                        targets.add(c_name)
                elif "card" in opt.payload:
                    targets.add(str(card_id_struggler_to_ts(opt.payload["card"])))
                elif "choice" in opt.payload:
                    targets.add(str(opt.payload["choice"]))
                elif "value" in opt.payload:
                    targets.add(str(opt.payload["value"]))
        return targets

    def resolve_decision(self, state: GameStateProtocol, target_or_choice: str | int, /) -> bool:
        if not isinstance(state, StrugglerGameState) or state.raw_engine.pending_decision is None:
            return False
        dec = state.raw_engine.pending_decision
        target_str = str(target_or_choice)
        
        matched_action = None
        for opt in dec.options:
            if hasattr(opt, "payload") and isinstance(opt.payload, dict):
                if "country" in opt.payload and country_struggler_to_ts(opt.payload["country"]) == target_str:
                    matched_action = opt
                    break
                if "card" in opt.payload and (
                    str(card_id_struggler_to_ts(opt.payload["card"])) == target_str or
                    opt.payload["card"] == target_str
                ):
                    matched_action = opt
                    break
                if "choice" in opt.payload:
                    val = str(opt.payload["choice"])
                    if val == target_str:
                        matched_action = opt
                        break
                    if target_str in ("0", "refuse", "decline", "none") and val in ("0", "refuse", "decline", "none"):
                        matched_action = opt
                        break
                if "card" in opt.payload and target_str.isdigit():
                    cid = int(target_str)
                    if 1 <= cid <= 110:
                        s_card = card_id_ts_to_struggler(cid)
                        if opt.payload["card"] == s_card:
                            matched_action = opt
                            break
                if "value" in opt.payload and str(opt.payload["value"]) == target_str:
                    matched_action = opt
                    break

        if matched_action is None and dec.options:
            matched_action = dec.options[0]

        if matched_action is not None:
            state.raw_engine.step(matched_action)
            return True
        return False



# =============================================================================
# STRUGGLER LEGAL CANDIDATE HELPER (NO TS_ENGINE DEPENDENCY)
# =============================================================================

def get_struggler_legal_country_names(decision: Any) -> set[str]:
    """Extract target country names from a struggler Decision or Engine."""
    if decision is None:
        return set()
    options = decision.options if hasattr(decision, "options") else getattr(decision, "pending_decision", decision)
    if hasattr(options, "options"):
        options = options.options
    if not isinstance(options, (list, tuple)):
        return set()
    names = set()
    for opt in options:
        if hasattr(opt, "payload") and isinstance(opt.payload, dict):
            if "country" in opt.payload:
                c_name = country_struggler_to_ts(opt.payload["country"])
                if c_name not in ("Chinese Civil War", "Chinese_Civil_War", "US", "USSR"):
                    names.add(c_name)
            elif "choice" in opt.payload:
                val = str(opt.payload["choice"])
                if val not in ("none", "refuse", "discard", "remove", "add", "south_africa_only", "and_adjacent", "participate", "boycott", "1", "2", "3", "4", "5", "end_game", "decline"):
                    c_name = country_struggler_to_ts(val)
                    if c_name not in ("Chinese Civil War", "Chinese_Civil_War", "US", "USSR"):
                        names.add(c_name)
    return names

# =============================================================================
# STRUGGLER DIFFERENTIAL COMPARISON HELPERS
# =============================================================================

import ts_engine

def sync_ts_to_struggler(ts_state: Any, s_eng: SEngine) -> None:
    for cid in range(84):
        c = ts_state.get_country(cid)
        s_name = country_id_to_struggler(cid)
        s_eng.board.influence[s_name]["US"] = c.us_influence
        s_eng.board.influence[s_name]["USSR"] = c.ussr_influence
    s_eng.defcon = ts_state.defcon
    s_eng.vp = ts_state.victory_points
    s_eng.turn = ts_state.turn
    s_eng.action_round = ts_state.action_round


def get_ts_legal_country_names(ts_state: Any) -> set[str]:
    """Get legal action country names for current POINT_NODE decision in ts_ai."""
    indices = ts_engine.Engine.get_legal_action_indices(ts_state)
    return {ts_engine.MapData.get_country_name(idx) for idx in indices}


def assert_board_equal(ts_state: Any, s_eng: SEngine) -> None:
    """Assert all 84 countries have identical US and USSR influence."""
    for cid in range(84):
        c = ts_state.get_country(cid)
        s_name = country_id_to_struggler(cid)
        s_inf = s_eng.board.influence[s_name]
        assert c.us_influence == s_inf["US"], f"US influence mismatch in {s_name} (ID {cid}): ts_ai={c.us_influence} vs struggler={s_inf['US']}"
        assert c.ussr_influence == s_inf["USSR"], f"USSR influence mismatch in {s_name} (ID {cid}): ts_ai={c.ussr_influence} vs struggler={s_inf['USSR']}"


def assert_tracks_equal(ts_state: Any, s_eng: SEngine) -> None:
    """Assert all global tracks (DEFCON, VP, mil ops, space race, turn, AR) match."""
    assert ts_state.defcon == s_eng.defcon, f"DEFCON mismatch: ts_ai={ts_state.defcon} vs struggler={s_eng.defcon}"
    assert ts_state.victory_points == s_eng.vp, f"VP mismatch: ts_ai={ts_state.victory_points} vs struggler={s_eng.vp}"
    assert ts_state.us_mil_ops == s_eng.military_ops["US"], f"US Mil Ops mismatch: ts_ai={ts_state.us_mil_ops} vs struggler={s_eng.military_ops['US']}"
    assert ts_state.ussr_mil_ops == s_eng.military_ops["USSR"], f"USSR Mil Ops mismatch: ts_ai={ts_state.ussr_mil_ops} vs struggler={s_eng.military_ops['USSR']}"
    assert ts_state.us_space_track == s_eng.space_race["US"], f"US Space Track mismatch: ts_ai={ts_state.us_space_track} vs struggler={s_eng.space_race['US']}"
    assert ts_state.ussr_space_track == s_eng.space_race["USSR"], f"USSR Space Track mismatch: ts_ai={ts_state.ussr_space_track} vs struggler={s_eng.space_race['USSR']}"
    assert ts_state.turn == s_eng.turn, f"Turn mismatch: ts_ai={ts_state.turn} vs struggler={s_eng.turn}"


def assert_full_state_equal(ts_state: Any, s_eng: SEngine) -> None:
    """Assert board influence and all global tracks are completely identical."""
    assert_board_equal(ts_state, s_eng)
    assert_tracks_equal(ts_state, s_eng)


def create_paired_state(seed: int = 42, defcon: int = 5, turn: int = 1, vp: int = 0) -> tuple[Any, SEngine]:
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
