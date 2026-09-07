"""
Native C++ Engine Adapter
=========================
Adapts the C++ ts_engine pybinding to the canonical EngineProtocol and GameStateProtocol.
This is the ONLY adapter that imports ts_engine.
"""

from __future__ import annotations

from typing import Any
import ts_engine
from tests.differential.engine_interface import (
    Player,
    Region,
    get_country_info,
    get_country_name,
    get_country_id,
    get_card_id,
    get_card_name,
    EFFECT_FLAG_BITS,
    EngineProtocol,
    GameStateProtocol,
    MicroAction,
    CountryStateProtocol,
    EFFECT_FLAG_BITS,
)


class NativeGameState(GameStateProtocol):
    """Wraps ts_engine.GameState providing full compatibility with GameStateProtocol."""

    def __init__(self, raw_state: ts_engine.GameState | None = None) -> None:
        self.raw_state: ts_engine.GameState = raw_state or ts_engine.GameState()

    @property
    def defcon(self) -> int:
        return self.raw_state.defcon

    @defcon.setter
    def defcon(self, val: int, /) -> None:
        self.raw_state.defcon = val

    @property
    def victory_points(self) -> int:
        return self.raw_state.victory_points

    @victory_points.setter
    def victory_points(self, val: int, /) -> None:
        self.raw_state.victory_points = val

    @property
    def turn(self) -> int:
        return self.raw_state.turn

    @turn.setter
    def turn(self, val: int, /) -> None:
        self.raw_state.turn = val

    @property
    def action_round(self) -> int:
        return self.raw_state.action_round

    @action_round.setter
    def action_round(self, val: int, /) -> None:
        self.raw_state.action_round = val

    @property
    def phase(self) -> str:
        if self.raw_state.current_phase == ts_engine.Phase.HEADLINE:
            return "headline"
        return "action_round"

    @phase.setter
    def phase(self, val: str, /) -> None:
        if val == "headline":
            self.raw_state.current_phase = ts_engine.Phase.HEADLINE
            self.raw_state.action_round = 0
        else:
            self.raw_state.current_phase = ts_engine.Phase.ACTION_ROUND
            if self.raw_state.action_round == 0:
                self.raw_state.action_round = 1
        self.raw_state.ctx().decision_type = ts_engine.DecisionType.NONE

    @property
    def us_mil_ops(self) -> int:
        return self.raw_state.us_mil_ops

    @us_mil_ops.setter
    def us_mil_ops(self, val: int, /) -> None:
        self.raw_state.us_mil_ops = val

    @property
    def ussr_mil_ops(self) -> int:
        return self.raw_state.ussr_mil_ops

    @ussr_mil_ops.setter
    def ussr_mil_ops(self, val: int, /) -> None:
        self.raw_state.ussr_mil_ops = val

    @property
    def us_space_track(self) -> int:
        return self.raw_state.us_space_track

    @us_space_track.setter
    def us_space_track(self, val: int, /) -> None:
        self.raw_state.us_space_track = val

    @property
    def ussr_space_track(self) -> int:
        return self.raw_state.ussr_space_track

    @ussr_space_track.setter
    def ussr_space_track(self, val: int, /) -> None:
        self.raw_state.ussr_space_track = val

    def get_country(self, cid: int, /) -> CountryStateProtocol:
        return self.raw_state.get_country(cid)

    def set_country(self, cid: int, us: int, ussr: int, /) -> None:
        self.raw_state.set_country(cid, us, ussr)

    def has_flag(self, flag: int | str, /) -> bool:
        bit = EFFECT_FLAG_BITS.get(flag, 0) if isinstance(flag, str) else flag
        return self.raw_state.has_flag(bit)

    def set_flag(self, flag: int | str, /) -> None:
        bit = EFFECT_FLAG_BITS.get(flag, 0) if isinstance(flag, str) else flag
        self.raw_state.set_flag(bit)

    def clear_flag(self, flag: int | str, /) -> None:
        bit = EFFECT_FLAG_BITS.get(flag, 0) if isinstance(flag, str) else flag
        self.raw_state.clear_flag(bit)

    def get_hand(self, player: Player, /) -> list[int]:
        loc = ts_engine.hand_of(ts_engine.Player.US) if player == Player.US else ts_engine.hand_of(ts_engine.Player.USSR)
        return [card for card in range(1, 111) if self.raw_state.get_card_location(card) == loc]

    def set_hand(self, player: Player, cards: list[int], /) -> None:
        loc = ts_engine.hand_of(ts_engine.Player.US) if player == Player.US else ts_engine.hand_of(ts_engine.Player.USSR)
        for card in range(1, 111):
            if self.raw_state.get_card_location(card) == loc:
                self.raw_state.set_card_location(card, ts_engine.CardLocation.DRAW_DECK)
        for card in cards:
            self.raw_state.set_card_location(card, loc)

    def get_discard_pile(self) -> list[int]:
        return [card for card in range(1, 111) if self.raw_state.get_card_location(card) == ts_engine.CardLocation.DISCARD_PILE]

    def to_dict(self) -> dict[str, Any]:
        return self.raw_state.to_dict()


def _get_raw(state: GameStateProtocol) -> ts_engine.GameState:
    if isinstance(state, NativeGameState):
        return state.raw_state
    elif isinstance(state, ts_engine.GameState):
        return state
    raise TypeError(f"Expected NativeGameState or ts_engine.GameState, got {type(state)}")


class NativeEngine(EngineProtocol):
    """Engine adapter for the native C++ ts_engine."""

    def init_game(self, state: GameStateProtocol, seed: int = 42, /) -> None:
        if isinstance(state, NativeGameState):
            ts_engine.Engine.init_game(state.raw_state, seed)
        elif isinstance(state, ts_engine.GameState):
            ts_engine.Engine.init_game(state, seed)

    def step(self, state: GameStateProtocol, action: MicroAction, /) -> bool:
        ts_action = ts_engine.MicroAction(
            ts_engine.DecisionType(action.decision_type.value),
            action.primary_id,
            action.secondary_id,
            action.flags,
        )
        if isinstance(state, NativeGameState):
            return ts_engine.Engine.step(state.raw_state, ts_action)
        elif isinstance(state, ts_engine.GameState):
            return ts_engine.Engine.step(state, ts_action)
        return False

    def is_terminal(self, state: GameStateProtocol, /) -> bool:
        if isinstance(state, NativeGameState):
            return ts_engine.Engine.is_terminal(state.raw_state)
        elif isinstance(state, ts_engine.GameState):
            return ts_engine.Engine.is_terminal(state)
        return False

    def get_legal_placements(self, state: GameStateProtocol, player: Player, /) -> set[str]:
        raw = _get_raw(state)
        ts_p = ts_engine.Player.US if player == Player.US else ts_engine.Player.USSR
        return {
            get_country_name(cid)
            for cid in range(84)
            if ts_engine.Operations.can_place_influence(raw, ts_p, cid)
        }

    def get_legal_coups(self, state: GameStateProtocol, player: Player, /, ops: int = 1) -> set[str]:
        raw = _get_raw(state)
        legal = set()
        for cid in range(84):
            c = raw.get_country(cid)
            opp_inf = c.ussr_influence if player == Player.US else c.us_influence
            if opp_inf == 0:
                continue
            c_info = get_country_info(cid)
            reg = c_info["region"]
            if reg == "Europe" and raw.defcon <= 4:
                continue
            if reg == "Asia" and raw.defcon <= 3:
                continue
            if reg == "Middle East" and raw.defcon <= 2:
                continue
            if cid == get_country_id("Japan") and player == Player.USSR and raw.has_flag(EFFECT_FLAG_BITS["us_japan"]):
                continue
            legal.add(c_info["name"])
        return legal

    def get_legal_realignments(self, state: GameStateProtocol, player: Player, /) -> set[str]:
        raw = _get_raw(state)
        legal = set()
        for cid in range(84):
            c = raw.get_country(cid)
            opp_inf = c.ussr_influence if player == Player.US else c.us_influence
            if opp_inf == 0:
                continue
            c_info = get_country_info(cid)
            reg = c_info["region"]
            if reg == "Europe" and raw.defcon <= 4:
                continue
            if reg == "Asia" and raw.defcon <= 3:
                continue
            if reg == "Middle East" and raw.defcon <= 2:
                continue
            if cid == get_country_id("Japan") and player == Player.USSR and raw.has_flag(EFFECT_FLAG_BITS["us_japan"]):
                continue
            legal.add(c_info["name"])
        return legal

    def score_region(self, state: GameStateProtocol, region: Region, /) -> int:
        raw = _get_raw(state)
        if region == Region.SOUTHEAST_ASIA:
            clone = raw.clone()
            v0 = clone.victory_points
            ts_engine.Scoring.score_southeast_asia(clone)
            return clone.victory_points - v0
        else:
            ts_reg = ts_engine.Region(region.value)
            summary = ts_engine.Scoring.evaluate_region(raw, ts_reg)
            return summary.us_score - summary.ussr_score

    def play_event(self, state: GameStateProtocol, player: Player, card_id: int, /) -> None:
        raw = _get_raw(state)
        raw.current_phase = ts_engine.Phase.ACTION_ROUND
        ts_p = ts_engine.Player.US if player == Player.US else ts_engine.Player.USSR
        ts_engine.CardHandlers.trigger_event(raw, card_id, ts_p)

    def lower_defcon(self, state: GameStateProtocol, player: Player, target_defcon: int = 2, /) -> None:
        raw = _get_raw(state)
        raw.defcon = target_defcon
        ts_p = ts_engine.Player.US if player == Player.US else ts_engine.Player.USSR
        raw.phasing_player = ts_p
        if raw.current_phase == ts_engine.Phase.ACTION_ROUND:
            if target_defcon == 2:
                if hasattr(raw, "defcon_dropped_to_2"):
                    raw.defcon_dropped_to_2 = 1
                else:
                    setattr(raw, "defcon_dropped_to_2_in_ar", 1)
            ts_engine.StateMachine.advance_after_action_round(raw)
        elif raw.current_phase == ts_engine.Phase.HEADLINE:
            ts_engine.StateMachine.advance_headline_step(raw)

    def has_pending_decision(self, state: GameStateProtocol, /) -> bool:
        raw = _get_raw(state)
        return raw.ctx().decision_type != ts_engine.DecisionType.NONE

    def get_pending_decision_type(self, state: GameStateProtocol, /) -> str:
        raw = _get_raw(state)
        dec = raw.ctx().decision_type
        if dec == ts_engine.DecisionType.POINT_NODE:
            return "country"
        elif dec == ts_engine.DecisionType.SELECT_CARD:
            return "card"
        elif dec == ts_engine.DecisionType.ROLL_DIE:
            return "roll"
        elif dec in (ts_engine.DecisionType.SELECT_PLAY_MODE, ts_engine.DecisionType.SELECT_OP_MODE):
            return "choice"
        elif dec == ts_engine.DecisionType.NONE:
            return "none"
        return str(dec.name).lower()

    def get_decision_targets(self, state: GameStateProtocol, /) -> set[str]:
        raw = _get_raw(state)
        dec = raw.ctx().decision_type
        if dec == ts_engine.DecisionType.POINT_NODE:
            indices = ts_engine.Engine.get_legal_action_indices(raw)
            targets = {get_country_name(idx) for idx in indices}
            if not targets and raw.ctx().allow_early_stop:
                targets.add("cancel")
            return targets
        elif dec == ts_engine.DecisionType.SELECT_CARD:
            indices = ts_engine.Engine.get_legal_action_indices(raw)
            return {str(idx) for idx in indices}
        elif dec == ts_engine.DecisionType.ROLL_DIE:
            return {"1", "2", "3", "4", "5", "6"}
        elif dec in (ts_engine.DecisionType.SELECT_PLAY_MODE, ts_engine.DecisionType.SELECT_OP_MODE):
            indices = ts_engine.Engine.get_legal_action_indices(raw)
            return {str(idx) for idx in indices}
        return set()

    def resolve_decision(self, state: GameStateProtocol, target_or_choice: str | int, /) -> bool:
        raw = _get_raw(state)
        dec = raw.ctx().decision_type
        if dec == ts_engine.DecisionType.NONE:
            return False
        act = ts_engine.MicroAction()
        act.decision_type = dec
        if dec == ts_engine.DecisionType.POINT_NODE:
            if str(target_or_choice).lower() in ("cancel", "done", "pass", "refuse", "decline"):
                act.flags = 0x80
                act.primary_id = 255
            else:
                act.primary_id = get_country_id(target_or_choice) if isinstance(target_or_choice, str) else int(target_or_choice)
        elif dec == ts_engine.DecisionType.SELECT_CARD:
            if str(target_or_choice).lower() in ("refuse", "decline", "none", "cancel", "0"):
                act.primary_id = 0
            elif isinstance(target_or_choice, str) and not target_or_choice.isdigit():
                act.primary_id = get_card_id(target_or_choice)
            else:
                act.primary_id = int(target_or_choice)
        elif dec == ts_engine.DecisionType.ROLL_DIE:
            act.primary_id = int(target_or_choice)
        else:
            if str(target_or_choice).lower() in ("refuse", "decline", "none", "0"):
                act.primary_id = 0
            else:
                act.primary_id = int(target_or_choice) if str(target_or_choice).isdigit() else 0
        ts_engine.CardHandlers.handle_event_step(raw, act)
        return True



# =============================================================================
# NATIVE COMPARISON & SYNC HELPERS (FOR DIFFERENTIAL TESTS)
# =============================================================================

def get_ts_legal_country_names(ts_state: ts_engine.GameState) -> set[str]:
    """Returns the set of legal country names for influence placement in ts_state."""
    indices = ts_engine.Engine.get_legal_action_indices(ts_state)
    return {ts_engine.MapData.get_country_name(idx) for idx in indices}


def assert_board_equal(ts_state: ts_engine.GameState, external: Any) -> None:
    """Asserts board influence between ts_engine.GameState and external state."""
    # If external is GameStateProtocol
    if isinstance(external, GameStateProtocol):
        for cid in range(84):
            c_ts = ts_state.get_country(cid)
            c_ext = external.get_country(cid)
            assert (c_ts.us_influence, c_ts.ussr_influence) == (c_ext.us_influence, c_ext.ussr_influence), (
                f"{ts_engine.MapData.get_country_name(cid)}: ts=({c_ts.us_influence}/{c_ts.ussr_influence}) vs ext=({c_ext.us_influence}/{c_ext.ussr_influence})"
            )
        return

    # If external is BlockchainClient or raw state dict
    state_dict = external.get_state() if hasattr(external, "get_state") else external
    if isinstance(state_dict, dict) and "countries" in state_dict:
        from tests.differential.blockchain_adapter import country_id_to_bc
        for cid in range(84):
            c_ts = ts_state.get_country(cid)
            bc_key = country_id_to_bc(cid)
            c_bc = state_dict["countries"][bc_key]
            assert (c_ts.us_influence, c_ts.ussr_influence) == (c_bc["us"], c_bc["ussr"]), (
                f"{ts_engine.MapData.get_country_name(cid)} ({bc_key}): ts=({c_ts.us_influence}/{c_ts.ussr_influence}) vs bc=({c_bc['us']}/{c_bc['ussr']})"
            )
        return

    # If external is Struggler SEngine
    if hasattr(external, "board") and hasattr(external.board, "influence"):
        from tests.differential.struggler_adapter import country_id_to_struggler
        for cid in range(84):
            c_ts = ts_state.get_country(cid)
            s_name = country_id_to_struggler(cid)
            inf = external.board.influence[s_name]
            assert (c_ts.us_influence, c_ts.ussr_influence) == (inf["US"], inf["USSR"]), (
                f"{ts_engine.MapData.get_country_name(cid)}: ts=({c_ts.us_influence}/{c_ts.ussr_influence}) vs s=({inf['US']}/{inf['USSR']})"
            )


def assert_tracks_equal(ts_state: ts_engine.GameState, external: Any) -> None:
    """Asserts status track values between ts_engine.GameState and external state."""
    if isinstance(external, GameStateProtocol):
        assert ts_state.defcon == external.defcon
        assert ts_state.victory_points == external.victory_points
        assert ts_state.turn == external.turn
        assert ts_state.us_mil_ops == external.us_mil_ops
        assert ts_state.ussr_mil_ops == external.ussr_mil_ops
        assert ts_state.us_space_track == external.us_space_track
        assert ts_state.ussr_space_track == external.ussr_space_track
        return

    state_dict = external.get_state() if hasattr(external, "get_state") else external
    if isinstance(state_dict, dict) and "defcon" in state_dict:
        assert ts_state.defcon == state_dict["defcon"]
        assert ts_state.victory_points == state_dict["vp"]
        assert ts_state.turn == state_dict["turn"]
        assert ts_state.us_mil_ops == state_dict["milops_us"]
        assert ts_state.ussr_mil_ops == state_dict["milops_ussr"]
        assert ts_state.us_space_track == state_dict["space_race_us"]
        assert ts_state.ussr_space_track == state_dict["space_race_ussr"]
        return

    if hasattr(external, "defcon") and hasattr(external, "vp"):
        assert ts_state.defcon == external.defcon
        assert ts_state.victory_points == external.vp
        assert ts_state.turn == external.turn
        assert ts_state.us_mil_ops == external.military_ops["US"]
        assert ts_state.ussr_mil_ops == external.military_ops["USSR"]
        assert ts_state.us_space_track == external.space_race["US"]
        assert ts_state.ussr_space_track == external.space_race["USSR"]


def assert_full_state_equal(ts_state: ts_engine.GameState, external: Any) -> None:
    assert_tracks_equal(ts_state, external)
    assert_board_equal(ts_state, external)


def create_paired_state(
    seed: int = 42,
    defcon: int = 5,
    turn: int = 1,
    vp: int = 0,
    bc_engine: Any = None,
) -> tuple[ts_engine.GameState, Any]:
    ts_state = ts_engine.GameState()
    ts_engine.Engine.init_game(ts_state, seed)
    ts_state.defcon = defcon
    ts_state.turn = turn
    ts_state.victory_points = vp
    ts_state.current_phase = ts_engine.Phase.ACTION_ROUND

    if bc_engine is not None:
        bc_engine.init_game(defcon=defcon, turn=turn, vp=vp)
        return ts_state, bc_engine

    from tests.differential.struggler_adapter import StrugglerGameState
    st = StrugglerGameState(seed=seed)
    st.defcon = defcon
    st.turn = turn
    st.victory_points = vp
    return ts_state, st
