"""Automated Full-Game Invariant & Rule-Compliance Verification Suite."""

import random
from typing import Dict, List, Any, Optional, Tuple
import pytest
import ts_engine
from bot import ExploratoryBot


def validate_legal_choices(state_dict: Dict[str, Any]) -> List[str]:
    """Exhaustively validates that the engine's legal_actions and decision_context
    conform strictly to Twilight Struggle rules for the given state.
    Returns a list of violation strings (empty if 100% compliant).
    """
    violations: List[str] = []
    legal = state_dict.get("legal_actions", {})
    ctx = state_dict.get("decision_context", {})
    d_type = legal.get("decision_type", 0)
    d_player = legal.get("decision_player", "NONE")
    valid_ids = legal.get("valid_ids", [])
    countries = state_dict.get("countries", {})
    defcon = state_dict.get("defcon", 5)
    phase = state_dict.get("phase", state_dict.get("current_phase", 0))

    if d_player not in ("US", "USSR"):
        violations.append(f"Invalid decision player: {d_player}")

    # 1. SELECT_CARD
    if d_type == 1:
        res_card = ctx.get("resolving_card", 0)
        hand = state_dict.get("hands", {}).get(f"{d_player}_cards", [])
        hand_ids = [c["id"] for c in hand] if hand and isinstance(hand[0], dict) else [c for c in hand]
        china = state_dict.get("china_card", {})
        discard = state_dict.get("discard_pile", [])

        opp_player = "USSR" if d_player == "US" else "US"
        opp_hand = state_dict.get("hands", {}).get(f"{opp_player}_cards", [])
        opp_hand_ids = [c["id"] for c in opp_hand] if opp_hand and isinstance(opp_hand[0], dict) else [c for c in opp_hand]

        if res_card in (43, 85):  # SALT_NEGOTIATIONS, STAR_WARS (pick from discard)
            for cid in valid_ids:
                if cid > 0 and cid not in discard:
                    violations.append(f"Card #{cid} proposed from discard for card #{res_card} ({ctx.get('resolving_card_name')}) but not in discard {discard}")
        elif res_card == 108:  # OUR_MAN_IN_TEHRAN
            pass
        elif res_card == 98:  # ALDRICH_AMES (USSR chooses from US hand)
            for cid in valid_ids:
                if cid > 0 and cid not in opp_hand_ids:
                    violations.append(f"Card #{cid} proposed to {d_player} for Aldrich Ames (#{res_card}) but not in opponent hand {opp_hand_ids}")
        elif res_card == 49:  # MISSILE_ENVY
            for cid in valid_ids:
                if cid > 0 and cid not in opp_hand_ids and cid not in hand_ids:
                    violations.append(f"Card #{cid} proposed to {d_player} for Missile Envy (#{res_card}) but not in hand {hand_ids} or opp {opp_hand_ids}")
        elif res_card == 10:  # BLOCKADE
            for cid in valid_ids:
                if cid > 0 and cid not in hand_ids:
                    violations.append(f"Blockade proposed card #{cid} but not in hand {hand_ids}")
        elif res_card == 95:  # LATIN_AMERICAN_DEBT_CRISIS
            for cid in valid_ids:
                if cid > 0 and cid not in hand_ids:
                    violations.append(f"Latin American Debt Crisis proposed card #{cid} but not in hand {hand_ids}")
        elif res_card == 77:  # ASK_NOT_WHAT_YOUR_COUNTRY_CAN_DO_FOR_YOU
            for cid in valid_ids:
                if cid > 0 and cid not in hand_ids:
                    violations.append(f"Ask Not proposed card #{cid} but not in hand {hand_ids}")
        elif res_card == 250:  # Space Walk discard
            for cid in valid_ids:
                if cid > 0 and cid not in hand_ids:
                    violations.append(f"Space Walk discard proposed card #{cid} but not in hand {hand_ids}")
        else:
            for cid in valid_ids:
                if cid == 0:
                    pass
                elif cid == 6:  # China Card
                    if china.get("holder") != d_player or not china.get("playable", False):
                        violations.append(f"The China Card proposed to {d_player} (res_card={res_card}) but holder={china.get('holder')}, playable={china.get('playable')}")
                else:
                    if phase != 0 and cid not in hand_ids and ctx.get("pending_op_card") != cid:
                        violations.append(f"Card #{cid} proposed to {d_player} (res_card={res_card}) but not in hand {hand_ids}")

    # 2. SELECT_PLAY_MODE
    elif d_type == 2:
        card = ctx.get("pending_op_card", 0)
        c_info = ts_engine.CardData.get_card_info(card) if 1 <= card <= 110 else {}
        is_scoring = c_info.get("is_scoring", False)
        if is_scoring:
            if 1 in valid_ids or 2 in valid_ids:
                violations.append(f"Scoring card #{card} allowed for Ops (1) or Space (2): valid_ids={valid_ids}")

    # 3. CHOOSE_TIMING_BRANCH
    elif d_type == 3:
        card = ctx.get("pending_op_card", 0)
        c_info = ts_engine.CardData.get_card_info(card) if 1 <= card <= 110 else {}
        opp = "USSR" if d_player == "US" else "US"
        if c_info.get("side") != opp:
            violations.append(f"CHOOSE_TIMING_BRANCH triggered for non-opponent card #{card} (side={c_info.get('side')})")

    # 4. POINT_NODE
    elif d_type == 5:
        op_mode = ctx.get("op_mode", 0)
        res_card = ctx.get("resolving_card", 0)

        if res_card > 0:
            if res_card == 16:  # WARSAW_PACT
                for nid in valid_ids:
                    if nid < 84:
                        c_info = ts_engine.MapData.get_country_info(nid)
                        if not c_info.get("in_eastern_europe"):
                            violations.append(f"Warsaw Pact allowed non-Eastern Europe country ID {nid}")

            elif res_card == 22:  # INDEPENDENT_REDS
                allowed_reds = {16, 17, 18, 19, 20}  # Czech, Hungary, Yugoslavia, Romania, Bulgaria
                for nid in valid_ids:
                    if nid not in allowed_reds and nid != 255 and nid != 0:
                        violations.append(f"Independent Reds allowed invalid country ID {nid}")

            elif res_card == 28:  # SUEZ_CRISIS
                allowed_suez = {1, 8, 23}  # UK, France, Israel
                for nid in valid_ids:
                    if nid not in allowed_suez and nid != 255 and nid != 0:
                        violations.append(f"Suez Crisis allowed invalid country ID {nid}")

            elif res_card == 29:  # EAST_EUROPEAN_UNREST
                for nid in valid_ids:
                    if nid < 84:
                        c_info = ts_engine.MapData.get_country_info(nid)
                        if not c_info.get("in_eastern_europe"):
                            violations.append(f"East European Unrest allowed non-Eastern Europe country ID {nid}")

            elif res_card == 30:  # DECOLONIZATION
                for nid in valid_ids:
                    if nid < 84:
                        c_info = ts_engine.MapData.get_country_info(nid)
                        if c_info.get("region") not in (3, "Africa") and not c_info.get("in_southeast_asia"):
                            violations.append(f"Decolonization allowed non-Africa/non-SE Asia country ID {nid}")

        elif phase == 0:  # SETUP
            pending_ops = ctx.get("pending_ops_value", 0)
            for nid in valid_ids:
                if nid < 84:
                    c_info = ts_engine.MapData.get_country_info(nid)
                    if d_player == "USSR" and not c_info.get("in_eastern_europe"):
                        violations.append(f"USSR Setup allowed non-Eastern Europe country ID {nid}")
                    elif d_player == "US":
                        # Stage 0: 7 in Western Europe. Stage 1: Bonus presence in any country with existing US influence.
                        if pending_ops == 0 and not c_info.get("in_western_europe"):
                            violations.append(f"US Setup Stage 0 allowed non-Western Europe country ID {nid}")

        else:
            # Action Round Ops
            if op_mode == 1:  # COUP
                # Tear Down This Wall (#96) grants the US free Coup or Realignment attempts
                # in Europe regardless of DEFCON (ops.cpp), so the Rule 8.1.5 regional
                # restriction does not apply while it is the card being resolved.
                tdtw = (ctx.get("pending_op_card") == 96 or res_card == 96)
                for nid in valid_ids:
                    if nid < 84:
                        c_info = ts_engine.MapData.get_country_info(nid)
                        r = c_info.get("region")
                        if defcon <= 4 and (r == 0 or r == "Europe") and not tdtw:
                            violations.append(f"Coup allowed in Europe at DEFCON {defcon}: ID {nid}")
                        if defcon <= 3 and (r == 1 or r == "Asia"):
                            violations.append(f"Coup allowed in Asia at DEFCON {defcon}: ID {nid}")
                        if defcon <= 2 and (r == 2 or r == "Middle East"):
                            violations.append(f"Coup allowed in Middle East at DEFCON {defcon}: ID {nid}")

            elif op_mode == 2:  # REALIGN
                opp = "USSR" if d_player == "US" else "US"
                for nid in valid_ids:
                    if nid < 84:
                        c_name = ts_engine.MapData.get_country_name(nid)
                        c_data = countries.get(c_name, {}) if isinstance(countries, dict) else {}
                        opp_inf = c_data.get("ussr_influence" if opp == "USSR" else "us_influence", 0)
                        if opp_inf == 0:
                            violations.append(f"Realignment allowed in {c_name} with 0 opponent influence")

    return violations


def run_invariant_audit_game(seed: int) -> Tuple[int, List[str]]:
    """Runs a single game using ExploratoryBots and asserts zero invariant violations."""
    state = ts_engine.GameState()
    ts_engine.Engine.init_game(state, seed)
    bot_us = ExploratoryBot("US", rng_seed=seed * 2 + 1)
    bot_ussr = ExploratoryBot("USSR", rng_seed=seed * 2 + 2)

    step_count = 0
    all_violations: List[str] = []

    while not ts_engine.Engine.is_terminal(state) and step_count < 1500:
        if state.ctx().decision_player == ts_engine.Player.NONE and state.ctx().decision_type == ts_engine.DecisionType.ROLL_DIE:
            ts_engine.Engine.step(state, ts_engine.MicroAction(ts_engine.DecisionType.ROLL_DIE, 0, 0, 0))
            continue
        d = state.to_dict()
        legal = d.get("legal_actions", {})
        d_player = "US" if state.ctx().decision_player == ts_engine.Player.US else "USSR"

        violations = validate_legal_choices(d)
        if violations:
            for v in violations:
                all_violations.append(f"[Seed {seed} | Step {step_count} | Turn {state.turn} | {d_player}]: {v}")

        bot = bot_us if d_player == "US" else bot_ussr
        action_dict = bot.select_action(d, legal)
        if action_dict is None:
            break

        action = ts_engine.MicroAction(
            ts_engine.DecisionType(int(action_dict["decision_type"])),
            int(action_dict["primary_id"]),
            int(action_dict["secondary_id"]),
            int(action_dict["flags"]),
        )

        success = ts_engine.Engine.step(state, action)
        if not success:
            all_violations.append(f"[Seed {seed} | Step {step_count}]: Engine rejected legal action {action_dict}")
            break

        step_count += 1

    return step_count, all_violations


@pytest.mark.parametrize("seed", [42, 101, 777, 2026, 9999])
def test_full_game_invariant_audit(seed: int):
    """Executes a full exploratory game asserting zero legal-action or state machine violations."""
    steps, violations = run_invariant_audit_game(seed)
    assert len(violations) == 0, f"Encountered {len(violations)} invariant violations:\n" + "\n".join(violations[:15])
    assert steps > 30, f"Game ended prematurely at step {steps}"


def test_canonical_game_ending_reasons():
    """Validates that classify_game_ending_reason cleanly distinguishes 20 VP from DEFCON 1 and outputs only canonical categories."""
    from tools.lib.tournament_evaluator import classify_game_ending_reason

    st = ts_engine.GameState()
    ts_engine.Engine.init_game(st, 42)

    # 1. DEFCON 1 (own decision) - e.g. unprovoked suicide
    st.defcon = 1
    st.victory_points = 20  # Engine sets VP to +/-20 on DEFCON 1
    assert classify_game_ending_reason(st) == "DEFCON 1 (own decision)"

    # 2. DEFCON 1 (opponent decision) - provoked event trap
    st.defcon = 1
    st.victory_points = -20
    st.set_flag(ts_engine.EffectBits.DEFCON_SUICIDE_PROVOKED)
    assert classify_game_ending_reason(st) == "DEFCON 1 (opponent decision)"

    # 3. 20 VP - Milestone VP swing or Europe Control at DEFCON > 1
    st.clear_flag(ts_engine.EffectBits.DEFCON_SUICIDE_PROVOKED)
    st.defcon = 2
    st.victory_points = 20
    assert classify_game_ending_reason(st) == "20 VP"

    st.victory_points = -20
    assert classify_game_ending_reason(st) == "20 VP"

    # 4. Final Scoring -- which the engine reaches holding turn *11*, not 10. finish_end_turn
    # increments the turn and only then tests `turn <= 10` before calling execute_final_scoring,
    # so a game that goes the distance terminates at turn 11 / AR 0. Driving 40 heuristic games
    # produces exactly that and never a turn-10 final scoring. This case previously asserted
    # turn 10, which is a position final scoring cannot occupy -- see case 5.
    st.defcon = 2
    st.victory_points = 6
    st.turn = 11
    st.current_phase = ts_engine.Phase.GAME_OVER
    assert classify_game_ending_reason(st) == "final scoring"

    # 5. Wargames (#100) - the game is over, short of final scoring, with abs(VP) < 20.
    st.turn = 8
    st.victory_points = 4
    st.current_phase = ts_engine.Phase.GAME_OVER
    assert classify_game_ending_reason(st) == "wargames"

    # 5b. Wargames played *in* turn 10 -- the case the `turn < 10` bound used to miss, sending
    # it to rule 4 and reporting it as final scoring. 3 of the 119 finished games in the human
    # corpus end this way (e.g. replay 163, turn 10 AR1).
    st.turn = 10
    st.victory_points = 1
    st.current_phase = ts_engine.Phase.GAME_OVER
    assert classify_game_ending_reason(st) == "wargames"
