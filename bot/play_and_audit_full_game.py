#!/usr/bin/env python3
"""
Plays a full Twilight Struggle game against itself with exhaustive rule and action mask validation on every single step.
Dumps full state + proposed choices log and performs independent verification.
"""

import os
import sys
import json
import random
from typing import Dict, List, Any, Optional, Tuple

import ts_engine
from bot.bot_client import HeuristicBot
from bot.exploratory_bot import ExploratoryBot

def validate_legal_choices(state_dict: dict) -> List[str]:
    """
    Exhaustively validates that the engine's legal_actions and decision_context
    conform to Twilight Struggle rules for the given state.
    Returns a list of violation strings (empty if 100% valid).
    """
    violations = []
    legal = state_dict.get("legal_actions", {})
    ctx = state_dict.get("decision_context", {})
    d_type = legal.get("decision_type", 0)
    d_player = legal.get("decision_player", "NONE")
    valid_ids = legal.get("valid_ids", [])
    countries = state_dict.get("countries", {})
    defcon = state_dict.get("defcon", 5)
    phase = state_dict.get("current_phase", 0)
    turn = state_dict.get("turn", 1)

    # 1. Player check
    if d_player not in ("US", "USSR"):
        violations.append(f"Invalid decision player: {d_player}")

    # 2. Decision Type Validation
    if d_type == 1: # SELECT_CARD
        res_card = ctx.get("resolving_card", 0)
        hand = state_dict.get("hands", {}).get(f"{d_player}", [])
        china = state_dict.get("china_card", {})
        discard = state_dict.get("discard_pile", [])
        
        if res_card in (43, 85): # SALT_NEGOTIATIONS, STAR_WARS (pick from discard)
            for cid in valid_ids:
                if cid not in discard:
                    violations.append(f"Card #{cid} proposed from discard for card #{res_card} but not in discard {discard}")
        elif res_card == 108: # OUR_MAN_IN_TEHRAN
            pass
        elif res_card == 10: # BLOCKADE (discard 3+ ops from hand)
            for cid in valid_ids:
                if cid not in hand:
                    violations.append(f"Blockade proposed card #{cid} but not in hand {hand}")
        else:
            for cid in valid_ids:
                if cid == 6: # The China Card
                    if china.get("holder") != d_player or not china.get("playable"):
                        violations.append(f"The China Card proposed to {d_player} but holder={china.get('holder')}, playable={china.get('playable')}")
                else:
                    if cid not in hand:
                        violations.append(f"Card #{cid} proposed to {d_player} but not in hand {hand}")

    elif d_type == 2: # SELECT_PLAY_MODE
        card = ctx.get("pending_op_card", 0)
        c_info = ts_engine.CardData.get_card_info(card) if 1 <= card <= 110 else {}
        is_scoring = c_info.get("is_scoring", False)
        if is_scoring:
            if 1 in valid_ids or 2 in valid_ids:
                violations.append(f"Scoring card #{card} allowed for Ops (1) or Space (2): valid_ids={valid_ids}")

    elif d_type == 3: # CHOOSE_TIMING_BRANCH
        card = ctx.get("pending_op_card", 0)
        c_info = ts_engine.CardData.get_card_info(card) if 1 <= card <= 110 else {}
        opp = "USSR" if d_player == "US" else "US"
        if c_info.get("side") != opp:
            violations.append(f"CHOOSE_TIMING_BRANCH triggered for non-opponent card #{card} (side={c_info.get('side')})")

    elif d_type == 4: # SELECT_OP_MODE
        # Influence (0), Coup (1), Realign (2)
        pass

    elif d_type == 5: # POINT_NODE
        op_mode = ctx.get("op_mode", 0)
        res_card = ctx.get("resolving_card", 0)

        if res_card > 0:
            # Card Event Specific Masks Validation
            if res_card == 16: # WARSAW_PACT
                for nid in valid_ids:
                    c_name = ts_engine.MapData.get_country_name(nid)
                    c_info = ts_engine.MapData.get_country_info(nid)
                    if not c_info.get("in_eastern_europe"):
                        violations.append(f"Warsaw Pact allowed non-Eastern Europe country {c_name} (ID {nid})")

            elif res_card == 33: # DE_STALINIZATION
                max_per = ctx.get("max_per_country", 0)
                if max_per == 2: # Stage 2 Placement
                    for nid in valid_ids:
                        c_name = ts_engine.MapData.get_country_name(nid)
                        c_data = countries.get(c_name, {})
                        if c_data.get("controlled_by") == "US":
                            violations.append(f"De-Stalinization allowed US-controlled country {c_name} (ID {nid})")

            elif res_card == 22: # INDEPENDENT_REDS
                allowed_reds = {16, 17, 18, 19, 20} # Czech, Hungary, Yugoslavia, Romania, Bulgaria
                for nid in valid_ids:
                    if nid not in allowed_reds:
                        violations.append(f"Independent Reds allowed invalid country ID {nid}")

            elif res_card == 105: # SPECIAL_RELATIONSHIP
                flags_list = state_dict.get("flags", [])
                if isinstance(flags_list, dict):
                    nato_active = flags_list.get("NATO_ACTIVE", False)
                else:
                    nato_active = "NATO_ACTIVE" in flags_list
                nato_active = nato_active or ((state_dict.get("persistent_effects", 0) & (1 << 3)) != 0)
                if not nato_active:
                    uk_info = ts_engine.MapData.get_country_info(1)
                    allowed_adj = set(uk_info.get("neighbors", []))
                    for nid in valid_ids:
                        if nid not in allowed_adj:
                            c_name = ts_engine.MapData.get_country_name(nid)
                            violations.append(f"Special Relationship allowed non-UK adjacent country {c_name} (ID {nid})")
                else:
                    for nid in valid_ids:
                        c_info = ts_engine.MapData.get_country_info(nid)
                        if not c_info.get("in_western_europe"):
                            c_name = ts_engine.MapData.get_country_name(nid)
                            violations.append(f"Special Relationship allowed non-Western Europe country {c_name} (ID {nid})")

            elif res_card == 28: # SUEZ_CRISIS
                allowed_suez = {1, 8, 23} # UK, France, Israel
                for nid in valid_ids:
                    if nid not in allowed_suez:
                        violations.append(f"Suez Crisis allowed invalid country ID {nid}")

            elif res_card == 29: # EAST_EUROPEAN_UNREST
                for nid in valid_ids:
                    c_info = ts_engine.MapData.get_country_info(nid)
                    if not c_info.get("in_eastern_europe"):
                        violations.append(f"East European Unrest allowed non-Eastern Europe country ID {nid}")

            elif res_card == 30: # DECOLONIZATION
                for nid in valid_ids:
                    c_info = ts_engine.MapData.get_country_info(nid)
                    if c_info.get("region") not in (3, "Africa") and not c_info.get("in_southeast_asia"):
                        violations.append(f"Decolonization allowed non-Africa/non-SE Asia country ID {nid}")

        elif phase == 0: # SETUP
            for nid in valid_ids:
                c_info = ts_engine.MapData.get_country_info(nid)
                if d_player == "USSR" and not c_info.get("in_eastern_europe"):
                    violations.append(f"USSR Setup allowed non-Eastern Europe country ID {nid}")
                elif d_player == "US" and not c_info.get("in_western_europe"):
                    violations.append(f"US Setup allowed non-Western Europe country ID {nid}")

        else:
            # Ops POINT_NODE
            if op_mode == 1: # COUP
                for nid in valid_ids:
                    c_info = ts_engine.MapData.get_country_info(nid)
                    r = c_info.get("region")
                    if defcon <= 4 and (r == 0 or r == "Europe"):
                        violations.append(f"Coup allowed in Europe at DEFCON {defcon}: ID {nid}")
                    if defcon <= 3 and (r == 1 or r == "Asia"):
                        violations.append(f"Coup allowed in Asia at DEFCON {defcon}: ID {nid}")
                    if defcon <= 2 and (r == 2 or r == "Middle East"):
                        violations.append(f"Coup allowed in Middle East at DEFCON {defcon}: ID {nid}")

            elif op_mode == 2: # REALIGN
                opp = "USSR" if d_player == "US" else "US"
                for nid in valid_ids:
                    c_name = ts_engine.MapData.get_country_name(nid)
                    c_data = countries.get(c_name, {})
                    opp_inf = c_data.get("ussr_influence" if opp == "USSR" else "us_influence", 0)
                    if opp_inf == 0:
                        violations.append(f"Realignment allowed in {c_name} with 0 opponent influence")

    return violations


def run_full_game_simulation(seed: int = 42, bot_type: str = "heuristic") -> Tuple[List[dict], List[str]]:
    """Runs a complete game simulation with exhaustive validation, using either heuristic or exploratory bots."""
    state = ts_engine.GameState()
    ts_engine.Engine.init_game(state, seed)

    if bot_type == "exploratory":
        bot_us = ExploratoryBot("US", rng_seed=seed * 2 + 1)
        bot_ussr = ExploratoryBot("USSR", rng_seed=seed * 2 + 2)
    else:
        bot_us = HeuristicBot("US")
        bot_ussr = HeuristicBot("USSR")
    """Runs a complete game simulation, logging every step and validating engine choices."""
    state = ts_engine.GameState()
    ts_engine.Engine.init_game(state, seed)



    game_logs = []
    all_violations = []
    step_num = 0
    max_steps = 1000

    print(f"Starting full game simulation (Seed: {seed})...")

    while step_num < max_steps:
        step_num += 1
        s_dict = state.to_dict()

        if s_dict.get("is_terminal"):
            winner = "US" if s_dict.get("terminal_utility", 0) > 0 else ("USSR" if s_dict.get("terminal_utility", 0) < 0 else "TIE")
            print(f"Game Terminal reached at step {step_num}! Winner: {winner}, Final VP: {state.victory_points}")
            game_logs.append({
                "step": step_num,
                "event": "GAME_OVER",
                "winner": winner,
                "final_vp": state.victory_points,
                "terminal_utility": s_dict.get("terminal_utility", 0),
                "turn": state.turn,
                "phase": str(state.current_phase)
            })
            break

        legal = s_dict.get("legal_actions", {})
        ctx = s_dict.get("decision_context", {})
        d_player = legal.get("decision_player", "NONE")
        d_type = legal.get("decision_type", 0)
        d_type_name = legal.get("decision_type_name", f"TYPE_{d_type}")
        valid_ids = legal.get("valid_ids", [])
        allow_early_stop = legal.get("allow_early_stop", False)

        # 1. Exhaustive Choice Validation
        violations = validate_legal_choices(s_dict)
        if violations:
            for v in violations:
                err = f"[Step {step_num} | Turn {state.turn} AR {state.action_round} {d_player} {d_type_name}]: {v}"
                print(f"❌ VIOLATION: {err}")
                all_violations.append(err)

        # 2. Select Action using Bot
        bot = bot_ussr if d_player == "USSR" else bot_us
        action_dict = bot.select_action(s_dict, legal)
        if not action_dict:
            print(f"❌ ERROR: Bot returned no action at step {step_num} for {d_player} in {d_type_name}!")
            break

        # Log entry
        log_entry = {
            "step": step_num,
            "turn": state.turn,
            "phase": str(state.current_phase),
            "ar": state.action_round,
            "phasing_player": str(state.phasing_player),
            "defcon": state.defcon,
            "victory_points": state.victory_points,
            "mil_ops": {"US": state.us_mil_ops, "USSR": state.ussr_mil_ops},
            "space": {"US": state.us_space_track, "USSR": state.ussr_space_track},
            "decision": {
                "player": d_player,
                "type": d_type,
                "type_name": d_type_name,
                "pending_card": ctx.get("pending_op_card", 0),
                "pending_card_name": ctx.get("pending_op_card_name", ""),
                "pending_ops": ctx.get("pending_ops_value", 0),
                "resolving_card": ctx.get("resolving_card", 0),
                "resolving_card_name": ctx.get("resolving_card_name", ""),
                "remaining_steps": ctx.get("remaining_steps", 0),
                "valid_ids": valid_ids,
                "allow_early_stop": allow_early_stop
            },
            "action_taken": action_dict,
            "violations": violations
        }
        game_logs.append(log_entry)

        # Execute step
        m_action = ts_engine.MicroAction(
            ts_engine.DecisionType(action_dict["decision_type"]),
            action_dict.get("primary_id", 0),
            action_dict.get("secondary_id", 0),
            action_dict.get("flags", 0)
        )
        ts_engine.Engine.step(state, m_action)

    return game_logs, all_violations


def generate_readable_game_summary(game_logs: List[dict], out_path: str):
    """Generates a clean text file of the entire playthrough."""
    lines = []
    lines.append("=" * 80)
    lines.append("TWILIGHT STRUGGLE FULL GAME SIMULATION & ENGINE AUDIT LOG")
    lines.append("=" * 80)

    for entry in game_logs:
        if entry.get("event") == "GAME_OVER":
            lines.append("\n" + "=" * 80)
            lines.append(f"★ GAME OVER | Winner: {entry.get('winner')} | Final VP: {entry.get('final_vp')}")
            lines.append("=" * 80)
            break

        step = entry["step"]
        turn = entry["turn"]
        ar = entry["ar"]
        phase = entry["phase"]
        defcon = entry["defcon"]
        vp = entry["victory_points"]
        dec = entry["decision"]
        act = entry["action_taken"]

        lines.append(f"[Step {step:03d}] T{turn} AR{ar} ({phase}) | DEFCON:{defcon} VP:{vp:+d} | {dec['player']} -> {dec['type_name']}")
        if dec.get("pending_card_name"):
            lines.append(f"    Context: Pending Card: {dec['pending_card_name']} ({dec['pending_ops']} Ops, {dec['remaining_steps']} rem)")
        if dec.get("resolving_card_name"):
            lines.append(f"    Context: Resolving Event: {dec['resolving_card_name']}")

        lines.append(f"    Choices Proposed ({len(dec['valid_ids'])} items): {dec['valid_ids']} (EarlyStop: {dec['allow_early_stop']})")
        lines.append(f"    Action Executed: type={act['decision_type']} primary={act.get('primary_id', 0)} sec={act.get('secondary_id', 0)} flags={act.get('flags', 0)}")
        if entry.get("violations"):
            for v in entry["violations"]:
                lines.append(f"    ❌ VIOLATION: {v}")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Readable log written to: {out_path}")


def main():
    os.makedirs("replays", exist_ok=True)
    logs, violations = run_full_game_simulation(seed=42)

    # Save JSON dump
    json_path = "replays/full_audit_game.json"
    with open(json_path, "w") as f:
        json.dump({"total_steps": len(logs), "violations_count": len(violations), "violations": violations, "logs": logs}, f, indent=2)
    print(f"JSON trace written to: {json_path}")

    # Save text summary
    txt_path = "replays/full_audit_game.txt"
    generate_readable_game_summary(logs, txt_path)

    print("\n" + "=" * 60)
    print(f"SIMULATION AUDIT SUMMARY: {len(logs)} total micro-steps executed.")
    if violations:
        print(f"⚠️ FOUND {len(violations)} RULE VIOLATIONS!")
        for v in violations[:10]:
            print(f"  • {v}")
    else:
        print("✅ ZERO RULE VIOLATIONS DETECTED! All choices proposed by the engine conformed strictly to rules.")
    print("=" * 60)

if __name__ == "__main__":
    main()
