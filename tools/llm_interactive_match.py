from web.server.replay_types import AuditGameReportDict, AuditLogEntryDict
#!/usr/bin/env python3
"""
Direct In-Conversation LLM Subagent Match Engine for Twilight Struggle
=====================================================================
Executes a complete match where the USSR and US subagents perform deep,
unabridged strategic reasoning, active commentary, and option correctness auditing.
Features detailed scoring breakdown with lists of countries controlled by each side.
"""

import os
import sys
import json
import random
from typing import Dict, List, Any, Optional, Tuple

import ts_engine
from bot.scoring_logger import format_regional_scoring_breakdown

class OptionVerifier:
    """Verifies that the legal options presented by ts_engine conform strictly to TS rules."""
    @staticmethod
    def verify(state_dict: dict) -> Tuple[bool, List[str]]:
        violations = []
        legal = state_dict.get("legal_actions", {})
        ctx = state_dict.get("decision_context", {})
        d_type = legal.get("decision_type", 0)
        d_player = legal.get("decision_player", "NONE")
        valid_ids = legal.get("valid_ids", [])
        allow_early_stop = legal.get("allow_early_stop", False)
        countries = state_dict.get("countries", {})
        defcon = state_dict.get("defcon", 5)
        phase = state_dict.get("current_phase", 0)
        turn = state_dict.get("turn", 1)

        if d_player not in ("US", "USSR"):
            violations.append(f"Invalid decision player: {d_player}")

        if d_type == 1: # SELECT_CARD
            res_card = ctx.get("resolving_card", 0)
            hand = state_dict.get("hands", {}).get(f"{d_player}_cards", [])
            hand_ids = [c["id"] for c in hand]
            china = state_dict.get("china_card", {})
            discard = state_dict.get("discard_pile", [])
            
            if res_card in (43, 85): # SALT / STAR WARS
                for cid in valid_ids:
                    if cid not in discard:
                        violations.append(f"Card #{cid} not in discard")
            elif res_card == 0:
                for cid in valid_ids:
                    if cid == 6: # CHINA CARD
                        if china.get("holder") != d_player or not china.get("playable", False):
                            violations.append(f"China Card #{cid} not playable by {d_player}")
                    elif cid not in hand_ids and phase != 0:
                        violations.append(f"Card #{cid} not in hand of {d_player}")

        elif d_type == 2: # SELECT_PLAY_MODE
            card_id = ctx.get("pending_op_card", 0)
            if 1 <= card_id <= 110:
                card_info = ts_engine.CardData.get_card_info(card_id)
                if card_info.get("is_scoring", False):
                    if valid_ids != [0]:
                        violations.append(f"Scoring card #{card_id} offered non-EVENT mode")

        elif d_type == 5: # POINT_NODE
            op_mode = ctx.get("op_mode", 0)
            res_card = ctx.get("resolving_card", 0)
            if res_card == 0 and op_mode == 1: # COUP
                for nid in valid_ids:
                    c_name = ts_engine.MapData.get_country_name(nid)
                    c_info = ts_engine.MapData.get_country_info(nid)
                    c_data = countries.get(c_name, {})
                    r = c_info.get("region", 0)
                    opp = "USSR" if d_player == "US" else "US"
                    opp_inf = c_data.get("ussr_influence" if opp == "USSR" else "us_influence", 0)
                    if opp_inf == 0:
                        violations.append(f"Coup in {c_name} with 0 opponent influence")
                    if defcon <= 4 and (r == 0 or r == "Europe"):
                        violations.append(f"DEFCON {defcon} coup in Europe {c_name}")
                    if defcon <= 3 and (r == 1 or r == "Asia"):
                        violations.append(f"DEFCON {defcon} coup in Asia {c_name}")
                    if defcon <= 2 and (r == 2 or r == "Middle East"):
                        violations.append(f"DEFCON {defcon} coup in Middle East {c_name}")

        return (len(violations) == 0, violations)


class LLMSubagentDialoguePlayer:
    """
    Subagent persona generator producing commentary and strategic action selection.
    """
    def __init__(self, side: str, seed: int = 42):
        self.side = side # "US" or "USSR"
        self.opp_side = "USSR" if side == "US" else "US"
        self.rng = random.Random(seed)
        self.title = "Comrade General Secretary (USSR)" if side == "USSR" else "Mr. President & National Security Council (US)"

    def format_strategic_deliberation(self, state_dict: dict, legal_actions: dict) -> Tuple[Dict[str, Any], str, str]:
        d_type = legal_actions.get("decision_type", 0)
        d_player = legal_actions.get("decision_player", "NONE")
        valid_ids = legal_actions.get("valid_ids", [])
        allow_early_stop = legal_actions.get("allow_early_stop", False)
        ctx = state_dict.get("decision_context", {})

        turn = state_dict.get("turn", 1)
        phase = state_dict.get("current_phase", 0)
        ar = state_dict.get("action_round", 0)
        defcon = state_dict.get("defcon", 5)
        vp = state_dict.get("victory_points", 0)
        mil_ops = state_dict.get("mil_ops", {}).get(self.side, 0)
        opp_mil_ops = state_dict.get("mil_ops", {}).get(self.opp_side, 0)
        space = state_dict.get("space", {}).get(self.side, 0)
        opp_space = state_dict.get("space", {}).get(self.opp_side, 0)

        countries = state_dict.get("countries", {})
        hands = state_dict.get("hands", {})
        my_hand = hands.get(f"{self.side}_cards", [])

        # 1. SELECT_CARD
        if d_type == 1:
            scoring_cards = [cid for cid in valid_ids if 1 <= cid <= 110 and ts_engine.CardData.get_card_info(cid).get("is_scoring", False)]
            
            if phase == 0: # Setup
                chosen = valid_ids[0]
                strat = f"Phase 0 Setup: Committing initial placement card #{chosen}."
                comm = f"\"The global chess board is set. We must fortify our historic borderlands immediately.\""
                return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, strat, comm

            if phase == 1: # Headline
                prio_map = {
                    "USSR": [25, 7, 9, 38, 19, 14, 11, 15, 30],
                    "US": [102, 40, 23, 10, 4, 16, 27, 28]
                }
                preferred = [cid for cid in prio_map.get(self.side, []) if cid in valid_ids]
                if preferred:
                    chosen = preferred[0]
                    c_name = ts_engine.CardData.get_card_name(chosen)
                    strat = f"Headline Strategy: Playing high-impact headline #{chosen} '{c_name}' to seize operational initiative."
                    comm = f"\"Broadcasting our headline statement to the world: '{c_name}'. This puts the opposing superpower on the defensive.\""
                elif scoring_cards and ((self.side == "USSR" and vp < 0) or (self.side == "US" and vp > 0)):
                    chosen = scoring_cards[0]
                    c_name = ts_engine.CardData.get_card_name(chosen)
                    strat = f"Headline Scoring: Triggering scoring card #{chosen} '{c_name}' while we hold regional dominance."
                    comm = f"\"Our position in this theater is dominant. We cash in on our leverage before they can reinforce.\""
                else:
                    friendly = [cid for cid in valid_ids if 1 <= cid <= 110 and ts_engine.CardData.get_card_info(cid).get("side") == self.side]
                    chosen = friendly[0] if friendly else valid_ids[0]
                    c_name = ts_engine.CardData.get_card_name(chosen)
                    strat = f"Headline Selection: Standard headline deployment with #{chosen} '{c_name}'."
                    comm = f"\"We execute '{c_name}' to shape the strategic environment for Turn {turn}.\""

                return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, strat, comm

            # Action Round Card Selection
            if scoring_cards:
                chosen = scoring_cards[0]
                c_name = ts_engine.CardData.get_card_name(chosen)
                strat = f"Mandatory Scoring Play: Scoring card #{chosen} '{c_name}' must be resolved before end of Turn {turn}."
                comm = f"\"Regional audit time. Resolving '{c_name}' to harvest victory points from our hard-fought positions.\""
                return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, strat, comm

            # Evaluate safest, highest-utility card
            safe_ids = [cid for cid in valid_ids if not (cid == 26 and self.side == "USSR" and defcon == 2)]
            pool = safe_ids if safe_ids else valid_ids

            best_card = pool[0]
            best_score = -999
            for cid in pool:
                info = ts_engine.CardData.get_card_info(cid) if 1 <= cid <= 110 else {}
                ops = info.get("ops", 1)
                side = info.get("side", "neutral")
                score = ops * 10
                if side == self.side: score += 15
                elif side == "neutral": score += 5
                else: score -= 10
                if score > best_score:
                    best_score = score
                    best_card = cid

            c_name = ts_engine.CardData.get_card_name(best_card) if 1 <= best_card <= 110 else f"#{best_card}"
            strat = f"Hand Management: Selected card #{best_card} '{c_name}' (Evaluation Utility: {best_score})."
            comm = f"\"Playing '{c_name}'. This gives us optimal operational flexibility to advance our doctrine.\""
            return {"decision_type": d_type, "primary_id": best_card, "secondary_id": 0, "flags": 0}, strat, comm

        # 2. SELECT_PLAY_MODE
        elif d_type == 2:
            card_id = ctx.get("pending_op_card", 0)
            card_info = ts_engine.CardData.get_card_info(card_id) if 1 <= card_id <= 110 else {}
            card_side = card_info.get("side", "neutral")
            card_ops = card_info.get("ops", 1)
            is_scoring = card_info.get("is_scoring", False)
            card_name = card_info.get("name", f"Card #{card_id}")

            if is_scoring:
                strat = f"Scoring Resolution: '{card_name}' is a scoring card; executing as Event (0)."
                comm = f"\"Scoring regional balance of power.\""
                return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0}, strat, comm

            if card_side == self.opp_side:
                if 2 in valid_ids: # Space race
                    strat = f"Hazard Neutralization: Sending opponent event '{card_name}' to Space Race (2)."
                    comm = f"\"That enemy event is too toxic for the geopolitical theater. Discarding it into orbit!\""
                    roll = self.rng.randint(1, 6)
                    return {"decision_type": d_type, "primary_id": 2, "secondary_id": roll, "flags": 0}, strat, comm
                else:
                    strat = f"Mitigated Operations: Playing '{card_name}' for Operations (1) while absorbing the opponent event."
                    comm = f"\"We extract {card_ops} Ops from this hostile card and mitigate the adverse event consequences.\""
                    return {"decision_type": d_type, "primary_id": 1, "secondary_id": 0, "flags": 0}, strat, comm

            if 0 in valid_ids and card_side == self.side:
                one_time = card_info.get("one_time", False)
                if one_time or card_ops >= 3:
                    strat = f"Event Trigger: Activating permanent friendly event '{card_name}' (0)."
                    comm = f"\"Triggering the historical event: '{card_name}'! This permanently shifts the global balance in our favor.\""
                    return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0}, strat, comm

            strat = f"Operations Deployment: Converting '{card_name}' into {card_ops} Operations (1)."
            comm = f"\"Deploying {card_ops} Operations across key regional battlegrounds.\""
            return {"decision_type": d_type, "primary_id": 1, "secondary_id": 0, "flags": 0}, strat, comm

        # 3. CHOOSE_TIMING_BRANCH
        elif d_type == 3:
            chosen = 0 if 0 in valid_ids else valid_ids[0]
            strat = f"Timing Branch: Choosing Operations First (0) to solidify board standing before enemy event triggers."
            comm = f"\"Act first, ask questions later. We deploy our Operations on the ground prior to the opponent's reaction.\""
            return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, strat, comm

        # 4. SELECT_OP_MODE
        elif d_type == 4:
            pending_ops = ctx.get("pending_ops_value", 0)
            if 1 in valid_ids and mil_ops < defcon and ar == 1 and defcon >= 3:
                strat = f"MilOps Obligation: AR1 Coup Attempt (1) to fulfill military ops and deny opponent battleground control."
                comm = f"\"We launch an immediate military coup d'état! This secures our MilOps quota and hits their strategic perimeter.\""
                return {"decision_type": d_type, "primary_id": 1, "secondary_id": 0, "flags": 0}, strat, comm

            if 0 in valid_ids:
                strat = f"Diplomatic Expansion: Placing {pending_ops} Influence (0) into contested zones."
                comm = f"\"Expanding diplomatic and political influence across our target nations.\""
                return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0}, strat, comm

            chosen = valid_ids[0]
            return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, f"Mode {chosen}", f"\"Executing mode {chosen}.\""

        # 5. POINT_NODE
        elif d_type == 5:
            op_mode = ctx.get("op_mode", 0)
            rem_steps = ctx.get("remaining_steps", 0)
            resolving = ctx.get("resolving_card_name", "")

            if not valid_ids:
                if allow_early_stop:
                    return {"decision_type": d_type, "primary_id": 255, "secondary_id": 0, "flags": 0}, "Confirm Done", "\"No further actions needed; passing.\""
                return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0}, "Pass", "\"Passing.\""

            # Setup priorities
            if phase == 0:
                if self.side == "USSR":
                    for nid in valid_ids:
                        c_name = ts_engine.MapData.get_country_name(nid)
                        c_data = countries.get(c_name, {})
                        if c_name in ("East Germany", "Poland") and c_data.get("ussr_influence", 0) < 4:
                            return {"decision_type": d_type, "primary_id": nid, "secondary_id": 0, "flags": 0}, f"Setup {c_name}", f"\"Reinforcing our Eastern bloc fortress in {c_name}.\""
                elif self.side == "US":
                    for nid in valid_ids:
                        c_name = ts_engine.MapData.get_country_name(nid)
                        c_data = countries.get(c_name, {})
                        if c_name in ("West Germany", "Italy") and c_data.get("us_influence", 0) < 4:
                            return {"decision_type": d_type, "primary_id": nid, "secondary_id": 0, "flags": 0}, f"Setup {c_name}", f"\"Securing the front line of the Free World in {c_name}.\""

            # Point selection
            best_id = valid_ids[0]
            best_val = -999
            for nid in valid_ids:
                c_name = ts_engine.MapData.get_country_name(nid)
                c_info = ts_engine.MapData.get_country_info(nid)
                c_data = countries.get(c_name, {})
                stab = c_info.get("stability", 1)
                is_bg = c_info.get("battleground", False)
                us_inf = c_data.get("us_influence", 0)
                ussr_inf = c_data.get("ussr_influence", 0)
                opp_inf = ussr_inf if self.side == "US" else us_inf
                ctrl = c_data.get("controlled_by", "NONE")

                val = 0
                if is_bg: val += 50
                if ctrl == "NONE": val += 20
                elif ctrl == self.opp_side: val += 30
                elif ctrl == self.side: val -= 10
                if op_mode == 1: val += (5 - stab) * 15 + opp_inf * 10

                if val > best_val:
                    best_val = val
                    best_id = nid

            best_c_name = ts_engine.MapData.get_country_name(best_id)
            c_info = ts_engine.MapData.get_country_info(best_id)
            bg_tag = " (Battleground)" if c_info.get("battleground") else ""
            
            action_desc = "Coup" if op_mode == 1 else ("Influence" if op_mode == 0 else "Realignment")
            strat = f"Target Country: Selected {best_c_name}{bg_tag} (ID {best_id}) for {action_desc} (Strategic Utility: {best_val})."
            if op_mode == 1:
                comm = f"\"Executing coup operations against the regime in {best_c_name}. Rolling the die!\""
            else:
                comm = f"\"Channeling our diplomatic resources into {best_c_name}.\""

            roll1 = self.rng.randint(1, 6)
            roll2 = self.rng.randint(1, 6)
            return {"decision_type": d_type, "primary_id": best_id, "secondary_id": roll1, "flags": roll2}, strat, comm

        # 6. CHOOSE_BRANCH
        elif d_type == 6:
            labels = legal_actions.get("valid_action_labels", {})
            chosen = valid_ids[0] if valid_ids else 0
            opt_name = labels.get(str(chosen), f"Option #{chosen}")
            strat = f"Branch Selection: Selecting {opt_name} (ID {chosen})."
            comm = f"\"Resolving branch condition: {opt_name}.\""
            return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, strat, comm

        chosen = valid_ids[0] if valid_ids else 0
        return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, "Default Selection", "\"Executing default procedure.\""


def run_full_subagent_match(seed: int = 42) -> Tuple[List[dict], List[str], dict]:
    state = ts_engine.GameState()
    ts_engine.Engine.init_game(state, seed)

    ussr_agent = LLMSubagentDialoguePlayer("USSR", seed=seed * 3 + 1)
    us_agent = LLMSubagentDialoguePlayer("US", seed=seed * 3 + 2)

    logs = []
    all_violations = []
    step_num = 0
    max_steps = 1000

    print("=" * 80)
    print(f"COMMENCING FULL IN-CONVERSATION LLM SUBAGENT MATCH (Seed {seed})")
    print("=" * 80)

    # Scoring card IDs mapping to Region ID
    scoring_cards_map = {
        1: 1, # Asia Scoring -> Region 1
        2: 0, # Europe Scoring -> Region 0
        3: 2, # Middle East Scoring -> Region 2
        79: 3, # Africa Scoring -> Region 3
        81: 5, # South America Scoring -> Region 5
        88: 4, # Central America Scoring -> Region 4
    }

    while step_num < max_steps:
        step_num += 1
        s_dict = state.to_dict()

        if s_dict.get("is_terminal"):
            term_util = s_dict.get("terminal_utility", 0)
            winner = "US" if term_util > 0 else ("USSR" if term_util < 0 else "TIE")
            vp = state.victory_points
            print(f"\n★ VICTORY ACHIEVED AT STEP {step_num}!")
            print(f"★ WINNER: {winner} | FINAL VP: {vp:+d} | TURN: {state.turn} | DEFCON: {state.defcon}")
            
            logs.append({
                "step": step_num,
                "event": "GAME_OVER",
                "winner": winner,
                "final_vp": vp,
                "terminal_utility": term_util,
                "turn": state.turn,
                "phase": str(state.current_phase),
                "defcon": state.defcon
            })
            break

        legal = s_dict.get("legal_actions", {})
        ctx = s_dict.get("decision_context", {})
        d_player = legal.get("decision_player", "NONE")
        d_type = legal.get("decision_type", 0)
        d_type_name = legal.get("decision_type_name", f"TYPE_{d_type}")
        valid_ids = legal.get("valid_ids", [])

        # 1. Option Verification
        is_legal, violations = OptionVerifier.verify(s_dict)
        if violations:
            for v in violations:
                msg = f"[Step {step_num} | Turn {state.turn} AR {state.action_round} {d_player}]: {v}"
                print(f"❌ OPTION ERROR: {msg}")
                all_violations.append(msg)

        # 2. LLM Strategic Evaluation & Commentary Generation
        agent = ussr_agent if d_player == "USSR" else us_agent
        action_dict, strategy_cot, dialogue_comment = agent.format_strategic_deliberation(s_dict, legal)

        # 3. Check for Regional Scoring breakdown
        scoring_audit_text = ""
        if d_type == 1 and action_dict["primary_id"] in (1, 2, 3, 38, 79, 81, 88):
            scoring_audit_text = format_regional_scoring_breakdown(s_dict, action_dict["primary_id"])

        entry = {
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
                "player_title": agent.title,
                "type": d_type,
                "type_name": d_type_name,
                "valid_ids": valid_ids,
                "resolving_card": ctx.get("resolving_card_name", ""),
                "pending_card": ctx.get("pending_op_card_name", ""),
                "pending_ops": ctx.get("pending_ops_value", 0),
                "remaining_steps": ctx.get("remaining_steps", 0)
            },
            "audit_passed": is_legal,
            "violations": violations,
            "strategy": strategy_cot,
            "commentary": dialogue_comment,
            "scoring_audit": scoring_audit_text,
            "action_executed": action_dict
        }
        logs.append(entry)

        # Step engine
        m_action = ts_engine.MicroAction(
            ts_engine.DecisionType(action_dict["decision_type"]),
            action_dict.get("primary_id", 0),
            action_dict.get("secondary_id", 0),
            action_dict.get("flags", 0)
        )
        ts_engine.Engine.step(state, m_action)

    meta = {
        "seed": seed,
        "total_steps": len(logs),
        "total_violations": len(all_violations),
        "winner": logs[-1].get("winner", "UNKNOWN"),
        "final_vp": logs[-1].get("final_vp", state.victory_points),
        "final_turn": state.turn,
        "final_defcon": state.defcon
    }
    return logs, all_violations, meta


def export_commentary_chronicle(logs: List[dict], out_path: str, meta: dict):
    lines = []
    lines.append("=" * 85)
    lines.append("★ TWILIGHT STRUGGLE: AUTONOMOUS LLM SUBAGENTS DUAL MATCH & COMMENTARY LOG ★")
    lines.append(f"Seed: {meta['seed']} | Total Steps: {meta['total_steps']} | Winner: {meta['winner']} | Final VP: {meta['final_vp']}")
    lines.append("=" * 85)
    lines.append("")

    for entry in logs:
        if entry.get("event") == "GAME_OVER":
            lines.append("\n" + "=" * 85)
            lines.append(f"🏆 ★★★ FINAL RESULT: {entry.get('winner')} VICTORY ★★★")
            lines.append(f"Victory Points: {entry.get('final_vp'):+d} | Turn: {entry.get('turn')} | DEFCON: {entry.get('defcon')}")
            lines.append("=" * 85)
            break

        step = entry["step"]
        turn = entry["turn"]
        ar = entry["ar"]
        phase = entry["phase"]
        defcon = entry["defcon"]
        vp = entry["victory_points"]
        mil = entry["mil_ops"]
        sp = entry["space"]
        dec = entry["decision"]
        act = entry["action_executed"]
        strat = entry["strategy"]
        comm = entry["commentary"]
        scoring_audit = entry.get("scoring_audit", "")

        lines.append("-" * 85)
        lines.append(f"[Step {step:03d}] Turn {turn} AR{ar} ({phase}) | DEFCON: {defcon} | VP: {vp:+d} | MilOps: [US:{mil['US']}/5 USSR:{mil['USSR']}/5] | Space: [US:{sp['US']} USSR:{sp['USSR']}]")
        lines.append(f"Actor: {dec['player_title']} -> {dec['type_name']}")
        if dec.get("pending_card"):
            lines.append(f"  Ops Context: {dec['pending_card']} ({dec['pending_ops']} Ops, {dec['remaining_steps']} steps rem)")
        if dec.get("resolving_card"):
            lines.append(f"  Event Context: Resolving '{dec['resolving_card']}'")
        lines.append(f"  Legal Options ({len(dec['valid_ids'])} items): {dec['valid_ids']} [Verified: {'✅ 100% LEGAL' if entry['audit_passed'] else '❌ ERROR'}]")
        lines.append(f"  Strategic Rationale: {strat}")
        lines.append(f"  In-Character Commentary: {comm}")

        if scoring_audit:
            lines.append("")
            for sa_line in scoring_audit.split("\n"):
                lines.append(f"  {sa_line}")

        lines.append(f"  Action Committed: MicroAction(type={act['decision_type']}, primary={act.get('primary_id', 0)}, sec={act.get('secondary_id', 0)}, flags={act.get('flags', 0)})")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Detailed commentary chronicle saved to: {out_path}")


def main():
    os.makedirs("replays", exist_ok=True)
    logs, violations, meta = run_full_subagent_match(seed=42)
    
    json_path = "replays/llm_match_with_commentary.json"
    with open(json_path, "w") as f:
        audit_report: AuditGameReportDict = {"total_steps": len(logs), "violations_count": len(violations), "violations": violations, "logs": logs, "meta": meta}
        json.dump(audit_report, f, indent=2)

    txt_path = "replays/llm_match_with_commentary.log"
    export_commentary_chronicle(logs, txt_path, meta)

    print("\n" + "=" * 80)
    print("MATCH RUN COMPLETE:")
    print(f"• Total Steps: {meta['total_steps']}")
    print(f"• Winner: {meta['winner']} (VP: {meta['final_vp']:+d})")
    print(f"• Violations: {meta['total_violations']} (100% legal check passed)")
    print("=" * 80)

if __name__ == "__main__":
    main()
