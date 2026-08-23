#!/usr/bin/env python3
"""
Enhanced Strategic Match Generator for Twilight Struggle
=========================================================
Generates a realistic, high-level strategic match:
- DEFCON stays at 2 for most of the game (early AR1 coups, safe play under DEFCON 2).
- Actively employs Realignments in vulnerable contested theaters.
- Strategic Space Race usage for toxic enemy events.
- Full recorded output saved to replays/strategic_llm_game_defcon2.tslog.json and .log.
"""

import os
import sys
import json
import random
from typing import Dict, List, Any, Optional, Tuple

import ts_engine
from bot.scoring_logger import format_regional_scoring_breakdown
from server.session import describe_action_and_deltas

class StrategicPlayer:
    def __init__(self, side: str, seed: int = 42):
        self.side = side
        self.opp_side = "USSR" if side == "US" else "US"
        self.rng = random.Random(seed)
        self.title = "Comrade General Secretary (USSR)" if side == "USSR" else "Mr. President & National Security Council (US)"

    def choose_action(self, state_dict: dict, legal_actions: dict) -> Tuple[Dict[str, Any], str, str]:
        d_type = legal_actions.get("decision_type", 0)
        valid_ids = legal_actions.get("valid_ids", [])
        allow_early_stop = legal_actions.get("allow_early_stop", False)
        ctx = state_dict.get("decision_context", {})

        turn = state_dict.get("turn", 1)
        phase = state_dict.get("current_phase", 0)
        ar = state_dict.get("action_round", 0)
        defcon = state_dict.get("defcon", 5)
        vp = state_dict.get("victory_points", 0)
        mil_ops = state_dict.get("mil_ops", {}).get(self.side, 0)
        space = state_dict.get("space", {}).get(self.side, 0)
        space_used = state_dict.get("space_turns_used", {}).get(self.side, 0)
        countries = state_dict.get("countries", {})

        # 1. SELECT_CARD
        if d_type == 1:
            res_card = ctx.get("resolving_card", 0)
            
            # Event sub-decisions (e.g. Blockade discard, SALT, Ask Not)
            if res_card > 0:
                if res_card == 250: # Space Walk (Box 6) turn-end discard
                    opp_cards = [cid for cid in valid_ids if cid > 0 and ts_engine.CardData.get_card_info(cid).get("side") == self.opp_side]
                    if opp_cards:
                        chosen = opp_cards[0]
                        c_name = ts_engine.CardData.get_card_name(chosen)
                        strat = f"Space Walk Privilege (Box 6): Discarding enemy card #{chosen} '{c_name}' at turn end."
                        comm = f"Utilizing our Space Walk advantage to safely discard enemy card '{c_name}'."
                        return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, strat, comm
                    elif allow_early_stop:
                        strat = "Space Walk Privilege (Box 6): Retaining current hand (passing discard)."
                        comm = "Our hand is strategically solid; no discard necessary."
                        return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 128}, strat, comm

                if res_card == 10: # Blockade
                    three_ops = [cid for cid in valid_ids if cid > 0 and ts_engine.CardData.get_card_info(cid).get("ops", 0) >= 3]
                    if three_ops:
                        chosen = three_ops[0]
                        c_name = ts_engine.CardData.get_card_name(chosen)
                        strat = f"Blockade Defense: Discarding #{chosen} '{c_name}' (3 Ops) to preserve West Germany."
                        comm = f"Berlin is the frontier of liberty. We sacrifice '{c_name}' to break the blockade."
                        return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, strat, comm
                    elif allow_early_stop:
                        strat = "Blockade Concession: Refusing / unable to discard 3-Ops card; allowing influence removal."
                        comm = "We cannot afford the operational sacrifice at this time."
                        return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 128}, strat, comm

                chosen = valid_ids[0] if valid_ids else 0
                return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, f"Event Sub-selection #{chosen}", "Resolving event card requirements."

            # Setup phase
            if phase == 0:
                chosen = valid_ids[0]
                strat = f"Setup: Placing initial influence card #{chosen}."
                comm = "Fortifying our historical ideological bastions."
                return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, strat, comm

            # Headline phase
            if phase == 1:
                opp_hl = state_dict.get("headline_ussr_card" if self.side == "US" else "headline_us_card", 0)
                opp_hl_intel = ""
                opp_hl_name = ""
                if opp_hl > 0:
                    opp_hl_name = ts_engine.CardData.get_card_name(opp_hl)
                    opp_hl_intel = f" [Space Box 4 Intel: Opponent committed '{opp_hl_name}' (#{opp_hl})]"

                scoring_cards = [cid for cid in valid_ids if 1 <= cid <= 110 and ts_engine.CardData.get_card_info(cid).get("is_scoring", False)]
                headline_pref = {
                    "USSR": [31, 7, 25, 9, 38, 14, 11, 15, 30, 50, 51],
                    "US": [103, 106, 23, 27, 40, 10, 4, 16, 28, 68, 96]
                }
                preferred = [cid for cid in headline_pref.get(self.side, []) if cid in valid_ids]
                if preferred:
                    chosen = preferred[0]
                    c_name = ts_engine.CardData.get_card_name(chosen)
                    strat = f"Headline Strategy: Deploying '{c_name}' (#{chosen}) for maximum initiative.{opp_hl_intel}"
                    comm = f"Broadcasting our doctrine to the world: '{c_name}'." if not opp_hl_intel else f"Responding to revealed enemy headline '{opp_hl_name}' by deploying '{c_name}'."
                elif scoring_cards and ((self.side == "USSR" and vp < 0) or (self.side == "US" and vp > 0)):
                    chosen = scoring_cards[0]
                    c_name = ts_engine.CardData.get_card_name(chosen)
                    strat = f"Headline Scoring: Resolving '{c_name}' while we maintain regional lead.{opp_hl_intel}"
                    comm = f"Harvesting victory points before the enemy can respond: '{c_name}'."
                else:
                    friendly = [cid for cid in valid_ids if 1 <= cid <= 110 and ts_engine.CardData.get_card_info(cid).get("side") == self.side]
                    chosen = friendly[0] if friendly else valid_ids[0]
                    c_name = ts_engine.CardData.get_card_name(chosen)
                    strat = f"Headline Card: Playing #{chosen} '{c_name}'.{opp_hl_intel}"
                    comm = f"Opening the turn with '{c_name}'." if not opp_hl_intel else f"Leveraging intelligence on '{opp_hl_name}' to counter with '{c_name}'."

                return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, strat, comm

            # Action Round Selection
            # Avoid DEFCON suicide cards when DEFCON == 2
            suicide_cards = []
            if defcon == 2:
                if self.side == "USSR":
                    suicide_cards = [4, 26, 62, 89]
                elif self.side == "US":
                    suicide_cards = [20, 50]
            
            safe_cards = [cid for cid in valid_ids if cid not in suicide_cards]
            pool = safe_cards if safe_cards else valid_ids

            scoring = [cid for cid in pool if 1 <= cid <= 110 and ts_engine.CardData.get_card_info(cid).get("is_scoring", False)]
            if scoring:
                chosen = scoring[0]
                c_name = ts_engine.CardData.get_card_name(chosen)
                strat = f"Scoring Round: Auditing regional dominance with #{chosen} '{c_name}'."
                comm = f"Executing regional scoring card '{c_name}'."
                return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, strat, comm

            best_card = pool[0]
            best_score = -999
            for cid in pool:
                info = ts_engine.CardData.get_card_info(cid) if 1 <= cid <= 110 else {}
                ops = info.get("ops", 1)
                side = info.get("side", "neutral")
                score = ops * 10
                if side == self.side: score += 15
                elif side == "neutral": score += 5
                else:
                    if space_used == 0: score += 8
                    else: score -= 15
                if score > best_score:
                    best_score = score
                    best_card = cid

            c_name = ts_engine.CardData.get_card_name(best_card) if 1 <= best_card <= 110 else f"#{best_card}"
            strat = f"Action Round {ar}: Selected card #{best_card} '{c_name}' (Evaluation Utility: {best_score})."
            comm = f"Playing '{c_name}' to advance our strategic objectives."
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
                return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0}, f"Scoring Event '{card_name}'", "Resolving scoring event."

            # Hostile card -> send to Space if available
            if card_side == self.opp_side and 2 in valid_ids:
                roll = self.rng.randint(1, 6)
                strat = f"Space Race Disposal: Discarding hostile card '{card_name}' to space track."
                comm = f"Neutralizing enemy card '{card_name}' into our space program."
                return {"decision_type": d_type, "primary_id": 2, "secondary_id": roll, "flags": 0}, strat, comm

            # Friendly powerful permanent event
            if 0 in valid_ids and card_side == self.side and (card_info.get("one_time", False) or card_ops >= 3):
                strat = f"Event Play: Triggering '{card_name}'."
                comm = f"Activating historical event: '{card_name}'!"
                return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0}, strat, comm

            strat = f"Operations Play: Using '{card_name}' for {card_ops} Operations."
            comm = f"Conducting {card_ops} Operations across contested regions."
            return {"decision_type": d_type, "primary_id": 1, "secondary_id": 0, "flags": 0}, strat, comm

        # 3. CHOOSE_TIMING_BRANCH
        elif d_type == 3:
            chosen = 0 if 0 in valid_ids else valid_ids[0]
            strat = "Timing Branch: Operations First (0)."
            comm = "Securing board position before opponent event triggers."
            return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, strat, comm

        # 4. SELECT_OP_MODE
        elif d_type == 4:
            pending_ops = ctx.get("pending_ops_value", 0)

            # Check if AR1 Coup is needed for MilOps and DEFCON is 3+
            if 1 in valid_ids and mil_ops < defcon and ar == 1 and defcon >= 3:
                strat = "AR1 Coup: Fulfilling MilOps requirement and dropping DEFCON to 2."
                comm = "Launching early Action Round Coup to secure MilOps and restrict DEFCON."
                return {"decision_type": d_type, "primary_id": 1, "secondary_id": 0, "flags": 0}, strat, comm

            # Realignments: Check if 2 is available and we have good realignment targets
            if 2 in valid_ids:
                has_good_realign = False
                for c_name, c_data in countries.items():
                    opp_inf = c_data.get("ussr_influence" if self.side == "US" else "us_influence", 0)
                    my_inf = c_data.get("us_influence" if self.side == "US" else "ussr_influence", 0)
                    if opp_inf > 0 and (my_inf > 0 or c_data.get("battleground", False)):
                        has_good_realign = True
                        break
                if has_good_realign and self.rng.random() < 0.45:
                    strat = f"Strategic Realignment: Using {pending_ops} Ops for Realignment rolls without degrading DEFCON."
                    comm = "Initiating targeted realignment maneuvers to purge enemy influence without triggering DEFCON backlash."
                    return {"decision_type": d_type, "primary_id": 2, "secondary_id": 0, "flags": 0}, strat, comm

            # Default: Influence Placement (0)
            if 0 in valid_ids:
                strat = f"Influence Placement: Deploying {pending_ops} Influence."
                comm = f"Strengthening political networks with {pending_ops} Influence."
                return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0}, strat, comm

            chosen = valid_ids[0]
            return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, f"Mode {chosen}", "Executing mode."

        # 5. POINT_NODE
        elif d_type == 5:
            op_mode = ctx.get("op_mode", 0)
            res_card = ctx.get("resolving_card", 0)

            # Independent Reds special target handling
            if res_card == 22:
                indep_targets = [nid for nid in valid_ids if nid in (19, 17, 12, 14, 13) and nid > 0]
                if indep_targets:
                    best_target = max(indep_targets, key=lambda cid: countries.get(ts_engine.MapData.get_country_name(cid), {}).get("ussr_influence", 0))
                    c_name = ts_engine.MapData.get_country_name(best_target)
                    inf_gain = countries.get(c_name, {}).get("ussr_influence", 0)
                    strat = f"Independent Reds Target: Adding {inf_gain} US Influence to {c_name}."
                    comm = f"Independent Reds event activates in {c_name}, matching {inf_gain} Soviet influence."
                    return {"decision_type": d_type, "primary_id": best_target, "secondary_id": 0, "flags": 0}, strat, comm
                else:
                    return {"decision_type": d_type, "primary_id": 255, "secondary_id": 0, "flags": 128}, "Independent Reds: No Target", "Passing event."

            # Filter non-zero country IDs for events or ops unless Canada (0) is explicitly the only valid target
            valid_country_ids = [nid for nid in valid_ids if nid > 0] if res_card > 0 else valid_ids
            if not valid_country_ids:
                if allow_early_stop or res_card > 0:
                    return {"decision_type": d_type, "primary_id": 255, "secondary_id": 0, "flags": 128}, "Confirm Done", "Passing."
                return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0}, "Pass", "Passing."

            # Setup priorities
            if phase == 0:
                if self.side == "USSR":
                    for nid in valid_ids:
                        c_name = ts_engine.MapData.get_country_name(nid)
                        c_data = countries.get(c_name, {})
                        if c_name in ("East Germany", "Poland") and c_data.get("ussr_influence", 0) < 4:
                            return {"decision_type": d_type, "primary_id": nid, "secondary_id": 0, "flags": 0}, f"Setup {c_name}", f"Fortifying {c_name}."
                elif self.side == "US":
                    for nid in valid_ids:
                        c_name = ts_engine.MapData.get_country_name(nid)
                        c_data = countries.get(c_name, {})
                        if c_name in ("West Germany", "Italy") and c_data.get("us_influence", 0) < 4:
                            return {"decision_type": d_type, "primary_id": nid, "secondary_id": 0, "flags": 0}, f"Setup {c_name}", f"Fortifying {c_name}."

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
                my_inf = us_inf if self.side == "US" else ussr_inf
                opp_inf = ussr_inf if self.side == "US" else us_inf
                ctrl = c_data.get("controlled_by", "NONE")

                val = 0
                if is_bg: val += 60
                if ctrl == "NONE":
                    val += 30
                    if my_inf + 1 >= stab + opp_inf: val += 40
                elif ctrl == self.opp_side:
                    val += 40
                elif ctrl == self.side:
                    val -= 10

                if op_mode == 2: # Realignment
                    if opp_inf > 0: val += 50 + opp_inf * 10
                    if my_inf > 0: val += 30

                if op_mode == 1: # Coup
                    val += (5 - stab) * 20 + opp_inf * 15

                if val > best_val:
                    best_val = val
                    best_id = nid

            best_c_name = ts_engine.MapData.get_country_name(best_id)
            c_info = ts_engine.MapData.get_country_info(best_id)
            bg_tag = " (Battleground)" if c_info.get("battleground") else ""
            
            action_desc = "Coup" if op_mode == 1 else ("Influence" if op_mode == 0 else "Realignment")
            strat = f"Target Country: Selected {best_c_name}{bg_tag} (ID {best_id}) for {action_desc} (Strategic Utility: {best_val})."
            comm = f"Executing {action_desc} in {best_c_name}."
            
            roll = self.rng.randint(1, 6) if op_mode in (1, 2) else 0
            return {"decision_type": d_type, "primary_id": best_id, "secondary_id": roll, "flags": 0}, strat, comm

        # 6. CHOOSE_BRANCH
        elif d_type == 6:
            chosen = valid_ids[0] if valid_ids else 0
            return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, f"Branch {chosen}", f"Selecting branch {chosen}."

        chosen = valid_ids[0] if valid_ids else 0
        return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, "Default Selection", "Executing action."


def generate_strategic_match(seed: int = 1989):
    state = ts_engine.GameState()
    ts_engine.Engine.init_game(state, seed)

    ussr_player = StrategicPlayer("USSR", seed=seed * 7 + 11)
    us_player = StrategicPlayer("US", seed=seed * 7 + 13)

    from server.replay import ReplayLogger
    replay_logger = ReplayLogger("strategic_llm_game_defcon2", seed, "US National Security Council", "USSR Central Committee")

    logs = []
    step_num = 0
    max_steps = 1500

    print("=" * 80)
    print(f"GENERATING HIGH-LEVEL STRATEGIC MATCH (DEFCON 2 & REALIGNMENTS) - Seed {seed}")
    print("=" * 80)

    while step_num < max_steps:
        step_num += 1
        s_dict = state.to_dict()

        if s_dict.get("is_terminal"):
            term_util = s_dict.get("terminal_utility", 0)
            winner = "US" if term_util > 0 else ("USSR" if term_util < 0 else "TIE")
            vp = state.victory_points
            print(f"\n★ VICTORY ACHIEVED AT STEP {step_num}!")
            print(f"★ WINNER: {winner} | FINAL VP: {vp:+d} | TURN: {state.turn} | DEFCON: {state.defcon}")
            
            replay_logger.set_result(winner, abs(vp), state.turn, f"Victory by {winner} (VP: {vp:+d})")
            logs.append({
                "step": step_num,
                "event": "GAME_OVER",
                "winner": winner,
                "final_vp": vp,
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

        agent = ussr_player if d_player == "USSR" else us_player
        action_dict, strat, comm = agent.choose_action(s_dict, legal)

        m_action = ts_engine.MicroAction(
            ts_engine.DecisionType(action_dict["decision_type"]),
            action_dict.get("primary_id", 0),
            action_dict.get("secondary_id", 0),
            action_dict.get("flags", 0)
        )

        state_before = s_dict
        ts_engine.Engine.step(state, m_action)
        state_after = state.to_dict()

        delta_logs = describe_action_and_deltas(state_before, state_after, m_action)
        action_desc = "; ".join(delta_logs) if delta_logs else strat

        step_turn = state_before.get("turn", state.turn)
        step_phase = str(state_before.get("current_phase_name", "ACTION"))
        step_ar = 0 if step_phase in ("HEADLINE", "SETUP") else state_before.get("action_round", state.action_round)

        replay_logger.log_step(
            step_index=step_num,
            turn=step_turn,
            ar=step_ar,
            phase=step_phase,
            player=d_player,
            action=action_dict,
            description=action_desc,
            state_snapshot=state_after
        )

        logs.append({
            "step": step_num,
            "turn": step_turn,
            "phase": step_phase,
            "ar": step_ar,
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
                "pending_card": ctx.get("pending_op_card_name", "")
            },
            "strategy": strat,
            "commentary": comm,
            "action_executed": action_dict,
            "description": action_desc
        })

    tslog_path = replay_logger.save()
    print(f"✅ Saved TSLog replay to: {tslog_path}")

    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "replays", "strategic_llm_game_defcon2.log")
    lines = []
    lines.append("=" * 85)
    lines.append("★ TWILIGHT STRUGGLE: STRATEGIC REASONED MATCH (DEFCON 2 & REALIGNMENTS) ★")
    lines.append(f"Seed: {seed} | Total Steps: {len(logs)} | Winner: {logs[-1].get('winner', 'UNKNOWN')} | Final VP: {logs[-1].get('final_vp', 0)}")
    lines.append("=" * 85)
    lines.append("")

    for entry in logs:
        if entry.get("event") == "GAME_OVER":
            lines.append("\n" + "=" * 85)
            lines.append(f"🏆 ★★★ FINAL RESULT: {entry.get('winner')} VICTORY ★★★")
            lines.append(f"Victory Points: {entry.get('final_vp'):+d} | Turn: {entry.get('turn')} | DEFCON: {entry.get('defcon')}")
            lines.append("=" * 85)
            break

        lines.append("-" * 85)
        lines.append(f"[Step {entry['step']:03d}] Turn {entry['turn']} AR{entry['ar']} ({entry['phase']}) | DEFCON: {entry['defcon']} | VP: {entry['victory_points']:+d} | MilOps: [US:{entry['mil_ops']['US']}/5 USSR:{entry['mil_ops']['USSR']}/5] | Space: [US:{entry['space']['US']} USSR:{entry['space']['USSR']}]")
        lines.append(f"Actor: {entry['decision']['player_title']} -> {entry['decision']['type_name']}")
        lines.append(f"  Description: {entry['description']}")
        lines.append(f"  Strategic Rationale: {entry['strategy']}")
        lines.append(f"  In-Character Commentary: \"{entry['commentary']}\"")
        lines.append(f"  Action Committed: MicroAction({entry['action_executed']})")
        lines.append("")

    with open(log_path, "w") as f:
        f.write("\n".join(lines))
    print(f"✅ Saved detailed chronicle to: {log_path}")

if __name__ == "__main__":
    generate_strategic_match(seed=1989)
