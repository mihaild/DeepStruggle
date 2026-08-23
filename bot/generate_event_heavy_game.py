#!/usr/bin/env python3
"""
Event-Heavy Game Trajectory Generator for Twilight Struggle
===========================================================
Generates a match trajectory where both superpowers prioritize playing
cards as Events whenever legal/possible:
- SELECT_PLAY_MODE: Always prefers PlayMode::EVENT (0) over Ops or Space.
- CHOOSE_TIMING_BRANCH: Always chooses TimingBranch::EVENT_FIRST (1) for opponent cards.
- SELECT_CARD: Maximizes event play frequency across Early, Mid, and Late War.
- Outputs full replay to replays/event_heavy_game.tslog.json and commentary log.
"""

import os
import sys
import json
import random
from typing import Dict, List, Any, Optional, Tuple

import ts_engine
from bot.scoring_logger import format_regional_scoring_breakdown
from server.session import describe_action_and_deltas
from server.replay import ReplayLogger

class EventHeavyPlayer:
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
        countries = state_dict.get("countries", {})

        # 1. SELECT_CARD
        if d_type == 1:
            res_card = ctx.get("resolving_card", 0)
            
            # Event sub-decisions (e.g. Blockade discard, SALT, Five Year Plan, Star Wars)
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

                if res_card == 10: # Blockade: US discards 3+ Ops card if held
                    three_ops = [cid for cid in valid_ids if cid > 0 and ts_engine.CardData.get_card_info(cid).get("ops", 0) >= 3]
                    if three_ops:
                        chosen = three_ops[0]
                        c_name = ts_engine.CardData.get_card_name(chosen)
                        strat = f"Blockade Response: Discarding #{chosen} '{c_name}' to maintain West Germany."
                        comm = f"Preserving the integrity of Berlin by sacrificing '{c_name}'."
                        return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, strat, comm
                    elif allow_early_stop:
                        strat = "Blockade Response: Permitting influence removal."
                        comm = "Accepting the blockade consequences."
                        return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 128}, strat, comm

                chosen = valid_ids[0] if valid_ids else 0
                c_name = ts_engine.CardData.get_card_name(chosen) if chosen > 0 else f"#{chosen}"
                return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, f"Event Sub-selection '{c_name}'", "Resolving event sub-requirements."

            # Setup phase
            if phase == 0:
                chosen = valid_ids[0]
                strat = f"Initial Setup: Placing starting influence card #{chosen}."
                comm = "Fortifying opening positions on the world stage."
                return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, strat, comm

            # Headline phase
            if phase == 1:
                # Prioritize event-rich headline cards
                headline_pref = {
                    "USSR": [7, 25, 31, 9, 38, 14, 11, 15, 30, 50, 51, 68, 87],
                    "US": [103, 106, 23, 27, 40, 10, 4, 16, 28, 68, 96, 97]
                }
                preferred = [cid for cid in headline_pref.get(self.side, []) if cid in valid_ids]
                if preferred:
                    chosen = preferred[0]
                else:
                    friendly = [cid for cid in valid_ids if 1 <= cid <= 110 and ts_engine.CardData.get_card_info(cid).get("side") == self.side]
                    chosen = friendly[0] if friendly else valid_ids[0]

                c_name = ts_engine.CardData.get_card_name(chosen)
                strat = f"Headline Event Priority: Triggering #{chosen} '{c_name}' as headline statement."
                comm = f"Setting the historical stage with '{c_name}'!"
                return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, strat, comm

            # Action Round: Prioritize playing cards with events
            scoring = [cid for cid in valid_ids if 1 <= cid <= 110 and ts_engine.CardData.get_card_info(cid).get("is_scoring", False)]
            
            # If late in turn (e.g. AR 6 or 7), must play scoring cards
            if scoring and ar >= 5:
                chosen = scoring[0]
                c_name = ts_engine.CardData.get_card_name(chosen)
                strat = f"Mandatory Scoring Audit: Triggering #{chosen} '{c_name}' before turn end."
                comm = f"Auditing regional balance with '{c_name}'."
                return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, strat, comm

            # Event-first prioritization: Friendly events > Neutral events > Opponent events > Scoring
            friendly_events = [cid for cid in valid_ids if 1 <= cid <= 110 and ts_engine.CardData.get_card_info(cid).get("side") == self.side and not ts_engine.CardData.get_card_info(cid).get("is_scoring", False)]
            neutral_events = [cid for cid in valid_ids if 1 <= cid <= 110 and ts_engine.CardData.get_card_info(cid).get("side") == "neutral" and not ts_engine.CardData.get_card_info(cid).get("is_scoring", False)]
            opp_events = [cid for cid in valid_ids if 1 <= cid <= 110 and ts_engine.CardData.get_card_info(cid).get("side") == self.opp_side and not ts_engine.CardData.get_card_info(cid).get("is_scoring", False)]

            # Avoid suicide cards at DEFCON 2
            if defcon == 2:
                if self.side == "USSR":
                    opp_events = [c for c in opp_events if c not in (4, 26, 62, 89)]
                    friendly_events = [c for c in friendly_events if c not in (26, 62)]
                elif self.side == "US":
                    opp_events = [c for c in opp_events if c not in (20, 50)]

            if friendly_events:
                chosen = friendly_events[0]
            elif neutral_events:
                chosen = neutral_events[0]
            elif opp_events:
                chosen = opp_events[0]
            elif scoring:
                chosen = scoring[0]
            else:
                chosen = valid_ids[0]

            c_name = ts_engine.CardData.get_card_name(chosen) if 1 <= chosen <= 110 else f"#{chosen}"
            strat = f"Event Play Drive: Selected #{chosen} '{c_name}' to trigger event in global arena."
            comm = f"Deploying '{c_name}' to enact historical geopolitical event."
            return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, strat, comm

        # 2. SELECT_PLAY_MODE: ALWAYS MAXIMIZE EVENT PLAY
        elif d_type == 2:
            card_id = ctx.get("pending_op_card", 0)
            card_info = ts_engine.CardData.get_card_info(card_id) if 1 <= card_id <= 110 else {}
            card_side = card_info.get("side", "neutral")
            card_ops = card_info.get("ops", 1)
            card_name = card_info.get("name", f"Card #{card_id}")

            # The China Card (Card #6) can ONLY be played for Operations (1)
            if card_id == 6:
                strat = f"China Card Play: Deploying {card_ops} Ops for regional operations."
                comm = f"Playing The China Card for {card_ops} Operations Points."
                return {"decision_type": d_type, "primary_id": 1, "secondary_id": 0, "flags": 0}, strat, comm

            # If EVENT (0) is legal (only for friendly/neutral cards), choose EVENT!
            if 0 in valid_ids:
                strat = f"Maximum Event Strategy: Triggering '{card_name}' directly for EVENT (0)."
                comm = f"Activating historical event: '{card_name}'!"
                return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0}, strat, comm

            # If opponent card or Ops only, play for Operations (1)
            strat = f"Operations Deployment: Playing '{card_name}' for Operations (1)."
            comm = f"Conducting Operations with '{card_name}'."
            return {"decision_type": d_type, "primary_id": 1, "secondary_id": 0, "flags": 0}, strat, comm

        # 3. CHOOSE_TIMING_BRANCH: ALWAYS CHOOSE EVENT FIRST
        elif d_type == 3:
            # 1 = EVENT_FIRST, 0 = OPS_FIRST
            chosen = 1 if 1 in valid_ids else (0 if 0 in valid_ids else valid_ids[0])
            strat = "Timing Strategy: Choosing EVENT FIRST (1) to maximize event precedence."
            comm = "Triggering the event consequences immediately before conducting ground ops."
            return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, strat, comm

        # 4. SELECT_OP_MODE
        elif d_type == 4:
            pending_ops = ctx.get("pending_ops_value", 0)

            # Use Realignments or Coups or Influence
            if 2 in valid_ids and self.rng.random() < 0.5:
                strat = f"Realignment Focus: Using {pending_ops} Ops for strategic realignments."
                comm = "Executing realignments to reshape regional influence."
                return {"decision_type": d_type, "primary_id": 2, "secondary_id": 0, "flags": 0}, strat, comm

            if 1 in valid_ids and ar == 1 and defcon >= 3:
                strat = "AR1 Military Coup: Conducting Coup for MilOps requirement."
                comm = "Launching military coup to challenge regional control."
                return {"decision_type": d_type, "primary_id": 1, "secondary_id": 0, "flags": 0}, strat, comm

            if 0 in valid_ids:
                strat = f"Influence Deployment: Placing {pending_ops} Influence."
                comm = f"Placing {pending_ops} Influence into key strategic countries."
                return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0}, strat, comm

            chosen = valid_ids[0]
            return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, f"Op Mode {chosen}", "Executing op mode."

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
                if is_bg: val += 50
                if ctrl == "NONE":
                    val += 25
                    if my_inf + 1 >= stab + opp_inf: val += 35
                elif ctrl == self.opp_side: val += 35
                elif ctrl == self.side: val -= 10

                if op_mode == 2: # Realignment
                    if opp_inf > 0: val += 40 + opp_inf * 10
                    if my_inf > 0: val += 20

                if op_mode == 1: # Coup
                    val += (5 - stab) * 20 + opp_inf * 15

                if val > best_val:
                    best_val = val
                    best_id = nid

            best_c_name = ts_engine.MapData.get_country_name(best_id)
            c_info = ts_engine.MapData.get_country_info(best_id)
            bg_tag = " (Battleground)" if c_info.get("battleground") else ""
            
            action_desc = "Coup" if op_mode == 1 else ("Influence" if op_mode == 0 else "Realignment")
            strat = f"Target Country: Selected {best_c_name}{bg_tag} (ID {best_id}) for {action_desc}."
            comm = f"Executing {action_desc} in {best_c_name}."
            
            roll = self.rng.randint(1, 6) if op_mode in (1, 2) else 0
            return {"decision_type": d_type, "primary_id": best_id, "secondary_id": roll, "flags": 0}, strat, comm

        # 6. CHOOSE_BRANCH
        elif d_type == 6:
            chosen = valid_ids[0] if valid_ids else 0
            return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, f"Branch Choice #{chosen}", f"Selecting branch option {chosen}."

        chosen = valid_ids[0] if valid_ids else 0
        return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, "Default Selection", "Executing action."


def generate_event_heavy_game(seed: int = 1975):
    state = ts_engine.GameState()
    ts_engine.Engine.init_game(state, seed)

    ussr_player = EventHeavyPlayer("USSR", seed=seed * 5 + 17)
    us_player = EventHeavyPlayer("US", seed=seed * 5 + 23)

    replay_logger = ReplayLogger("event_heavy_game", seed, "US National Security Council (Event Drive)", "USSR Central Committee (Event Drive)")

    logs = []
    step_num = 0
    max_steps = 1500
    events_triggered_count = 0

    print("=" * 80)
    print(f"GENERATING EVENT-HEAVY MATCH TRAJECTORY (MAX EVENT DENSITY) - Seed {seed}")
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
            print(f"★ TOTAL EVENTS TRIGGERED: {events_triggered_count}")
            
            replay_logger.set_result(winner, abs(vp), state.turn, f"Victory by {winner} (VP: {vp:+d}) in Event-Heavy Campaign")
            logs.append({
                "step": step_num,
                "event": "GAME_OVER",
                "winner": winner,
                "final_vp": vp,
                "turn": state.turn,
                "phase": str(state.current_phase),
                "defcon": state.defcon,
                "total_events": events_triggered_count
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

        if action_dict["decision_type"] == 2 and action_dict.get("primary_id") == 0:
            events_triggered_count += 1
        elif action_dict["decision_type"] == 3 and action_dict.get("primary_id") == 1:
            events_triggered_count += 1

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

    # Save replay and commentary log
    tslog_path = replay_logger.save()
    print(f"✅ Saved Event-Heavy TSLog replay to: {tslog_path}")

    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "replays", "event_heavy_game.log")
    lines = []
    lines.append("=" * 85)
    lines.append("★ TWILIGHT STRUGGLE: EVENT-HEAVY MATCH CHRONICLE (MAX EVENT DENSITY) ★")
    lines.append(f"Seed: {seed} | Total Steps: {len(logs)} | Winner: {logs[-1].get('winner', 'UNKNOWN')} | Final VP: {logs[-1].get('final_vp', 0)} | Total Events: {events_triggered_count}")
    lines.append("=" * 85)
    lines.append("")

    for entry in logs:
        if entry.get("event") == "GAME_OVER":
            lines.append("\n" + "=" * 85)
            lines.append(f"🏆 ★★★ FINAL RESULT: {entry.get('winner')} VICTORY ★★★")
            lines.append(f"Victory Points: {entry.get('final_vp'):+d} | Turn: {entry.get('turn')} | DEFCON: {entry.get('defcon')} | Total Events: {entry.get('total_events')}")
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
    print(f"✅ Saved detailed Event-Heavy chronicle to: {log_path}")

if __name__ == "__main__":
    generate_event_heavy_game(seed=1975)
