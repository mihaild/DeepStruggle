"""StrategicBot: High-level realistic strategic agent maintaining DEFCON-2 containment, realignments, and coups."""

import random
from typing import Dict, Any, Optional, Tuple, List
import os
import sys

try:
    import ts_engine
except ImportError:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _build = os.path.join(_root, "build")
    if os.path.exists(_build) and _build not in sys.path:
        sys.path.insert(0, _build)
    import ts_engine

from bot.base_bot import BaseBot


class StrategicBot(BaseBot):
    """High-level strategic agent prioritizing DEFCON-2 containment, coups, realignments, and Space Race safety."""

    def __init__(self, role: str, rng_seed: Optional[int] = None, name: Optional[str] = None):
        super().__init__(role, name=name)
        self.opp_role = "USSR" if self.role == "US" else "US"
        self.rng = random.Random(rng_seed)
        self.last_strategy: str = ""
        self.last_commentary: str = ""

    def select_action(self, state: Dict[str, Any], legal_actions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        action_dict, strat, comm = self.choose_action(state, legal_actions)
        self.last_strategy = strat
        self.last_commentary = comm
        return action_dict

    def choose_action(self, state_dict: Dict[str, Any], legal_actions: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str, str]:
        d_type = legal_actions.get("decision_type", 0)
        valid_ids = legal_actions.get("valid_ids", [])
        allow_early_stop = legal_actions.get("allow_early_stop", False)
        ctx = state_dict.get("decision_context", {})

        turn = state_dict.get("turn", 1)
        phase = state_dict.get("phase", state_dict.get("current_phase", 0))
        ar = state_dict.get("action_round", 0)
        defcon = state_dict.get("defcon", 5)
        vp = state_dict.get("victory_points", 0)
        mil_ops = state_dict.get("mil_ops", {}).get(self.role, 0)
        space = state_dict.get("space", {}).get(self.role, 0)
        space_used = state_dict.get("space_turns_used", {}).get(self.role, 0)
        countries = state_dict.get("countries", {})

        # 1. SELECT_CARD
        if d_type == 1:
            res_card = ctx.get("resolving_card", 0)

            # Event sub-decisions (e.g. Blockade discard, Space Walk)
            if res_card > 0:
                if res_card == 250:  # Space Walk (Box 6) turn-end discard
                    opp_cards = [
                        cid
                        for cid in valid_ids
                        if cid > 0 and ts_engine.CardData.get_card_info(cid).get("side") == self.opp_role
                    ]
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

                if res_card == 10:  # Blockade
                    three_ops = [
                        cid
                        for cid in valid_ids
                        if cid > 0 and ts_engine.CardData.get_card_info(cid).get("ops", 0) >= 3
                    ]
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
                chosen = valid_ids[0] if valid_ids else 0
                strat = f"Setup: Placing initial influence card #{chosen}."
                comm = "Fortifying our historical ideological bastions."
                return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, strat, comm

            # Headline phase
            if phase == 1:
                scoring_cards = [
                    cid
                    for cid in valid_ids
                    if 1 <= cid <= 110 and ts_engine.CardData.get_card_info(cid).get("is_scoring", False)
                ]
                headline_pref = {
                    "USSR": [31, 7, 25, 9, 38, 14, 11, 15, 30, 50, 51],
                    "US": [103, 106, 23, 27, 40, 10, 4, 16, 28, 68, 96],
                }
                preferred = [cid for cid in headline_pref.get(self.role, []) if cid in valid_ids]
                if preferred:
                    chosen = preferred[0]
                    c_name = ts_engine.CardData.get_card_name(chosen)
                    strat = f"Headline Strategy: Deploying '{c_name}' (#{chosen}) for maximum initiative."
                    comm = f"Broadcasting our doctrine to the world: '{c_name}'."
                elif scoring_cards and ((self.role == "USSR" and vp < 0) or (self.role == "US" and vp > 0)):
                    chosen = scoring_cards[0]
                    c_name = ts_engine.CardData.get_card_name(chosen)
                    strat = f"Headline Scoring: Resolving '{c_name}' while we maintain regional lead."
                    comm = f"Harvesting victory points before the enemy can respond: '{c_name}'."
                else:
                    friendly = [
                        cid
                        for cid in valid_ids
                        if 1 <= cid <= 110 and ts_engine.CardData.get_card_info(cid).get("side") == self.role
                    ]
                    chosen = friendly[0] if friendly else (valid_ids[0] if valid_ids else 0)
                    c_name = ts_engine.CardData.get_card_name(chosen) if 1 <= chosen <= 110 else f"#{chosen}"
                    strat = f"Headline Card: Playing #{chosen} '{c_name}'."
                    comm = f"Opening the turn with '{c_name}'."

                return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, strat, comm

            # Action Round Selection - Avoid DEFCON suicide cards when DEFCON == 2
            suicide_cards = []
            if defcon == 2:
                if self.role == "USSR":
                    suicide_cards = [4, 26, 62, 89]
                elif self.role == "US":
                    suicide_cards = [20, 50]

            safe_cards = [cid for cid in valid_ids if cid not in suicide_cards]
            pool = safe_cards if safe_cards else valid_ids

            scoring = [
                cid
                for cid in pool
                if 1 <= cid <= 110 and ts_engine.CardData.get_card_info(cid).get("is_scoring", False)
            ]
            if scoring:
                chosen = scoring[0]
                c_name = ts_engine.CardData.get_card_name(chosen)
                strat = f"Scoring Round: Auditing regional dominance with #{chosen} '{c_name}'."
                comm = f"Executing regional scoring card '{c_name}'."
                return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, strat, comm

            best_card = pool[0] if pool else 0
            best_score = -999
            for cid in pool:
                info = ts_engine.CardData.get_card_info(cid) if 1 <= cid <= 110 else {}
                ops = info.get("ops", 1)
                side = info.get("side", "neutral")
                score = ops * 10
                if side == self.role:
                    score += 15
                elif side == "neutral":
                    score += 5
                else:
                    if space_used == 0:
                        score += 8
                    else:
                        score -= 15
                if score > best_score:
                    best_score = score
                    best_card = cid

            c_name = ts_engine.CardData.get_card_name(best_card) if 1 <= best_card <= 110 else f"#{best_card}"
            strat = f"Action Round {ar}: Selected card #{best_card} '{c_name}'."
            comm = f"Playing '{c_name}' to advance our strategic objectives."
            return {"decision_type": d_type, "primary_id": best_card, "secondary_id": 0, "flags": 0}, strat, comm

        # 2. SELECT_PLAY_MODE
        elif d_type == 2:
            card_id = ctx.get("pending_op_card", 0)
            card_info = ts_engine.CardData.get_card_info(card_id) if 1 <= card_id <= 110 else {}
            card_side = card_info.get("side", "neutral")
            card_ops = int(card_info.get("ops", 1))
            is_scoring = card_info.get("is_scoring", False)
            card_name = card_info.get("name", f"Card #{card_id}")

            if is_scoring:
                return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0}, f"Scoring Event '{card_name}'", "Resolving scoring event."

            # Hostile card -> send to Space if available
            if card_side == self.opp_role and 2 in valid_ids:
                roll = self.rng.randint(1, 6)
                strat = f"Space Race Disposal: Discarding hostile card '{card_name}' to space track."
                comm = f"Neutralizing enemy card '{card_name}' into our space program."
                return {"decision_type": d_type, "primary_id": 2, "secondary_id": roll, "flags": 0}, strat, comm

            # Friendly powerful permanent event
            if 0 in valid_ids and card_side == self.role and (card_info.get("one_time", False) or card_ops >= 3):
                strat = f"Event Play: Triggering '{card_name}'."
                comm = f"Activating historical event: '{card_name}'!"
                return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0}, strat, comm

            strat = f"Operations Play: Using '{card_name}' for {card_ops} Operations."
            comm = f"Conducting {card_ops} Operations across contested regions."
            return {"decision_type": d_type, "primary_id": 1, "secondary_id": 0, "flags": 0}, strat, comm

        # 3. CHOOSE_TIMING_BRANCH
        elif d_type == 3:
            chosen = 0 if 0 in valid_ids else (valid_ids[0] if valid_ids else 0)
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

            # Realignments: Check if 2 is available
            if 2 in valid_ids and self.rng.random() < 0.4:
                strat = f"Strategic Realignment: Using {pending_ops} Ops for Realignment rolls."
                comm = "Initiating targeted realignment maneuvers to purge enemy influence."
                return {"decision_type": d_type, "primary_id": 2, "secondary_id": 0, "flags": 0}, strat, comm

            # Default: Influence Placement (0)
            if 0 in valid_ids:
                strat = f"Influence Placement: Deploying {pending_ops} Influence."
                comm = f"Strengthening political networks with {pending_ops} Influence."
                return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0}, strat, comm

            chosen = valid_ids[0] if valid_ids else 0
            return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, f"Mode {chosen}", "Executing mode."

        # 5. POINT_NODE
        elif d_type == 5:
            res_card = ctx.get("resolving_card", 0)
            if not valid_ids:
                if allow_early_stop:
                    return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 128}, "Confirm Done", "Passing."
                return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0}, "Pass", "Passing."

            # Setup priorities
            if phase == 0:
                if self.role == "USSR":
                    for nid in valid_ids:
                        c_name = ts_engine.MapData.get_country_name(nid)
                        if c_name in ("East Germany", "Poland"):
                            return {"decision_type": d_type, "primary_id": nid, "secondary_id": 0, "flags": 0}, f"Setup {c_name}", f"Fortifying {c_name}."
                elif self.role == "US":
                    for nid in valid_ids:
                        c_name = ts_engine.MapData.get_country_name(nid)
                        if c_name in ("West Germany", "Italy"):
                            return {"decision_type": d_type, "primary_id": nid, "secondary_id": 0, "flags": 0}, f"Setup {c_name}", f"Fortifying {c_name}."

            best_id = valid_ids[0]
            best_val = -999
            for nid in valid_ids:
                if nid >= 84:
                    continue
                c_info = ts_engine.MapData.get_country_info(nid)
                stab = c_info.get("stability", 1)
                is_bg = c_info.get("battleground", False)
                val = (100 if is_bg else 30) - stab * 5
                if val > best_val:
                    best_val = val
                    best_id = nid

            c_name = ts_engine.MapData.get_country_name(best_id) if best_id < 84 else f"#{best_id}"
            strat = f"Strategic Target: Targeting {c_name} (ID {best_id})."
            comm = f"Concentrating resources on {c_name}."
            return {"decision_type": d_type, "primary_id": best_id, "secondary_id": 0, "flags": 0}, strat, comm

        # Fallback
        chosen = valid_ids[0] if valid_ids else 0
        return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, f"Action {chosen}", "Executing action."
