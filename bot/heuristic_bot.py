"""HeuristicBot: Expert rule-based baseline agent for Twilight Struggle."""

import random
from typing import Dict, Any, Optional
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


class HeuristicBot(BaseBot):
    """Rule-based heuristic bot prioritizing key battlegrounds, scoring cards, and event timing."""

    def select_action(self, state: Dict[str, Any], legal_actions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        valid_ids = legal_actions.get("valid_ids", [])
        d_type = legal_actions.get("decision_type", 0)
        allow_early_stop = legal_actions.get("allow_early_stop", False)

        if not valid_ids:
            if allow_early_stop:
                return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0x80}
            return None

        # 1. SETUP: Prefer key battlegrounds
        if state.get("current_phase") == 0:  # SETUP
            if self.role == "USSR":
                # East Germany (14), Poland (15)
                for preferred in [14, 15]:
                    if preferred in valid_ids:
                        return {"decision_type": d_type, "primary_id": preferred, "secondary_id": 0, "flags": 0}
            elif self.role == "US":
                # Iran (25), West Germany (7), Italy (10), France (8)
                for preferred in [25, 7, 10, 8]:
                    if preferred in valid_ids:
                        return {"decision_type": d_type, "primary_id": preferred, "secondary_id": 0, "flags": 0}

        # 2. SELECT_CARD: Play scoring cards if holding them or highest Ops card
        if d_type == 1:  # SELECT_CARD
            # Check for scoring cards
            for cid in valid_ids:
                if 1 <= cid <= 110:
                    info = ts_engine.CardData.get_card_info(cid)
                    if info.get("is_scoring"):
                        return {"decision_type": d_type, "primary_id": cid, "secondary_id": 0, "flags": 0}

            # Else pick highest Ops card
            best_cid = valid_ids[0]
            best_ops = -1
            for cid in valid_ids:
                if 1 <= cid <= 110:
                    info = ts_engine.CardData.get_card_info(cid)
                    if info.get("ops", 0) > best_ops:
                        best_ops = info.get("ops", 0)
                        best_cid = cid
            return {"decision_type": d_type, "primary_id": best_cid, "secondary_id": 0, "flags": 0}

        # 3. SELECT_PLAY_MODE: If friendly event, play as Event (0), else Ops (1)
        if d_type == 2:  # SELECT_PLAY_MODE
            ctx = state.get("decision_context", {})
            card_id = ctx.get("pending_op_card", 0)
            if 1 <= card_id <= 110:
                info = ts_engine.CardData.get_card_info(card_id)
                if info.get("side") == self.role and 0 in valid_ids:
                    return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0}
            if 1 in valid_ids:  # OPS
                return {"decision_type": d_type, "primary_id": 1, "secondary_id": 0, "flags": 0}

        # 4. CHOOSE_TIMING_BRANCH: OPS_FIRST (0)
        if d_type == 3:  # CHOOSE_TIMING_BRANCH
            return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0}

        # 5. SELECT_OP_MODE: Prefer Influence (0) unless DEFCON allows safe Coup (1)
        if d_type == 4:  # SELECT_OP_MODE
            if 0 in valid_ids:  # INFLUENCE
                return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0}

        # 6. POINT_NODE: Prefer controlling new battlegrounds, avoid overcontrol
        if d_type == 5:  # POINT_NODE
            best_node = None
            best_score = -999999
            countries = state.get("countries", {})
            for node_id in valid_ids:
                if 0 <= node_id < 84:
                    c_info = ts_engine.MapData.get_country_info(node_id)
                    is_bg = c_info.get("battleground", False)
                    stab = c_info.get("stability", 2)
                    c_data = countries.get(str(node_id), {}) if isinstance(countries, dict) else {}
                    my_inf = c_data.get("us_influence" if self.role == "US" else "ussr_influence", 0)
                    opp_inf = c_data.get("ussr_influence" if self.role == "US" else "us_influence", 0)
                    my_ctrl = (my_inf >= opp_inf + stab)
                    deficit = max(stab - my_inf, opp_inf + stab - my_inf)

                    if is_bg:
                        if not my_ctrl:
                            score = 1000 - deficit * 10 - stab
                        elif my_inf == opp_inf + stab:
                            score = 500 - stab
                        else:
                            score = 10 - (my_inf - opp_inf - stab) * 5
                    else:
                        if not my_ctrl:
                            score = 300 - deficit * 10 - stab
                        elif my_inf == opp_inf + stab:
                            score = 100
                        else:
                            score = 5 - (my_inf - opp_inf - stab) * 5

                    if score > best_score:
                        best_score = score
                        best_node = node_id

            if best_node is not None:
                return {"decision_type": d_type, "primary_id": best_node, "secondary_id": 0, "flags": 0}

        # Fallback to random
        return {
            "decision_type": d_type,
            "primary_id": random.choice(valid_ids),
            "secondary_id": 0,
            "flags": 0,
        }
