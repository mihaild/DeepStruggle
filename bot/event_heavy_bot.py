"""EventHeavyBot: Heuristic agent maximizing card event play and event-first timing."""

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


class EventHeavyBot(BaseBot):
    """Agent prioritizing card events over operations and choosing EVENT_FIRST timing on opponent cards."""

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
        ctx = state_dict.get("decision_context", {})

        turn = state_dict.get("turn", 1)
        phase = state_dict.get("phase", state_dict.get("current_phase", 0))
        ar = state_dict.get("action_round", 0)
        defcon = state_dict.get("defcon", 5)
        vp = state_dict.get("victory_points", 0)

        # 1. SELECT_CARD
        if d_type == 1:
            res_card = ctx.get("resolving_card", 0)
            if res_card > 0:
                chosen = valid_ids[0] if valid_ids else 0
                return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, f"Resolving #{res_card}", "Resolving event sub-decision."

            # Headline: prefer friendly high-impact events
            if phase == 1:
                friendly = [cid for cid in valid_ids if 1 <= cid <= 110 and ts_engine.CardData.get_card_info(cid).get("side") == self.role]
                chosen = friendly[0] if friendly else (valid_ids[0] if valid_ids else 0)
                c_name = ts_engine.CardData.get_card_name(chosen) if 1 <= chosen <= 110 else f"#{chosen}"
                return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, f"Headline Event: '{c_name}'", f"Deploying headline event '{c_name}'."

            # Action round: prefer friendly events or highest ops
            friendly = [cid for cid in valid_ids if 1 <= cid <= 110 and ts_engine.CardData.get_card_info(cid).get("side") == self.role]
            scoring = [cid for cid in valid_ids if 1 <= cid <= 110 and ts_engine.CardData.get_card_info(cid).get("is_scoring", False)]
            if scoring:
                chosen = scoring[0]
            elif friendly:
                chosen = friendly[0]
            else:
                chosen = valid_ids[0] if valid_ids else 0

            c_name = ts_engine.CardData.get_card_name(chosen) if 1 <= chosen <= 110 else f"#{chosen}"
            return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, f"Event-Heavy Selection: #{chosen} '{c_name}'", f"Playing '{c_name}'."

        # 2. SELECT_PLAY_MODE: Always prefer EVENT (0)
        elif d_type == 2:
            card_id = ctx.get("pending_op_card", 0)
            c_name = ts_engine.CardData.get_card_name(card_id) if 1 <= card_id <= 110 else f"#{card_id}"
            if 0 in valid_ids:
                return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0}, f"Play as Event: '{c_name}'", f"Triggering event for '{c_name}'."
            if 1 in valid_ids:
                return {"decision_type": d_type, "primary_id": 1, "secondary_id": 0, "flags": 0}, f"Play as Ops: '{c_name}'", f"Using '{c_name}' for operations."
            chosen = valid_ids[0] if valid_ids else 0
            return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, f"Mode {chosen}", "Selecting mode."

        # 3. CHOOSE_TIMING_BRANCH: Always choose EVENT_FIRST (1)
        elif d_type == 3:
            chosen = 1 if 1 in valid_ids else (0 if 0 in valid_ids else (valid_ids[0] if valid_ids else 0))
            strat = "Timing: EVENT_FIRST (1)." if chosen == 1 else "Timing: OPS_FIRST (0)."
            return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, strat, "Resolving event timing branch."

        # 4. SELECT_OP_MODE: Prefer Influence (0)
        elif d_type == 4:
            chosen = 0 if 0 in valid_ids else (valid_ids[0] if valid_ids else 0)
            return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, f"Op Mode {chosen}", "Conducting operations."

        # 5. POINT_NODE
        elif d_type == 5:
            if not valid_ids:
                # The engine guarantees CONFIRM_DONE is legal whenever no other action is
                # (its own anti-deadlock fallback), regardless of allow_early_stop -- flags=0
                # here would be a real (illegal) target country, not a pass.
                return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 128}, "Confirm Done", "Passing."
            chosen = self.rng.choice(valid_ids)
            c_name = ts_engine.MapData.get_country_name(chosen) if chosen < 84 else f"#{chosen}"
            return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, f"Targeting {c_name}", f"Placing on {c_name}."

        chosen = valid_ids[0] if valid_ids else 0
        return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}, f"Action {chosen}", "Executing action."
