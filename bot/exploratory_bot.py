"""ExploratoryBot: Baseline agent exploring diverse decision paths, edge cases, space race, and coups."""

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


class ExploratoryBot(BaseBot):
    """AI agent designed to explore diverse decision paths, Space Race, Realignments, and Coups."""

    def __init__(self, role: str, rng_seed: Optional[int] = None, name: Optional[str] = None):
        super().__init__(role, name=name)
        self.rng = random.Random(rng_seed)

    def select_action(self, state: Dict[str, Any], legal_actions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        d_type = legal_actions.get("decision_type", 0)
        valid_ids = legal_actions.get("valid_ids", [])
        allow_early_stop = legal_actions.get("allow_early_stop", False)
        ctx = state.get("decision_context", {})

        if not valid_ids:
            # The engine guarantees CONFIRM_DONE is legal whenever no other action is
            # (its own anti-deadlock fallback), regardless of allow_early_stop -- flags=0
            # here would be a real (illegal) target/card/mode, not a pass.
            return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0x80}

        # 1. SELECT_CARD (1)
        if d_type == 1:
            # Randomly select among legal cards, with a preference to score held scoring cards
            scoring_cards = [
                cid
                for cid in valid_ids
                if 1 <= cid <= 110 and ts_engine.CardData.get_card_info(cid).get("is_scoring", False)
            ]
            if scoring_cards and self.rng.random() < 0.4:
                chosen = self.rng.choice(scoring_cards)
            else:
                chosen = self.rng.choice(valid_ids)
            return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}

        # 2. SELECT_PLAY_MODE (2)
        elif d_type == 2:
            card = ctx.get("pending_op_card", 0)
            is_scoring = (
                ts_engine.CardData.get_card_info(card).get("is_scoring", False) if 1 <= card <= 110 else False
            )
            if is_scoring:
                return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0}  # EVENT

            # Bias: 60% Ops, 25% Event, 15% Space -- unchanged in total. P17 merged the Ops
            # mode into this node, so the 60 is split across the three Ops resolutions by the
            # same 45/35/20 this bot uses at SELECT_OP_MODE, which keeps the joint distribution
            # over (play mode, op mode) exactly what it was before the merge.
            weights = []
            for m in valid_ids:
                if m == 0:                      # EVENT
                    weights.append(25)
                elif m == 1:                    # SPACE
                    weights.append(15)
                elif m == 2:                    # OPS_INFLUENCE  (60 * 0.45)
                    weights.append(27)
                elif m == 3:                    # OPS_COUP       (60 * 0.35)
                    weights.append(21)
                elif m == 4:                    # OPS_REALIGN    (60 * 0.20)
                    weights.append(12)
                else:
                    weights.append(10)
            chosen = self.rng.choices(valid_ids, weights=weights, k=1)[0]
            sec_id = self.rng.randint(1, 6) if chosen == 1 else 0  # Space roll
            return {"decision_type": d_type, "primary_id": chosen, "secondary_id": sec_id, "flags": 0}

        # 4. SELECT_OP_MODE (4)
        elif d_type == 4:
            # 0 = INFLUENCE, 1 = COUP, 2 = REALIGN
            # Bias: 45% Influence, 35% Coup, 20% Realign
            weights = []
            for m in valid_ids:
                if m == 0:
                    weights.append(45)
                elif m == 1:
                    weights.append(35)
                elif m == 2:
                    weights.append(20)
                else:
                    weights.append(10)
            chosen = self.rng.choices(valid_ids, weights=weights, k=1)[0] if valid_ids else 0
            return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}

        # 5. POINT_NODE (5)
        elif d_type == 5:
            # Check if stopping early (e.g. 5% chance if allowed); valid_ids is non-empty here,
            # the empty case having already returned CONFIRM_DONE above.
            if allow_early_stop and self.rng.random() < 0.05:
                return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0x80}

            chosen_id = self.rng.choice(valid_ids)
            roll1 = self.rng.randint(1, 6)
            roll2 = self.rng.randint(1, 6)
            return {"decision_type": d_type, "primary_id": chosen_id, "secondary_id": roll1, "flags": roll2}

        # 6. CHOOSE_BRANCH (6) / Other
        else:
            chosen = self.rng.choice(valid_ids) if valid_ids else 0
            return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}
