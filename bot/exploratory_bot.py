#!/usr/bin/env python3
"""
ExploratoryBot: An AI client designed to explore diverse decision paths, edge cases,
Space Race, Realignments, Coups, and complex Event sub-decisions in Twilight Struggle.
"""

import random
from typing import Dict, Any, Optional

import ts_engine

class ExploratoryBot:
    def __init__(self, side: str, rng_seed: Optional[int] = None):
        self.side = side # "US" or "USSR"
        self.rng = random.Random(rng_seed)

    def select_action(self, state_dict: dict, legal_actions: dict) -> Dict[str, Any]:
        d_type = legal_actions.get("decision_type", 0)
        valid_ids = legal_actions.get("valid_ids", [])
        allow_early_stop = legal_actions.get("allow_early_stop", False)
        ctx = state_dict.get("decision_context", {})

        if not valid_ids and not allow_early_stop:
            return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0}

        # 1. SELECT_CARD (1)
        if d_type == 1:
            # Randomly select among legal cards, with a preference to score held scoring cards
            scoring_cards = [cid for cid in valid_ids if 1 <= cid <= 110 and ts_engine.CardData.get_card_info(cid).get('is_scoring', False)]
            if scoring_cards and self.rng.random() < 0.4:
                chosen = self.rng.choice(scoring_cards)
            else:
                chosen = self.rng.choice(valid_ids)
            return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}

        # 2. SELECT_PLAY_MODE (2)
        elif d_type == 2:
            card = ctx.get("pending_op_card", 0)
            is_scoring = ts_engine.CardData.get_card_info(card).get('is_scoring', False) if 1 <= card <= 110 else False
            if is_scoring:
                return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0} # EVENT

            # Otherwise explore among EVENT (0), OPS (1), SPACE (2)
            # Bias: 60% Ops, 25% Event, 15% Space
            weights = []
            for m in valid_ids:
                if m == 1: weights.append(60)
                elif m == 0: weights.append(25)
                elif m == 2: weights.append(15)
                else: weights.append(10)
            chosen = self.rng.choices(valid_ids, weights=weights, k=1)[0]
            sec_id = self.rng.randint(1, 6) if chosen == 2 else 0 # Space roll
            return {"decision_type": d_type, "primary_id": chosen, "secondary_id": sec_id, "flags": 0}

        # 3. CHOOSE_TIMING_BRANCH (3)
        elif d_type == 3:
            # 0 = OPS_FIRST, 1 = EVENT_FIRST
            chosen = self.rng.choice(valid_ids) if valid_ids else 0
            return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}

        # 4. SELECT_OP_MODE (4)
        elif d_type == 4:
            # 0 = INFLUENCE, 1 = COUP, 2 = REALIGN
            # Bias: 45% Influence, 35% Coup, 20% Realign
            weights = []
            for m in valid_ids:
                if m == 0: weights.append(45)
                elif m == 1: weights.append(35)
                elif m == 2: weights.append(20)
                else: weights.append(10)
            chosen = self.rng.choices(valid_ids, weights=weights, k=1)[0] if valid_ids else 0
            return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}

        # 5. POINT_NODE (5)
        elif d_type == 5:
            # Check if stopping early (e.g. 5% chance if allowed and at least some placements done)
            if allow_early_stop and valid_ids and self.rng.random() < 0.05:
                return {"decision_type": d_type, "primary_id": 255, "secondary_id": 0, "flags": 0}

            if not valid_ids:
                if allow_early_stop:
                    return {"decision_type": d_type, "primary_id": 255, "secondary_id": 0, "flags": 0}
                return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0}

            chosen_id = self.rng.choice(valid_ids)
            roll1 = self.rng.randint(1, 6)
            roll2 = self.rng.randint(1, 6)
            return {"decision_type": d_type, "primary_id": chosen_id, "secondary_id": roll1, "flags": roll2}

        # 6. CHOOSE_BRANCH (6) / Other
        else:
            chosen = self.rng.choice(valid_ids) if valid_ids else 0
            return {"decision_type": d_type, "primary_id": chosen, "secondary_id": 0, "flags": 0}
