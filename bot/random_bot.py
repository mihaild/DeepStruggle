"""RandomBot: Baseline stochastic agent selecting random valid actions."""

import random
from typing import Dict, Any, Optional
from bot.base_bot import BaseBot


class RandomBot(BaseBot):
    """Uniform random decision-maker across all valid micro-action IDs."""

    def select_action(self, state: Dict[str, Any], legal_actions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        valid_ids = legal_actions.get("valid_ids", [])
        d_type = legal_actions.get("decision_type", 0)
        allow_early_stop = legal_actions.get("allow_early_stop", False)

        if not valid_ids and not allow_early_stop:
            return None

        # 15% chance to stop early if allowed
        if allow_early_stop and random.random() < 0.15:
            return {
                "decision_type": d_type,
                "primary_id": 0,
                "secondary_id": 0,
                "flags": 0x80,  # CONFIRM_DONE
            }

        if not valid_ids:
            if allow_early_stop:
                return {
                    "decision_type": d_type,
                    "primary_id": 0,
                    "secondary_id": 0,
                    "flags": 0x80,
                }
            return None

        choice = random.choice(valid_ids)
        return {
            "decision_type": d_type,
            "primary_id": choice,
            "secondary_id": 0,
            "flags": 0,
        }
