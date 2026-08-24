"""NeuralBot: Deep Reinforcement Learning Player Client for Twilight Struggle."""

import os
import sys
from typing import Dict, Optional, Any
import numpy as np
import torch

try:
    import ts_engine as ts
except ImportError:
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    import ts_engine as ts

from ai.models.coldwar_net import ColdWarNet, create_coldwar_net
from ai.env.action_encoder import ActionEncoder
from bot.bot_client import BaseBot


class NeuralBot(BaseBot):
    """Neural network player driven by ColdWarNet (NashPG/BC weights)."""

    def __init__(
        self,
        role: str,
        model_path: Optional[str] = None,
        device: str = "cuda",
        temperature: float = 0.05,
    ):
        super().__init__(role)
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = create_coldwar_net(self.device)
        self.temperature = temperature

        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"[NeuralBot] Loaded trained checkpoint from {model_path}")
        else:
            if model_path:
                print(f"[NeuralBot] Warning: checkpoint {model_path} not found; using initialized model.")
        self.model.eval()

    def select_action(self, state_dict: dict, legal_actions_dict: dict) -> Optional[dict]:
        """Selects micro-action using ColdWarNet forward inference."""
        valid_ids = legal_actions_dict.get("valid_ids", [])
        d_type = legal_actions_dict.get("decision_type", 0)
        allow_early_stop = legal_actions_dict.get("allow_early_stop", False)

        if not valid_ids and not allow_early_stop:
            return None

        # Build ts.GameState from session state dict or reconstruct
        # For direct GameState evaluation, we can parse state or use session GameState
        # Let's construct state or use legal mask
        p = ts.Player.US if self.role == "US" else ts.Player.USSR

        # Fallback to legal action choice if state serialization is partial
        if not valid_ids:
            if allow_early_stop:
                return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0x80}
            return None

        # If choice is unique, return immediately
        if len(valid_ids) == 1 and not allow_early_stop:
            return {"decision_type": d_type, "primary_id": valid_ids[0], "secondary_id": 0, "flags": 0}

        # Build flat mask for valid IDs
        mask = np.zeros(212, dtype=np.uint8)
        for vid in valid_ids:
            if d_type == int(ts.DecisionType.SELECT_CARD):
                if 1 <= vid <= 110:
                    mask[vid - 1] = 1
                elif vid == 0:
                    mask[ActionEncoder.CONFIRM_DONE_INDEX] = 1
            elif d_type == int(ts.DecisionType.SELECT_PLAY_MODE):
                if vid < 4:
                    mask[ActionEncoder.PLAY_MODE_OFFSET + vid] = 1
            elif d_type == int(ts.DecisionType.CHOOSE_TIMING_BRANCH):
                if vid < 2:
                    mask[ActionEncoder.TIMING_OFFSET + vid] = 1
            elif d_type == int(ts.DecisionType.SELECT_OP_MODE):
                if vid < 3:
                    mask[ActionEncoder.OP_MODE_OFFSET + vid] = 1
            elif d_type == int(ts.DecisionType.POINT_NODE):
                if vid < 84:
                    mask[ActionEncoder.NODE_OFFSET + vid] = 1
            elif d_type == int(ts.DecisionType.CHOOSE_BRANCH):
                if vid < 8:
                    mask[ActionEncoder.BRANCH_OFFSET + vid] = 1

        if allow_early_stop:
            mask[ActionEncoder.CONFIRM_DONE_INDEX] = 1

        # Build approximate/direct observation from state dict
        obs = np.zeros(4293, dtype=np.float32)

        # Global features
        obs[3672 + 0] = float(state_dict.get("victory_points", 0)) / 20.0
        obs[3672 + 1] = float(state_dict.get("defcon", 5)) / 5.0
        obs[3672 + 2] = float(state_dict.get("us_mil_ops", 0)) / 5.0
        obs[3672 + 3] = float(state_dict.get("ussr_mil_ops", 0)) / 5.0
        obs[3672 + 6] = float(state_dict.get("turn", 1)) / 10.0
        obs[3672 + 7] = float(state_dict.get("action_round", 0)) / 8.0
        obs[3672 + 8] = 1.0 if self.role == "US" else -1.0
        obs[4292] = 1.0 if self.role == "US" else -1.0

        # Country features
        countries = state_dict.get("countries", [])
        for c in countries:
            cid = c.get("id", 0)
            if 0 <= cid < 84:
                offset = cid * 28
                us_inf = float(c.get("us_influence", 0))
                ussr_inf = float(c.get("ussr_influence", 0))
                obs[offset + 0] = us_inf / 10.0
                obs[offset + 1] = ussr_inf / 10.0
                obs[offset + 2] = (us_inf - ussr_inf) / 10.0

        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            actions_t, _, _, _, _ = self.model.sample_action(
                obs_t, mask_t, temperature=self.temperature, deterministic=(self.temperature == 0)
            )

        chosen_flat = int(actions_t.item())

        if chosen_flat == ActionEncoder.CONFIRM_DONE_INDEX:
            return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0x80}

        if d_type == int(ts.DecisionType.SELECT_CARD):
            card_id = chosen_flat + 1
            return {"decision_type": d_type, "primary_id": card_id, "secondary_id": 0, "flags": 0}

        if d_type == int(ts.DecisionType.SELECT_PLAY_MODE):
            mode_id = chosen_flat - ActionEncoder.PLAY_MODE_OFFSET
            return {"decision_type": d_type, "primary_id": mode_id, "secondary_id": 0, "flags": 0}

        if d_type == int(ts.DecisionType.CHOOSE_TIMING_BRANCH):
            timing_id = chosen_flat - ActionEncoder.TIMING_OFFSET
            return {"decision_type": d_type, "primary_id": timing_id, "secondary_id": 0, "flags": 0}

        if d_type == int(ts.DecisionType.SELECT_OP_MODE):
            op_id = chosen_flat - ActionEncoder.OP_MODE_OFFSET
            return {"decision_type": d_type, "primary_id": op_id, "secondary_id": 0, "flags": 0}

        if d_type == int(ts.DecisionType.POINT_NODE):
            country_id = chosen_flat - ActionEncoder.NODE_OFFSET
            return {"decision_type": d_type, "primary_id": country_id, "secondary_id": 0, "flags": 0}

        if d_type == int(ts.DecisionType.CHOOSE_BRANCH):
            branch_id = chosen_flat - ActionEncoder.BRANCH_OFFSET
            return {"decision_type": d_type, "primary_id": branch_id, "secondary_id": 0, "flags": 0}

        # Fallback
        return {"decision_type": d_type, "primary_id": valid_ids[0], "secondary_id": 0, "flags": 0}
