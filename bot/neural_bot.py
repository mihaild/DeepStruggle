"""NeuralBot: Deep Reinforcement Learning Player Client for Twilight Struggle."""

import base64
import logging
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
from ai.models.coldwar_net_v2 import ColdWarNetV2, create_coldwar_net_v2
from ai.models.coldwar_net_v3 import ColdWarNetV3, create_coldwar_net_v3
from ai.models.coldwar_net_v4 import ColdWarNetV4, create_coldwar_net_v4
from bindings.action_encoder import ActionEncoder
from bot.base_bot import BaseBot

logger = logging.getLogger(__name__)


class NeuralBot(BaseBot):
    """Neural network player driven by ColdWarNet (NashPG/BC weights, supporting V1, V2, and V3 architectures)."""

    def __init__(
        self,
        role: str,
        model_path: Optional[str] = None,
        device: str = "cpu",
        temperature: float = 0.3,
        name: Optional[str] = None,
    ):
        super().__init__(role, name=name)
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.temperature = temperature

        if model_path and os.path.exists(model_path):
            weights_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            if any("belief_head" in k or "card_transformer" in k for k in weights_dict.keys()):
                self.model = create_coldwar_net_v4(self.device)
            elif any("node_pointer_proj" in k or "cross_b2c" in k for k in weights_dict.keys()):
                self.model = create_coldwar_net_v3(self.device)
            elif any("cross_attn" in k or "cross_card_proj" in k for k in weights_dict.keys()):
                self.model = create_coldwar_net_v2(self.device)
            else:
                self.model = create_coldwar_net(self.device)
            self.model.load_state_dict(weights_dict)
            print(f"[NeuralBot] Loaded trained checkpoint ({type(self.model).__name__}) from {model_path}")
        else:
            self.model = create_coldwar_net(self.device)
            if model_path:
                print(f"[NeuralBot] Warning: checkpoint {model_path} not found; using initialized model.")
        self.model.eval()

    def select_action(self, state: Dict[str, Any], legal_actions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Selects micro-action using ColdWarNet forward inference."""
        valid_ids = legal_actions.get("valid_ids", [])
        d_type = legal_actions.get("decision_type", 0)
        allow_early_stop = legal_actions.get("allow_early_stop", False)

        if not valid_ids:
            # The engine guarantees CONFIRM_DONE is legal whenever no other action is
            # (its own anti-deadlock fallback), regardless of allow_early_stop.
            return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0x80}

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

        # Prefer the engine's own observation when the server supplies it. The Python
        # reconstruction below duplicates Observation::extract across ~4300 fields, and a
        # policy fed even a slightly different encoding than it was trained on plays close
        # to randomly -- replays generated through this path once ended on turn 1 while the
        # same checkpoint played to turn 10 through the engine extractor. The fallback is
        # kept only so an older server still works.
        engine_obs = state.get("observation_b64")
        if isinstance(engine_obs, str) and engine_obs:
            try:
                decoded = np.frombuffer(base64.b64decode(engine_obs), dtype=np.float32)
                if decoded.size == 4293:
                    return self._select_from_observation(decoded, mask, d_type, valid_ids,
                                                         allow_early_stop)
                logger.warning("observation_b64 had %d floats, expected 4293; "
                               "falling back to local reconstruction", decoded.size)
            except Exception as exc:
                logger.warning("could not decode observation_b64 (%s); "
                               "falling back to local reconstruction", exc)
        else:
            logger.debug("server sent no observation_b64; using local reconstruction, "
                         "which may drift from the engine")

        # Fallback: rebuild the observation from the state dict.
        obs = np.zeros(4293, dtype=np.float32)
        my_is_us = (self.role == "US")
        side_sign = 1.0 if my_is_us else -1.0

        opp_role = "USSR" if my_is_us else "US"
        countries = state.get("countries", {})
        c_list = countries.values() if isinstance(countries, dict) else countries
        for c_data in c_list:
            cid = c_data.get("id", 0)
            if 0 <= cid < 84:
                offset = cid * 28
                us_inf = float(c_data.get("us_influence", 0))
                ussr_inf = float(c_data.get("ussr_influence", 0))
                my_inf = us_inf if my_is_us else ussr_inf
                opp_inf = ussr_inf if my_is_us else us_inf
                stab = max(1.0, float(c_data.get("stability", 1)))

                obs[offset + 0] = my_inf / 10.0
                obs[offset + 1] = opp_inf / 10.0
                obs[offset + 2] = (my_inf - opp_inf) / 10.0
                obs[offset + 3] = stab / 5.0
                obs[offset + 4] = 1.0 if c_data.get("battleground", False) else 0.0

                controlled_by = c_data.get("controlled_by", "NONE")
                obs[offset + 5] = 1.0 if (controlled_by == self.role) else 0.0
                obs[offset + 6] = 1.0 if (controlled_by not in (self.role, "NONE")) else 0.0
                obs[offset + 7] = 1.0 if (controlled_by == "NONE") else 0.0

                # 26 & 27: Influence deficits to control
                my_def = max(0.0, stab - my_inf, opp_inf + stab - my_inf)
                opp_def = max(0.0, stab - opp_inf, my_inf + stab - opp_inf)
                obs[offset + 26] = min(my_def / 5.0, 2.0)
                obs[offset + 27] = min(opp_def / 5.0, 2.0)

        # Populate Card features: offset 2352..3671 (110 * 12 features)
        hands = state.get("hands", {})
        my_hand = set(hands.get(self.role, []))
        opp_hand = set(hands.get(opp_role, []))
        discard_pile = set(state.get("discard_pile", []))
        removed_cards = set(state.get("removed_cards", []))
        card_locs = state.get("card_locations", {})

        for cid in range(1, 111):
            c_offset = 2352 + (cid - 1) * 12
            c_info = ts.CardData.get_card_info(cid)

            if cid in my_hand or (isinstance(card_locs, dict) and card_locs.get(str(cid)) == ("HAND_US" if my_is_us else "HAND_USSR")):
                canon_loc = 1
            elif cid in discard_pile or (isinstance(card_locs, dict) and card_locs.get(str(cid)) == "DISCARD_PILE"):
                canon_loc = 3
            elif cid in removed_cards or (isinstance(card_locs, dict) and card_locs.get(str(cid)) == "REMOVED_FROM_GAME"):
                canon_loc = 4
            else:
                canon_loc = 0

            obs[c_offset + canon_loc] = 1.0
            ops = float(c_info.get("ops", 0))
            side_str = c_info.get("side", "NEUTRAL")
            rel_side = 1.0 if side_str == self.role else (-1.0 if side_str == opp_role else 0.0)
            era_val = 0.0 if c_info.get("era") == "EARLY" else (0.5 if c_info.get("era") == "MID" else 1.0)

            obs[c_offset + 7] = ops / 4.0
            obs[c_offset + 8] = rel_side
            obs[c_offset + 9] = era_val
            obs[c_offset + 10] = 1.0 if c_info.get("one_time", False) else 0.0
            obs[c_offset + 11] = 1.0 if c_info.get("is_scoring", False) else 0.0

        raw_vp = float(state.get("victory_points", 0))
        my_vp = raw_vp if my_is_us else -raw_vp
        obs[3672 + 0] = my_vp / 20.0
        obs[3672 + 1] = float(state.get("defcon", 5)) / 5.0

        my_mil = float(state.get("mil_ops", {}).get(self.role, 0))
        opp_mil = float(state.get("mil_ops", {}).get("USSR" if my_is_us else "US", 0))
        obs[3672 + 2] = my_mil / 5.0
        obs[3672 + 3] = opp_mil / 5.0

        obs[3672 + 6] = float(state.get("turn", 1)) / 10.0
        obs[3672 + 7] = float(state.get("action_round", 0)) / 8.0

        obs[3672 + 61] = 1.0 if my_is_us else 0.0
        obs[3672 + 62] = 1.0 if not my_is_us else 0.0
        obs[3672 + 63] = side_sign
        obs[3672 + 70] = float(len(opp_hand)) / 10.0
        obs[4292] = side_sign

        return self._select_from_observation(obs, mask, d_type, valid_ids, allow_early_stop)

    def _select_from_observation(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        d_type: int,
        valid_ids: Any,
        allow_early_stop: bool,
    ) -> Optional[Dict[str, Any]]:
        """Runs the policy on an observation and decodes the flat action for this node."""
        obs_t = torch.from_numpy(np.array(obs, dtype=np.float32, copy=True)).float().unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            actions_t, _, _, _, _ = self.model.sample_action(
                obs_t, mask_t, temperature=self.temperature, deterministic=False
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

        return {"decision_type": d_type, "primary_id": valid_ids[0], "secondary_id": 0, "flags": 0}
