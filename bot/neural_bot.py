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
from bindings.action_encoder import ActionEncoder
from tools.lib.player_agent import reject_retired_architecture
from bot.base_bot import BaseBot
from web.server.replay_types import ReplayPolicyDict

logger = logging.getLogger(__name__)


class NeuralBot(BaseBot):
    """Neural network player driven by ColdWarNet (NashPG/BC weights).

    The architecture is detected from the checkpoint's own weights, so V1 and V2 both load.
    Retired architectures are refused outright by `reject_retired_architecture` rather than
    partially loaded, which would otherwise run a network that is not the one that trained.
    """

    def __init__(
        self,
        role: str,
        model_path: Optional[str] = None,
        device: str = "cpu",
        temperature: float = 0.3,
        name: Optional[str] = None,
        trace: bool = False,
    ):
        super().__init__(role, name=name)
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.temperature = temperature
        #: When set, `last_readout` carries what the policy believed at the node it just played,
        #: for a caller that records replays. Off by default: a live game must not hand a policy
        #: distribution to anyone, since it is a read on this bot's own hand.
        self.trace = trace
        self.last_readout: Optional["ReplayPolicyDict"] = None

        if model_path and os.path.exists(model_path):
            weights_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            reject_retired_architecture(weights_dict)
            if any("cross_attn" in k or "cross_card_proj" in k for k in weights_dict.keys()):
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

        # Cleared up front, not only on the path that sets it: the early returns below answer
        # without consulting the network, and a readout left over from the previous node would
        # be recorded against this one -- a distribution describing a different decision.
        self.last_readout = None

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
                if vid < 5:
                    mask[ActionEncoder.PLAY_MODE_OFFSET + vid] = 1
            elif d_type == int(ts.DecisionType.SELECT_OP_MODE):
                # The deferred Ops choice shares the three OPS_* resolution slots.
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

        # The engine's own observation, or nothing. There used to be a fallback here that
        # rebuilt the observation in Python across ~4300 fields when an older server sent none.
        # It reproduced the retired legacy layout, which no model reads any more, and it never
        # reproduced it exactly even then -- replays generated through that path ended on turn 1
        # while the same checkpoint played to turn 10 through the engine extractor. A duplicate
        # encoder that can only be wrong is worse than no encoder.
        engine_obs = state.get("observation_b64")
        if not isinstance(engine_obs, str) or not engine_obs:
            raise ValueError(
                "the server sent no observation_b64. A neural bot needs the engine's own "
                "observation: it is the only thing that produces the layout the policy was "
                "trained on.")
        decoded = np.frombuffer(base64.b64decode(engine_obs), dtype=np.float32)
        want = int(ts.OBS_SIZE)
        if decoded.size != want:
            raise ValueError(
                f"observation_b64 carried {decoded.size} floats, not {want}. A policy reads "
                f"fixed slices, so the wrong width is misread rather than rejected -- most "
                f"likely the server is running a different engine.")
        return self._select_from_observation(decoded, mask, d_type, valid_ids,
                                             allow_early_stop)


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

        if self.trace:
            from ai.eval.policy_readout import read_policy
            # Same sampling, one pass, plus the distribution it came from. The action names are
            # filled in by the caller, which has the engine state this bot only sees as JSON.
            chosen_flat, self.last_readout = read_policy(
                self.model, obs_t, mask_t, temperature=self.temperature, deterministic=False)
        else:
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
