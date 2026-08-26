def resolve_device(device: Optional[Union[torch.device, str]] = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device
    if isinstance(device, str):
        if "cuda" in device and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device)
    return torch.device("cpu")

"""Player Agent Abstractions for Twilight Struggle Bots and Neural Models."""

from typing import Protocol, Optional, Dict, Any, Union
import os
import numpy as np
import torch
import torch.nn as nn

import ts_engine as ts
from ai.env.action_encoder import ActionEncoder
from ai.training.behavioral_cloning import HeuristicPolicy, OldHeuristicPolicy
from ai.models.coldwar_net import ColdWarNet, create_coldwar_net
from ai.models.coldwar_net_v2 import ColdWarNetV2, create_coldwar_net_v2


class PlayerAgent(Protocol):
    """Abstract player interface for action selection."""

    name: str

    def select_action(
        self,
        state: ts.GameState,
        player: ts.Player,
        temperature: float = 0.1,
    ) -> int:
        """Selects a legal flat action index [0..211]."""
        ...


class RandomAgent:
    """Uniformly samples from legal action mask."""

    def __init__(self, name: str = "RandomBot"):
        self.name = name

    def select_action(
        self,
        state: ts.GameState,
        player: ts.Player,
        temperature: float = 0.1,
    ) -> int:
        mask = ActionEncoder.get_legal_mask(state)
        legal = np.where(mask > 0)[0]
        if len(legal) == 0:
            return ActionEncoder.CONFIRM_DONE_INDEX
        return int(np.random.choice(legal))


class OldHeuristicAgent:
    """Legacy rule-based heuristic baseline player (prior to overcontrol prevention)."""

    def __init__(self, name: str = "OldHeuristicBot"):
        self.name = name

    def select_action(
        self,
        state: ts.GameState,
        player: ts.Player,
        temperature: float = 0.1,
    ) -> int:
        return int(OldHeuristicPolicy.select_action(state))


class HeuristicAgent:
    """Rule-based heuristic bot prioritizing Battlegrounds, DEFCON safety, and scoring timing."""

    def __init__(self, name: str = "HeuristicBot"):
        self.name = name

    def select_action(
        self,
        state: ts.GameState,
        player: ts.Player,
        temperature: float = 0.1,
    ) -> int:
        return int(HeuristicPolicy.select_action(state))


class NeuralAgent:
    """Neural network agent supporting ColdWarNet (V1) and ColdWarNetV2 (Cross-Attention)."""

    def __init__(
        self,
        model: nn.Module,
        name: str = "NeuralBot",
        device: Optional[torch.device | str] = None,
    ):
        self.device = resolve_device(device) if device is not None else next(model.parameters()).device
        self.model = model.to(self.device)
        self.model.eval()
        self.name = name

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        name: Optional[str] = None,
        device: torch.device | str = "cuda",
    ) -> "NeuralAgent":
        """Loads a NeuralAgent from checkpoint with automatic architecture detection."""
        dev = resolve_device(device)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=dev, weights_only=True)
        # Architecture detection: ColdWarNetV2 contains country_cross_attn
        is_v2 = any("cross_attn" in k or "cross_card_proj" in k for k in state_dict.keys())

        if is_v2:
            model = create_coldwar_net_v2(dev)
        else:
            model = create_coldwar_net(dev)

        model.load_state_dict(state_dict)
        model.to(dev)
        agent_name = name or os.path.splitext(os.path.basename(checkpoint_path))[0]
        return cls(model=model, name=agent_name, device=dev)

    def select_action(
        self,
        state: ts.GameState,
        player: ts.Player,
        temperature: float = 0.1,
    ) -> int:
        obs = ts.extract_observation(state, player)
        mask = ActionEncoder.get_legal_mask(state)

        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_t, _, _, _, _ = self.model.sample_action(obs_t, mask_t, temperature=temperature, deterministic=(temperature <= 0.05))
        return int(action_t.item())


def load_agent(spec: str, device: torch.device | str = "cuda") -> PlayerAgent:
    """Factory function loading agents from string specifier (random, heuristic, or checkpoint path)."""
    s = spec.strip()
    if s.lower() in ["random", "randombot", "rand"]:
        return RandomAgent()
    if s.lower() in ["old_heuristic", "old_heuristicbot", "old_heur", "oldheuristic"]:
        return OldHeuristicAgent()
    if s.lower() in ["heuristic", "heuristicbot", "heur", "new_heuristic", "new_heuristicbot"]:
        return HeuristicAgent()
    return NeuralAgent.from_checkpoint(s, device=device)
