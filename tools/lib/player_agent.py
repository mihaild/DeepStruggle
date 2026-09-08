"""Player Agent Abstractions for Twilight Struggle Bots and Neural Models."""

from typing import Protocol, Optional, Dict, Any, Union, cast
import os
import numpy as np
import torch
import torch.nn as nn

import ts_engine as ts
from bindings.action_encoder import ActionEncoder
from ai.training.behavioral_cloning import HeuristicPolicy, OldHeuristicPolicy
from ai.models.coldwar_net import ColdWarNet, create_coldwar_net
from ai.models.coldwar_net_v2 import ColdWarNetV2, create_coldwar_net_v2
from ai.models.coldwar_net_v3 import ColdWarNetV3, create_coldwar_net_v3
from ai.models.coldwar_net_v4 import ColdWarNetV4, create_coldwar_net_v4

ColdWarModel = Union[ColdWarNet, ColdWarNetV2, ColdWarNetV3, ColdWarNetV4]


def load_checkpoint_into(model: nn.Module, state_dict: Dict[str, Any]) -> None:
    """Load a checkpoint, tolerating an absent auxiliary DEFCON-risk head.

    The head was added after these checkpoints were written, so their state dicts have no
    weights for it. Everything else must still match exactly: only keys belonging to the
    aux head may be missing, and unexpected keys are never allowed.
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    stale = [k for k in missing if not k.startswith("defcon_risk_head.")]
    if stale or unexpected:
        raise RuntimeError(
            f"checkpoint does not match {type(model).__name__}: "
            f"missing {stale}, unexpected {list(unexpected)}"
        )


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


class HeuristicV2Agent:
    """HeuristicBot with an instant-win / instant-loss safety layer.

    The v1 policy walks into forced losses: measured on constructed positions it plays
    Duck and Cover for Ops at DEFCON 2 with probability 1.000, and Ortega for Ops with
    probability 1.000, both of which end the game against it on the spot.

    That matters beyond the bot's own strength. HeuristicBot is the Elo anchor and a
    self-play reference, so an opponent that neither commits nor punishes these mistakes
    makes them invisible to training and to evaluation alike.

    This wrapper changes nothing else: it takes a forced win when one exists, removes
    losing moves from consideration otherwise, and delegates every remaining decision to
    the same v1 policy.
    """

    def __init__(self, name: str = "HeuristicBotV2", avoid_risky: bool = True):
        self.name = name
        self.avoid_risky = avoid_risky

    def select_action(
        self,
        state: ts.GameState,
        player: ts.Player,
        temperature: float = 0.1,
    ) -> int:
        from ai.eval.safety import find_instant_win, safe_actions

        win = find_instant_win(state, player)
        if win is not None:
            return int(win)

        choice = int(HeuristicPolicy.select_action(state))
        allowed = safe_actions(state, player, avoid_risky=self.avoid_risky)
        if choice in allowed or not allowed:
            return choice
        # The policy picked a losing move. Re-ask it with the losing options masked out, so
        # the fallback still reflects its preferences rather than an arbitrary legal move.
        return int(HeuristicPolicy.select_action_restricted(state, allowed))


class NeuralAgent:
    """Neural network agent supporting ColdWarNet (V1, V2, V3, V4)."""

    model: ColdWarModel

    def __init__(
        self,
        model: Union[ColdWarModel, nn.Module],
        name: str = "NeuralBot",
        device: Optional[Union[torch.device, str]] = None,
    ):
        self.device = resolve_device(device) if device is not None else next(model.parameters()).device
        self.model = cast(ColdWarModel, model.to(self.device))
        self.model.eval()
        self.name = name

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        name: Optional[str] = None,
        device: Union[torch.device, str] = "cuda",
    ) -> "NeuralAgent":
        """Loads a NeuralAgent from checkpoint with automatic architecture detection."""
        dev = resolve_device(device)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=dev, weights_only=True)
        # Architecture detection: V4 contains belief_head/card_transformer, V3 contains node_pointer_proj, V2 contains cross_attn
        is_v4 = any("belief_head" in k or "card_transformer" in k for k in state_dict.keys())
        is_v3 = any("node_pointer_proj" in k or "cross_b2c" in k for k in state_dict.keys())
        is_v2 = any("cross_attn" in k or "cross_card_proj" in k for k in state_dict.keys())

        model: ColdWarModel
        if is_v4:
            model = create_coldwar_net_v4(dev)
        elif is_v3:
            model = create_coldwar_net_v3(dev)
        elif is_v2:
            # Both the card block width and the presence of the history branch are read off the
            # checkpoint's own weights rather than assumed. The two layouts are indistinguishable
            # from a filename, and building at defaults raises a shape error for a v2.1 policy --
            # or worse, would silently reinterpret every card feature by one position if the
            # widths happened to match.
            card_features = int(state_dict["card_fc.0.weight"].shape[1]) \
                if "card_fc.0.weight" in state_dict else ColdWarNetV2.CARD_FEATURES
            use_history = any(k.startswith("hist_conv.") for k in state_dict)
            model = create_coldwar_net_v2(dev, card_features=card_features,
                                          use_history=use_history)
        else:
            model = create_coldwar_net(dev)

        load_checkpoint_into(model, state_dict)
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


def load_agent(spec: str, device: Union[torch.device, str] = "cuda") -> PlayerAgent:
    """Factory function loading agents from string specifier (random, heuristic, or checkpoint path)."""
    s = spec.strip()
    if s.lower() in ["random", "randombot", "rand"]:
        return RandomAgent()
    if s.lower() in ["old_heuristic", "old_heuristicbot", "old_heur", "oldheuristic"]:
        return OldHeuristicAgent()
    if s.lower() in ["heuristic", "heuristicbot", "heur", "new_heuristic", "new_heuristicbot"]:
        return HeuristicAgent()
    if s.lower() in ["heuristic_v2", "heuristicbotv2", "heuristicv2", "heur2"]:
        return HeuristicV2Agent()
    return NeuralAgent.from_checkpoint(s, device=device)
