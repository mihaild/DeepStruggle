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
from ai.models.coldwar_net_v2 import (ColdWarNetV2, check_checkpoint_layout,
                                     create_coldwar_net_v2)
from bindings.ts_env import check_obs_width

ColdWarModel = Union[ColdWarNet, ColdWarNetV2]


#: Weight names that identify a retired architecture. A checkpoint carrying one of these must be
#: refused rather than fall through to the V1 branch, which would load *some* of it and run.
_RETIRED_ARCH_KEYS = {
    "belief_head": "V4 (card transformer + oracle critic + belief head)",
    "card_transformer": "V4 (card transformer + oracle critic + belief head)",
    "node_pointer_proj": "V3 (dual pointer co-attention)",
    "cross_b2c": "V3 (dual pointer co-attention)",
}


def reject_retired_architecture(state_dict: Dict[str, Any]) -> None:
    """Raises if this checkpoint was trained on an architecture that no longer exists.

    V3 and V4 were removed: neither ever produced a logged result, and both predate the
    starred-card engine fix and the single observation layout, so their checkpoints could not be
    run even if the classes were still here. The oracle critic and belief head that rode with V4
    are not maintained here; the old implementation is recoverable from version control.
    """
    for key, what in _RETIRED_ARCH_KEYS.items():
        if any(key in name for name in state_dict):
            raise ValueError(
                f"this checkpoint was trained on {what}, which has been removed. It also "
                f"predates both the starred-card engine fix and the single observation layout, "
                f"so it cannot be run.")


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
        # There is one observation layout, so there is nothing to select -- but a model whose
        # width is not the engine's is still worth catching here rather than at the first
        # forward pass, because it means a checkpoint from a retired layout.
        self.obs_size = check_obs_width(self.model)
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
        # `keep_idx` is derived from drop_static, not learned. It was briefly a *persistent*
        # buffer, so checkpoints exist both with and without it; dropping it here lets both eras
        # load. Buffers that are a pure function of the configuration should never be persisted.
        state_dict.pop("keep_idx", None)
        # Architecture detection by weight name: V2 carries the cross-attention block, V1 does
        # not. A retired architecture is refused rather than allowed to fall through to V1.
        reject_retired_architecture(state_dict)
        is_mlp = any(k.startswith("mlp_in.") for k in state_dict)
        is_v2 = is_mlp or any("cross_attn" in k or "cross_card_proj" in k for k in state_dict)

        model: ColdWarModel
        if is_v2:
            # Refuses a checkpoint from a retired layout by its own weights. A checkpoint is a
            # bare state dict and names no layout, and a model handed the wrong width does not
            # fail -- it reads fixed slices, so the observation is misread and the network merely
            # plays badly.
            check_checkpoint_layout(state_dict)
            # The value head is detected the same way the architecture is: by weight name. A
            # categorical checkpoint carries `value_dist_head` and no `val_vp_head`, and a model
            # built the other way round refuses it outright -- which is right, but it meant the
            # tournament and every probe could not read a P1 categorical arm at all.
            categorical = any(k.startswith("value_dist_head") for k in state_dict)
            ident = state_dict.get("card_identity.weight")
            identity_dim = int(ident.shape[1]) if ident is not None else 0
            if is_mlp:
                from ai.models.coldwar_net_v2 import (create_coldwar_net_mlp,
                                                      static_input_mask)
                w = state_dict.get("mlp_in.0.weight")
                narrowed = int(w.shape[1]) if w is not None else 0
                drop_static = narrowed and narrowed < int(static_input_mask().numel())
                model = create_coldwar_net_mlp(dev, categorical_value=categorical,
                                               drop_static=bool(drop_static))
            else:
                # Both architecture variants are detected by weight name, like everything else
                # here. A model built without them loads the checkpoint's other tensors fine and
                # silently drops these, so the probe would rate a different network than the one
                # that trained -- the failure mode that killed arms E and F.
                self_transform = any(k.endswith("self_linear.weight") for k in state_dict)
                ro_q = state_dict.get("ro_query.weight")
                attn_readout = int(ro_q.shape[0]) if ro_q is not None else 0
                pe = state_dict.get("pe_trunk.weight")
                per_entity_heads = int(pe.shape[0]) if pe is not None else 0
                model = create_coldwar_net_v2(dev, categorical_value=categorical,
                                              identity_dim=identity_dim,
                                              self_transform=self_transform,
                                              attn_readout=attn_readout,
                                              per_entity_heads=per_entity_heads)
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
