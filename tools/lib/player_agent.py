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
from tools.lib.checkpoint_id import checkpoint_label

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
        """Selects a legal flat action index [0..FLAT_ACTION_SIZE-1]."""
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
                # Graph depth by weight name: two conv layers, one, or a plain
                # per-country encoder with no adjacency at all.
                if any(k.startswith("board_fc.") for k in state_dict):
                    graph_layers = 0
                elif any(k.startswith("gconv2.") for k in state_dict):
                    graph_layers = 2
                else:
                    graph_layers = 1
                pe = state_dict.get("pe_trunk.weight")
                per_entity_heads = int(pe.shape[0]) if pe is not None else 0
                model = create_coldwar_net_v2(dev, categorical_value=categorical,
                                              identity_dim=identity_dim,
                                              self_transform=self_transform,
                                              attn_readout=attn_readout,
                                              per_entity_heads=per_entity_heads,
                                              graph_layers=graph_layers)
        else:
            model = create_coldwar_net(dev)

        load_checkpoint_into(model, state_dict)
        model.to(dev)
        # Run-attributed, not the bare stem: two runs produce identically-named snapshots, so a
        # basename cannot identify a checkpoint. See tools/lib/checkpoint_id.py.
        agent_name = name or checkpoint_label(checkpoint_path)
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
    if s.lower().startswith("temp:"):
        # temp:<T>:<rest-of-spec> -- pins this agent's sampling temperature, overriding the
        # tournament-wide --temperature. Exists so one checkpoint can be played against itself at
        # two temperatures, which a single shared setting cannot express. The name carries the
        # value so the two rows are distinguishable in a report.
        _, t_str, rest = s.split(":", 2)
        agent = load_agent(rest, device=device)
        # setattr rather than a declared field: PlayerAgent is a Protocol, and adding a required
        # attribute there would oblige every bot in bot/ to carry a tournament-only concern. The
        # batch runner reads it with getattr(agent, "temperature", None), so an agent that has
        # never heard of it behaves exactly as before.
        setattr(agent, "temperature", float(t_str))
        setattr(agent, "name", f"{agent.name}@T{float(t_str):g}")
        return agent
    if s.lower().startswith("search:"):
        # search:<checkpoint>[:sims[:determinize[:node_filter[:subsample]]]]
        #
        # `node_filter` is "all" (every decision -- what the ~+27pp measurement used) or
        # "card" (SELECT_CARD / SELECT_PLAY_MODE only, P3's proposal). `subsample` is the
        # fraction of those actually searched, e.g. 0.125 for P3's "1 in 8". Where search is
        # skipped the agent plays its own greedy policy, so a coverage sweep varies one thing.
        #
        # Exposed here rather than left to callers so that search games go through the same
        # CLIs, and therefore the same replay writer, as every other match.
        parts = s.split(":")
        path = parts[1]
        sims = int(parts[2]) if len(parts) > 2 and parts[2] else 64
        determinize = len(parts) > 3 and parts[3].lower().startswith("determin")
        node_filter = "all"
        if len(parts) > 4 and parts[4]:
            node_filter = "card_playmode" if parts[4].lower().startswith("card") else parts[4]
        subsample = float(parts[5]) if len(parts) > 5 and parts[5] else 1.0
        from ai.search.batched_mcts import BatchedMCTSAgent, BatchedMCTSConfig

        base = NeuralAgent.from_checkpoint(path, device=device)
        # advance_root=False because the CLIs hand over a state they have NOT settled --
        # tools/tournament.py only auto-advances under --auto-advance, and play_match.py steps
        # decision by decision. With the default True the searcher settles its own root, so at a
        # node whose mask holds a single legal action `auto_advance_step` consumes it, the root
        # moves to the successor, and the search returns an action that is legal there and
        # illegal in the caller's state. Observed as "engine refused flat action 104 at a
        # POINT_NODE whose mask has 1 legal action: 211". Settling an already-settled state is a
        # no-op, so False is also correct when the caller does settle.
        cfg = BatchedMCTSConfig(simulations=sims, temperature=0.0,
                                auto_advance=True, advance_root=False,
                                determinize=determinize,
                                node_filter=node_filter, subsample=subsample)
        tag = "" if node_filter == "all" else "-card"
        tag += "" if subsample >= 1.0 else f"-{subsample:g}"
        label = f"search{sims}{'-det' if determinize else ''}{tag}"
        return BatchedMCTSAgent(base.model, name=label, device=device, config=cfg)
    if s.lower().startswith("legacy:"):
        # legacy:<checkpoint> -- play a PRE-P17 checkpoint on the post-P17 engine.
        #
        # P17 merged the three-step card play into one resolution node. The observation is
        # untouched by that refactor, so an old checkpoint's input stays valid and only its
        # action semantics differ. The adapter reconstructs the old masks from the merged one
        # and asks the old policy the questions it was trained on, drawing in the same places
        # it used to -- which is what makes a result comparable against the old engine rather
        # than a measurement of how many times the distribution got sampled.
        #
        # Exists so the old-vs-new check runs through this CLI like every other match, instead
        # of an ad-hoc script.
        path = s.split(":", 1)[1]
        from tools.lib.p17_adapter import LegacyPolicyAdapter

        base = NeuralAgent.from_checkpoint(path, device=device)
        adapter = LegacyPolicyAdapter(base.model, device=str(device))
        setattr(adapter, "name", f"legacy-{base.name}")
        return adapter
    if s.lower() in ["random", "randombot", "rand"]:
        return RandomAgent()
    if s.lower() in ["old_heuristic", "old_heuristicbot", "old_heur", "oldheuristic"]:
        return OldHeuristicAgent()
    if s.lower() in ["heuristic", "heuristicbot", "heur", "new_heuristic", "new_heuristicbot"]:
        return HeuristicAgent()
    if s.lower() in ["heuristic_v2", "heuristicbotv2", "heuristicv2", "heur2"]:
        return HeuristicV2Agent()
    # heuristic_mcts[:sims] -- MCTS with a rules-derived leaf value and no checkpoint. Registered
    # here as well as in tools/play_match.py so it can enter a tournament: a reference opponent
    # that only works in one-off matches cannot anchor a ladder.
    #
    # Perfect information: it searches the true state and sees the opponent's hand. Legitimate for
    # a benchmark, disqualifying for deployment, and a win rate against it is not a claim about
    # play under the real information set.
    if s.lower() == "heuristic_mcts" or s.lower().startswith("heuristic_mcts:"):
        from ai.search.heuristic_mcts import HeuristicMCTSConfig, make_heuristic_mcts_agent

        parts = s.split(":")
        sims = int(parts[1]) if len(parts) > 1 and parts[1] else 64
        return make_heuristic_mcts_agent(
            name=f"HeuristicMCTS{sims}", config=HeuristicMCTSConfig(simulations=sims))
    return NeuralAgent.from_checkpoint(s, device=device)
