"""Player Agent Abstractions for Twilight Struggle Bots and Neural Models."""

from typing import (Protocol, Optional, Dict, Any, List, Sequence, Tuple, Union, cast,
                    runtime_checkable)
import os
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

import ts_engine as ts
from bindings.action_encoder import ActionEncoder
from ai.eval.safety import find_instant_win, safe_actions
from ai.training.behavioral_cloning import HeuristicPolicy, OldHeuristicPolicy
from ai.models.coldwar_net import ColdWarNet, create_coldwar_net
from ai.models.coldwar_net_v2 import (ColdWarNetV2, check_checkpoint_layout,
                                     create_coldwar_net_mlp, create_coldwar_net_v2,
                                     static_input_mask)
from ai.models.ladder_net import create_ladder_net, ladder_config_from_state_dict
from ai.search.batched_mcts import BatchedMCTSAgent, BatchedMCTSConfig
from ai.search.heuristic_mcts import HeuristicMCTSConfig, make_heuristic_mcts_agent
from bindings.ts_env import check_obs_width, model_obs_features
from tools.lib.action_view import checkpoint_merged_influence
from tools.lib.checkpoint_id import checkpoint_label
from tools.lib.openings import OPENINGS
from tools.lib.player_spec import Gumbel, Search
from tools.lib.p17_adapter import LegacyPolicyAdapter

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


# The interfaces an agent MAY offer beyond `select_action`. Harnesses test for them with
# isinstance, which for a runtime-checkable Protocol checks that the attributes exist -- the same
# test a hasattr would make, but with a type on the other side of it.

@runtime_checkable
class BatchSelector(Protocol):
    """Decides a batch of positions in one call (a searcher: one tree per position, stepped
    together so the network calls batch)."""

    name: str

    def select_actions_batch(self, states: Sequence[ts.GameState]) -> List[int]:
        ...


@runtime_checkable
class BatchActor(Protocol):
    """Decides a batch from observations and masks outside torch (`OnnxAgent`). `obs_size` is the
    width it reads, the first that many floats of each row."""

    name: str
    obs_size: int

    def act_batch(self, obs: np.ndarray, masks: np.ndarray, temperature: float,
                  greedy: bool) -> np.ndarray:
        ...


@runtime_checkable
class Reseedable(Protocol):
    """Carries a random stream of its own between games, and can restart it."""

    def reseed(self, seed: int) -> None:
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
        merged_influence: bool = False,
    ):
        #: P23 / E4.1: whether this network decides in the merged-influence view. Every harness
        #: that builds masks for it must use this view (tools/lib/action_view.py). Set from the
        #: checkpoint's run directory by `from_checkpoint`.
        self.merged_influence = bool(merged_influence)
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
        """Loads a NeuralAgent from checkpoint, with its architecture detected from the weights
        and its action view (P23) from the run directory."""

        agent = cls._load_checkpoint(checkpoint_path, name=name, device=device)
        agent.merged_influence = checkpoint_merged_influence(checkpoint_path)
        return agent

    @classmethod
    def _load_checkpoint(
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

        # A P21 ladder rung is checked FIRST. `lad_cross_attn.*` contains the substring
        # `cross_attn`, so a cross-attention rung matches the v2 test below and would be rebuilt
        # as a ColdWarNetV2 -- loading most tensors, silently dropping the rest, and rating a
        # different network than the one that trained. Like everything else here the
        # configuration is recovered from the weights, not from a recorded config.
        ladder_cfg = ladder_config_from_state_dict(state_dict)
        if ladder_cfg is not None:
            check_checkpoint_layout(state_dict)
            ladder = create_ladder_net(dev, **ladder_cfg)
            ladder.load_state_dict(state_dict, strict=True)
            ladder.eval()
            return cls(model=ladder, device=dev,
                       name=name or checkpoint_label(checkpoint_path))

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
        # In the model's own view (the view spec): the base layout plus its appended blocks.
        obs = ts.extract_observation_features(state, player, model_obs_features(self.model))
        mask = ActionEncoder.get_legal_mask(state)

        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_t, _, _, _, _ = self.model.sample_action(obs_t, mask_t, temperature=temperature, deterministic=(temperature <= 0.05))
        return int(action_t.item())


class OnnxAgent:
    """A network exported by tools/export_onnx.py, run in ONNX Runtime -- the file the workbench
    plays, and the only form in which the published models exist (the Hugging Face repo carries
    no .pt).

    The graph takes (obs, mask) and returns masked logits, so action selection is
    `ColdWarNetV2.sample_action` over those logits: argmax when greedy, else a sample at the given
    temperature. The export itself is verified against torch (same favourite move, probabilities
    within 1e-3), so a greedy game matches the checkpoint's; a sampled one draws from this agent's
    own generator rather than torch's, and matches in distribution only.

    Everything needed to run the file travels in its metadata, and is checked here as the page
    checks it: the format tag, and the observation width against this engine's.
    """

    FORMAT = "ts-onnx-v1"

    def __init__(self, path: str, name: Optional[str] = None, seed: int = 0):

        if not os.path.exists(path):
            raise FileNotFoundError(f"ONNX model not found: {path}")
        opts = ort.SessionOptions()
        # One thread per torch thread: a tournament worker pins torch to one thread so that N
        # workers use N cores, and ONNX Runtime would otherwise start a pool the size of the
        # machine in each of them.
        opts.intra_op_num_threads = max(1, torch.get_num_threads())
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(path, sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        meta: Dict[str, str] = dict(self.session.get_modelmeta().custom_metadata_map)
        if meta.get("ts.format") != self.FORMAT:
            raise ValueError(
                f"{path} is not a {self.FORMAT} export (ts.format={meta.get('ts.format')!r}); "
                f"make one with tools/export_onnx.py")
        self.obs_size = int(meta["ts.obs_size"])
        if self.obs_size != int(ts.OBS_SIZE):
            raise ValueError(
                f"{path} reads an observation of width {self.obs_size}, but this engine emits "
                f"{int(ts.OBS_SIZE)}: it was exported from a retired layout and would misread "
                f"every slot")
        action_size = int(meta.get("ts.action_size", ActionEncoder.FLAT_ACTION_SIZE))
        if action_size != ActionEncoder.FLAT_ACTION_SIZE:
            raise ValueError(f"{path} has {action_size} actions, this engine "
                             f"{ActionEncoder.FLAT_ACTION_SIZE}")
        #: P23 / E4.1: decides in the merged-influence view; read by every harness building masks.
        self.merged_influence = meta.get("ts.merged_influence") == "true"
        self.name = name or meta.get("ts.label") or os.path.splitext(os.path.basename(path))[0]
        self.engine_fingerprint = meta.get("ts.engine_fingerprint", "")
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def logits(self, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        out = self.session.run(["logits"], {"obs": np.ascontiguousarray(obs, dtype=np.float32),
                                            "mask": np.ascontiguousarray(masks, dtype=np.uint8)})
        return np.asarray(out[0], dtype=np.float64)

    def act_batch(self, obs: np.ndarray, masks: np.ndarray, temperature: float,
                  greedy: bool) -> np.ndarray:
        """One action per row, as `sample_action` would choose it from these logits."""
        logits = self.logits(obs, masks)
        if greedy:
            return logits.argmax(axis=1).astype(np.int32)
        z = logits / max(temperature, 1e-4)
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        cdf = np.cumsum(p, axis=1)
        u = self.rng.random(len(cdf)) * cdf[:, -1]
        # The first index whose cdf exceeds u. An illegal action adds nothing to the cdf, so it
        # can never be that index; a u that rounds up to the total falls back to the argmax.
        picks = (cdf <= u[:, None]).sum(axis=1)
        over = picks >= logits.shape[1]
        picks[over] = logits[over].argmax(axis=1)
        return picks.astype(np.int32)

    def select_action(self, state: ts.GameState, player: ts.Player,
                      temperature: float = 0.1) -> int:
        obs = np.asarray(ts.extract_observation(state, player), dtype=np.float32)[None, :]
        mask = np.asarray(ActionEncoder.get_legal_mask(state, self.merged_influence),
                          dtype=np.uint8)[None, :]
        return int(self.act_batch(obs, mask, temperature, temperature <= 0.05)[0])


def search_spec_config(spec: str) -> Tuple[str, BatchedMCTSConfig, str]:
    """(checkpoint, search configuration, label) of a `search:` or `gumbel:` agent spec. One parser
    for every CLI that plays a searcher (`load_agent`, tools/play_match.py), so a spec means the
    same thing in a tournament and in a single match.

    search:<checkpoint>[:sims[:determinize[:node_filter[:subsample[:backend[:fpu]]]]]]

      `node_filter` is "all" (every decision -- what the ~+27pp measurement used) or "card"
      (SELECT_CARD / SELECT_PLAY_MODE only, P3's proposal). `subsample` is the fraction of those
      actually searched, e.g. 0.125 for P3's "1 in 8". Where search is skipped the agent plays its
      own greedy policy, so a coverage sweep varies one thing. `backend` is "cpp" (the default,
      ts_engine.BatchedSearch) or "python" (the reference tree). `fpu` is the first-play urgency
      reduction, 0 by default.

    gumbel:<checkpoint>[:sims[:k[:fpu[:node_filter[:worlds]]]]]

      Honest search with the move chosen by a noise-free Gumbel root
      (ai/search/gumbel_root.py): the k most probable moves, sequential halving over `sims`
      network evaluations. Defaults 256 evaluations, k = 8, first-play urgency 0.2
      (research/log/E7_gumbel_headroom.md); both kinds' defaults live in tools/lib/player_spec.py.
      `node_filter` as for `search:` -- "all" (the default, every decision) or "card"
      (SELECT_CARD / SELECT_PLAY_MODE only, the agent's own greedy policy elsewhere). The
      leaderboard's Gumbel spec has no such field, so its players always search every decision.
      `worlds` (default 1) is `BatchedMCTSConfig.gumbel_worlds`: independent draws of the hidden
      cards and the dice each candidate is searched in per halving phase, its evaluations split
      over them -- the same budget, averaged over more chance.

    Every searcher here has advance_root=False because the CLIs hand over a state they have NOT
    settled -- tools/tournament.py only auto-advances under --auto-advance, and play_match.py steps
    decision by decision. With the default True the searcher settles its own root, so at a node
    whose mask holds a single legal action `auto_advance_step` consumes it, the root moves to the
    successor, and the search returns an action that is legal there and illegal in the caller's
    state. Observed as "engine refused flat action 104 at a POINT_NODE whose mask has 1 legal
    action: 211". Settling an already-settled state is a no-op, so False is also correct when the
    caller does settle.
    """
    parts = spec.strip().split(":")
    kind = parts[0].lower()
    path = parts[1]

    def field(i: int) -> Optional[str]:
        return parts[i] if len(parts) > i and parts[i] else None

    if kind == "gumbel":
        g = Gumbel()
        sims = int(field(2) or g.sims)
        k = int(field(3) or g.k)
        fpu = float(field(4) or g.fpu)
        g_filter = "card_playmode" if (field(5) or "all").lower().startswith("card") else "all"
        worlds = int(field(6) or 1)
        cfg = BatchedMCTSConfig(simulations=sims, temperature=0.0, auto_advance=True,
                                advance_root=False, determinize=True, node_filter=g_filter,
                                gumbel_k=k, gumbel_scale=0.0, fpu_reduction=fpu,
                                gumbel_worlds=worlds)
        label = (f"gumbel{sims}-k{k}" + ("" if fpu == g.fpu else f"-fpu{fpu:g}")
                 + ("" if g_filter == "all" else "-card") + ("" if worlds == 1 else f"-w{worlds}"))
        return path, cfg, label
    if kind != "search":
        raise ValueError(f"not a search spec: {spec!r}")
    d = Search()
    sims = int(field(2) or d.sims)
    determinize = (field(3) or "").lower().startswith("determin") if field(3) else d.determinize
    node_filter = field(4) or d.node_filter
    if node_filter.lower().startswith("card"):
        node_filter = "card_playmode"
    subsample = float(field(5) or d.subsample)
    backend = (field(6) or d.backend).lower()
    fpu = float(field(7) or d.fpu)
    cfg = BatchedMCTSConfig(simulations=sims, temperature=0.0, auto_advance=True,
                            advance_root=False, determinize=determinize,
                            node_filter=node_filter, subsample=subsample, backend=backend,
                            fpu_reduction=fpu)
    tag = "" if node_filter == "all" else "-card"
    tag += "" if subsample >= 1.0 else f"-{subsample:g}"
    tag += "" if backend == "cpp" else f"-{backend}"
    tag += "" if fpu == 0.0 else f"-fpu{fpu:g}"
    return path, cfg, f"search{sims}{'-det' if determinize else ''}{tag}"


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
    if s.lower().startswith("script:"):
        # script:<name>:<rest-of-spec> -- a named scripted rule (tools/lib/batch_tournament.SCRIPTS)
        # played on top of the agent: it forces the moves it covers and leaves the rest alone.
        _, script_name, rest = s.split(":", 2)
        agent = load_agent(rest, device=device)
        setattr(agent, "script", script_name)
        setattr(agent, "name", f"{agent.name}+{script_name}")
        return agent
    if s.lower().startswith("headline:"):
        # headline:<card>:<rest-of-spec> -- a scripted rule over the agent: whenever it chooses its
        # headline holding <card>, it headlines <card>. Everything else is the agent's own choice.
        # Read by the batch runner with getattr(agent, "headline_card", None), like the two above;
        # exists to measure what one headline rule is worth on top of a policy.
        # <card> may carry a condition, <card>+<name> (tools/lib/batch_tournament.HEADLINE_CONDITIONS),
        # e.g. 40+oppcantpay: headline Cuban Missile Crisis only when the opponent cannot pay it off.
        _, card_str, rest = s.split(":", 2)
        card_s, _, condition = card_str.partition("+")
        agent = load_agent(rest, device=device)
        setattr(agent, "headline_card", int(card_s))
        if condition:
            setattr(agent, "headline_condition", condition)
        setattr(agent, "name", f"{agent.name}+headline{int(card_s)}" + (f"-{condition}" if condition else ""))
        return agent
    if s.lower().startswith("name:"):
        # name:<label>:<rest-of-spec> -- the entrant's name in reports, for two entrants whose specs
        # would otherwise share one (two searchers of the same configuration over two checkpoints).
        _, label, rest = s.split(":", 2)
        agent = load_agent(rest, device=device)
        setattr(agent, "name", label)
        return agent
    if s.lower().startswith("opening:"):
        # opening:<name>:<rest-of-spec> -- this agent's setup is the named opening from
        # tools/lib/openings.py instead of its own placements, for checkpoints trained with
        # --forced-opening, whose setup was never learned. Read by the batch runner with
        # getattr(agent, "forced_opening", None), like temperature above.
        _, name, rest = s.split(":", 2)
        if name not in OPENINGS:
            raise ValueError(f"unknown opening {name!r}; known: {sorted(OPENINGS)}")
        agent = load_agent(rest, device=device)
        setattr(agent, "forced_opening", name)
        setattr(agent, "name", f"{agent.name}+{name}")
        return agent
    if s.lower().startswith("rollout:"):
        # rollout:<checkpoint>[:k[:worlds[:horizon[:rule]]]] -- the network's top k moves played out
        # by the network in sampled worlds, a cautious choice (ai/search/rollout_root.py).
        from ai.search.rollout_root import RolloutRootAgent, rollout_spec_config
        path, rcfg, label = rollout_spec_config(s)
        base = NeuralAgent.from_checkpoint(path, device=device)
        return RolloutRootAgent(base.model, label, rcfg, device=resolve_device(device))
    if s.lower().startswith(("search:", "gumbel:")):
        path, cfg, label = search_spec_config(s)
        base = NeuralAgent.from_checkpoint(path, device=device)
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

        parts = s.split(":")
        sims = int(parts[1]) if len(parts) > 1 and parts[1] else 64
        return make_heuristic_mcts_agent(
            name=f"HeuristicMCTS{sims}", config=HeuristicMCTSConfig(simulations=sims))
    # A tools/export_onnx.py export: the form the workbench plays and Hugging Face publishes.
    if s.lower().endswith(".onnx"):
        return OnnxAgent(s)
    return NeuralAgent.from_checkpoint(s, device=device)
