"""Live model analysis for the workbench: what a chosen checkpoint thinks of the position on screen.

The replay trace (`ai/eval/policy_readout.py`) records what a model believed at each step of a
game that has already been played. This answers the same two questions -- the distribution over
every legal action, and both critic heads from both perspectives -- for the position a person is
exploring *now*, one decision at a time, so a human can play either side (or both) with the
model's opinion painted on the board.

Three things here are easy to get subtly wrong, and each would still produce plausible numbers:

* **The action view.** A checkpoint trained in the E4.1 merged-influence view (P23) reads a NODE
  slot at an op-choice node as "ops for influence, first point here". Its mask must be built in
  that view, and its favourite action applied in it, or the probabilities land on the wrong
  things. The view comes from the run directory via `NeuralAgent.from_checkpoint`, exactly as for
  every other harness.
* **The reader.** Policy and critic go through `read_policy` / `read_critic`, the functions the
  inline trace and the annotator use, so "what the model said" means one thing everywhere.
  `deterministic=True`: nothing is being sampled, and the torch RNG is left alone.
* **Which checkpoint.** Only files under the shared checkpoints tree can be named, by a path
  relative to it, so a URL can carry the choice and cannot point the server at an arbitrary file.

The position token (`encode_position` / `decode_position`) is the engine's named-field save,
`GameState.to_save_dict`, which round-trips a mid-game position exactly (decision stack included)
and tolerates missing and unknown keys, so a link keeps opening across engine builds that add a
field. JSON, zlib, base64url: about 1 KB for a mid-game position.
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import re
import threading
import zlib
from collections import OrderedDict
from typing import Any, List, Optional, Tuple, cast

import numpy as np
import torch

import ts_engine as ts
from ai.eval.policy_readout import read_critic, read_policy
from bindings.action_encoder import ActionEncoder
from tools.lib.data_root import checkpoints_dir
from tools.lib.player_agent import NeuralAgent
from web.server.replay_types import (
    AnalysisChoiceDict,
    AnalysisModelListDict,
    AnalysisRunDict,
    LiveAnalysisDict,
    ReplayPolicyDict,
)

#: The flat slot the merged view composes onto: at an op-choice node it is "Ops for influence",
#: whether the node is the play-mode resolution or the deferred op-mode choice.
OPS_INFLUENCE_SLOT: int = ActionEncoder.OP_MODE_OFFSET

#: A position token never needs more than this once inflated; a token that does is refused
#: before it can allocate much.
_MAX_POSITION_BYTES: int = 64 * 1024

_SNAPSHOT_STEPS = re.compile(r"snapshot_(\d+)steps\.pt$")


class AnalysisError(ValueError):
    """A request the analysis layer refuses: an unknown model, or a malformed position."""


# -- position tokens ----------------------------------------------------------------------------

def encode_position(state: ts.GameState) -> str:
    """A compact, URL-safe token for the position: zlib'd `to_save_dict` JSON, base64url."""
    raw = json.dumps(state.to_save_dict(), separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode("ascii").rstrip("=")


def decode_position(token: str) -> ts.GameState:
    """The position a token names. Raises `AnalysisError` on anything that does not round-trip.

    A save that does not survive `state_from_save_dict` -> `to_save_dict` unchanged holds values
    the engine would have silently truncated or ignored, so it is not the position it claims to be
    and is refused rather than opened approximately.
    """
    token = token.strip()
    try:
        packed = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
        inflater = zlib.decompressobj()
        raw = inflater.decompress(packed, _MAX_POSITION_BYTES)
        if inflater.unconsumed_tail:
            raise AnalysisError("position token inflates past the size limit")
        save = json.loads(raw.decode("utf-8"))
    except (binascii.Error, zlib.error, UnicodeDecodeError, ValueError) as exc:
        if isinstance(exc, AnalysisError):
            raise
        raise AnalysisError(f"not a position token: {exc}") from exc
    if not isinstance(save, dict) or not str(save.get("format", "")).startswith("ts_save"):
        raise AnalysisError("position token does not hold an engine save")
    try:
        state = ts.state_from_save_dict(save)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise AnalysisError(f"engine refused the saved position: {exc}") from exc
    if state.to_save_dict() != save:
        raise AnalysisError("saved position does not round-trip through the engine")
    return state


# -- which checkpoints can be named -----------------------------------------------------------

def models_root() -> str:
    """The checkpoints tree models are named relative to. `$TS_CHECKPOINTS_DIR` overrides the
    shared one, resolved per call, so a test can point the server at checkpoints it just wrote."""
    return os.environ.get("TS_CHECKPOINTS_DIR") or checkpoints_dir()


def _is_weights_file(name: str) -> bool:
    # `resume_*.pt` hold optimizer and scheduler state for continuing a run, not a network.
    return name.endswith(".pt") and not name.startswith("resume_")


def _snapshot_sort_key(name: str) -> tuple[int, int, str]:
    m = _SNAPSHOT_STEPS.search(name)
    if m:
        return (1, int(m.group(1)), name)
    if name == "snapshot_final.pt":
        return (2, 0, name)
    return (0, 0, name)


def list_models(root: Optional[str] = None) -> AnalysisModelListDict:
    """Every `.pt` under the checkpoints tree, grouped by run directory, newest run first."""
    root = root or models_root()
    runs: List[AnalysisRunDict] = []
    loose: List[str] = []
    if os.path.isdir(root):
        dated: List[tuple[float, AnalysisRunDict]] = []
        for entry in os.scandir(root):
            if entry.is_file() and _is_weights_file(entry.name):
                loose.append(entry.name)
            elif entry.is_dir():
                try:
                    snaps = [f.name for f in os.scandir(entry.path)
                             if f.is_file() and _is_weights_file(f.name)]
                except OSError:
                    continue
                if snaps:
                    snaps.sort(key=_snapshot_sort_key)
                    dated.append((entry.stat().st_mtime, {"run": entry.name, "snapshots": snaps}))
        dated.sort(key=lambda t: t[0], reverse=True)
        runs = [r for _, r in dated]
    loose.sort()
    return {"root": root, "runs": runs, "loose": loose}


def resolve_model_path(rel: str, root: Optional[str] = None) -> str:
    """The absolute path of a checkpoint named relative to the checkpoints tree.

    Anything that escapes the tree, or is not an existing `.pt` file, is refused.
    """
    root = os.path.realpath(root or models_root())
    path = os.path.realpath(os.path.join(root, rel))
    if (not path.startswith(root + os.sep) or not _is_weights_file(os.path.basename(path))
            or not os.path.isfile(path)):
        raise AnalysisError(f"no checkpoint {rel!r} under {root}")
    return path


# -- loaded models ----------------------------------------------------------------------------

def _analysis_device() -> str:
    # CPU by default: a single position is cheap, and the GPU usually belongs to a training run.
    return os.environ.get("TS_ANALYSIS_DEVICE", "cpu")


class ModelCache:
    """A few loaded checkpoints, least recently used evicted. Loading takes seconds; a
    forward pass on one position takes milliseconds, so every client shares one copy."""

    def __init__(self, capacity: int = 4) -> None:
        self.capacity = capacity
        self._agents: "OrderedDict[str, NeuralAgent]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, rel: str) -> NeuralAgent:
        path = resolve_model_path(rel)
        with self._lock:
            agent = self._agents.get(path)
            if agent is not None:
                self._agents.move_to_end(path)
                return agent
        try:
            agent = NeuralAgent.from_checkpoint(path, device=_analysis_device())
        except (RuntimeError, ValueError, KeyError, OSError) as exc:
            raise AnalysisError(f"could not load {rel}: {exc}") from exc
        with self._lock:
            self._agents[path] = agent
            self._agents.move_to_end(path)
            while len(self._agents) > self.capacity:
                self._agents.popitem(last=False)
        return agent


MODEL_CACHE = ModelCache()


# -- the readout ------------------------------------------------------------------------------

def _decider(state: ts.GameState) -> Optional[ts.Player]:
    p = state.ctx().decision_player
    return None if p == ts.Player.NONE else p


def _choice(state: ts.GameState, idx: int, p: float, merged: bool) -> AnalysisChoiceDict:
    """One legal action, with the MicroAction a click would send, so the client can attach its
    probability to the button/card/country without re-deriving flat offsets."""
    entry: AnalysisChoiceDict = {"idx": idx, "p": p}
    if merged and ts.ActionMask.is_merged_influence_action(state, idx):
        # Composed E4.1 action: "Ops for influence" plus its first placement. Not a single click
        # in the E4 interface, so it names both halves.
        commit = ts.decode_flat_action(state, OPS_INFLUENCE_SLOT)
        entry["composed"] = True
        entry["decision_type"] = int(commit.decision_type)
        entry["primary_id"] = int(commit.primary_id)
        entry["flags"] = 0
        if idx != OPS_INFLUENCE_SLOT:
            cid = idx - ActionEncoder.NODE_OFFSET
            entry["country_id"] = cid
            entry["name"] = f"Ops → Influence, first point in {ts.MapData.get_country_name(cid)}"
        else:
            entry["name"] = "Ops → Influence, place nothing"
        return entry
    ma = ts.decode_flat_action(state, idx)
    entry["decision_type"] = int(ma.decision_type)
    entry["primary_id"] = int(ma.primary_id)
    entry["flags"] = int(ma.flags)
    entry["name"] = ActionEncoder.get_action_name(state, idx)
    return entry


def analyze(agent: NeuralAgent, state: ts.GameState, model_rel: str) -> LiveAnalysisDict:
    """The model's policy over the decision now open, and its critic on the position."""
    model: Any = agent.model
    out: LiveAnalysisDict = {
        "model": model_rel,
        "label": agent.name,
        "merged_influence": agent.merged_influence,
        "critic": read_critic(model, state, at="position"),
        "choices": [],
    }
    decider = _decider(state)
    if ts.Engine.is_terminal(state) or decider is None:
        return out

    mask_np = np.asarray(ActionEncoder.get_legal_mask(state, agent.merged_influence)).reshape(1, -1)
    obs_np = np.asarray(ts.extract_observation(state, decider), dtype=np.float32).reshape(1, -1)
    device = next(model.parameters()).device
    _, policy = read_policy(
        model,
        torch.from_numpy(obs_np).to(device),
        torch.from_numpy(mask_np).to(device),
        temperature=1.0,
        deterministic=True,   # nothing is sampled; the argmax is the favourite
        state=None,           # names come from _choice, which knows the view
        source="analysis",
    )
    top = policy.pop("top", [])
    out["policy"] = cast(ReplayPolicyDict, policy)
    out["decision_player"] = "US" if decider == ts.Player.US else "USSR"
    out["decision_type"] = int(state.ctx().decision_type)
    out["choices"] = [_choice(state, int(e["idx"]), float(e["p"]), agent.merged_influence)
                      for e in top]
    return out


def flat_to_steps(state: ts.GameState, flat_idx: int,
                  merged: bool) -> Tuple[ts.MicroAction, Optional[int]]:
    """The E4 step a flat action begins with, and -- for a composed E4.1 action -- the E4 slot of
    its second half (the placement, or the stop), to be decoded on the state the first leads to.

    Mirrors `Engine::step_flat`'s composition, but as two ordinary steps, so each is logged and
    can be undone like a click. Raises `AnalysisError` when the action is not legal in the view.
    """
    mask = ActionEncoder.get_legal_mask(state, merged)
    if not (0 <= flat_idx < len(mask)) or not mask[flat_idx]:
        raise AnalysisError(f"flat action {flat_idx} is not legal here "
                            f"({'merged' if merged else 'E4'} view)")
    if merged and ts.ActionMask.is_merged_influence_action(state, flat_idx):
        second = ActionEncoder.CONFIRM_DONE_INDEX if flat_idx == OPS_INFLUENCE_SLOT else flat_idx
        return ts.decode_flat_action(state, OPS_INFLUENCE_SLOT), second
    return ts.decode_flat_action(state, flat_idx), None
