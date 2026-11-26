"""Branching a trained network into RELATIVE_EFFECTS without changing what it computes (owner, 2026-11-26).

RELATIVE_EFFECTS (`relative_effects`, ts.OBS_FEATURE_RELATIVE_EFFECTS) puts the persistent-effect bits (globals 12-56)
in the observer's frame: a USSR observer's mirrored pairs (ts.OBS_RELATIVE_EFFECT_PAIRS) are swapped, and the two
space-attempt slots (bits 43/44) read the observer's own attempts. A US observer's view is unchanged.

A network that reads only USSR rows -- an A39 (--ladder-side-nets) network's USSR copy, or the USSR network taken out
alone -- is branched by swapping the columns of each pair in every input layer that reads the global block raw
(`GLOBAL_INPUT_WEIGHTS`), and in their Adam moments: W' v' = W P P v = W v, so it computes exactly what it did wherever
the information is the same. The attempt slots are not a permutation (a USSR observer read the US's attempts and now
reads its own; both are also in globals 59/60, its own and the opponent's space turns used), so their columns stay.
A network that reads both sides through one set of weights (A8, A38's shared trunk) cannot be branched this way --
the swap would be right for its USSR rows and wrong for its US rows -- and is refused."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

import torch
from torch import nn

import ts_engine as ts

RELATIVE_EFFECTS = int(ts.OBS_FEATURE_RELATIVE_EFFECTS)
#: Input layers that read the 100-float global block raw, by parameter name (without a side prefix): the grouped and
#: entity trunks' global projection, and the token path's global token.
GLOBAL_INPUT_WEIGHTS = ("global_proj.0.weight", "tok_global_in.weight")
#: Input layers that read the whole observation raw: a network with one cannot be branched (refused).
_WHOLE_INPUT_WEIGHTS = ("lad_in.0.weight", "mlp_in.0.weight")
USSR_PREFIX = "ussr_net."


def swapped_columns() -> List[Tuple[int, int]]:
    """The global-block columns RELATIVE_EFFECTS swaps for a USSR observer: (the US's effect, the USSR's)."""
    g = int(ts.OBS_GLOBAL_EFFECT_BITS)
    return [(g + int(a), g + int(b)) for a, b in ts.OBS_RELATIVE_EFFECT_PAIRS]


def ussr_prefixes(keys: List[str], side_only: Optional[str]) -> List[str]:
    """The key prefixes of the networks in a state that read USSR rows only. Raises for a network reading both."""
    if any(k.startswith(USSR_PREFIX) for k in keys):
        return [USSR_PREFIX]
    if side_only == "ussr":
        return [""]
    if side_only == "us":
        return []
    raise ValueError("relative_effects can be added on resume only to a --ladder-side-nets (A39) state or one side's "
                     "network taken out of one: a network reading both sides through one set of weights cannot be "
                     "permuted to compute what it did (the swap is right for its USSR rows and wrong for its US rows)")


def _swap_(t: torch.Tensor, cols: List[Tuple[int, int]]) -> torch.Tensor:
    out = t.clone()
    for a, b in cols:
        out[:, a], out[:, b] = t[:, b], t[:, a]
    return out


def _input_keys(keys: List[str], prefixes: List[str]) -> List[str]:
    for p in prefixes:
        bad = [p + n for n in _WHOLE_INPUT_WEIGHTS if p + n in keys]
        if bad:
            raise ValueError(f"relative_effects cannot be added to a network reading the whole observation through "
                             f"one layer ({', '.join(bad)}): only the global block's own projections are permuted")
    return [p + n for p in prefixes for n in GLOBAL_INPUT_WEIGHTS if p + n in keys]


def permute_state_dict_(sd: MutableMapping[str, Any], prefixes: List[str]) -> List[str]:
    """Swap the pair columns of every global input layer under `prefixes`, in place; returns the keys permuted."""
    keys = _input_keys(list(sd), prefixes)
    cols = swapped_columns()
    for k in keys:
        sd[k] = _swap_(sd[k], cols)
    return keys


def permute_optimizer_(opt: Mapping[str, Any], names: List[str], keys: List[str]) -> int:
    """Swap the same columns of those parameters' Adam moments (indexed by `optimizer_param_names`); returns how
    many parameters had moments."""
    cols = swapped_columns()
    flat = [int(i) for g in opt["param_groups"] for i in g["params"]]
    done = 0
    for k in keys:
        if k not in names:
            continue
        st = opt["state"].get(flat[names.index(k)])
        if not st:
            continue
        for m in ("exp_avg", "exp_avg_sq"):
            if m in st:
                st[m] = _swap_(st[m], cols)
        done += 1
    return done


@torch.no_grad()
def permute_module_(net: nn.Module) -> List[str]:
    """`permute_state_dict_` on a live network's parameters (a restored pool member): its USSR copy's global input
    layers. Refuses a network reading both sides through one set of weights."""
    params = dict(net.named_parameters())
    prefixes = ussr_prefixes(list(params), getattr(net, "side_only", None))
    keys = _input_keys(list(params), prefixes)
    cols = swapped_columns()
    for k in keys:
        params[k].copy_(_swap_(params[k].detach(), cols))
    return keys


def branch_resume_blob_(blob: Dict[str, Any]) -> List[str]:
    """Permute a resume state for RELATIVE_EFFECTS: the network, pi_ref, any weight average and the Adam moments.
    Returns the keys permuted in the network."""
    side_only = blob.get("side_only")
    sd = blob["model_state_dict"]
    prefixes = ussr_prefixes(list(sd), side_only)
    keys = permute_state_dict_(sd, prefixes)
    for other in ("reference_state_dict", "ema_state_dict"):
        if blob.get(other) is not None:
            permute_state_dict_(blob[other], prefixes)
    if blob.get("optimizer_state_dict") is not None and keys:
        names = blob.get("optimizer_param_names")
        if names is None:
            raise ValueError("relative_effects on resume needs the state's optimizer_param_names to permute the Adam "
                             "moments; pass --resume-fresh-optimizer to drop them")
        permute_optimizer_(blob["optimizer_state_dict"], list(names), keys)
    for refused in ("head_init_state",):
        if blob.get(refused):
            raise ValueError(f"relative_effects on resume does not permute {refused}")
    return keys
