"""muP for LadderNet (P39 arm 4, research/plans/P39_deep_trunk_optimisation.md), and the per-group rates of arm 5.

The ladder scaled one global learning rate with width; muP (Yang et al., Tensor Programs V) scales each layer's rate
with its own fan-in, so the step a layer takes keeps its size relative to its weights as the network widens and the
rates tuned at a base width transfer. For Adam the rule is: a weight matrix whose fan-in grows by a factor m gets its
rate divided by m (hidden and output layers alike); a layer whose fan-in does not grow (an input projection reading
the observation) keeps the base rate; vectors (biases, LayerNorm gains) keep it too. The output layers start at zero
(muP's readout at init variance 1/fan_in^2 taken to its limit, a common choice), so a wider network does not start
with larger logits.

The base is the A8 control (entity projections 256, fusion 480): every rate is relative to the shape the same
parameter has there, so at the base widths every multiplier is 1 and `--mup` changes nothing but the zeroed readouts.
A parameter the base network does not have (a sphere block, a residual block) is scaled by its fan-in against the
base fusion width.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Mapping, Optional

import torch
import torch.nn as nn

#: The policy heads, by top-level module: what `--policy-head-lr-mult` scales (P39 arm 5: the heads saturate first).
POLICY_HEAD_PREFIXES = ("policy_head.", "pe_trunk.", "pe_country.", "pe_card.", "branch_head_net.",
                        "play_mode_head_net.", "head_norms.")

#: The readouts muP starts at zero: the last linear layer of each head that emits logits or values. The branch and
#: play-mode heads' readouts are already zero-initialised by LadderNet.
OUTPUT_LAYERS = ("policy_head.3", "pe_country.2", "pe_card.2", "val_win_head.3", "val_vp_head.3")


def fan_in(p: torch.Tensor) -> int:
    """A Linear weight's input width (its second dimension); 0 for anything that is not a matrix."""
    return int(p.shape[1]) if p.ndim == 2 else 0


def mup_lr_mults(model: nn.Module, base_shapes: Mapping[str, torch.Size], base_width: int) -> Dict[str, float]:
    """Each matrix parameter's rate multiplier: base fan-in over its fan-in where the base has the parameter, else
    `base_width` over its fan-in when that is wider; 1 for embeddings, vectors and layers whose fan-in did not grow."""
    embeddings = {f"{n}.weight" for n, m in model.named_modules() if isinstance(m, nn.Embedding)}
    out: Dict[str, float] = {}
    for name, p in model.named_parameters():
        f = fan_in(p)
        if not f or name in embeddings:
            continue
        base = base_shapes.get(name)
        base_f = int(base[1]) if base is not None and len(base) == 2 else int(base_width)
        if f > base_f:
            out[name] = base_f / f
    return out


def base_shapes_for(config: Mapping[str, object], base_proj_dim: int, base_hidden_dim: int,
                    build: Callable[..., nn.Module]) -> Dict[str, torch.Size]:
    """The parameter shapes of the same network at the base widths, without its extra depth (no sphere or residual
    blocks): `build(**config)` makes a LadderNet on the CPU."""
    cfg = dict(config)
    cfg.update(entity_proj_dim=int(base_proj_dim), hidden_dim=int(base_hidden_dim), num_res_blocks=0)
    for k in ("sphere_blocks", "sphere_expansion", "sphere_out_scale", "sphere_carry_norm", "trunk_norm",
              "res_post_norm"):
        cfg.pop(k, None)
    base = build(**cfg)
    return {n: p.shape for n, p in base.named_parameters()}


@torch.no_grad()
def zero_output_layers(model: nn.Module, names: Iterable[str] = OUTPUT_LAYERS) -> List[str]:
    """Zero the readout layers the model has (weight and bias); returns the ones zeroed."""
    done: List[str] = []
    for name in names:
        try:
            mod = model.get_submodule(name)
        except AttributeError:
            continue
        if isinstance(mod, nn.Linear):
            mod.weight.zero_()
            if mod.bias is not None:
                mod.bias.zero_()
            done.append(name)
    return done


def policy_head_lr_mults(model: nn.Module, mult: float) -> Dict[str, float]:
    """`mult` for every parameter of a policy head; empty at 1."""
    if mult == 1.0:
        return {}
    return {n: float(mult) for n, _ in model.named_parameters() if n.startswith(POLICY_HEAD_PREFIXES)}


def combine_lr_mults(*mults: Optional[Mapping[str, float]]) -> Optional[Dict[str, float]]:
    """The product of several multiplier maps, or None when every one is empty (the old single-group optimiser)."""
    out: Dict[str, float] = {}
    for m in mults:
        for k, v in (m or {}).items():
            out[k] = out.get(k, 1.0) * float(v)
    return out or None
