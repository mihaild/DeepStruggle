"""muP for LadderNet (P39 arm 4, research/plans/P39_deep_trunk_optimisation.md), and the per-group rates of arm 5.

The ladder scaled one global learning rate with width; muP (Yang et al., Tensor Programs V) scales each layer's rate
with its own fan-in, so the step a layer takes keeps its size relative to its weights as the network widens and the
rates tuned at a base width transfer. For Adam the rule is: a weight matrix whose fan-in grows by a factor m gets its
rate divided by m (hidden and output layers alike); a layer whose fan-in does not grow (an input projection reading
the observation) keeps the base rate; vectors (biases, LayerNorm gains) keep it too. The output layers start at zero
(muP's readout at init variance 1/fan_in^2 taken to its limit, a common choice), so a wider network does not start
with larger logits.

The base is the same network at the A8 control's widths (entity projections 256, fusion 480), its depth and every
other setting kept: every rate is relative to the shape the same parameter has there, so at the base widths every
multiplier is 1 and `--mup` changes nothing but the zeroed readouts. Before 2026-11-22 the base dropped the sphere
and residual blocks and scaled them against the base fusion width, which read a sphere block's expansion factor as
width: its w2 (fan-in 4 x hidden) got 1/4 of w1's rate at every width (audit M-1; the A30/A32 sphere runs under
--mup). A parameter the base still lacks is scaled by its fan-in against the base fusion width.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Mapping, Optional

import torch
import torch.nn as nn

from ai.models.head_registry import HEAD_NORMS, POLICY_HEADS, READOUTS, copy_prefixes

#: The policy heads, by top-level module, and their head norms: what `--policy-head-lr-mult` scales (P39 arm 5: the
#: heads saturate first). From the head registry (ai/models/head_registry.py).
POLICY_HEAD_PREFIXES: tuple[str, ...] = tuple(f"{n}." for n in POLICY_HEADS) + (f"{HEAD_NORMS}.",)

#: The readouts muP starts at zero: the output layer of each policy and value head (the head registry's). The branch,
#: play-mode, macro and region-attention heads' are already zero-initialised by LadderNet, so zeroing them is a no-op.
#: A side-split network's USSR copies (`ussr_heads.` / `ussr_net.`) are zeroed with their originals.
OUTPUT_LAYERS: tuple[str, ...] = READOUTS


def policy_head_prefixes(model: nn.Module) -> tuple[str, ...]:
    """POLICY_HEAD_PREFIXES under every copy of `model`'s heads (a side-split network's USSR copy too)."""
    return tuple(pre + p for pre in copy_prefixes(model) for p in POLICY_HEAD_PREFIXES)


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
    """The parameter shapes of the same network at the base widths: only the true widths (the entity projection and
    the fusion width) change; depth (sphere and residual blocks), a sphere block's expansion factor and every other
    setting are the network's own, so a matrix whose fan-in is a multiple of the width (a sphere block's w2) keeps
    that multiple in the base and gets the same multiplier as its width's other matrices. `build(**config)` makes a
    LadderNet on the CPU. A side-split network's base is side-split too (the side flags are in its config), so each
    USSR copy has its original's base shape and multiplier."""
    cfg = dict(config)
    # A token-main network has no grouped projections to set a base width for (P41 J5).
    cfg.update(entity_proj_dim=0 if cfg.get("token_main") else int(base_proj_dim), hidden_dim=int(base_hidden_dim))
    base = build(**cfg)
    return {n: p.shape for n, p in base.named_parameters()}


@torch.no_grad()
def zero_output_layers(model: nn.Module, names: Iterable[str] = OUTPUT_LAYERS) -> List[str]:
    """Zero the readout layers the model has (weight and bias), in every copy of its heads (a side-split network's
    USSR copy under its prefix); returns the ones zeroed."""
    done: List[str] = []
    for name in [pre + n for pre in copy_prefixes(model) for n in names]:
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
    prefixes = policy_head_prefixes(model)
    return {n: float(mult) for n, _ in model.named_parameters() if n.startswith(prefixes)}


def combine_lr_mults(*mults: Optional[Mapping[str, float]]) -> Optional[Dict[str, float]]:
    """The product of several multiplier maps, or None when every one is empty (the old single-group optimiser)."""
    out: Dict[str, float] = {}
    for m in mults:
        for k, v in (m or {}).items():
            out[k] = out.get(k, 1.0) * float(v)
    return out or None
