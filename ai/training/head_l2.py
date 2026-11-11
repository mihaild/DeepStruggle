"""L2 toward the initial weights on the policy and value heads (`--head-l2-init-coef`, P44 step 1's arm).

Juliani and Ash 2024 (arXiv:2405.19153), the one systematic study of plasticity loss in PPO, found that resets,
plasticity injection and dormant-unit recycling failed while L2 toward the initial weights and LayerNorm worked,
best combined. P37a placed this project's plateau in the policy heads (weights 14-50x a fresh head's, dormant
units, own-head random-label fit 36.7% against a fresh head's 99%), so the term acts on the heads only:
coef * sum over their parameters of ||theta - theta_0||^2, with theta_0 the weights the run started from.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Tuple

import torch
import torch.nn as nn

from ai.models.coldwar_net_v2 import POLICY_HEAD_MODULES

#: The modules the term pulls toward their initial weights: the policy heads (the base head, the per-entity heads
#: and their trunk context, the branch and play-mode heads), --ladder-head-norm's LayerNorms in them, and the two
#: value heads. The trunk is left free.
HEAD_L2_MODULES: Tuple[str, ...] = POLICY_HEAD_MODULES + ("head_norms", "val_win_head", "val_vp_head")


def head_init_params(net: nn.Module) -> Dict[str, torch.Tensor]:
    """A detached copy of every parameter of the HEAD_L2_MODULES the network has, by name: theta_0."""
    out = {n: p.detach().clone() for n, p in net.named_parameters() if n.split(".")[0] in HEAD_L2_MODULES}
    if not out:
        raise ValueError(f"{type(net).__name__} has none of the modules L2 toward the initial weights acts on "
                         f"({', '.join(HEAD_L2_MODULES)})")
    return out


def head_pairs(net: nn.Module, init: Mapping[str, torch.Tensor]) -> List[Tuple[nn.Parameter, torch.Tensor]]:
    """(parameter, its initial value) for every parameter in `init`; refuses a name the network lacks or a shape
    that differs, so a state from another network cannot be pulled toward silently."""
    params = dict(net.named_parameters())
    pairs: List[Tuple[nn.Parameter, torch.Tensor]] = []
    for name, p0 in init.items():
        p = params.get(name)
        if p is None or p.shape != p0.shape:
            raise ValueError(f"the initial head weights do not match the network: {name} "
                             f"{'missing' if p is None else f'{tuple(p.shape)} against {tuple(p0.shape)}'}")
        pairs.append((p, p0))
    return pairs


def head_l2_to_init(pairs: List[Tuple[nn.Parameter, torch.Tensor]]) -> torch.Tensor:
    """sum ||theta - theta_0||^2 over the pairs, in float32."""
    return torch.stack([(p.float() - p0.float()).pow(2).sum() for p, p0 in pairs]).sum()
