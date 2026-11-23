"""The heads of ColdWarNetV2 / LadderNet, in one registry (audit M-4, 2026-11-22).

Every tool that treats "the heads" differently from the trunk reads its set from here: muP's zeroed readouts and the
policy-head rate (`ai/models/mup.py`), the head L2 term (`ai/training/head_l2.py`), the policy-head reset and the
head-norm metrics (`POLICY_HEAD_MODULES`, `ai/models/coldwar_net_v2.py`, `ai/training/generic_trainer.py`),
--ladder-head-norm's LayerNorms (`ai/models/ladder_net.py`) and the token-growing resume (`ai/models/grow.py`).
Before, each kept its own list, and the heads added later (P42's `pe_macro`, Cell A's `region_attn`, the card head
`pe_card`, the categorical `value_dist_head`) were missing from some of them.

A head is a top-level module; a network that does not have one (a ColdWarNetV2 has no `pe_macro`) skips it. The
training-only heads (aux, card-event, opponent-legality, prior gain) are not policy or value heads and are not in
HEADS; they are listed in AUX_HEADS.

**Side copies** (owner, 2026-11-23): a side-split LadderNet keeps a second copy of its heads or of its whole network for
the USSR rows, under one prefix -- `ussr_heads.` (--ladder-side-heads: every head in HEADS, the head norms and every
AUX_HEADS head) or `ussr_net.` (--ladder-side-nets: a complete second LadderNet) -- while the US copy keeps the plain
names. A consumer that keys on a head's top-level name reads it through `copy_prefixes(model)` (every copy's prefix,
"" first) or `head_of(param_name)` (the head a parameter belongs to, through a copy's prefix), so each copy is treated
as its original is: zeroed by muP, scaled by the policy-head rate, pulled by the head L2 term, reset, measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class HeadSpec:
    #: The top-level module's attribute name.
    name: str
    #: "policy" or "value".
    kind: str
    #: The submodule (relative to the head) of its output layer, which muP starts at zero; None for a head with no
    #: readout of its own (the per-entity heads' trunk context `pe_trunk`).
    readout: Optional[str] = None
    #: --ladder-head-norm puts a LayerNorm between its first Linear and the GELU (a Linear -> GELU -> Linear head that
    #: has no norm of its own).
    norm: bool = False
    #: A per-entity head: its first layer reads (identity, token, raw slots, trunk context), so a grown token path
    #: widens it (`grow.ENTITY_HEADS`).
    entity: bool = False


#: In model order where it matters: the head norms are a ModuleDict built in this order (pe_macro last, so every
#: earlier network's norms keep their parameter order).
HEADS: Tuple[HeadSpec, ...] = (
    HeadSpec("policy_head", "policy", readout="3"),
    HeadSpec("pe_trunk", "policy"),
    HeadSpec("pe_country", "policy", readout="2", norm=True, entity=True),
    HeadSpec("pe_card", "policy", readout="2", norm=True, entity=True),
    HeadSpec("branch_head_net", "policy", readout="2", norm=True),
    HeadSpec("play_mode_head_net", "policy", readout="2", norm=True),
    HeadSpec("pe_macro", "policy", readout="2", norm=True, entity=True),
    HeadSpec("region_attn", "policy", readout="out"),
    HeadSpec("val_win_head", "value", readout="3"),
    HeadSpec("val_vp_head", "value", readout="3"),
    HeadSpec("value_dist_head", "value", readout="3"),
)

#: The module holding --ladder-head-norm's LayerNorms (one per `norm` head the network has).
HEAD_NORMS = "head_norms"

#: The policy heads' modules.
POLICY_HEADS: Tuple[str, ...] = tuple(h.name for h in HEADS if h.kind == "policy")
#: The value heads' modules.
VALUE_HEADS: Tuple[str, ...] = tuple(h.name for h in HEADS if h.kind == "value")
#: The heads --ladder-head-norm normalises, in the order their norms are registered.
NORM_HEADS: Tuple[str, ...] = tuple(h.name for h in HEADS if h.norm)
#: The per-entity heads.
ENTITY_HEADS: Tuple[str, ...] = tuple(h.name for h in HEADS if h.entity)
#: Each head's output layer, as a submodule path of the network.
READOUTS: Tuple[str, ...] = tuple(f"{h.name}.{h.readout}" for h in HEADS if h.readout is not None)

#: The training-only heads that read the trunk: the DEFCON-risk head, P29's ownership and VP-margin heads, P30's
#: card-event head, P31's opponent-legality head and P46c's prior gain. Neither policy nor value heads, but heads all the
#: same: --ladder-side-heads gives each a USSR copy.
AUX_HEADS: Tuple[str, ...] = ("defcon_risk_head", "aux_own_head", "aux_vp_head", "card_aux_head", "opp_legal_head",
                              "prior_gain_head")

#: Every module --ladder-side-heads copies per side, in the order the copies are registered.
SIDE_HEAD_MODULES: Tuple[str, ...] = tuple(h.name for h in HEADS) + (HEAD_NORMS,) + AUX_HEADS

#: The attribute holding a side-split network's USSR copy: its heads (--ladder-side-heads) or its whole network
#: (--ladder-side-nets). The US copy keeps the plain names.
SIDE_HEADS_COPY = "ussr_heads"
SIDE_NET_COPY = "ussr_net"
SIDE_COPIES: Tuple[str, ...] = (SIDE_HEADS_COPY, SIDE_NET_COPY)


def copy_prefixes(model: Any) -> Tuple[str, ...]:
    """The module-path prefix of each copy of `model`'s heads: "" (the plain names, the US copy of a side-split
    network), then "ussr_heads." or "ussr_net." when the network has one."""
    return ("",) + tuple(f"{c}." for c in SIDE_COPIES if getattr(model, c, None) is not None)


def head_of(name: str) -> str:
    """The top-level module a parameter (or module path) belongs to, read through a side copy's prefix:
    "ussr_heads.pe_country.0.weight" and "pe_country.0.weight" are both "pe_country"."""
    parts = name.split(".")
    if parts[0] in SIDE_COPIES and len(parts) > 1:
        return parts[1]
    return parts[0]


def is_side_split(model: Any) -> bool:
    """Whether `model` keeps a USSR copy of its heads or of its network (--ladder-side-heads / --ladder-side-nets)."""
    return len(copy_prefixes(model)) > 1


def refuse_side_split(model: Any, what: str) -> None:
    """Refuse a side-split network -- a module, or a ladder configuration dict -- in a tool that reads one copy of the
    heads (or of the trunk) by name, which would read the US copy for every row."""
    split = (bool(model.get("side_heads") or model.get("side_nets")) if isinstance(model, dict)
             else is_side_split(model))
    if split:
        raise ValueError(f"{what} reads one copy of the network's heads by name; a side-split network "
                         f"(--ladder-side-heads / --ladder-side-nets) has one per side, and this tool does not "
                         f"handle them")
