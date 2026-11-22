"""The heads of ColdWarNetV2 / LadderNet, in one registry (audit M-4, 2026-11-22).

Every tool that treats "the heads" differently from the trunk reads its set from here: muP's zeroed readouts and the
policy-head rate (`ai/models/mup.py`), the head L2 term (`ai/training/head_l2.py`), the policy-head reset and the
head-norm metrics (`POLICY_HEAD_MODULES`, `ai/models/coldwar_net_v2.py`, `ai/training/generic_trainer.py`),
--ladder-head-norm's LayerNorms (`ai/models/ladder_net.py`) and the token-growing resume (`ai/models/grow.py`).
Before, each kept its own list, and the heads added later (P42's `pe_macro`, Cell A's `region_attn`, the card head
`pe_card`, the categorical `value_dist_head`) were missing from some of them.

A head is a top-level module; a network that does not have one (a ColdWarNetV2 has no `pe_macro`) skips it. The
training-only heads (aux, card-event, opponent-legality, prior gain) are not policy or value heads and are not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


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
