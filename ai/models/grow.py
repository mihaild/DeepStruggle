"""Growing a trained LadderNet in place: the grouped input projections made wider without changing what it plays.

`--resume-widen` (owner, 2026-10-13): the board and card projections (`lad_board`, `lad_card`: the whole
board and the whole card block each into `entity_proj_dim` numbers through one linear layer) are where
the trunk loses state (research/findings/training/trunk_depth_and_state.md), and a run resumed into a
model whose projections are k times wider continues from its state instead of restarting.

Each new unit is a copy of an old one (new unit i is old unit i mod p), and the fusion layer reads each
copy with 1/k of the old unit's weight. A LayerNorm over a vector tiled k times has the same mean and
variance, so every copy carries exactly the old unit's activation and the fusion layer's input sum is
unchanged: the widened network computes the same function. Copies with identical weights would get
identical gradients and never separate, so each copy's incoming weights get a perturbation that sums to
zero over its copies -- the output then moves only at second order.

Adam's moments follow the gradients: a copy's incoming weights and LayerNorm parameters see 1/k of the old
unit's gradient (its outgoing weight is 1/k), so their first moment is scaled by 1/k and the second by
1/k^2, which keeps each element's step the size it was; a fusion column sees the old column's gradient
and keeps its moments.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import torch

#: The grouped projections, in the order the fusion layer reads them (board, then cards).
PROJECTIONS = ("lad_board", "lad_card")
FUSION = "fusion_in.0.weight"


def widen_factor(saved: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor]) -> int:
    """How many times wider the target's projections are than the saved ones; 1 when equal."""
    old = int(saved["lad_board.0.weight"].shape[0])
    new = int(target["lad_board.0.weight"].shape[0])
    if new % old or int(target["lad_card.0.weight"].shape[0]) != new or int(saved["lad_card.0.weight"].shape[0]) != old:
        raise ValueError(f"the target's projections ({new}) are not an integer multiple of the saved ones ({old}), "
                         "or the board and card projections differ in width")
    return new // old


def _tile_rows(t: torch.Tensor, k: int) -> torch.Tensor:
    return t.repeat((k,) + (1,) * (t.dim() - 1))


def _fusion_cols(w: torch.Tensor, p: int, k: int, scale: float) -> torch.Tensor:
    """Fusion weight columns: the board and card blocks tiled k times (times `scale`), the rest as they were."""
    board, card, rest = w[:, :p], w[:, p:2 * p], w[:, 2 * p:]
    return torch.cat([board.repeat(1, k) * scale, card.repeat(1, k) * scale, rest], dim=1)


def widen_state_dict(sd: Mapping[str, torch.Tensor], k: int, noise: float = 1e-3,
                     generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
    """The state dict of the same network with its grouped projections k times wider.

    `noise` is the perturbation's size relative to each projection weight's own standard deviation; the
    perturbation of a unit's k copies sums to zero. The same `generator` state gives the same perturbation,
    so a network and an identical reference widen identically.
    """
    if k < 1:
        raise ValueError("the widening factor must be >= 1")
    out = {name: t.clone() for name, t in sd.items()}
    if k == 1:
        return out
    p = int(sd["lad_board.0.weight"].shape[0])
    for proj in PROJECTIONS:
        w = sd[f"{proj}.0.weight"]
        tiled = _tile_rows(w, k)
        if noise > 0:
            eps = torch.randn((k,) + tuple(w.shape), generator=generator, dtype=torch.float32).to(w.device, w.dtype)
            eps = (eps - eps.mean(dim=0, keepdim=True)) * (noise * float(w.float().std()))
            tiled = tiled + eps.reshape(tiled.shape)
        out[f"{proj}.0.weight"] = tiled
        for leaf in (f"{proj}.0.bias", f"{proj}.1.weight", f"{proj}.1.bias"):
            out[leaf] = _tile_rows(sd[leaf], k)
    out[FUSION] = _fusion_cols(sd[FUSION], p, k, 1.0 / k)
    return out


def widen_optimizer_state(saved: Dict[str, Any], names: Sequence[str], p: int, k: int) -> Dict[str, Any]:
    """An Adam state dict for the widened network: `names` are the model's parameter names in
    `named_parameters()` order (the order the saved state indexes), `p` the saved projection width."""
    row_params = {f"{proj}.{leaf}" for proj in PROJECTIONS for leaf in ("0.weight", "0.bias", "1.weight", "1.bias")}
    state: Dict[Any, Any] = {}
    for idx, st in saved["state"].items():
        name = names[int(idx)]
        new_st = dict(st)
        for key, power in (("exp_avg", 1), ("exp_avg_sq", 2)):
            if key not in st:
                continue
            t = st[key]
            if name in row_params:
                new_st[key] = _tile_rows(t, k) / float(k ** power)
            elif name == FUSION:
                new_st[key] = _fusion_cols(t, p, k, 1.0)
        state[idx] = new_st
    return {"state": state, "param_groups": [dict(g) for g in saved["param_groups"]]}


# ---- token layers grown at identity (P35 item 1's grown read) ---------------------------------------

#: The per-entity heads' first layers: their input is (identity, token, raw slots, trunk context), so a
#: grown token path widens them at [identity_dim, identity_dim + token_dim).
ENTITY_HEADS = ("pe_country.0.weight", "pe_card.0.weight")


def _insert_zero_cols(w: torch.Tensor, at: int, n: int) -> torch.Tensor:
    return torch.cat([w[:, :at], w.new_zeros(w.shape[0], n), w[:, at:]], dim=1)


def grow_tokens_state_dict(saved: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor],
                           token_dim: int, identity_dim: int) -> Dict[str, torch.Tensor]:
    """The state dict of `target` -- the saved network with `--ladder-token-layers` added -- that computes
    exactly what the saved network did: every saved tensor kept, the token modules as `target` initialised
    them, and zero weights wherever the fusion layer (its last `token_dim` inputs: the global token) and the
    per-entity heads (inputs [identity_dim, identity_dim + token_dim): each entity's token) read the new
    path. The zero weights take gradient at once, so the token path starts learning from the first step
    (the Cell A pattern)."""
    if any(k.startswith("region_attn.") for k in target):
        raise ValueError("growing token layers under region attention is not defined: its input widens too")
    extra = sorted(set(saved) - set(target) - {"keep_idx"})
    if extra:
        raise ValueError(f"the saved state has tensors the grown network lacks: {extra[:5]}")
    out: Dict[str, torch.Tensor] = {}
    for name, t in target.items():
        s = saved.get(name)
        if s is not None and s.shape == t.shape:
            out[name] = s.clone()
        elif name == FUSION and s is not None and s.shape[1] + token_dim == t.shape[1]:
            out[name] = _insert_zero_cols(s, s.shape[1], token_dim)
        elif name in ENTITY_HEADS and s is not None and s.shape[1] + token_dim == t.shape[1]:
            out[name] = _insert_zero_cols(s, identity_dim, token_dim)
        elif s is None and name.startswith("tok_"):
            out[name] = t.clone()
        else:
            raise ValueError(f"cannot grow {name}: saved {None if s is None else tuple(s.shape)}, "
                             f"target {tuple(t.shape)}")
    return out


def grow_tokens_optimizer_state(saved: Dict[str, Any], names: Sequence[str], token_dim: int,
                                identity_dim: int) -> Dict[str, Any]:
    """Adam's state for the grown network: `names` are its parameter names in `named_parameters()` order.
    The saved state indexed the same names less the token modules; a widened weight's moments get zero
    columns where its weights did, the token modules start without state (Adam initialises it)."""
    old = [n for n in names if not n.startswith("tok_")]
    state: Dict[Any, Any] = {}
    for i, name in enumerate(names):
        if name.startswith("tok_"):
            continue
        st = saved["state"].get(old.index(name))
        if st is None:
            continue
        new_st = dict(st)
        for key in ("exp_avg", "exp_avg_sq"):
            if key not in st:
                continue
            m = st[key]
            if name == FUSION:
                new_st[key] = _insert_zero_cols(m, m.shape[1], token_dim)
            elif name in ENTITY_HEADS:
                new_st[key] = _insert_zero_cols(m, identity_dim, token_dim)
        state[i] = new_st
    groups = [dict(g) for g in saved["param_groups"]]
    groups[0]["params"] = list(range(len(names)))
    return {"state": state, "param_groups": groups}


#: P42: the dense policy output, which grows from the base flat width to the macro view's.
POLICY_OUT = ("policy_head.3.weight", "policy_head.3.bias")
#: The macro view's per-country head, absent from a base network.
MACRO_HEAD = "pe_macro."
#: The grown until-control rows' bias: on J5's 700M positions -10 left up to 1.8e-4 of probability on them, past the
#: identity bar of 1e-4; -12 is ~e^-2 of that.
MACRO_START_BIAS = -12.0


def grow_macro_state_dict(saved: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor],
                          base_size: int, bias: float = MACRO_START_BIAS) -> Dict[str, torch.Tensor]:
    """The state dict of `target` -- the saved base network in P42's macro view -- that plays what the saved one
    did on every base slot: every saved tensor kept, the dense policy output's new rows (the until-control slots)
    at weight 0 and bias `bias`, and the macro head as `target` initialised it (its output layer is zero). The
    until-control slots then start at ~e^bias against the base options, and the zero-avoiding term lifts them."""
    extra = sorted(set(saved) - set(target) - {"keep_idx"})
    if extra:
        raise ValueError(f"the saved state has tensors the grown network lacks: {extra[:5]}")
    out: Dict[str, torch.Tensor] = {}
    for name, t in target.items():
        s = saved.get(name)
        if s is not None and s.shape == t.shape:
            out[name] = s.clone()
        elif name in POLICY_OUT and s is not None and s.shape[0] == base_size and t.shape[0] > base_size:
            new = t.new_full((t.shape[0] - base_size,) + tuple(t.shape[1:]),
                             0.0 if name.endswith("weight") else bias)
            out[name] = torch.cat([s.to(t.dtype), new], dim=0)
        elif s is None and name.startswith(MACRO_HEAD):
            out[name] = t.clone()
        else:
            raise ValueError(f"cannot grow {name} into the macro view: saved "
                             f"{None if s is None else tuple(s.shape)}, target {tuple(t.shape)}")
    return out


def grow_macro_optimizer_state(saved: Dict[str, Any], names: Sequence[str], sizes: Mapping[str, int],
                               base_size: int) -> Dict[str, Any]:
    """Adam's state for the grown network: `names` are its parameter names in `named_parameters()` order, `sizes`
    the grown dense output's row count per name. The saved state indexed the same names less the macro head; the
    dense output's moments get zero rows for the new slots, the macro head starts without state. The groups come back
    as one in model order, which the trainer maps onto its own (per-rate) groups, as for any added head."""
    old = [n for n in names if not n.startswith(MACRO_HEAD)]
    new_index = {name: i for i, name in enumerate(names)}
    state: Dict[Any, Any] = {}
    for j, name in enumerate(old):
        st = saved["state"].get(j)
        if st is None:
            continue
        new_st = dict(st)
        if name in POLICY_OUT:
            for key in ("exp_avg", "exp_avg_sq"):
                m = st.get(key)
                if m is None:
                    continue
                if m.shape[0] != base_size:
                    raise ValueError(f"{name}'s moments have {m.shape[0]} rows, expected {base_size}")
                pad = m.new_zeros((sizes[name] - base_size,) + tuple(m.shape[1:]))
                new_st[key] = torch.cat([m, pad], dim=0)
        state[new_index[name]] = new_st
    flat = [int(i) for g in saved["param_groups"] for i in g["params"]]
    if flat != list(range(len(old))):
        raise ValueError(f"the saved optimizer does not index the {len(old)} parameters of the network less its macro "
                         f"head in model order")
    group = dict(saved["param_groups"][0])
    group["params"] = list(range(len(names)))
    return {"state": state, "param_groups": [group]}
