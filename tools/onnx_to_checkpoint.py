#!/usr/bin/env python3
"""Rebuild a torch checkpoint from a published ONNX export, checked against the export.

Some published models exist only as ONNX, and a searcher (`search:` / `gumbel:` specs) needs the
torch network. The export carries every weight the forward pass reads: most under their parameter
names, a few folded by the exporter into anonymous constants (a per-entity head's weights become
`val_*`, transposed). This starts from a **template** checkpoint of the same architecture,
overwrites each parameter with the export's -- by name, or else by the one anonymous constant of
the transposed shape -- and keeps the template's tensors the export does not carry: fixed buffers
(the map's adjacency, a centring flag) and heads the forward pass of play does not use (DEFCON
risk). Those are listed.

Nothing is written unless the rebuilt network reproduces the export: logits, v_win and v_vp on
positions from real self-play, to `--tol`, with the same favourite move at every one. A name that
matches nothing, or a constant that fits two parameters, refuses rather than guesses.

    PYTHONPATH=.:build/release python tools/onnx_to_checkpoint.py --onnx newest.onnx \\
        --template shallow_E7-02+03+04+05_1200M.pt --out newest.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import ts_engine as ts
from onnx import numpy_helper

from bindings.action_encoder import ActionEncoder
from bindings.ts_env import legal_mask, model_obs_features, observe
from tools.lib.player_agent import NeuralAgent


def rebuild(onnx_path: str, template_path: str) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
    """(state dict, the parameters taken from anonymous constants, the template tensors kept)."""
    model = onnx.load(onnx_path)
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    sd = torch.load(template_path, map_location="cpu", weights_only=True)
    anonymous = {k: v for k, v in inits.items() if k not in sd and v.dtype == np.float32 and v.ndim >= 1}
    used: set = set()
    out: Dict[str, torch.Tensor] = {}
    folded: List[str] = []
    kept: List[str] = []
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
            continue
        if k in inits:
            arr = inits[k]
            if tuple(arr.shape) != tuple(v.shape):
                raise SystemExit(f"{k}: the export holds {arr.shape}, the template {tuple(v.shape)} -- "
                                 f"not the same architecture")
            out[k] = torch.from_numpy(arr.copy()).to(v.dtype)
            continue
        if v.ndim == 2:
            fits = [n for n, a in anonymous.items() if n not in used and a.shape == (v.shape[1], v.shape[0])]
            if len(fits) > 1:
                raise SystemExit(f"{k}: {len(fits)} anonymous constants fit it transposed ({fits}); refusing to guess")
            if fits:
                used.add(fits[0])
                out[k] = torch.from_numpy(anonymous[fits[0]].T.copy()).to(v.dtype)
                folded.append(f"{k} <- {fits[0]}ᵀ")
                continue
        out[k] = v
        kept.append(k)
    return out, folded, kept


def _positions(n: int, seed: int) -> List:
    """Decision positions from a few random games -- real positions, not random tensors."""
    rng = np.random.default_rng(seed)
    out: List = []
    g = 0
    while len(out) < n:
        st = ts.GameState()
        ts.Engine.init_game(st, int(seed * 1000 + g))
        g += 1
        for _ in range(2000):
            ts.Engine.auto_advance_step(st)
            if ts.Engine.is_terminal(st):
                break
            mask = np.asarray(ActionEncoder.get_legal_mask(st))
            legal = np.flatnonzero(mask)
            if len(legal) > 1 and rng.random() < 0.1:
                out.append(st.clone())
            ts.Engine.step_flat(st, int(rng.choice(legal)), True)
    return out[:n]


def verify(sd: Dict[str, torch.Tensor], onnx_path: str, n: int, tol: float) -> Tuple[float, float, float]:
    """Max |torch - onnx| over logits (legal moves), v_win and v_vp; raises on a different favourite."""
    tmp = onnx_path + ".verify.pt"
    torch.save(sd, tmp)
    try:
        net = NeuralAgent.from_checkpoint(tmp, device="cpu").model.eval()
    finally:
        os.remove(tmp)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    # In the network's own view, as the export was verified: its observation feature set (which
    # must be the one the export says it reads) and its action view.
    meta: Dict[str, str] = dict(sess.get_modelmeta().custom_metadata_map)
    features = model_obs_features(net)
    if int(meta.get("ts.obs_features", "0")) != features:
        raise SystemExit(f"the rebuilt network reads observation features {features:#x}, the export "
                         f"{int(meta.get('ts.obs_features', '0')):#x}: the template is not this network's")
    merged = meta.get("ts.merged_influence") == "true"
    states = _positions(n, 7)
    mover = [s.ctx().decision_player if s.ctx().decision_player != ts.Player.NONE else s.phasing_player
             for s in states]
    obs = np.stack([observe(s, m, features) for s, m in zip(states, mover)])
    masks = np.stack([legal_mask(s, merged, None) for s in states])
    lo, vw, vv = (np.asarray(x, dtype=np.float64) for x in
                  sess.run(["logits", "v_win", "v_vp"], {"obs": obs, "mask": masks}))
    with torch.no_grad():
        lt, vwt, vvt = net(torch.from_numpy(obs), torch.from_numpy(masks))[:3]
    legal = masks.astype(bool)
    dl = float(np.abs(np.where(legal, lt.numpy() - lo, 0.0)).max())
    dw = float(np.abs(vwt.numpy().reshape(-1) - vw.reshape(-1)).max())
    dv = float(np.abs(vvt.numpy().reshape(-1) - vv.reshape(-1)).max())
    same = (np.where(legal, lt.numpy(), -np.inf).argmax(1) == np.where(legal, lo, -np.inf).argmax(1)).all()
    if not same or max(dl, dw, dv) > tol:
        raise SystemExit(f"the rebuilt network does not reproduce the export over {n} positions: "
                         f"|Δlogits| {dl:.2e}, |Δv_win| {dw:.2e}, |Δv_vp| {dv:.2e}, same favourite: {bool(same)}")
    return dl, dw, dv


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--template", required=True, help="a .pt checkpoint of the same architecture")
    ap.add_argument("--out", required=True)
    ap.add_argument("--positions", type=int, default=512)
    ap.add_argument("--tol", type=float, default=1e-3)
    a = ap.parse_args(argv)
    sd, folded, kept = rebuild(a.onnx, a.template)
    dl, dw, dv = verify(sd, a.onnx, a.positions, a.tol)
    torch.save(sd, a.out)
    print(f"{a.out}: {len(sd)} tensors; from anonymous constants: {', '.join(folded) or 'none'}; "
          f"kept from the template: {', '.join(kept) or 'none'}", file=sys.stderr)
    print(f"matches {a.onnx} over {a.positions} positions: |Δlogits| {dl:.1e}, |Δv_win| {dw:.1e}, "
          f"|Δv_vp| {dv:.1e}, same favourite everywhere", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
