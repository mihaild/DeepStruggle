#!/usr/bin/env python3
"""P30: can the architecture represent card x board interactions, and has RL learned them?

Reads a dataset from `card_board_targets.py` (positions; per held card, T2 ops arithmetic,
T3 event can fire, T4 event outcome) and answers two questions on held-out games.

**1. Representable?** Four architectures trained from scratch, supervised, on the same targets,
at several data sizes:

* `shallow` -- the E6-12-44 trunk (grouped projections, no residual blocks) with a per-card head;
* `m2d` -- the same with four residual blocks (E6-03-44's trunk);
* `attention` -- every country and card a token (its raw row plus a learned identity), two
  transformer layers over all 195 tokens with a global token, each card read from its own token:
  an explicit card <-> country mechanism;
* `mlp` -- two 1024-wide layers over the raw observation.

**2. Learned?** The same head fitted on the frozen hidden vector of trained nets (and of an
untrained one, the floor), and a behaviour test: at card-selection decisions, does the policy's
preference for a card move with that card's board-dependent event outcome? The preference is the
card's logit less the mean over the held cards, the outcome a T4 quantity, and both are demeaned
per card, so a card's fixed popularity does not count -- only the board-dependent part.

Every score is given overall and **within card** (each card's mean removed from target and
prediction first), which isolates the interaction: how the target varies with the board for a
fixed card.

    PYTHONPATH=.:build/release python tools/scripts/card_board_probe.py --data d.npz \\
        --frozen a.pt b.pt --behaviour a.pt --sizes 25000 100000 0 --output-json o.json --output-md o.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from ai.models.hyperspherical import register_sphere_projection  # noqa: E402
from ai.models.ladder_net import create_ladder_net  # noqa: E402
from tools.lib.player_agent import NeuralAgent  # noqa: E402
from ai.training.card_event_targets import (BOARD_W, CARD_OFFSET, CARD_W, GLOBAL_OFFSET, N_CARDS,  # noqa: E402
                                            N_COUNTRIES, N_T2, SCORING, T2_NAMES, T4_NAMES)
from tools.scripts.trunk_state_probe import auc  # noqa: E402

D2, D4 = N_T2, 12
D = D2 + 1 + D4
SHALLOW: Dict[str, Any] = {
    "input_mode": "grouped", "aggregation": "flatten", "entity_dim": 16, "entity_proj_dim": 256,
    "card_self_attention": False, "cross_attention": False, "per_entity_heads": 64, "head_context": True,
    "head_static": True, "head_entities": "country", "head_center": True, "identity_dim": 0,
    "drop_static": True, "hidden_dim": 480, "num_res_blocks": 0, "num_attn_heads": 4, "card_lookup": False,
    "card_lookup_heads": 0, "card_lookup_dim": 0, "card_lookup_identity_dim": 0,
    "categorical_value": False}


class CardHead(nn.Module):
    """hidden vector -> (B, 110, D)."""

    def __init__(self, d_in: int, width: int = 512) -> None:
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d_in, width), nn.GELU(), nn.Linear(width, N_CARDS * D))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.f(h).view(-1, N_CARDS, D)


class LadderCard(nn.Module):
    def __init__(self, blocks: int) -> None:
        super().__init__()
        self.trunk: Any = create_ladder_net("cpu", **{**SHALLOW, "num_res_blocks": blocks})
        self.head = CardHead(480)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk.extract_features(x))


class MLPCard(nn.Module):
    def __init__(self, width: int = 1024) -> None:
        super().__init__()
        self.f = nn.Sequential(nn.Linear(GLOBAL_OFFSET + 100, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.head = CardHead(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.f(x))


class AttentionCard(nn.Module):
    """Countries, cards and one global token; full self-attention; each card read from its token."""

    def __init__(self, d: int = 128, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.country_in = nn.Linear(BOARD_W, d)
        self.card_in = nn.Linear(CARD_W, d)
        self.global_in = nn.Linear(100, d)
        self.country_id = nn.Parameter(torch.randn(N_COUNTRIES, d) * 0.02)
        self.card_id = nn.Parameter(torch.randn(N_CARDS, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        c = self.country_in(x[:, :CARD_OFFSET].view(b, N_COUNTRIES, BOARD_W)) + self.country_id
        k = self.card_in(x[:, CARD_OFFSET:GLOBAL_OFFSET].view(b, N_CARDS, CARD_W)) + self.card_id
        g = self.global_in(x[:, GLOBAL_OFFSET:GLOBAL_OFFSET + 100]).unsqueeze(1)
        t = self.norm(self.enc(torch.cat([g, c, k], 1)))
        cards = t[:, 1 + N_COUNTRIES:]
        return self.out(torch.cat([cards, t[:, :1].expand(-1, N_CARDS, -1)], -1))


class CrossCard(nn.Module):
    """The cheap C1 candidate: card tokens (and one global token) query the country tokens in one
    cross-attention block; countries do not attend. About a quarter of the cost of a full layer."""

    def __init__(self, d: int = 64, heads: int = 4) -> None:
        super().__init__()
        self.country_in = nn.Linear(BOARD_W, d)
        self.card_in = nn.Linear(CARD_W, d)
        self.global_in = nn.Linear(100, d)
        self.country_id = nn.Parameter(torch.randn(N_COUNTRIES, d) * 0.02)
        self.card_id = nn.Parameter(torch.randn(N_CARDS, d) * 0.02)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        c = self.country_in(x[:, :CARD_OFFSET].view(b, N_COUNTRIES, BOARD_W)) + self.country_id
        k = self.card_in(x[:, CARD_OFFSET:GLOBAL_OFFSET].view(b, N_CARDS, CARD_W)) + self.card_id
        g = self.global_in(x[:, GLOBAL_OFFSET:GLOBAL_OFFSET + 100]).unsqueeze(1)
        q = torch.cat([g, k], 1)
        a, _ = self.att(self.ln(q), c, c, need_weights=False)
        q = q + a
        t = self.norm(q + self.ff(q))
        return self.out(torch.cat([t[:, 1:], t[:, :1].expand(-1, N_CARDS, -1)], -1))


class Targets:
    """Held-card targets, standardised on the fitting rows."""

    def __init__(self, d: Dict[str, np.ndarray], fit: np.ndarray, dev: torch.device) -> None:
        self.cards = torch.from_numpy(d["cards"].astype(np.int64)).to(dev)                 # (N, 12), 0 = none
        m2, m4 = d["m2"], d["m4"]
        f2 = d["y2"][fit][m2[fit]]
        f4 = d["y4"][fit][m4[fit]]
        self.mu2, self.sd2 = f2.mean(0), f2.std(0) + 1e-6
        self.mu4, self.sd4 = f4.mean(0), f4.std(0) + 1e-6
        self.y2 = torch.from_numpy((d["y2"] - self.mu2) / self.sd2).float().to(dev)
        self.y4 = torch.from_numpy((d["y4"] - self.mu4) / self.sd4).float().to(dev)
        self.y3 = torch.from_numpy(d["y3"]).float().to(dev)
        self.m2 = torch.from_numpy(m2).to(dev)
        self.m3 = torch.from_numpy(d["m3"]).to(dev)
        self.m4 = torch.from_numpy(m4).to(dev)

    def gather(self, out: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """(B, 110, D) at the held cards of rows idx -> (B, 12, D)."""
        c = (self.cards[idx] - 1).clamp_min(0)
        return torch.gather(out, 1, c.unsqueeze(-1).expand(-1, -1, D))

    def loss(self, out: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        g = self.gather(out, idx)
        m2, m3, m4 = self.m2[idx], self.m3[idx], self.m4[idx]
        l2 = ((g[..., :D2] - self.y2[idx]) ** 2).mean(-1)[m2].mean() if m2.any() else g.sum() * 0
        l3 = F.binary_cross_entropy_with_logits(g[..., D2][m3], self.y3[idx][m3]) if m3.any() else g.sum() * 0
        l4 = ((g[..., D2 + 1:] - self.y4[idx]) ** 2).mean(-1)[m4].mean() if m4.any() else g.sum() * 0
        return l2 + l3 + l4


def fit_model(model: nn.Module, inputs: Callable[[torch.Tensor], torch.Tensor], tg: Targets, fit_idx: np.ndarray,
              val_idx: np.ndarray, dev: torch.device, lr: float, max_epochs: int, seed: int,
              batch: int = 512) -> Tuple[nn.Module, Dict[str, Any]]:
    torch.manual_seed(seed)
    model = model.to(dev)
    opt = register_sphere_projection(torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4), model)
    fi = torch.from_numpy(fit_idx).to(dev)
    vi = torch.from_numpy(val_idx).to(dev)

    def val_loss() -> float:
        model.eval()
        with torch.no_grad():
            tot = 0.0
            for k in range(0, vi.numel(), 4096):
                j = vi[k:k + 4096]
                tot += float(tg.loss(model(inputs(j)), j)) * j.numel()
        return tot / max(1, vi.numel())

    best, best_state, stale, epochs = float("inf"), None, 0, 0
    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        order = fi[torch.randperm(fi.numel(), device=dev)]
        for k in range(0, order.numel(), batch):
            j = order[k:k + batch]
            loss = tg.loss(model(inputs(j)), j)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        v = val_loss()
        epochs = ep + 1
        if v < best - 1e-4:
            best, stale = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 3:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, {"epochs": epochs, "val_loss": best, "seconds": round(time.time() - t0, 1),
                   "params": int(sum(p.numel() for p in model.parameters()))}


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    v = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - ((y - p) ** 2).sum() / v) if v > 0 else float("nan")


def _within(y: np.ndarray, p: np.ndarray, card: np.ndarray) -> float:
    """R2 after removing each card's mean from target and prediction."""
    yy, pp = y.copy(), p.copy()
    for c in np.unique(card):
        s = card == c
        yy[s] -= y[s].mean()
        pp[s] -= p[s].mean()
    return _r2(yy, pp)


def evaluate(model: nn.Module, inputs: Callable[[torch.Tensor], torch.Tensor], tg: Targets,
             d: Dict[str, np.ndarray], test_idx: np.ndarray, dev: torch.device) -> Dict[str, Any]:
    ti = torch.from_numpy(test_idx).to(dev)
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for k in range(0, ti.numel(), 4096):
            j = ti[k:k + 4096]
            outs.append(tg.gather(model(inputs(j)), j).float().cpu().numpy())
    g = np.concatenate(outs)
    cards = d["cards"][test_idx].astype(np.int64)
    res: Dict[str, Any] = {}
    m2 = d["m2"][test_idx]
    y2 = d["y2"][test_idx][m2]
    p2 = g[..., :D2][m2] * tg.sd2 + tg.mu2
    c2 = cards[m2]
    res["T2"] = {n: {"r2": _r2(y2[:, i], p2[:, i]), "within": _within(y2[:, i], p2[:, i], c2)}
                 for i, n in enumerate(T2_NAMES)}
    m3 = d["m3"][test_idx]
    res["T3"] = {"auc": auc(g[..., D2][m3], d["y3"][test_idx][m3]), "base_rate": float(d["y3"][test_idx][m3].mean())}
    m4 = d["m4"][test_idx]
    y4 = d["y4"][test_idx][m4]
    p4 = g[..., D2 + 1:][m4] * tg.sd4 + tg.mu4
    c4 = cards[m4]
    res["T4"] = {n: {"r2": _r2(y4[:, i], p4[:, i]), "within": _within(y4[:, i], p4[:, i], c4)}
                 for i, n in enumerate(T4_NAMES)}
    sc = np.isin(c4, SCORING)
    res["T1"] = {"r2": _r2(y4[sc, 0], p4[sc, 0]), "within": _within(y4[sc, 0], p4[sc, 0], c4[sc]), "n": int(sc.sum())}
    return res


def summary_row(r: Dict[str, Any]) -> Dict[str, float]:
    t2w = float(np.nanmean([v["within"] for v in r["T2"].values()]))
    t4w = float(np.nanmean([v["within"] for v in r["T4"].values()]))
    t4 = float(np.nanmean([v["r2"] for v in r["T4"].values()]))
    return {"T1 within": r["T1"]["within"], "T2 within": t2w, "T3 AUC": r["T3"]["auc"], "T4 R2": t4, "T4 within": t4w}


@torch.no_grad()
def behaviour(model: Any, d: Dict[str, np.ndarray], dev: torch.device) -> Dict[str, Any]:
    """At card selections: within-card correlation of the card's relative logit with its T4 outcome."""
    sel = np.flatnonzero(d["decision"] == 1)
    lg: List[np.ndarray] = []
    for k in range(0, len(sel), 4096):
        j = sel[k:k + 4096]
        o = torch.from_numpy(d["obs"][j]).float().to(dev)
        m = torch.from_numpy(d["mask"][j]).to(dev)
        lg.append(model(o, m)[0].float()[:, :N_CARDS].cpu().numpy())
    logits = np.concatenate(lg)
    rel, card, out = [], [], []
    for r, i in enumerate(sel):
        cs = d["cards"][i].astype(np.int64)
        legal = [(j, c) for j, c in enumerate(cs) if c > 0 and d["mask"][i][c - 1]]
        if len(legal) < 2:
            continue
        ls = np.array([logits[r, c - 1] for _, c in legal])
        ls = ls - ls.mean()
        for (j, c), l in zip(legal, ls):
            if d["m4"][i, j]:
                y = d["y4"][i, j]
                rel.append(l)
                card.append(c)
                out.append([y[0], y[2:8].sum(), y[8] - y[9], y[10] - y[11]])
    rel_a, card_a, out_a = np.array(rel), np.array(card), np.array(out)
    names = ("VP", "sum of regional margins", "battleground balance", "influence balance")
    res: Dict[str, Any] = {"n": int(len(rel_a))}
    for k, n in enumerate(names):
        a, b = rel_a.copy(), out_a[:, k].copy()
        for c in np.unique(card_a):
            s = card_a == c
            a[s] -= a[s].mean()
            b[s] -= b[s].mean()
        ok = np.abs(b) > 0
        res[n] = {"within_corr": float(np.corrcoef(a, b)[0, 1]) if b.std() > 0 else float("nan"),
                  "varies_frac": float(ok.mean())}
    return res


def render(res: Dict[str, Any]) -> str:
    L = ["# Card × board targets: representable, and learned?", "",
         f"Data: `{res['data']}` -- {res['positions']} positions, test {res['test_positions']} "
         f"(held-out games). *within*: R² with each card's mean removed from target and prediction, "
         f"i.e. only the board-dependent part. T3's base rate is {100 * res['t3_base']:.1f}% can fire.", "",
         "## 1. Trained from scratch (representable?)", "",
         "| architecture | positions | params | T1 within | T2 within | T3 AUC | T4 R² | **T4 within** | epochs |",
         "|:---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in res["from_scratch"]:
        s = row["summary"]
        L.append(f"| {row['arch']} | {row['n_fit']} | {row['fit']['params']:,} | {s['T1 within']:.3f} | "
                 f"{s['T2 within']:.3f} | {s['T3 AUC']:.3f} | {s['T4 R2']:.3f} | **{s['T4 within']:.3f}** | "
                 f"{row['fit']['epochs']} |")
    L += ["", "## 2a. The same head on a frozen trunk (learned?)", "",
          "| trunk | T1 within | T2 within | T3 AUC | T4 R² | **T4 within** |", "|:---|---:|---:|---:|---:|---:|"]
    for row in res["frozen"]:
        s = row["summary"]
        L.append(f"| {row['name']} | {s['T1 within']:.3f} | {s['T2 within']:.3f} | {s['T3 AUC']:.3f} | "
                 f"{s['T4 R2']:.3f} | **{s['T4 within']:.3f}** |")
    L += ["", "## 2b. Behaviour at card selections", "",
          "Within-card correlation of the card's relative logit with its event outcome (both demeaned per card).", "",
          "| net | n | VP | sum of regional margins | battleground balance | influence balance |",
          "|:---|---:|---:|---:|---:|---:|"]
    for name, b in res["behaviour"].items():
        L.append(f"| {name} | {b['n']} | " + " | ".join(
            f"{b[k]['within_corr']:+.3f}" for k in ("VP", "sum of regional margins", "battleground balance",
                                                   "influence balance")) + " |")
    L += ["", "## Per-quantity detail (largest from-scratch fit of each architecture; within-card R²)", "",
          "| quantity | " + " | ".join(r["arch"] for r in res["detail"]) + " |",
          "|:---|" + "---:|" * len(res["detail"])]
    for n in T2_NAMES:
        L.append(f"| T2 {n} | " + " | ".join(f"{r['eval']['T2'][n]['within']:.3f}" for r in res["detail"]) + " |")
    for n in T4_NAMES:
        L.append(f"| T4 {n} | " + " | ".join(f"{r['eval']['T4'][n]['within']:.3f}" for r in res["detail"]) + " |")
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--archs", nargs="+", default=["shallow", "m2d", "attention", "mlp"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[25_000, 100_000, 0], help="fitting positions; 0 = all")
    ap.add_argument("--frozen", nargs="*", default=[], help="checkpoints whose frozen trunk gets the head")
    ap.add_argument("--behaviour", nargs="*", default=[], help="checkpoints whose policy is tested")
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--output-md", default=None)
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    z = np.load(a.data)
    d = {k: z[k] for k in z.files if k != "policies"}
    n = d["obs"].shape[0]
    rng = np.random.default_rng(a.seed)
    games = np.unique(d["game"])
    test_g = rng.choice(games, int(len(games) * a.test_frac), replace=False)
    test = np.isin(d["game"], test_g)
    rest_g = np.setdiff1d(games, test_g)
    val_g = rng.choice(rest_g, max(1, len(rest_g) // 10), replace=False)
    val = np.isin(d["game"], val_g) & ~test
    fit = ~test & ~val
    fit_all, val_idx, test_idx = np.flatnonzero(fit), np.flatnonzero(val), np.flatnonzero(test)
    tg = Targets(d, fit, dev)
    x_all = torch.from_numpy(d["obs"]).to(dev)                        # float16 on the device

    def raw(j: torch.Tensor) -> torch.Tensor:
        return x_all[j].float()

    res: Dict[str, Any] = {"data": a.data, "positions": n, "test_positions": int(test.sum()),
                           "t3_base": float(d["y3"][d["m3"]].mean()), "from_scratch": [], "frozen": [],
                           "behaviour": {}, "detail": []}
    makers: Dict[str, Tuple[Callable[[], nn.Module], float]] = {
        "shallow": (lambda: LadderCard(0), 1e-3), "m2d": (lambda: LadderCard(4), 1e-3),
        "attention": (lambda: AttentionCard(), 5e-4), "mlp": (lambda: MLPCard(), 1e-3),
        # P30 C1 throughput candidates (full attention costs ~5x the training throughput)
        "attn_d64_l1": (lambda: AttentionCard(64, 1), 5e-4), "attn_d32_l1": (lambda: AttentionCard(32, 1), 5e-4),
        "attn_d128_l1": (lambda: AttentionCard(128, 1), 5e-4),
        "attn_d64_l2": (lambda: AttentionCard(64, 2), 5e-4), "attn_d32_l2": (lambda: AttentionCard(32, 2), 5e-4),
        "cross_d64": (lambda: CrossCard(64), 5e-4), "cross_d128": (lambda: CrossCard(128), 5e-4)}
    for arch in a.archs:
        make, lr = makers[arch]
        best_eval: Optional[Dict[str, Any]] = None
        for size in a.sizes:
            fi = fit_all if size <= 0 or size >= len(fit_all) else np.sort(rng.choice(fit_all, size, replace=False))
            model, info = fit_model(make(), raw, tg, fi, val_idx, dev, lr, a.max_epochs, a.seed)
            ev = evaluate(model, raw, tg, d, test_idx, dev)
            res["from_scratch"].append({"arch": arch, "n_fit": int(len(fi)), "fit": info, "summary": summary_row(ev)})
            best_eval = ev
            print(f"{arch} @ {len(fi)}: {summary_row(ev)} ({info['seconds']}s)", flush=True)
            del model
            torch.cuda.empty_cache()
        assert best_eval is not None
        res["detail"].append({"arch": arch, "eval": best_eval})

    for path in a.frozen:
        ag = NeuralAgent.from_checkpoint(path, device=str(dev))
        net: Any = ag.model.eval()
        hs: List[torch.Tensor] = []
        with torch.no_grad():
            for k in range(0, n, 4096):
                hs.append(net.extract_features(x_all[k:k + 4096].float()).float())
        h = torch.cat(hs)
        del net
        head, info = fit_model(CardHead(h.shape[1]), lambda j: h[j], tg, fit_all, val_idx, dev, 1e-3,
                               a.max_epochs, a.seed)
        ev = evaluate(head, lambda j: h[j], tg, d, test_idx, dev)
        res["frozen"].append({"name": ag.name, "fit": info, "summary": summary_row(ev), "eval": ev})
        print(f"frozen {ag.name}: {summary_row(ev)}", flush=True)
        del h, head
        torch.cuda.empty_cache()

    for path in a.behaviour:
        ag = NeuralAgent.from_checkpoint(path, device=str(dev))
        res["behaviour"][ag.name] = behaviour(ag.model.eval(), d, dev)
        print(f"behaviour {ag.name}: {res['behaviour'][ag.name]}", flush=True)

    md = render(res)
    print(md)
    if a.output_json:
        with open(a.output_json, "w") as f:
            json.dump(res, f, indent=1, default=float)
    if a.output_md:
        with open(a.output_md, "w") as f:
            f.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
