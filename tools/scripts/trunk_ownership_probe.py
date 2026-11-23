#!/usr/bin/env python3
"""P29 bet 2: does a trunk's hidden vector encode who will control each country?

`aux_ownership_probe.py` found the online ownership head no better than a per-country habit. That
is either the trunk (its pooled hidden vector does not carry per-country control) or the aux
training (too weak to fit a head, let alone shape the trunk). This separates the two: each trunk
is frozen, and a fresh head is fitted on its hidden vector to convergence on the same positions,
so the trunks differ only in what they encode.

* positions: self-play of each `--policies` checkpoint at `--temperature`, sampled decisions,
  labelled with every country's final controller when the game ends; split by game into a
  fitting and a test set (`--test-frac`);
* per trunk (`--trunks`), two probes on the frozen hidden vector `h` (what the policy, value and aux
  heads read): **linear** (h -> 84 x 3) and **mlp** (h -> 256 -> 84 x 3, the aux head's shape);
* references on the same test positions: an MLP fitted on the **raw observation** from scratch
  (what the data allows without any trunk), each trunk's own trained aux head if it has one,
  and the per-country **constant** and **current** baselines of `aux_ownership_probe.py`.

Labels and probes are in the mover's frame, as the aux target is; scores are absolute.

    PYTHONPATH=.:build/release python tools/scripts/trunk_ownership_probe.py \\
        --trunks a.pt b.pt --policies a.pt b.pt --games 1500 --output-json out.json --output-md out.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import ts_engine as ts  # noqa: E402

from ai.models.head_registry import refuse_side_split  # noqa: E402
from bindings.ts_env import TsVectorizedEnv  # noqa: E402
from tools.lib.player_agent import NeuralAgent  # noqa: E402
from tools.lib.engine_fingerprint import require_fresh  # noqa: E402
from tools.scripts.aux_ownership_probe import N_COUNTRIES, _absolute_class, _control, analyse  # noqa: E402


@torch.no_grad()
def collect(agent: NeuralAgent, games: int, envs: int, seed: int, sample_frac: float,
            temperature: float, dev: torch.device, game_offset: int) -> Dict[str, np.ndarray]:
    """Sampled positions of `agent`'s self-play: observation, legal mask, mover, board then, final board, winner."""
    model: Any = agent.model
    env = TsVectorizedEnv(num_envs=envs, base_seed=seed)
    env.record_final_control = True
    obs, masks, _ = env.reset_all()
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    pending: List[List[Dict[str, Any]]] = [[] for _ in range(envs)]
    rows: List[Dict[str, Any]] = []
    done_games = 0
    sampling = np.ones(envs, dtype=bool)          # as in aux_ownership_probe: finish games in flight
    while done_games < games or any(pending):
        dp = np.asarray(env.runner.get_decision_players(), dtype=np.int8)
        ob = np.asarray(obs)
        mk = np.asarray(masks)
        for i in np.flatnonzero(sampling & (dp != 0) & (rng.random(envs) < sample_frac)):
            st = env.runner.get_state(int(i))
            pending[int(i)].append({"obs": ob[i].astype(np.float16), "mask": mk[i].astype(bool),
                                    "mover": int(dp[i]), "turn": int(st.turn),
                                    "now": _absolute_class(_control(st))})
        o = torch.from_numpy(ob).float().to(dev)
        m = torch.from_numpy(np.asarray(masks)).to(dev)
        logits = model(o, m)[0].float()
        acts = torch.multinomial(torch.softmax(logits / temperature, -1), 1,
                                 generator=gen).squeeze(-1).cpu().numpy()
        obs, masks, _r, _d, info = env.step(acts)
        for ep in info.get("completed_episodes", []):
            i = int(ep["env_idx"])
            if not sampling[i]:
                continue
            final = _absolute_class(np.asarray(ep["final_control"]))
            winner = {"US": 1, "USSR": -1}.get(str(ep.get("winner")), 0)
            for r in pending[i]:
                rows.append({**r, "final": final, "winner": winner, "game": game_offset + done_games})
            pending[i] = []
            done_games += 1
            if done_games >= games:
                sampling[i] = False
    return {k: np.stack([r[k] for r in rows])
            for k in ("obs", "mask", "mover", "turn", "now", "final", "winner", "game")}


def mover_frame(final: np.ndarray, mover: np.ndarray) -> np.ndarray:
    """Absolute US / USSR / none -> mine / theirs / neither, the aux head's classes."""
    us = mover[:, None] == 1
    return np.where(final == 2, 2, np.where((final == 0) == us, 0, 1)).astype(np.int64)


def to_absolute(p_mover: np.ndarray, mover: np.ndarray) -> np.ndarray:
    """(n, 84, mine/theirs/neither) -> (n, 84, US/USSR/none)."""
    out = p_mover.copy()
    ussr = mover == -1
    out[ussr, :, 0], out[ussr, :, 1] = p_mover[ussr, :, 1], p_mover[ussr, :, 0]
    return out


def fit_probe(make: Callable[[], nn.Module], x_fit: torch.Tensor, y_fit: torch.Tensor, game_fit: np.ndarray,
              x_test: torch.Tensor, epochs: int, dev: torch.device, seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Adam on cross-entropy, early-stopped on a tenth of the fitting set's *games* (positions of one
    game are near-duplicates, so holding out positions would let the stop follow memorisation)."""
    torch.manual_seed(seed)
    games = np.unique(game_fit)
    held = np.isin(game_fit, np.random.default_rng(seed + 1).choice(games, max(1, len(games) // 10), replace=False))
    val = torch.from_numpy(np.flatnonzero(held)).to(dev)
    tr = torch.from_numpy(np.flatnonzero(~held)).to(dev)
    # Standardise each input feature on the fitting set: trunks differ in scale by orders of
    # magnitude (E6-10-44's hidden vector grows ~38x over training), and an unscaled probe then
    # measures conditioning rather than content. Constant features are left at unit scale.
    s1 = torch.zeros(x_fit.shape[1], device=dev, dtype=torch.float64)
    s2 = torch.zeros_like(s1)
    for k in range(0, x_fit.shape[0], 8192):
        xb = x_fit[k:k + 8192].double()
        s1 += xb.sum(0)
        s2 += (xb * xb).sum(0)
    mu = s1 / x_fit.shape[0]
    sd = (s2 / x_fit.shape[0] - mu * mu).clamp_min(0).sqrt()
    mu, sd = mu.float(), torch.where(sd > 1e-6, sd, torch.ones_like(sd)).float()
    inner = make().to(dev)

    def net(x: torch.Tensor) -> torch.Tensor:
        return inner((x.float() - mu) / sd)
    opt = torch.optim.AdamW(inner.parameters(), lr=1e-3, weight_decay=1e-4)

    def ce(idx: torch.Tensor) -> float:
        with torch.no_grad():
            tot = 0.0
            for k in range(0, idx.numel(), 8192):
                j = idx[k:k + 8192]
                tot += float(F.cross_entropy(net(x_fit[j].float()).view(-1, 3), y_fit[j].view(-1),
                                             reduction="sum"))
            return tot / (idx.numel() * N_COUNTRIES)

    best, best_state, stale, used = float("inf"), None, 0, 0
    for ep in range(epochs):
        inner.train()
        order = tr[torch.randperm(tr.numel(), device=dev)]
        for k in range(0, order.numel(), 1024):
            j = order[k:k + 1024]
            loss = F.cross_entropy(net(x_fit[j].float()).view(-1, 3), y_fit[j].view(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        inner.eval()
        v = ce(val)
        used = ep + 1
        if v < best - 1e-4:
            best, stale = v, 0
            best_state = {k: t.detach().clone() for k, t in inner.state_dict().items()}
        else:
            stale += 1
            if stale >= 3:
                break
    assert best_state is not None
    inner.load_state_dict(best_state)
    inner.eval()
    with torch.no_grad():
        p = torch.cat([torch.softmax(net(x_test[k:k + 8192].float()).view(-1, N_COUNTRIES, 3), -1)
                       for k in range(0, x_test.shape[0], 8192)]).cpu().numpy()
    return p, {"epochs": used, "val_ce": best}


class _Linear(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.f = nn.Linear(d, N_COUNTRIES * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x).view(-1, N_COUNTRIES, 3)


class _MLP(nn.Module):
    def __init__(self, d: int, width: int) -> None:
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, width), nn.GELU(), nn.Linear(width, N_COUNTRIES * 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x).view(-1, N_COUNTRIES, 3)


@torch.no_grad()
def hidden(model: Any, obs: np.ndarray, dev: torch.device) -> torch.Tensor:
    refuse_side_split(model, "the trunk probe")
    return torch.cat([model._encode(torch.from_numpy(obs[k:k + 4096]).float().to(dev))[0].float()
                      for k in range(0, obs.shape[0], 4096)])


@torch.no_grad()
def own_head(model: Any, obs: np.ndarray, dev: torch.device) -> np.ndarray:
    return torch.cat([torch.softmax(model.forward_aux(torch.from_numpy(obs[k:k + 4096]).float().to(dev))[0].float(), -1)
                      for k in range(0, obs.shape[0], 4096)]).cpu().numpy()


def summary(a: Dict[str, Any]) -> Dict[str, Any]:
    return {"all": a["all"], "surprises": a["surprises"], "changes": a["changes"],
            "late": a["by_stage"]["turns 8-10"]["all"], "early": a["by_stage"]["turns 1-3"]["all"],
            "surprise_by_country": {c["name"]: c["surprises"].get("head") for c in a["countries"]}}


def _ll(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.3f}"


def render(res: Dict[str, Any]) -> str:
    L = ["# Ownership from a frozen trunk", "",
         f"Positions: self-play of {', '.join(res['policies'])} at temperature {res['temperature']}; "
         f"{res['fit_positions']} fitting and {res['test_positions']} test positions "
         f"({res['test_games']} test games, split by game). Scores on the test set, absolute frame.", "",
         "Accuracy in %; LL = log-loss in nats. *surprises*: the country did not end with its usual "
         "controller; *changes*: its controller changed after the position.", "",
         "| predictor | all | all LL | turns 1-3 | turns 8-10 | changes | surprises | surprises LL |",
         "|:---|---:|---:|---:|---:|---:|---:|---:|"]
    base = res["baselines"]
    for name, key in (("constant (per-country usual)", "constant"), ("current controller", "current")):
        L.append(f"| {name} | {100 * base['all'][key]:.1f} | {_ll(base['all'].get(key + '_logloss'))} | "
                 f"{100 * base['early'][key]:.1f} | {100 * base['late'][key]:.1f} | "
                 f"{100 * base['changes'][key]:.1f} | {100 * base['surprises'][key]:.1f} | |")
    for name, s in res["predictors"].items():
        L.append(f"| {name} | {100 * s['all']['head']:.1f} | {s['all']['head_logloss']:.3f} | "
                 f"{100 * s['early']['head']:.1f} | {100 * s['late']['head']:.1f} | "
                 f"{100 * s['changes']['head']:.1f} | {100 * s['surprises']['head']:.1f} | "
                 f"{s['surprises']['head_logloss']:.3f} |")
    L += ["", "## Surprises in the countries with a near-certain usual controller", "",
          "| predictor | " + " | ".join(res["focus"]) + " |", "|:---|" + "---:|" * len(res["focus"])]
    for name, s in res["predictors"].items():
        L.append(f"| {name} | " + " | ".join(
            "—" if s["surprise_by_country"].get(c) is None else f"{100 * s['surprise_by_country'][c]:.1f}"
            for c in res["focus"]) + " |")
    L += ["", "Probe fits (epochs, held-out CE): " + "; ".join(
        f"{k}: {v['epochs']}, {v['val_ce']:.3f}" for k, v in res["fits"].items())]
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trunks", nargs="+", required=True)
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=1500, help="games per policy")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--sample-frac", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--obs-mlp-width", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--output-md", default=None)
    a = ap.parse_args(argv)
    require_fresh()
    dev = torch.device(a.device)

    parts: List[Dict[str, np.ndarray]] = []
    names: List[str] = []
    for k, path in enumerate(a.policies):
        ag = NeuralAgent.from_checkpoint(path, device=str(dev))
        names.append(ag.name)
        parts.append(collect(ag, a.games, a.envs, a.seed + 1000 * k, a.sample_frac, a.temperature, dev,
                             game_offset=k * 10 * a.games))
        print(f"collected {parts[-1]['final'].shape[0]} positions from {ag.name}", flush=True)
    d = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    games = np.unique(d["game"])
    test_games = np.random.default_rng(a.seed).choice(games, int(len(games) * a.test_frac), replace=False)
    is_test = np.isin(d["game"], test_games)
    fit, test = ~is_test, is_test
    y_fit = torch.from_numpy(mover_frame(d["final"][fit], d["mover"][fit])).to(dev)
    test_rows = {k: d[k][test] for k in ("now", "final", "turn", "game")}
    mover_test = d["mover"][test]

    def score(p_mover: np.ndarray) -> Dict[str, Any]:
        return summary(analyse({**test_rows, "p": to_absolute(p_mover, mover_test)}))

    predictors: Dict[str, Dict[str, Any]] = {}
    fits: Dict[str, Dict[str, Any]] = {}
    # the raw-observation reference, no trunk
    x_fit = torch.from_numpy(d["obs"][fit]).to(dev)
    x_test = torch.from_numpy(d["obs"][test]).to(dev)
    w = x_fit.shape[1]
    p, fits["raw observation, mlp"] = fit_probe(lambda: _MLP(w, a.obs_mlp_width), x_fit, y_fit, d["game"][fit], x_test,
                                                a.epochs, dev, a.seed)
    predictors["raw observation, mlp"] = score(p)
    print("fitted raw observation", flush=True)
    del x_fit, x_test

    for path in a.trunks:
        ag = NeuralAgent.from_checkpoint(path, device=str(dev))
        model: Any = ag.model.eval()
        h_fit, h_test = hidden(model, d["obs"][fit], dev), hidden(model, d["obs"][test], dev)
        dim = h_fit.shape[1]
        for kind, make in (("linear", lambda: _Linear(dim)), ("mlp", lambda: _MLP(dim, 256))):
            key = f"{ag.name} trunk, {kind}"
            p, fits[key] = fit_probe(make, h_fit, y_fit, d["game"][fit], h_test, a.epochs, dev, a.seed)
            predictors[key] = score(p)
            print(f"fitted {key}", flush=True)
        if getattr(model, "aux_heads", False):
            predictors[f"{ag.name} own aux head (online)"] = score(own_head(model, d["obs"][test], dev))
        del h_fit, h_test

    base = analyse({**test_rows, "p": np.full((int(test.sum()), N_COUNTRIES, 3), 1.0 / 3, dtype=np.float32)})
    focus = [c["name"] for c in sorted(base["countries"], key=lambda c: -c["usual_rate"])
             if 0.75 <= c["usual_rate"] < 0.99][:8]
    res = {"policies": names, "temperature": a.temperature, "fit_positions": int(fit.sum()),
           "test_positions": int(test.sum()), "test_games": int(len(test_games)),
           "baselines": summary(base), "predictors": predictors, "fits": fits, "focus": focus}
    md = render(res)
    print(md)
    if a.output_json:
        with open(a.output_json, "w") as f:
            json.dump(res, f, indent=1)
    if a.output_md:
        with open(a.output_md, "w") as f:
            f.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
