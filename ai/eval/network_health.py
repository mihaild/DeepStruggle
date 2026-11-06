"""Early stability checks for a network being trained from scratch (research/method/network_stability.md).

Two modes, both cheap enough to run at 30-60M and again at 400M:

* ``--runs DIR ...`` reads only the training logs: per window, the log-normaliser's max (``logit_lse_absmax``), the
  KL to pi_ref's max, the entropy and the critic's explained variance -- the series that flagged A22 within 30M.
* ``--checkpoints FILE ...`` measures the networks on positions from a few self-play games of the first one: the
  logits' level and spread by decision type, the geometry of what the heads read (RMS, largest dimension over the
  median one, effective rank, top-10 principal components; beside the same shape freshly initialised, from which the
  rank should grow) and of the residual stream before the trunk norm, the
  policy head's weight norms against a fresh head of the same shape, and dormant units in the head and the trunk.
  (Not the train/eval gap of residual dropout: a checkpoint's weights do not record its dropout, so a loaded network
  has the historical 0.05 whatever it trained with.)

    PYTHONPATH=.:build/release python -m ai.eval.network_health --runs <run dir> <control run dir>
    PYTHONPATH=.:build/release python -m ai.eval.network_health --checkpoints <a.pt> <control.pt> --out report.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

import ts_engine as ts

from ai.eval.plasticity_probe import POLICY_HEAD, dormant
from ai.models.coldwar_net_v2 import create_like, policy_head_norms
from ai.eval.loading import load_for_eval
from bindings.ts_env import legal_mask, model_obs_features, observe
from tools.lib.corpus_driver import selfplay

#: Training-log series read per window, and how each window is summarised.
LOG_SERIES: Dict[str, str] = {"logit_lse_absmax": "max", "kl_div": "max", "entropy": "median",
                              "explained_variance": "median"}


def effective_rank(h: torch.Tensor) -> float:
    """exp of the entropy of the normalised squared singular values of the centred rows (Roy & Vetterli)."""
    s = torch.linalg.svdvals((h - h.mean(0)).float())
    p = s ** 2 / (s ** 2).sum().clamp_min(1e-12)
    return float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))


def geometry(h: torch.Tensor) -> Dict[str, float]:
    """The size and shape of a batch of feature vectors (rows)."""
    rms_dim = h.float().pow(2).mean(0).sqrt()
    c = (h - h.mean(0)).float()
    s = torch.linalg.svdvals(c)
    var = s ** 2 / (s ** 2).sum().clamp_min(1e-12)
    return {"rms": float(h.float().pow(2).mean().sqrt()),
            "max_over_median_dim_rms": float(rms_dim.max() / rms_dim.median().clamp_min(1e-12)),
            "effective_rank": effective_rank(h), "width": float(h.shape[1]),
            "top10_variance": float(var[:10].sum())}


def logit_stats(raw: torch.Tensor, masks: torch.Tensor, dtypes: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Per decision type: the log-normaliser over the legal actions (median, |max|) and the legal logits' spread
    (max minus min; median, max). Masked logits at or below -1e8 are not legal."""
    legal = masks.bool() & (raw > -1e8)
    lse = torch.logsumexp(raw.masked_fill(~legal, -1e9), -1)
    spread = raw.masked_fill(~legal, -1e30).max(-1).values - raw.masked_fill(~legal, 1e30).min(-1).values
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, d in enumerate(dtypes):
        groups[d].append(i)
    groups["all"] = list(range(len(dtypes)))
    out: Dict[str, Dict[str, float]] = {}
    for d, ii in groups.items():
        t = torch.tensor(ii)
        out[d] = {"n": float(len(ii)), "lse_median": float(lse[t].median()), "lse_absmax": float(lse[t].abs().max()),
                  "spread_median": float(spread[t].median()), "spread_max": float(spread[t].max())}
    return out


def head_dormant(model: nn.Module, obs: torch.Tensor, masks: torch.Tensor, tau: float = 0.1) -> float:
    """ReDo's dormant fraction over the policy head's activation units (as `plasticity_probe.dormant` for the trunk)."""
    stats: List[torch.Tensor] = []
    hooks = []
    for top in POLICY_HEAD:
        mod = getattr(model, top, None)
        if not isinstance(mod, nn.Module):
            continue
        for sub in mod.modules():
            if isinstance(sub, (nn.GELU, nn.Tanh)):
                hooks.append(sub.register_forward_hook(
                    lambda _m, _i, out: stats.append(out.detach().abs().reshape(-1, out.shape[-1]).mean(0))))
    with torch.no_grad():
        model(obs, masks)
    for h in hooks:
        h.remove()
    units = sum(s.numel() for s in stats)
    return sum(int(((s / s.mean().clamp_min(1e-12)) <= tau).sum()) for s in stats) / max(1, units)


def measure(model: nn.Module, obs: torch.Tensor, masks: torch.Tensor, dtypes: Sequence[str],
            seed: int = 3737) -> Dict[str, Any]:
    """Every checkpoint measure of the module docstring, on one batch of positions."""
    model.eval()
    stream: Dict[str, torch.Tensor] = {}
    norm = getattr(model, "trunk_norm", None)
    hook = (norm.register_forward_hook(lambda _m, inp, _o: stream.update(pre=inp[0].detach()))
            if isinstance(norm, nn.Module) else None)
    extract = getattr(model, "extract_features")
    with torch.no_grad():
        h = extract(obs)
        raw = model(obs, masks)[0].float()
    if hook is not None:
        hook.remove()
    torch.manual_seed(seed)
    fresh = create_like(model).eval()
    fresh_extract = getattr(fresh, "extract_features")
    with torch.no_grad():
        h_fresh = fresh_extract(obs)
    fresh_norms = policy_head_norms(fresh)
    out: Dict[str, Any] = {
        "logits": logit_stats(raw, masks, dtypes),
        "head_input": geometry(h),
        # The same network freshly initialised, on the same positions: the effective rank should grow from here.
        "head_input_fresh": geometry(h_fresh),
        "head_norm_over_fresh": {k: v / max(1e-12, fresh_norms.get(k, 0.0)) for k, v in policy_head_norms(model).items()},
        "head_dormant_tau_0.1": head_dormant(model, obs, masks),
        "trunk_dormant_tau_0.1": dormant(model, obs, masks)["all"]["tau_0.1"],
    }
    if "pre" in stream:
        out["stream_before_trunk_norm"] = geometry(stream["pre"])
    return out


class _Positions:
    def __init__(self) -> None:
        self.states: List[Any] = []

    def observe(self, st: Any, action: int) -> None:
        self.states.append(ts.state_from_raw_bytes(bytes(st.raw_bytes())))


def selfplay_positions(model: nn.Module, games: int, seed: int) -> List[Any]:
    """Every position of `games` self-play games of `model` at temperature 1 (as training samples)."""
    feats = model_obs_features(model)

    def logits_fn(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            lg = model(torch.tensor(obs, dtype=torch.float32), torch.tensor(masks).bool())[0].float()
        return lg.masked_fill(~torch.tensor(masks).bool(), -1e9).numpy()

    recs = selfplay(logits_fn, feats, games=games, seed=seed, batch=games, temperature=1.0, factory=_Positions)
    return [s for r in recs for s in r.states]


def check_checkpoints(paths: Sequence[str], games: int, seed: int, max_positions: int) -> Dict[str, Any]:
    # The masks below are E4: a merged-view checkpoint is refused at load.
    models = [load_for_eval(p, "cpu", merged_ok=False).model for p in paths]
    states = selfplay_positions(models[0], games, seed)
    rng = np.random.default_rng(seed)
    if len(states) > max_positions:
        states = [states[i] for i in sorted(rng.choice(len(states), size=max_positions, replace=False))]
    dtypes = [str(s.ctx().decision_type).split(".")[-1] for s in states]
    masks = torch.tensor(np.stack([legal_mask(s, False, None).astype(bool) for s in states]))
    report: Dict[str, Any] = {"positions": len(states), "games": games, "seed": seed, "networks": {}}
    for path, m in zip(paths, models):
        feats = model_obs_features(m)
        obs = torch.tensor(np.stack([observe(s, s.ctx().decision_player, feats) for s in states]))
        report["networks"][path] = measure(m, obs, masks, dtypes, seed)
    return report


def log_windows(run_dir: str, window_m: int) -> List[Dict[str, float]]:
    """The training log's stability series per `window_m` million steps."""
    rows = [json.loads(line) for line in open(os.path.join(run_dir, "training_metrics.jsonl"))
            if '"total_steps"' in line]
    out: List[Dict[str, float]] = []
    if not rows:
        return out
    end = rows[-1]["total_steps"]
    for lo in range(0, int(end), window_m * 1_000_000):
        w = [r for r in rows if lo < r["total_steps"] <= lo + window_m * 1_000_000]
        if not w:
            continue
        rec: Dict[str, float] = {"from_m": lo / 1e6, "to_m": lo / 1e6 + window_m}
        for k, how in LOG_SERIES.items():
            vals = [float(r[k]) for r in w if k in r]
            if vals:
                rec[k] = max(vals) if how == "max" else statistics.median(vals)
        out.append(rec)
    return out


def _print_checkpoints(report: Dict[str, Any]) -> None:
    print(f"{report['positions']} positions from {report['games']} self-play games")
    for path, r in report["networks"].items():
        g = r["head_input"]
        lg = r["logits"]
        worst = max((v for k, v in lg.items() if k != "all"), key=lambda v: v["spread_max"])
        print(f"\n{path}")
        print(f"  logits: |lse| max {lg['all']['lse_absmax']:.1f}, spread median {lg['all']['spread_median']:.1f}, "
              f"worst decision type's spread max {worst['spread_max']:.1f}")
        print(f"  head input: RMS {g['rms']:.2f}, max/median dim {g['max_over_median_dim_rms']:.1f}, effective rank "
              f"{g['effective_rank']:.1f} of {g['width']:.0f} (fresh {r['head_input_fresh']['effective_rank']:.1f}), "
              f"top-10 PCs {100 * g['top10_variance']:.1f}%")
        if "stream_before_trunk_norm" in r:
            s = r["stream_before_trunk_norm"]
            print(f"  stream before the trunk norm: RMS {s['rms']:.2f}, max/median dim {s['max_over_median_dim_rms']:.1f}")
        print("  head norm / fresh: " + ", ".join(f"{k} {v:.1f}x" for k, v in r["head_norm_over_fresh"].items()))
        print(f"  dormant (tau 0.1): head {100 * r['head_dormant_tau_0.1']:.1f}%, trunk "
              f"{100 * r['trunk_dormant_tau_0.1']:.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[], help="run directories: the training-log series per window")
    ap.add_argument("--window", type=int, default=50, help="window for --runs, in millions of steps")
    ap.add_argument("--checkpoints", nargs="*", default=[], help="checkpoints: measured on shared positions")
    ap.add_argument("--games", type=int, default=16, help="self-play games of the first checkpoint for positions")
    ap.add_argument("--max-positions", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default=None, help="write the full report as JSON")
    a = ap.parse_args()
    if not a.runs and not a.checkpoints:
        ap.error("give --runs and/or --checkpoints")
    torch.set_num_threads(a.threads)
    report: Dict[str, Any] = {}
    if a.runs:
        report["runs"] = {d: log_windows(d, a.window) for d in a.runs}
        for d, rows in report["runs"].items():
            print(f"\n{os.path.basename(d.rstrip('/'))}")
            print(f"  {'window':>13s} {'lse absmax':>10s} {'KL max':>7s} {'entropy':>7s} {'expl var':>8s}")
            for r in rows:
                print(f"  {r['from_m']:6.0f}-{r['to_m']:5.0f}M {r.get('logit_lse_absmax', float('nan')):10.1f} "
                      f"{r.get('kl_div', float('nan')):7.3f} {r.get('entropy', float('nan')):7.3f} "
                      f"{r.get('explained_variance', float('nan')):8.3f}")
    if a.checkpoints:
        report["checkpoints"] = check_checkpoints(a.checkpoints, a.games, a.seed, a.max_positions)
        _print_checkpoints(report["checkpoints"])
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
