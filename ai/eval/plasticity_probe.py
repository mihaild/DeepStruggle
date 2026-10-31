"""P37a, the plasticity probe (CPU): can the plateau network's trunk still fit new targets?

Lyle et al. 2023's measure: the plateau trunk and a fresh trunk of the same shape, each given the same freshly
initialised policy head, fit (i) random labels on real positions -- one random legal move per position, a target
with no structure, so how far a network gets is its capacity to fit at all -- and (ii) P36c's targets, the keyed
rollout root's improved policy at 128 worlds; each with the trunk frozen (only the head trains) and fully
fine-tuned, to convergence. Alongside, the fraction of dormant trunk units (Sokar et al. 2023, ReDo): unit i's
score is its mean |activation| over a batch divided by the layer's mean, dormant at score <= tau.

Positions are P36c's (data/datasets/p36c/targets.npz): raw GameState bytes. They are decoded by this build only
after checking it can -- every state's bytes round-trip, and the network that labelled them picks the argmax the
dataset recorded -- since raw bytes are only meaningful to a build with the writer's GameState layout.

    PYTHONPATH=.:build/release python ai/eval/plasticity_probe.py --positions 20000 --out data/reports/p37a.json
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ts_engine as ts

from ai.models.coldwar_net_v2 import POLICY_HEAD_MODULES, create_like
from ai.eval.loading import load_for_eval
from bindings.ts_env import legal_mask, model_obs_features, observe

PLATEAU = "/workspace/data/checkpoints/E7-A8-R1-S44@6400M+R34@7200M+R48@9170M+A13_20261014_105711/swa_12730-12800M.pt"
DATASET = "/workspace/data/datasets/p36c/targets.npz"
#: Snapshots along the heads line, for the dormant-unit trend.
TREND = {
    "heads run 400M": "/workspace/data/checkpoints/E7-A8-R1-S44_20261007_092825/swa_320-400M.pt",
    "heads run 2,000M": "/workspace/data/checkpoints/E7-A8-R1-S44_20261007_162257/swa_1920-2000M.pt",
    "heads run 6,400M": "/workspace/data/checkpoints/E7-A8-R1-S44_20261007_162257/swa_6320-6400M.pt",
}
TRUNK = ("lad_board", "lad_card", "global_proj", "tok_main", "fusion_in", "res_blocks")
POLICY_HEAD = POLICY_HEAD_MODULES
A = 220


def load(path: str) -> nn.Module:
    """A network for this probe: the positions' masks are E4, so a merged-view one is refused."""
    return load_for_eval(path, "cpu", merged_ok=False).model


def observations(states: Sequence[Any], features: int) -> np.ndarray:
    """Each state's observation for its decider, in the feature set `features`."""
    return np.stack([observe(st, st.ctx().decision_player, features) for st in states])


def positions(n: int, seed: int, features: int) -> Dict[str, Any]:
    """n positions from P36c's training games (not its held-out ones): observations, legal masks, the
    recorded argmax and the 128-world target, densified over the 220 actions."""
    z = np.load(DATASET)
    train = np.flatnonzero(~z["heldout"])
    idx = np.sort(np.random.default_rng(seed).choice(train, size=n, replace=False))
    states = z["states"][idx]
    masks = np.zeros((n, A), dtype=np.uint8)
    decoded: List[Any] = []
    for i in range(n):
        raw = states[i].tobytes()
        st = ts.state_from_raw_bytes(raw)
        if bytes(st.raw_bytes()) != raw:
            raise SystemExit(f"position {idx[i]}: raw bytes do not round-trip on this build")
        decoded.append(st)
        masks[i] = legal_mask(st, False)
    # In the feature set of the network these are first fed to (the plateau, and its fresh copy
    # and head); a network of another set reads `observations(d["states"], its set)` instead.
    obs = observations(decoded, features)
    target = np.zeros((n, A), dtype=np.float32)
    acts, probs = z["acts_128"][idx], z["probs_128"][idx]
    for i in range(n):
        for a, p in zip(acts[i], probs[i]):
            if a >= 0:
                target[i, a] += p
    target *= masks
    target /= np.maximum(target.sum(1, keepdims=True), 1e-9)
    return {"obs": obs, "masks": masks, "net_argmax": z["net_argmax"][idx].astype(np.int64), "target": target,
            "states": decoded}


def logits_of(model: nn.Module, obs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    return model(obs, masks)[0].float().masked_fill(~masks, -1e9)


def check_decoding(model: nn.Module, d: Dict[str, Any]) -> float:
    """The labelling network's argmax on the decoded positions against the one the dataset recorded."""
    agree = 0
    with torch.no_grad():
        for lo in range(0, len(d["obs"]), 2048):
            o = torch.from_numpy(d["obs"][lo:lo + 2048])
            m = torch.from_numpy(d["masks"][lo:lo + 2048]).bool()
            agree += int((logits_of(model, o, m).argmax(1).numpy() == d["net_argmax"][lo:lo + 2048]).sum())
    return agree / len(d["obs"])


def with_head(trunk_model: nn.Module, head_source: nn.Module) -> nn.Module:
    """`trunk_model` with its policy head replaced by `head_source`'s (copied, so every fit starts equal)."""
    m = copy.deepcopy(trunk_model)
    src = dict(head_source.named_children())
    for name in POLICY_HEAD:
        if name in src:
            setattr(m, name, copy.deepcopy(src[name]))
    return m


def fit(model: nn.Module, d: Dict[str, Any], labels: np.ndarray, soft: bool, frozen: bool,
        epochs: int, lr: float, batch: int, seed: int) -> List[Dict[str, float]]:
    """Train the policy toward `labels` (hard: one action per row; soft: a distribution) and return per-epoch
    fit: accuracy (argmax = label or = target's top) and, for soft targets, the departures' switch rate."""
    torch.manual_seed(seed)
    model.train()
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        p.requires_grad_(top in POLICY_HEAD or (not frozen and top in TRUNK))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    obs = torch.from_numpy(d["obs"])
    masks = torch.from_numpy(d["masks"]).bool()
    y = torch.from_numpy(labels)
    top = torch.from_numpy(d["target"].argmax(1)) if soft else y
    departure = torch.from_numpy(d["target"].argmax(1) != d["net_argmax"]) if soft else None
    n = len(obs)
    rng = np.random.default_rng(seed)
    curve: List[Dict[str, float]] = []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        order = torch.from_numpy(rng.permutation(n))
        total = 0.0
        for lo in range(0, n, batch):
            b = order[lo:lo + batch]
            lg = logits_of(model, obs[b], masks[b])
            if soft:
                loss = -(y[b] * F.log_softmax(lg, dim=1)).sum(1).mean()
            else:
                loss = F.cross_entropy(lg, y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(b)
        with torch.no_grad():
            pred = torch.cat([logits_of(model, obs[lo:lo + 2048], masks[lo:lo + 2048]).argmax(1)
                              for lo in range(0, n, 2048)])
        rec = {"epoch": float(ep), "loss": total / n, "fit": float((pred == top).float().mean()),
               "seconds": time.time() - t0}
        if soft and departure is not None:
            rec["departures_switched"] = float((pred[departure] == top[departure]).float().mean())
        curve.append(rec)
        print(f"    epoch {ep:3d}: loss {rec['loss']:.4f}  fit {100 * rec['fit']:.1f}%"
              + (f"  departures {100 * rec['departures_switched']:.1f}%" if soft else "")
              + f"  ({rec['seconds']:.0f} s)", flush=True)
        if ep >= 6 and curve[-1]["fit"] - curve[-6]["fit"] < 0.003:     # converged: <0.3 points in 5 epochs
            break
    model.eval()
    return curve


def dormant(model: nn.Module, obs: torch.Tensor, masks: torch.Tensor,
            taus: Sequence[float] = (0.0, 0.025, 0.1)) -> Dict[str, Any]:
    """ReDo's dormant fraction over the trunk's activation units (GELU / Tanh outputs; a unit is a channel,
    averaged over the batch and over entities where the layer is per-entity)."""
    stats: List[Tuple[str, torch.Tensor]] = []
    hooks = []
    for top in TRUNK:
        mod = getattr(model, top, None)
        if mod is None:
            continue
        for name, sub in mod.named_modules():
            if isinstance(sub, (nn.GELU, nn.Tanh)):
                def hook(_m: nn.Module, _i: Any, out: torch.Tensor, key: str = f"{top}.{name}") -> None:
                    a = out.detach().abs()
                    stats.append((key, a.reshape(-1, a.shape[-1]).mean(0)))
                hooks.append(sub.register_forward_hook(hook))
    with torch.no_grad():
        model(obs, masks)
    for h in hooks:
        h.remove()
    out: Dict[str, Any] = {"layers": {}}
    units = 0
    counts = {t: 0 for t in taus}
    for key, s in stats:
        score = s / s.mean().clamp_min(1e-12)
        layer = {f"tau_{t}": float((score <= t).float().mean()) for t in taus}
        layer["units"] = int(s.numel())
        out["layers"][key] = layer
        units += s.numel()
        for t in taus:
            counts[t] += int((score <= t).sum())
    out["all"] = {f"tau_{t}": counts[t] / max(1, units) for t in taus}
    out["units"] = units
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--positions", type=int, default=20_000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=3737)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    plateau = load(PLATEAU)
    torch.manual_seed(a.seed)
    fresh = create_like(plateau).eval()                 # same shape, fresh weights
    print(f"positions: {a.positions} from P36c's training games", flush=True)
    d = positions(a.positions, a.seed, model_obs_features(plateau))
    agree = check_decoding(plateau, d)
    print(f"decoding check: the labelling network picks the recorded argmax at {100 * agree:.2f}%", flush=True)
    if agree < 0.99:
        raise SystemExit("this build does not decode the dataset's positions as the writer did")
    rng = np.random.default_rng(a.seed + 1)
    random_labels = np.array([rng.choice(np.flatnonzero(m)) for m in d["masks"]], dtype=np.int64)

    report: Dict[str, Any] = {"positions": a.positions, "decoding_agreement": agree, "lr": a.lr, "batch": a.batch,
                              "plateau": PLATEAU, "fits": {}, "dormant": {}}
    probe_obs = torch.from_numpy(d["obs"][:4096])
    probe_masks = torch.from_numpy(d["masks"][:4096]).bool()
    report["dormant"]["fresh"] = dormant(fresh, probe_obs, probe_masks)
    report["dormant"]["plateau (A13+R48 SWA 12,730-12,800M)"] = dormant(plateau, probe_obs, probe_masks)
    for label, path in TREND.items():
        net = load(path)
        own_obs = torch.from_numpy(observations(d["states"][:4096], model_obs_features(net)))
        report["dormant"][label] = dormant(net, own_obs, probe_masks)
    for k, v in report["dormant"].items():
        print(f"dormant, {k}: " + ", ".join(f"{t} {100 * x:.1f}%" for t, x in v["all"].items()), flush=True)

    head = create_like(plateau)                         # one fresh head for every fit
    for trunk_name, trunk in (("plateau", plateau), ("fresh", fresh)):
        for task, labels, soft in (("random labels", random_labels, False), ("P36c targets", d["target"], True)):
            for frozen in (True, False):
                key = f"{trunk_name} trunk, {task}, {'frozen trunk' if frozen else 'full fine-tune'}"
                print(f"== {key}", flush=True)
                curve = fit(with_head(trunk, head), d, labels, soft, frozen, a.epochs, a.lr, a.batch, a.seed)
                report["fits"][key] = curve
                json.dump(report, open(a.out, "w"), indent=1)
    json.dump(report, open(a.out, "w"), indent=1)
    print("\nfinal fits:")
    for key, curve in report["fits"].items():
        c = curve[-1]
        print(f"  {key:60s} {100 * c['fit']:5.1f}%"
              + (f"  departures {100 * c['departures_switched']:5.1f}%" if "departures_switched" in c else "")
              + f"  ({int(c['epoch'])} epochs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
