"""P36c: fit a student's policy to the rollout root's improved policy -- signal or capacity?

`tools/distill_targets.py` writes the targets: on the network's own entry decisions, the keyed root's
improved policy (R32's form) at several world counts, held out by game, the held-out rows searched twice.
This trains a student on one world count's targets by cross-entropy, on the policy heads only (the value
heads frozen), **to convergence**: the stop is the training fit no longer rising, not the held-out loss --
the question is whether the student *can* fit the targets, which an early-stopped fit never answers (the
earlier offline rounds' two epochs at 1e-4 underfit by design). Students, after the plan: the current
network fine-tuned (`init`'s weights); the same shape from scratch (`reinit`); a larger shape (`reinit`
with `arch`).

Three numbers per evaluation, the plan's cell readings:

1. **training fit** -- on a fixed sample of training positions where the teacher departs (its target's top
   move is not the network's argmax), the share where the student's argmax is the teacher's top move;
2. **held-out agreement** -- the same on held-out games' departures (and on all held-out positions);
3. **the teacher's self-agreement** on those held-out positions -- its second seed's top move against its
   first, on the same departures: the ceiling (2) can reach.

The z2 picks are read alike (`*_pick`): where the root's own decision departs from the network's move.
"""

from __future__ import annotations

import copy
import json
import math
import os
import pathlib
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from bindings.action_encoder import ActionEncoder
import torch.nn.functional as F

import ts_engine as ts
from ai.models.ladder_net import create_ladder_net, ladder_config_from_state_dict
from ai.training.generic_trainer import _resolve_run_dir
from ai.training.npz_mmap import load_npz_mmap
from ai.training.value_fit import Codes, _git
from tools.lib.engine_fingerprint import check_raw_states

ACTIONS = ActionEncoder.BASE_ACTION_SIZE


@dataclass
class PolicyFitConfig:
    dataset: str                    # a tools/distill_targets.py .npz
    worlds: int                     # which world count's targets (P36c's teachers, named by world count)
    init: str                       # an RL checkpoint: its weights (fine-tune) or its shape (reinit)
    reinit: bool = False
    #: Changes to the init network's ladder config (LadderNet keyword -> value), with reinit only.
    arch: Optional[Dict[str, Any]] = None
    batch: int = 4096
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_epochs: float = 60.0
    eval_every: int = 200
    patience: int = 10              # evaluations without the training fit rising by min_gain
    min_gain: float = 0.002
    fit_sample: int = 20000         # training positions the fit is read on
    #: The share of the training games used (whole games, the lowest game ids): the data-scaling cells.
    train_fraction: float = 1.0
    #: Freeze the whole network and train only a correction head on its trunk features, added to its logits
    #: (zero at the start): the corrections as a residual that cannot damage the policy.
    head_only: bool = False
    #: Weight of KL(the initial network's policy || the student's) on every position: a fine-tune held near
    #: the policy it started from.
    kl_anchor: float = 0.0
    #: Which teacher's targets by name (`tools/distill_targets.py --teacher NAME=...`); None = `worlds`.
    target: Optional[str] = None
    #: Re-initialise the policy heads (the trunk kept): plasticity -- can fresh heads on the trained trunk read
    #: the corrections the fine-tuned ones do not?
    reinit_heads: bool = False
    seed: int = 44
    device: str = "cuda"
    bf16: bool = True


def dense_targets(acts: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """(n, 16) sparse improved policies -> (n, 220) dense rows."""
    out = np.zeros((acts.shape[0], ACTIONS), dtype=np.float32)
    rows = np.repeat(np.arange(acts.shape[0], dtype=np.int64), acts.shape[1])
    a = acts.reshape(-1).astype(np.int64)
    p = probs.reshape(-1)
    ok = a >= 0
    out[rows[ok], a[ok]] = p[ok]            # a row's moves are distinct
    return out


#: The modules that turn the trunk's features into logits (whichever of them a LadderNet has); `region_attn`
#: feeds only the per-country head.
POLICY_MODULES = ("policy_head", "pe_trunk", "pe_card", "pe_country", "play_mode_head_net", "branch_head_net",
                  "region_attn")


def reinit_policy_heads(model: nn.Module, fresh: nn.Module) -> int:
    """The policy modules' weights replaced by those of `fresh`, a newly built network of the same shape (the
    trunk and the value heads kept): a true fresh start, the layers LadderNet starts at zero zeroed again
    (`reset_parameters` would randomise them, and a centred head's final bias, which gets no gradient, would
    keep its random value). Returns the parameters reset."""
    n = 0
    for name in POLICY_MODULES:
        mod, new = getattr(model, name, None), getattr(fresh, name, None)
        if not isinstance(mod, nn.Module):
            continue
        assert isinstance(new, nn.Module), f"the fresh network has no {name}"
        mod.load_state_dict(new.state_dict(), strict=True)
        n += sum(p.numel() for p in mod.parameters())
    if n == 0:
        raise SystemExit("--policy-reinit-heads: this network has none of the policy modules " + ", ".join(POLICY_MODULES))
    return n


def masks_of(states: np.ndarray) -> np.ndarray:
    out = np.empty((states.shape[0], ACTIONS), dtype=np.uint8)
    for i in range(states.shape[0]):
        out[i] = np.asarray(ts.ActionMask.generate_flat_mask(ts.state_from_raw_bytes(states[i].tobytes())))[:ACTIONS]
    return out


def run_policy_fit(cfg: PolicyFitConfig, run_name: Optional[str], description: Optional[str],
                   output_dir: Optional[str] = None, cli: Optional[Dict[str, Any]] = None) -> str:
    """Fit to convergence, read the three numbers, keep the last and the best held-out; returns the run dir."""
    dev = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    sd = torch.load(cfg.init, map_location="cpu", weights_only=True)
    sd.pop("keep_idx", None)
    lcfg = ladder_config_from_state_dict(sd)
    if lcfg is None:
        raise SystemExit(f"{cfg.init}: policy fit takes a LadderNet checkpoint")
    if cfg.arch:
        if not cfg.reinit:
            raise SystemExit("a changed architecture (--policy-arch) trains from fresh weights: add --policy-reinit")
        unknown = sorted(set(cfg.arch) - set(lcfg))
        if unknown:
            raise SystemExit(f"--policy-arch: not ladder settings: {unknown}")
        lcfg = {**lcfg, **cfg.arch}
    model = create_ladder_net(dev, **lcfg)
    if not cfg.reinit:
        model.load_state_dict(sd, strict=True)
    # The anchor is the initial network's policy, so it is copied before any head is re-initialised.
    anchor = None
    if cfg.kl_anchor > 0:
        anchor = copy.deepcopy(model).eval()
        for p in anchor.parameters():
            p.requires_grad_(False)
    if cfg.reinit_heads:
        if cfg.reinit or cfg.head_only:
            raise SystemExit("--policy-reinit-heads keeps the trained trunk: not with --policy-reinit or --policy-head-only")
        n_reset = reinit_policy_heads(model, create_ladder_net(dev, **lcfg))
        print(f"policy heads re-initialised: {n_reset / 1e6:.2f}M parameters", flush=True)
    # The value path frozen: the student is read and trained on its policy only.
    for name, p in model.named_parameters():
        if name.startswith(("val_win_head", "val_vp_head", "val_logits_head")):
            p.requires_grad_(False)
    head: Optional[nn.Module] = None
    if cfg.head_only:
        for p in model.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            width = int(getattr(model, "_encode")(torch.zeros(1, model.TOTAL_OBS_SIZE, device=dev))[0].shape[1])
        head = nn.Sequential(nn.Linear(width, 512), nn.GELU(), nn.Linear(512, ACTIONS)).to(dev)
        out_layer = head[-1]
        assert isinstance(out_layer, nn.Linear)
        nn.init.zeros_(out_layer.weight)
        nn.init.zeros_(out_layer.bias)
        model.eval()
    else:
        model.train()
    params = list(head.parameters()) if head is not None else [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in model.parameters()) + (sum(p.numel() for p in head.parameters()) if head is not None else 0)

    t0 = time.time()
    check_raw_states(cfg.dataset)
    d = load_npz_mmap(cfg.dataset)          # memory-mapped: members are read as they are used
    w = cfg.target or str(cfg.worlds)
    if f"acts_{w}" not in d:
        names = sorted(k[5:] for k in d if k.startswith("acts_"))
        raise SystemExit(f"{cfg.dataset} has no targets {w!r} (it has {names})")
    heldout = d["heldout"].astype(bool)
    codes = Codes(int(getattr(model, "obs_feature_bits", 0)), dev)
    x_all = codes.encode(d["states"])
    m_all = masks_of(d["states"])
    y_all = dense_targets(d[f"acts_{w}"], d[f"probs_{w}"])
    top = d[f"acts_{w}"][:, 0].astype(np.int64)                 # the target's top move (sorted by probability)
    top2 = d[f"acts2_{w}"][:, 0].astype(np.int64)
    net = d["net_argmax"].astype(np.int64)
    pick, pick2 = d[f"pick_{w}"].astype(np.int64), d[f"pick2_{w}"].astype(np.int64)
    tr, ho = np.flatnonzero(~heldout), np.flatnonzero(heldout)
    if cfg.train_fraction < 1.0:
        games = np.unique(d["game"][tr])
        keep_games = games[:max(1, int(round(cfg.train_fraction * games.size)))]
        tr = tr[np.isin(d["game"][tr], keep_games)]
    dep_tr = tr[top[tr] != net[tr]]
    fit_rows = np.sort(rng.choice(dep_tr, size=min(cfg.fit_sample, dep_tr.size), replace=False)) if dep_tr.size else dep_tr
    dep_ho = ho[top[ho] != net[ho]]
    pdep_tr = tr[pick[tr] != net[tr]]
    pdep_ho = ho[pick[ho] != net[ho]]
    ceiling = float((top2[dep_ho] == top[dep_ho]).mean()) if dep_ho.size else float("nan")
    ceiling_all = float((top2[ho] == top[ho]).mean()) if ho.size else float("nan")
    ceiling_pick = float((pick2[pdep_ho] == pick[pdep_ho]).mean()) if pdep_ho.size else float("nan")
    print(f"policy fit to {w}: {tr.size:,} train / {ho.size:,} held-out positions; teacher departures "
          f"{100 * dep_tr.size / max(1, tr.size):.1f}% / {100 * dep_ho.size / max(1, ho.size):.1f}%; the teacher's own "
          f"held-out agreement {100 * ceiling:.1f}% on departures, {100 * ceiling_all:.1f}% overall, z2 picks "
          f"{100 * ceiling_pick:.1f}%; {n_params / 1e6:.2f}M parameters; loaded in {time.time() - t0:.0f}s", flush=True)

    x_dev = torch.from_numpy(x_all).to(dev)
    m_dev = torch.from_numpy(m_all).to(dev)
    y_dev = torch.from_numpy(y_all).to(dev)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, tr.size // cfg.batch)
    max_steps = int(cfg.max_epochs * steps_per_epoch)

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        frac = (step - cfg.warmup_steps) / max(1, max_steps - cfg.warmup_steps)
        return cfg.lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, frac))))

    def logits(rows: torch.Tensor) -> torch.Tensor:
        obs = codes.decode(x_dev[rows])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16 and dev.type == "cuda"):
            if head is not None:
                with torch.no_grad():
                    base = model(obs, m_dev[rows])[0]
                    h = getattr(model, "_encode")(obs)[0]
                lg = base.float() + head(h.float()).float()
            else:
                lg = model(obs, m_dev[rows])[0]
        return lg.float().masked_fill(m_dev[rows] == 0, float("-inf"))

    def set_train(on: bool) -> None:
        if head is not None:
            head.train(on)
        else:
            model.train(on)

    @torch.no_grad()
    def argmax_of(rows: np.ndarray) -> np.ndarray:
        set_train(False)
        out = [logits(torch.from_numpy(rows[lo:lo + 8192]).to(dev)).argmax(-1).cpu().numpy()
               for lo in range(0, rows.size, 8192)]
        set_train(True)
        return np.concatenate(out) if out else np.zeros(0, np.int64)

    @torch.no_grad()
    def held_ce() -> float:
        set_train(False)
        tot, n = 0.0, 0
        for lo in range(0, ho.size, 8192):
            r = torch.from_numpy(ho[lo:lo + 8192]).to(dev)
            lp = torch.log_softmax(logits(r), -1).nan_to_num(neginf=0.0)
            tot += float(-(y_dev[r] * lp).sum().item())
            n += int(r.numel())
        set_train(True)
        return tot / max(1, n)

    def evaluate() -> Dict[str, float]:
        out = {"held_ce": held_ce()}
        if fit_rows.size:
            out["train_fit"] = float((argmax_of(fit_rows) == top[fit_rows]).mean())
        if dep_ho.size:
            a_dep = argmax_of(dep_ho)
            out["held_agree_dep"] = float((a_dep == top[dep_ho]).mean())
            out["held_keep_dep"] = float((a_dep == net[dep_ho]).mean())     # still the network's own move
        a_all = argmax_of(ho)
        out["held_agree_all"] = float((a_all == top[ho]).mean())
        agree_ho = top[ho] == net[ho]
        if agree_ho.any():
            out["held_leave_agree"] = float((a_all[agree_ho] != net[ho][agree_ho]).mean())   # damage
        if pdep_ho.size:
            out["held_agree_pick"] = float((argmax_of(pdep_ho) == pick[pdep_ho]).mean())
        if pdep_tr.size:
            sample = pdep_tr[:cfg.fit_sample]
            out["train_fit_pick"] = float((argmax_of(sample) == pick[sample]).mean())
        out.update(ceiling_dep=ceiling, ceiling_all=ceiling_all, ceiling_pick=ceiling_pick)
        return out

    run_dir = _resolve_run_dir(output_dir, run_name, "policyfit", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    meta: Dict[str, Any] = dict(cli or {})
    if not cfg.reinit:
        meta["resumed_from"] = cfg.init
    meta |= {"run_name": run_name, "mode": "policy-fit", "description": description, "seed": cfg.seed,
             "git_commit": _git("rev-parse", "HEAD"),
             "git_dirty": bool(_git("status", "--porcelain", "--", "ai", "tools", "bindings", "engine")),
             "dataset": cfg.dataset, "dataset_meta": json.loads(pathlib.Path(cfg.dataset + ".meta.json").read_text())
             if pathlib.Path(cfg.dataset + ".meta.json").exists() else None,
             "init": cfg.init, "config": asdict(cfg), "ladder_config": lcfg, "parameters": n_params,
             "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (pathlib.Path(run_dir) / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    log = open(os.path.join(run_dir, "policy_fit_metrics.jsonl"), "a", encoding="utf-8")
    first = evaluate()
    first.update(step=0, epoch=0.0)
    print("  step 0:", {k: round(v, 4) for k, v in first.items()}, flush=True)
    log.write(json.dumps(first) + "\n")
    log.flush()
    best_fit, best_ce, bad, step, cursor = -1.0, math.inf, 0, 0, 0
    perm = rng.permutation(tr)
    t_fit = time.time()
    last: Dict[str, float] = first
    while step < max_steps:
        if cursor + cfg.batch > perm.size:
            perm, cursor = rng.permutation(tr), 0
        idx = torch.from_numpy(np.sort(perm[cursor:cursor + cfg.batch])).to(dev)
        cursor += cfg.batch
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        lp = torch.log_softmax(logits(idx), -1).nan_to_num(neginf=0.0)
        loss = -(y_dev[idx] * lp).sum(-1).mean()
        if anchor is not None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16 and dev.type == "cuda"):
                la = anchor(codes.decode(x_dev[idx]), m_dev[idx])[0]
            pa = torch.softmax(la.float().masked_fill(m_dev[idx] == 0, float("-inf")), -1)
            kl = (pa * (torch.log(pa.clamp(min=1e-12)) - lp)).sum(-1).mean()
            loss = loss + cfg.kl_anchor * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if step % cfg.eval_every == 0 or step == max_steps:
            ev = evaluate()
            ev.update(step=step, epoch=round(step / steps_per_epoch, 2), lr=lr_at(step), train_ce=float(loss.item()),
                      seconds=round(time.time() - t_fit, 1))
            log.write(json.dumps(ev) + "\n")
            log.flush()
            last = ev
            print(f"  step {step} epoch {ev['epoch']}: train fit {100 * ev.get('train_fit', float('nan')):.1f}%  held-out "
                  f"{100 * ev.get('held_agree_dep', float('nan')):.1f}% on departures (teacher {100 * ceiling:.1f}%), "
                  f"{100 * ev['held_agree_all']:.1f}% overall (teacher {100 * ceiling_all:.1f}%)  held-out CE "
                  f"{ev['held_ce']:.4f}", flush=True)
            if ev["held_ce"] < best_ce:
                best_ce = ev["held_ce"]
                torch.save(model.state_dict(), os.path.join(run_dir, "policy_fit_best_heldout.pt"))
            if ev.get("train_fit", 0.0) > best_fit + cfg.min_gain:
                best_fit, bad = ev.get("train_fit", 0.0), 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    print(f"  converged: the training fit has not risen by {cfg.min_gain} in {bad} evaluations", flush=True)
                    break
    torch.save(model.state_dict(), os.path.join(run_dir, "policy_fit_last.pt"))
    if head is not None:
        torch.save(head.state_dict(), os.path.join(run_dir, "policy_fit_head_last.pt"))
    meta.update(finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), steps=step, initial=first, final=last)
    (pathlib.Path(run_dir) / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    log.close()
    print(f"policy fit done at step {step}: train fit {100 * last.get('train_fit', float('nan')):.1f}%, held-out "
          f"{100 * last.get('held_agree_dep', float('nan')):.1f}% (teacher {100 * ceiling:.1f}%) -> {run_dir}")
    return run_dir


def counts(model_cfg: Dict[str, Any]) -> Tuple[int, int]:
    """Parameters of a ladder shape and of the policy path alone (for sizing students)."""
    m = create_ladder_net("cpu", **model_cfg)
    total = sum(p.numel() for p in m.parameters())
    value = sum(p.numel() for n, p in m.named_parameters() if n.startswith(("val_win_head", "val_vp_head", "val_logits_head")))
    return total, total - value
