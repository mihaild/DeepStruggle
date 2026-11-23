"""An action-value head fitted to the rollout root's per-candidate leads -- search distilled as values.

`tools/distill_targets.py` writes, for every root teacher, each candidate's paired lead over the network's
own move and that lead's standard error (`cand_T`, `lead_T`, `lead_se_T`). A pick or an improved policy is a
nonlinear function of those noisy means, so their noise becomes bias (the winner's curse; borderline
departures clearing the z2 rule); a regression on the leads themselves is unbiased, and weighting each by
1 / SE^2 is the textbook way to use labels of known, unequal noise. This fits Q(s, a) so that
Q(s, a_j) - Q(s, a_0) is the lead of candidate j over the network's move a_0, on a **copy** of the network:
the network's own policy and value never move, so a fitted head cannot damage them.

Heads (`head`):

* `mlp` -- a fresh two-layer head on the trunk's features, 220 outputs (one value per flat action), zero at
  the start (every candidate level with the network's move). With the trunk frozen its features are
  computed once, so the fit runs on a CPU.
* `policy` -- the network's own policy logits times a learned scale, read as values: the card-conditioned
  heads' structure kept, the scale started at the least-squares fit of the leads on the logit differences.

`train_trunk` trains the copy's trunk too (the GPU cell); otherwise only the head (and for `policy`, the
policy heads and the scale) trains. Stopping is on a validation slice of the training games (one game in
`val_every`), never on the held-out games.

Readings on the held-out games, whose rows were searched twice (two seeds, independent noise):

* `r_dis` -- the correlation of the predicted leads with the two seeds' mean, over (position, candidate)
  pairs, divided by the square root of that mean's reliability (the seeds' split-half correlation
  stepped up by Spearman-Brown): 1.0 is everything the search's leads say that is not noise. The same for
  the starting network's logit differences (`r_dis_logits`) and for a baseline teacher's leads
  (`r_dis_base`, `baseline`: h0, the critic one step on) -- the bar.
* `sign_acc` -- the share of well-determined pairs (the seeds' mean lead beyond 2 SE) whose sign the head
  gets right; the same for the baselines.
* `r_within_dis` -- the same with each position's leads centred on their mean (the network's move
  included): only the part that ranks one position's moves, which is all a root acts on; disattenuated by
  the centred leads' own split-half reliability.
* `gain_*` -- the judge's two-seed mean lead of the move the Q root plays, over every held-out row: what
  it is worth per decision over the network, unbiased since the head never saw those seeds. The bars are
  in `fixed`: `gain_judge_xseed` (the judge's root, one seed's pick valued by the other) and `gain_base`
  (the baseline teacher's pick). A bar averages only the rows where the judge searched the move it plays
  (`*_coverage`; h0 at k 16 often plays a move outside h4's four candidates), so where a bar's coverage is
  below 1 the Q root's gain is also read on exactly that bar's rows (`gain_<theta>_on_base_rows`,
  `gain_<theta>_on_judge_xseed_rows`): the like-for-like comparison.
  `gain_se_*` is its standard error over rows (a game's rows are not independent: a lower bound).
* The **Q root's** move: the candidate with the largest predicted lead when it clears `theta`, else the
  network's. Against the teacher's z2 pick where both seeds agree: `take_*` (it plays the teacher's move
  on the stable departures), `leave_*` (it leaves the network's move where both seeds keep it -- damage),
  `depart_*` (how often it departs at all), at each threshold in `thetas`.
"""

from __future__ import annotations

import copy
import json
import math
import os
import pathlib
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ai.models.head_registry import VALUE_HEADS, refuse_side_split
from ai.models.hyperspherical import register_sphere_projection
from ai.models.ladder_net import create_ladder_net, ladder_config_from_state_dict
from ai.training.generic_trainer import _resolve_run_dir
from ai.training.npz_mmap import load_npz_mmap
from ai.training.policy_fit import ACTIONS, POLICY_MODULES, masks_of, refuse_macro
from ai.training.value_fit import Codes, _git
from tools.lib.engine_fingerprint import check_raw_states

#: Fix verification V-7.2 (2026-11-22): the registry's value heads (the old list named a nonexistent
#: "val_logits_head" and left value_dist_head out).
VALUE_PREFIXES = VALUE_HEADS


@dataclass
class QFitConfig:
    dataset: str                    # a tools/distill_targets.py .npz with a root teacher's statistics
    target: str                     # the teacher by name (its `lead_T`)
    init: str                       # an RL checkpoint: the trunk copied, its policy and value never moved
    head: str = "mlp"               # mlp | policy
    train_trunk: bool = False
    weighting: str = "inverse_variance"   # inverse_variance | unit
    #: Added in quadrature to every lead's standard error: no pair weighs more than 1 / se_floor^2.
    se_floor: float = 0.01
    baseline: Optional[str] = "h0"  # a second teacher's leads read as the bar (absent from the file: skipped)
    #: The teacher whose two seeds judge the held-out rows (default the target): a fit to 32-world leads
    #: judged by the 128-world ones asks whether the regression recovers what four times the search sees.
    judge: Optional[str] = None
    batch: int = 1024               # positions (each with its candidates)
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_epochs: float = 40.0
    eval_every: int = 200
    patience: int = 10              # evaluations without the validation loss falling
    val_every: int = 10             # one training game in this many is the validation slice
    thetas: List[float] = field(default_factory=lambda: [0.0, 0.01, 0.02, 0.04, 0.08])
    #: Fit on this share of the training games (whole games, the first by id): the data-scaling cells.
    #: The validation slice and the held-out games are unchanged.
    train_fraction: float = 1.0
    seed: int = 44
    device: str = "cuda"
    bf16: bool = True


class _Pairs:
    """Positions with their candidates, leads and weights as dense (n, K) arrays; slot 0 is the network's move."""

    def __init__(self, cand: np.ndarray, lead: np.ndarray, se: np.ndarray, cfg: QFitConfig) -> None:
        self.cand = cand.astype(np.int64)
        self.lead = np.nan_to_num(lead.astype(np.float32), nan=0.0)
        ok = (cand >= 0) & np.isfinite(lead) & np.isfinite(se)
        ok[:, 0] = False                       # the network's move: lead 0 by definition
        if cfg.weighting == "unit":
            w = np.ones_like(self.lead)
        elif cfg.weighting == "inverse_variance":
            w = 1.0 / (np.nan_to_num(se.astype(np.float32), nan=1.0) ** 2 + cfg.se_floor ** 2)
        else:
            raise SystemExit(f"--q-weighting {cfg.weighting!r}: inverse_variance or unit")
        self.w = np.where(ok, w, 0.0).astype(np.float32)
        self.ok = ok


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _match(cand: np.ndarray, other_cand: np.ndarray, other_lead: np.ndarray) -> np.ndarray:
    """The other teacher's lead for each of `cand`'s moves (NaN where it did not consider the move)."""
    out = np.full(cand.shape, np.nan, dtype=np.float32)
    for j in range(other_cand.shape[1]):
        hit = (cand == other_cand[:, j:j + 1]) & (cand >= 0)
        out = np.where(hit, other_lead[:, j:j + 1], out)
    return out


def run_q_fit(cfg: QFitConfig, run_name: Optional[str], description: Optional[str],
              output_dir: Optional[str] = None, cli: Optional[Dict[str, Any]] = None) -> str:
    """Fit to the validation minimum, read the held-out numbers; keeps the best and the last; returns the run dir."""
    if cfg.head not in ("mlp", "policy"):
        raise SystemExit(f"--q-head {cfg.head!r}: mlp or policy")
    dev = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    sd = torch.load(cfg.init, map_location="cpu", weights_only=True)
    sd.pop("keep_idx", None)
    lcfg = ladder_config_from_state_dict(sd)
    if lcfg is None:
        raise SystemExit(f"{cfg.init}: the Q fit takes a LadderNet checkpoint")
    refuse_macro(lcfg, "the Q fit")
    refuse_side_split(lcfg, "the Q fit")
    model = create_ladder_net(dev, **lcfg)
    model.load_state_dict(sd, strict=True)
    base_net = copy.deepcopy(model).eval()          # the starting network: its logits are a baseline
    for p in base_net.parameters():
        p.requires_grad_(False)

    t0 = time.time()
    meta_path = pathlib.Path(cfg.dataset + ".meta.json")
    check_raw_states(cfg.dataset)
    d = load_npz_mmap(cfg.dataset)          # memory-mapped: members are read as they are used
    t = cfg.target
    if f"lead_{t}" not in d:
        names = sorted(k[5:] for k in d if k.startswith("lead_") and not k.startswith("lead_se"))
        raise SystemExit(f"{cfg.dataset} has no candidate statistics for {t!r} (it has {names}); "
                         "they are written by tools/distill_targets.py since 42fbf65f")
    searched = (d[f"worlds_{t}"] > 0) & ((d[f"cand_{t}"] >= 0).sum(1) >= 2)
    heldout = d["heldout"].astype(bool)
    game = d["game"].astype(np.int64)
    val_mask = ~heldout & (game % cfg.val_every == cfg.val_every // 2)
    tr = np.flatnonzero(~heldout & ~val_mask & searched)
    if cfg.train_fraction < 1.0:
        games = np.unique(game[tr])
        tr = tr[np.isin(game[tr], games[:max(1, int(round(cfg.train_fraction * games.size)))])]
    va = np.flatnonzero(val_mask & searched)
    jt = cfg.judge or t
    if f"lead2_{jt}" not in d:
        raise SystemExit(f"{cfg.dataset} has no second-seed statistics for the judge {jt!r}")
    ho = np.flatnonzero(heldout & searched & (d[f"worlds2_{jt}"] > 0))
    P = _Pairs(d[f"cand_{t}"], d[f"lead_{t}"], d[f"lead_se_{t}"], cfg)

    def judged(cand_key: str, key: str) -> np.ndarray:
        """The judge's `key` for each of the target's candidates on the held-out rows (NaN where it has none)."""
        return _match(P.cand[ho], d[f"{cand_key}_{jt}"][ho].astype(np.int64), d[f"{key}_{jt}"][ho].astype(np.float32))

    net = d["net_argmax"].astype(np.int64)
    pick, pick2 = d[f"pick_{jt}"].astype(np.int64), d[f"pick2_{jt}"].astype(np.int64)
    base_lead: Optional[np.ndarray] = None
    if cfg.baseline and f"lead_{cfg.baseline}" in d:
        base_lead = _match(P.cand, d[f"cand_{cfg.baseline}"].astype(np.int64), d[f"lead_{cfg.baseline}"].astype(np.float32))

    codes = Codes(int(getattr(model, "obs_feature_bits", 0)), dev)
    x_all = codes.encode(d["states"])
    m_all = masks_of(d["states"])
    x_dev = torch.from_numpy(x_all).to(dev)
    m_dev = torch.from_numpy(m_all).to(dev)
    cand_dev = torch.from_numpy(np.maximum(P.cand, 0)).to(dev)
    lead_dev = torch.from_numpy(P.lead).to(dev)
    w_dev = torch.from_numpy(P.w).to(dev)
    autocast = cfg.bf16 and dev.type == "cuda"

    @torch.no_grad()
    def base_logits(rows: np.ndarray) -> np.ndarray:
        out = []
        for lo in range(0, rows.size, 8192):
            r = torch.from_numpy(rows[lo:lo + 8192]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                lg = base_net(codes.decode(x_dev[r]), m_dev[r])[0]
            out.append(torch.gather(lg.float(), 1, cand_dev[r]).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, P.cand.shape[1]), np.float32)

    # The trainable copy: the head, and with train_trunk the trunk; never the value heads.
    for name, p in model.named_parameters():
        p.requires_grad_(cfg.train_trunk and not name.startswith(VALUE_PREFIXES))
    head: Optional[nn.Module] = None
    scale = torch.zeros((), device=dev)
    feats: Optional[torch.Tensor] = None
    if cfg.head == "mlp":
        with torch.no_grad():
            width = int(getattr(model, "_encode")(torch.zeros(1, model.TOTAL_OBS_SIZE, device=dev))[0].shape[1])
        head = nn.Sequential(nn.Linear(width, 512), nn.GELU(), nn.Linear(512, ACTIONS)).to(dev)
        out_layer = head[-1]
        assert isinstance(out_layer, nn.Linear)
        nn.init.zeros_(out_layer.weight)
        nn.init.zeros_(out_layer.bias)
        if not cfg.train_trunk:             # frozen trunk: its features once, the fit is a head on a table
            model.eval()
            with torch.no_grad():
                parts = []
                for lo in range(0, x_dev.shape[0], 4096):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                        parts.append(getattr(model, "_encode")(codes.decode(x_dev[lo:lo + 4096]))[0].float())
                feats = torch.cat(parts)
    else:
        if not cfg.train_trunk:
            for name in POLICY_MODULES:
                mod = getattr(model, name, None)
                if isinstance(mod, nn.Module):
                    for p in mod.parameters():
                        p.requires_grad_(True)
        lg0 = base_logits(tr[:20000])
        x = (lg0[:, 1:] - lg0[:, :1])[P.ok[tr[:20000], 1:]]
        y = P.lead[tr[:20000], 1:][P.ok[tr[:20000], 1:]]
        s0 = float((x * y).sum() / max(1e-9, (x * x).sum()))
        scale = torch.tensor(s0, device=dev, requires_grad=True)
        print(f"  policy head: the leads' least-squares scale on the logit differences {s0:.4f}", flush=True)
    params: List[torch.Tensor] = [p for p in model.parameters() if p.requires_grad]
    if head is not None:
        params += list(head.parameters())
    if cfg.head == "policy":
        params.append(scale)
    if not params:
        raise SystemExit("nothing to train")
    n_train = sum(p.numel() for p in params)

    def values(rows: torch.Tensor) -> torch.Tensor:
        """(len(rows), K) predicted values of each row's candidates."""
        if head is not None and feats is not None:
            q = head(feats[rows])
        else:
            obs = codes.decode(x_dev[rows])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                if head is not None:
                    q = head(getattr(model, "_encode")(obs)[0])
                else:
                    q = model(obs, m_dev[rows])[0]
            q = q.float()
            if head is None:
                q = q.masked_fill(m_dev[rows] == 0, 0.0) * scale
        return torch.gather(q.float(), 1, cand_dev[rows])

    def leads(rows: torch.Tensor) -> torch.Tensor:
        v = values(rows)
        return v - v[:, :1]

    def loss_of(rows: torch.Tensor) -> torch.Tensor:
        w = w_dev[rows]
        return (w * (leads(rows) - lead_dev[rows]) ** 2).sum() / w.sum().clamp(min=1e-9)

    def set_train(on: bool) -> None:
        if head is not None:
            head.train(on)
        if cfg.train_trunk:
            model.train(on)

    @torch.no_grad()
    def predict(rows: np.ndarray) -> np.ndarray:
        set_train(False)
        out = [leads(torch.from_numpy(rows[lo:lo + 8192]).to(dev)).cpu().numpy() for lo in range(0, rows.size, 8192)]
        set_train(True)
        return np.concatenate(out) if out else np.zeros((0, P.cand.shape[1]), np.float32)

    @torch.no_grad()
    def val_loss() -> float:
        set_train(False)
        num, den = 0.0, 0.0
        for lo in range(0, va.size, 8192):
            r = torch.from_numpy(va[lo:lo + 8192]).to(dev)
            w = w_dev[r]
            num += float((w * (leads(r) - lead_dev[r]) ** 2).sum())
            den += float(w.sum())
        set_train(True)
        return num / max(den, 1e-9)

    # The held-out judge: its two seeds' mean lead and that mean's reliability, computed once.
    l1, l2 = judged("cand", "lead"), judged("cand2", "lead2")
    se1, se2 = judged("cand", "lead_se"), judged("cand2", "lead_se2")
    ok_ho = P.ok[ho] & np.isfinite(l1) & np.isfinite(l2) & np.isfinite(se1) & np.isfinite(se2)
    l1, l2, se1, se2 = (np.nan_to_num(x, nan=0.0) for x in (l1, l2, se1, se2))
    mean12 = 0.5 * (l1 + l2)
    se12 = 0.5 * np.sqrt(se1 ** 2 + se2 ** 2)
    r12 = _corr(l1[ok_ho], l2[ok_ho])
    rel = 2 * r12 / (1 + r12) if r12 == r12 and r12 > 0 else float("nan")
    sure = ok_ho & (np.abs(mean12) > 2 * se12)
    stable = pick[ho] == pick2[ho]
    stable_dep = stable & (pick[ho] != net[ho])
    stable_keep = stable & (pick[ho] == net[ho])

    def centred(x: np.ndarray, m: np.ndarray) -> np.ndarray:
        """`x` less its row's mean over `m` and the network's move (slot 0, lead 0): what ranks one position's moves."""
        mm = m.copy()
        mm[:, 0] = m.any(1)
        return x - (x * mm).sum(1, keepdims=True) / np.maximum(mm.sum(1, keepdims=True), 1)

    within_ok = ok_ho.copy()
    within_ok[:, 0] = ok_ho.any(1)
    r12_within = _corr(centred(l1, ok_ho)[within_ok], centred(l2, ok_ho)[within_ok])
    rel_within = 2 * r12_within / (1 + r12_within) if r12_within == r12_within and r12_within > 0 else float("nan")

    def readings(pred: np.ndarray, prefix: str, sel: Optional[np.ndarray] = None) -> Dict[str, float]:
        pairs = ok_ho if sel is None else ok_ho & sel
        well = sure & pairs
        r = _corr(pred[pairs], mean12[pairs])
        wp = pairs.copy()
        wp[:, 0] = pairs.any(1)
        rw = _corr(centred(pred, pairs)[wp], centred(mean12, pairs)[wp])
        return {f"r{prefix}": r, f"r_dis{prefix}": r / math.sqrt(rel) if rel == rel else float("nan"),
                f"r_within_dis{prefix}": rw / math.sqrt(rel_within) if rel_within == rel_within else float("nan"),
                f"sign_acc{prefix}": float((np.sign(pred[well]) == np.sign(mean12[well])).mean()) if well.any()
                else float("nan")}

    def q_root(pred: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        masked = np.where(ok_ho, pred, -np.inf)
        best = masked.argmax(1)
        lead_best = masked[np.arange(best.size), best]
        for th in cfg.thetas:
            move = np.where(lead_best > th, P.cand[ho][np.arange(best.size), best], net[ho])
            out[f"depart_{th:g}"] = float((move != net[ho]).mean())
            out[f"take_{th:g}"] = float((move[stable_dep] == pick[ho][stable_dep]).mean()) if stable_dep.any() else float("nan")
            out[f"leave_{th:g}"] = float((move[stable_keep] != net[ho][stable_keep]).mean()) if stable_keep.any() else float("nan")
            gain = np.where(lead_best > th, mean12[np.arange(best.size), best], 0.0)
            out[f"gain_{th:g}"] = float(gain.mean())
            out[f"gain_se_{th:g}"] = float(gain.std() / math.sqrt(max(1, gain.size)))
            for bar, rows in bar_rows.items():
                out[f"gain_{th:g}_on_{bar}_rows"] = float(gain[rows].mean()) if rows.any() else float("nan")
        return out

    def gain_of(moves: np.ndarray, cands: List[str], leads_: List[str]) -> Tuple[Dict[str, float], np.ndarray]:
        """The judge's lead (averaged over the seeds named) of each held-out row's `moves`, the network's move 0,
        over the rows where the judge has the move; and those rows."""
        vals = [_match(moves[:, None], d[f"{c}_{jt}"][ho].astype(np.int64), d[f"{k}_{jt}"][ho].astype(np.float32))[:, 0]
                for c, k in zip(cands, leads_)]
        stack = np.stack(vals)
        seen = np.isfinite(stack).sum(0)
        v = np.where(seen > 0, np.nansum(stack, 0) / np.maximum(seen, 1), np.nan)   # a mean over the seeds that have it
        v = np.where(moves == net[ho], 0.0, v)
        have = np.isfinite(v)
        return {"gain": float(v[have].mean()) if have.any() else float("nan"), "coverage": float(have.mean())}, have

    lg_ho = base_logits(ho)
    fixed: Dict[str, float] = {"reliability_single": r12, "reliability_mean": rel, "reliability_within": rel_within,
                               "stable_departures": int(stable_dep.sum()), "sure_pairs": int(sure.sum()),
                               "train_positions": int(tr.size)}
    # The bars for `gain_*`: the judge's own root, each seed's pick valued by the other seed (no winner's curse),
    # and the baseline teacher's pick valued by both of the judge's seeds (independent searches).
    # A bar's rows: where it has a value. With coverage below 1 the Q root is also read on those rows alone.
    bar_rows: Dict[str, np.ndarray] = {}
    (xs1, have1), (xs2, have2) = gain_of(pick[ho], ["cand2"], ["lead2"]), gain_of(pick2[ho], ["cand"], ["lead"])
    fixed["gain_judge_xseed"] = 0.5 * (xs1["gain"] + xs2["gain"])
    fixed["gain_judge_xseed_coverage"] = 0.5 * (xs1["coverage"] + xs2["coverage"])
    if not (have1 & have2).all():
        bar_rows["judge_xseed"] = have1 & have2
    if cfg.baseline and f"pick_{cfg.baseline}" in d:
        gb, have_b = gain_of(d[f"pick_{cfg.baseline}"][ho].astype(np.int64), ["cand", "cand2"], ["lead", "lead2"])
        fixed["gain_base"], fixed["gain_base_coverage"] = gb["gain"], gb["coverage"]
        if not have_b.all():
            bar_rows["base"] = have_b
    fixed.update(readings(lg_ho - lg_ho[:, :1], "_logits"))
    if base_lead is not None:
        bl = base_lead[ho]
        have = ok_ho & np.isfinite(bl)
        if have.any():
            fixed["baseline_coverage"] = float(have.sum() / max(1, ok_ho.sum()))
            fixed.update(readings(np.nan_to_num(bl), "_base", have))
            fixed.update({f"base_{k}": v for k, v in q_root(np.where(have, np.nan_to_num(bl), -np.inf)).items()})
    print(f"Q fit to {t}, judged by {jt}: {tr.size:,} train / {va.size:,} validation / {ho.size:,} held-out positions; "
          f"held-out lead reliability {r12:.3f} single seed, {rel:.3f} two-seed mean; logits r_dis "
          f"{fixed.get('r_dis_logits', float('nan')):.3f}, {cfg.baseline} r_dis {fixed.get('r_dis_base', float('nan')):.3f}; "
          f"{n_train / 1e6:.2f}M trained parameters; loaded in {time.time() - t0:.0f}s", flush=True)

    def evaluate() -> Dict[str, float]:
        pred = predict(ho)
        out = {"val_loss": val_loss()}
        out.update(readings(pred, ""))
        out.update(q_root(pred))
        return out

    run_dir = _resolve_run_dir(output_dir, run_name, "qfit", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    meta: Dict[str, Any] = dict(cli or {})
    meta["resumed_from"] = cfg.init          # a copy of the network: the run branches from it
    meta |= {"run_name": run_name, "mode": "q-fit", "description": description, "seed": cfg.seed,
             "git_commit": _git("rev-parse", "HEAD"),
             "git_dirty": bool(_git("status", "--porcelain", "--", "ai", "tools", "bindings", "engine")),
             "dataset": cfg.dataset, "dataset_meta": json.loads(meta_path.read_text()) if meta_path.exists() else None,
             "init": cfg.init, "config": asdict(cfg), "ladder_config": lcfg, "trained_parameters": n_train,
             "fixed": fixed, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (pathlib.Path(run_dir) / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")

    def save(name: str) -> None:
        state = {"model": model.state_dict(), "head": head.state_dict() if head is not None else None,
                 "scale": float(scale.detach()), "config": asdict(cfg), "ladder_config": lcfg}
        torch.save(state, os.path.join(run_dir, name))

    opt = register_sphere_projection(torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay), model)
    steps_per_epoch = max(1, tr.size // cfg.batch)
    max_steps = int(cfg.max_epochs * steps_per_epoch)

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        frac = (step - cfg.warmup_steps) / max(1, max_steps - cfg.warmup_steps)
        return cfg.lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, frac))))

    log = open(os.path.join(run_dir, "q_fit_metrics.jsonl"), "a", encoding="utf-8")
    first = evaluate()
    first.update(step=0, epoch=0.0)
    log.write(json.dumps(first) + "\n")
    log.flush()
    print("  step 0:", {k: round(v, 4) for k, v in first.items()}, flush=True)
    best_val, bad, step, cursor = first["val_loss"], 0, 0, 0
    best: Dict[str, float] = first
    last: Dict[str, float] = first
    perm = rng.permutation(tr)
    t_fit = time.time()
    set_train(True)
    save("q_fit_best.pt")
    while step < max_steps:
        if cursor + cfg.batch > perm.size:
            perm, cursor = rng.permutation(tr), 0
        idx = torch.from_numpy(np.sort(perm[cursor:cursor + cfg.batch])).to(dev)
        cursor += cfg.batch
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss = loss_of(idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if step % cfg.eval_every == 0 or step == max_steps:
            ev = evaluate()
            ev.update(step=step, epoch=round(step / steps_per_epoch, 2), lr=lr_at(step), train_loss=float(loss.item()),
                      seconds=round(time.time() - t_fit, 1))
            log.write(json.dumps(ev) + "\n")
            log.flush()
            last = ev
            print(f"  step {step} epoch {ev['epoch']}: val {ev['val_loss']:.5f}  r_dis {ev['r_dis']:.3f} (logits "
                  f"{fixed.get('r_dis_logits', float('nan')):.3f}, {cfg.baseline} {fixed.get('r_dis_base', float('nan')):.3f})  "
                  f"sign {ev['sign_acc']:.3f}  within {ev['r_within_dis']:.3f}  take/leave@0 {ev['take_0']:.3f}/"
                  f"{ev['leave_0']:.3f}  gain@0.02 {ev['gain_0.02']:+.4f} (root {fixed.get('gain_judge_xseed', float('nan')):+.4f})",
                  flush=True)
            if ev["val_loss"] < best_val:
                best_val, best, bad = ev["val_loss"], ev, 0
                save("q_fit_best.pt")
            else:
                bad += 1
                if bad >= cfg.patience:
                    print(f"  converged: the validation loss has not fallen in {bad} evaluations", flush=True)
                    break
    save("q_fit_last.pt")
    meta.update(finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), steps=step, initial=first, best=best,
                final=last)
    (pathlib.Path(run_dir) / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    log.close()
    print(f"Q fit done at step {step}: best val {best_val:.5f}, held-out r_dis {best['r_dis']:.3f} (logits "
          f"{fixed.get('r_dis_logits', float('nan')):.3f}, {cfg.baseline} {fixed.get('r_dis_base', float('nan')):.3f}) -> {run_dir}")
    return run_dir

