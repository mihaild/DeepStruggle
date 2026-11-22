"""P34's supervised value fit: an evaluator trained on playout-labelled positions.

The dataset is tools/value_labels.py's (train.npz, holdout.npz, siblings.npz, manifest.json). The
network is a LadderNet, either fine-tuned from an RL checkpoint's weights or the same shape freshly
initialised; what is trained is its value path -- `v_win` toward the label's mean score, `v_vp`
toward the mean final VP (/20) -- plus a small auxiliary head on the same trunk features (VP and
DEFCON at the turn's end, each region's swing at its next scoring), kept beside the checkpoint and
not part of it. The sibling slice adds a paired loss: in each world the candidates' leaves were
played on the same deal and dice, so the difference of their results is a low-noise target for the
difference of their values (P33's value probe: the paired term is what lifted sibling correlation).

The output is an ordinary checkpoint -- the network's full state dict -- so the rollout root reads it
as a leaf network unchanged. Its policy heads are untouched by the fit and not meant to be played.

Observations are stored as exact one-byte codes (ts.ObsCodebook, one table per run) and decoded on
the GPU; the sibling leaves are encoded per batch from their raw states.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ts_engine as ts
from ai.models.ladder_net import LadderNet, create_ladder_net, ladder_config_from_state_dict
from ai.models.hyperspherical import register_sphere_projection
from ai.training.generic_trainer import _resolve_run_dir
from ai.training.npz_mmap import load_npz_mmap
from tools.lib.engine_fingerprint import fingerprint

#: Region order of the dataset's swing columns (tools/value_labels.REGIONS).
N_REGIONS = 7

#: The decisions the rollout root reads at its leaves (k=4, 16 worlds, stop=earlier, leaf=own,
#: chance=keyed; A13+R48's SWA on 3,000 positions of P34's generation-1 set, 2026-10-17). The dataset,
#: one decision in eight of self-play, holds them at 0.225 / 0.037 / 0.21 / 0.48 / 0.04: almost half
#: of it is placement micro-states the root never reads.
LEAF_SHARE: Dict[str, float] = {"SELECT_CARD": 0.708, "HEADLINE": 0.266, "SELECT_PLAY_MODE": 0.012,
                                "POINT_NODE": 0.011, "CHOOSE_BRANCH": 0.002, "SELECT_OP_MODE": 0.001}


def decision_kind(st: ts.GameState) -> str:
    """LEAF_SHARE's kinds: the decision type, a card choice in the headline counted as HEADLINE."""
    k = str(st.ctx().decision_type).split(".")[-1]
    return "HEADLINE" if k == "SELECT_CARD" and st.current_phase == ts.Phase.HEADLINE else k


def leaf_weights(states: np.ndarray, data_kinds: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, Dict[str, float]]:
    """Per row, LEAF_SHARE[kind] / the set's own share of that kind, normalised to mean 1 (the shares
    taken from `data_kinds` when given, so a held-out split is weighted by the training set's)."""
    kinds = [decision_kind(ts.state_from_raw_bytes(b.tobytes())) for b in states]
    u, inv = np.unique(np.asarray(kinds), return_inverse=True)
    share = data_kinds or {str(k): float((inv == i).mean()) for i, k in enumerate(u)}
    w = np.asarray([LEAF_SHARE.get(str(k), 0.0) / max(share.get(str(k), 1.0), 1e-9) for k in u])[inv]
    return (w / max(float(w.mean()), 1e-12)).astype(np.float32), share


@dataclass
class ValueFitConfig:
    dataset: str
    init: str                       # an RL checkpoint: its weights (fine-tune) or its shape (--value-reinit)
    reinit: bool = False
    #: Changes to the init network's ladder config (LadderNet keyword -> value), with reinit only:
    #: the deep shape is {"hidden_dim": 768, "num_res_blocks": 8} (A3's), the token shape
    #: {"token_layers": 2, "token_dim": 128} (A6's).
    arch: Optional[Dict[str, Any]] = None
    batch: int = 4096
    sibling_groups: int = 256       # (position, world) groups of k leaves per step
    sibling_coef: float = 10.0
    vp_coef: float = 1.0
    aux_coef: float = 0.1
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 300
    max_epochs: float = 20.0
    eval_every: int = 250
    patience: int = 6               # evaluations without a better held-out loss
    seed: int = 44
    device: str = "cuda"
    bf16: bool = True
    #: P34 revision 2. Siblings from another dataset directory (D2, the keyed slice) instead of the
    #: main set's own.
    siblings: Optional[str] = None
    #: "world": the paired loss per (decision, world), as R52 trained; "decision": per decision, the
    #: mean over its shared worlds of each candidate's leaf value against the mean of its results --
    #: a single world's difference is a +/-1 coin with sd ~0.6, the 16-world mean is what carries the
    #: move's effect.
    sibling_loss: str = "world"
    sibling_decisions: int = 32     # decisions a step under "decision" (x candidates x worlds leaves)
    leaf_coef: float = 0.0          # each sibling leaf's own cross-entropy to its world's result
    #: Weight the main rows by how often the rollout root reads their kind of decision at a leaf
    #: (LEAF_SHARE) against how often the dataset holds it.
    leaf_weights: bool = False


class AuxHead(nn.Module):
    """Turn-end VP (/20), turn-end DEFCON (/5) and the regions' swings (/10) from trunk features."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 2 + N_REGIONS))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


def _win_logit(model: LadderNet, h: torch.Tensor) -> torch.Tensor:
    """v_win's pre-tanh value z: v_win = tanh(z) and P(win) = sigmoid(2z), so the cross-entropy is
    taken on 2z without clamping a probability."""
    head = model.val_win_head
    if not isinstance(head, nn.Sequential) or not isinstance(head[-1], nn.Tanh):
        raise SystemExit("value fit needs a scalar v_win head ending in tanh (not a categorical model)")
    return head[:-1](h).squeeze(-1)


def _vp(model: LadderNet, h: torch.Tensor) -> torch.Tensor:
    head = model.val_vp_head
    if head is None:
        raise SystemExit("value fit needs a scalar v_vp head")
    return head(h).squeeze(-1)


def _load_split(path: pathlib.Path) -> Dict[str, np.ndarray]:
    """A split's arrays, memory-mapped (ai/training/npz_mmap.py): read as they are used, never all at once."""
    return load_npz_mmap(path)


class Codes:
    """The run's one codebook, its values on the device, and encoding of raw states."""

    def __init__(self, features: int, device: torch.device) -> None:
        self.book = ts.ObsCodebook()
        self.features = features
        self.device = device
        self.values = torch.zeros((256, 1), dtype=torch.float32, device=device)
        self._n = 0

    def encode(self, states: np.ndarray) -> np.ndarray:
        out = []
        for lo in range(0, states.shape[0], 65536):
            out.append(self.book.encode_raw(np.ascontiguousarray(states[lo:lo + 65536]), self.features))
        self._sync()
        return np.concatenate(out) if out else np.zeros((0, ts.OBS_SIZE), np.uint8)

    def _sync(self) -> None:
        if self.book.size() != self._n:
            v = self.book.values()
            self.values[:v.size, 0] = torch.from_numpy(v).to(self.device)
            self._n = int(v.size)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return F.embedding(codes.int(), self.values).squeeze(-1)


def _targets(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    swing = np.nan_to_num(d["swing"], nan=0.0) / 10.0
    return {
        "score": d["score"].astype(np.float32),
        "vp": d["final_vp"].astype(np.float32),
        "aux": np.concatenate([d["turn_vp"][:, None], (d["turn_defcon"] / 5.0)[:, None], swing], axis=1).astype(np.float32),
        "aux_w": np.concatenate([np.ones((d["score"].size, 2), np.float32), d["swing_frac"].astype(np.float32)], axis=1),
    }


def _sibling_groups(s: Dict[str, np.ndarray], holdout: bool) -> Tuple[np.ndarray, np.ndarray]:
    """(groups x k row indices, groups x k scores) for the (position, world) groups whose every
    candidate kept a leaf, in the requested split."""
    meta = s["meta"]
    keep = (s["holdout"] == holdout) & (meta[:, 2] == 0)
    key = s["id"] * 4096 + meta[:, 1]
    order = np.lexsort((meta[:, 0], key))
    order = order[keep[order]]
    k = int(meta[:, 0].max()) + 1 if meta.size else 0
    rows: List[np.ndarray] = []
    if k == 0:
        return np.zeros((0, 0), np.int64), np.zeros((0, 0), np.float32)
    keys = key[order]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:], keys.size]
    for a, b in zip(starts, ends):
        g = order[a:b]
        if g.size == k and (meta[g, 3] == 1).all():
            rows.append(g)
    idx = np.stack(rows) if rows else np.zeros((0, k), np.int64)
    return idx, s["cols"][idx, 0].astype(np.float32) if idx.size else np.zeros((0, k), np.float32)


def sibling_decisions(s: Dict[str, np.ndarray], holdout: bool) -> Dict[str, np.ndarray]:
    """The sibling slice as decisions: `row` (decisions x k x worlds, -1 where absent), `y` the result,
    `leaf` whether that row kept a leaf, `ok` (decisions x worlds) the worlds every candidate finished.
    A row without a leaf ended before its mover decided again: its result is its exact value."""
    meta = s["meta"]
    sel = np.flatnonzero(s["holdout"] == holdout)
    if sel.size == 0:
        z = np.zeros((0, 0, 0), np.int64)
        return {"row": z, "y": z.astype(np.float32), "leaf": z.astype(bool), "ok": np.zeros((0, 0), bool)}
    k = int(meta[:, 0].max()) + 1
    w = int(meta[:, 1].max()) + 1
    u, inv = np.unique(s["id"][sel], return_inverse=True)
    row = np.full((u.size, k, w), -1, np.int64)
    row[inv, meta[sel, 0], meta[sel, 1]] = sel
    present = row >= 0
    safe = np.where(present, row, 0)
    failed = np.where(present, meta[safe, 2] == 1, True)
    ok = (~failed).all(axis=1)
    return {"row": row, "y": np.where(present, s["cols"][safe, 0], 0.0).astype(np.float32),
            "leaf": present & (meta[safe, 3] == 1), "ok": ok}


def parse_arch(items: Optional[List[str]]) -> Dict[str, Any]:
    """`key=value` ladder settings: integers, floats, true/false, else strings."""
    out: Dict[str, Any] = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--value-arch takes key=value, got {item!r}")
        value: Any = raw
        if raw in ("true", "false"):
            value = raw == "true"
        else:
            for conv in (int, float):
                try:
                    value = conv(raw)
                    break
                except ValueError:
                    pass
        out[key] = value
    return out


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@dataclass(frozen=True)
class _DecisionMeans:
    """Sibling decisions' per-candidate means over their shared worlds (decisions x k), and the
    leaves behind them for the per-leaf loss."""
    pred: torch.Tensor          # mean leaf value as a score (a leafless row's: its result)
    target: torch.Tensor        # mean result
    leaf_logits: torch.Tensor   # the leaves' win logits
    leaf_results: torch.Tensor  # their results


def run_value_fit(cfg: ValueFitConfig, run_name: Optional[str], description: Optional[str],
                  output_dir: Optional[str] = None, cli: Optional[Dict[str, Any]] = None) -> str:
    """Fit, evaluate on held-out games, keep the best; returns the run directory. `cli` is the
    tools/train.py settings, recorded top-level in metadata.json under their own names so
    launch_flags.py and run_codes.py read a value fit like any other run."""
    dev = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    ds = pathlib.Path(cfg.dataset)
    sd = torch.load(cfg.init, map_location="cpu", weights_only=True)
    sd.pop("keep_idx", None)
    lcfg = ladder_config_from_state_dict(sd)
    if lcfg is None:
        raise SystemExit(f"{cfg.init}: value fit takes a LadderNet checkpoint")
    if cfg.arch:
        if not cfg.reinit:
            raise SystemExit("a changed architecture (--value-arch) trains from fresh weights: add --value-reinit")
        unknown = sorted(set(cfg.arch) - set(lcfg))
        if unknown:
            raise SystemExit(f"--value-arch: not ladder settings: {unknown}")
        lcfg = {**lcfg, **cfg.arch}
    model = create_ladder_net(dev, **lcfg)
    if not cfg.reinit:
        model.load_state_dict(sd, strict=True)
    model.train()
    manifest = json.loads((ds / "manifest.json").read_text())
    engine = manifest["engine"]
    if engine != fingerprint():
        raise SystemExit(f"{ds}: its raw states were written by engine {engine[:8]}, this build is {fingerprint()[:8]} -- "
                         "raw state bytes are read only by the build that wrote them (tools/value_labels.same_games); "
                         "relabel on this build")
    features = int(getattr(model, "obs_feature_bits", 0))
    width = model.TOTAL_OBS_SIZE

    t0 = time.time()
    codes = Codes(features, dev)
    train = _load_split(ds / "train.npz")
    hold = _load_split(ds / "holdout.npz")
    train_states = train.pop("state")
    hold_states = hold.pop("state")
    x_train = codes.encode(train_states)
    x_hold = codes.encode(hold_states)
    y_train, y_hold = _targets(train), _targets(hold)
    if cfg.leaf_weights:
        w_train, kinds = leaf_weights(train_states)
        w_hold, _ = leaf_weights(hold_states, kinds)
    else:
        w_train = np.ones(x_train.shape[0], np.float32)
        w_hold = np.ones(x_hold.shape[0], np.float32)
    sib_dir = pathlib.Path(cfg.siblings) if cfg.siblings else ds
    sib_path = sib_dir / "siblings.npz"
    sib = _load_split(sib_path) if sib_path.exists() else None
    sib_manifest = json.loads((sib_dir / "manifest.json").read_text()) if sib is not None else None
    if sib is not None:
        g_train, gy_train = _sibling_groups(sib, holdout=False)
        g_hold, gy_hold = _sibling_groups(sib, holdout=True)
        dec_train, dec_hold = sibling_decisions(sib, holdout=False), sibling_decisions(sib, holdout=True)
    else:
        g_train = g_hold = np.zeros((0, 0), np.int64)
        gy_train = gy_hold = np.zeros((0, 0), np.float32)
        dec_train = dec_hold = sibling_decisions({"meta": np.zeros((0, 5), np.int32), "holdout": np.zeros(0, bool),
                                                  "id": np.zeros(0, np.int64), "cols": np.zeros((0, 4), np.float32)},
                                                 holdout=False)
    print(f"value fit: {x_train.shape[0]:,} train / {x_hold.shape[0]:,} held-out positions, "
          f"{g_train.shape[0]:,} / {g_hold.shape[0]:,} sibling groups, {dec_train['row'].shape[0]:,} / "
          f"{dec_hold['row'].shape[0]:,} sibling decisions (from {sib_dir.name}), codebook {codes.book.size()}, "
          f"loaded in {time.time() - t0:.0f}s", flush=True)

    aux = AuxHead(model.val_win_head[0].in_features).to(dev)   # type: ignore[index]
    params = list(model.parameters()) + list(aux.parameters())
    opt = register_sphere_projection(torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay), model)
    steps_per_epoch = max(1, x_train.shape[0] // cfg.batch)
    max_steps = int(cfg.max_epochs * steps_per_epoch)

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        frac = (step - cfg.warmup_steps) / max(1, max_steps - cfg.warmup_steps)
        return cfg.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, frac))))

    y_dev = {k: torch.from_numpy(v).to(dev) for k, v in y_train.items()}
    w_dev = torch.from_numpy(w_train).to(dev)
    del train_states, hold_states

    def heads(obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16 and dev.type == "cuda"):
            h = model._encode(obs)[0]
            z = _win_logit(model, h)
            vp = _vp(model, h)
            a = aux(h)
        return z.float(), vp.float(), a.float()

    def leaf_obs(rows: np.ndarray) -> torch.Tensor:
        assert sib is not None
        c = codes.book.encode_raw(np.ascontiguousarray(sib["leaf"][rows.reshape(-1)]), features)
        codes._sync()
        return codes.decode(torch.from_numpy(c).to(dev, non_blocking=True))

    def paired(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(2.0 * z)
        return (((p - p.mean(1, keepdim=True)) - (y - y.mean(1, keepdim=True))) ** 2).mean()

    def decision_means(dec: Dict[str, np.ndarray], di: np.ndarray) -> _DecisionMeans:
        """For decisions `di`: the per-candidate means of the leaf values and of the results, and the
        leaves' win logits and results. A leafless row's value is its result."""
        row, y, leaf = dec["row"][di], dec["y"][di], dec["leaf"][di]
        ok = np.broadcast_to(dec["ok"][di][:, None, :], row.shape)
        use = leaf & ok
        z_leaf = torch.zeros(0, device=dev)
        if use.any():
            z_leaf, _, _ = heads(leaf_obs(row[use]))
        p = torch.from_numpy(y.copy()).to(dev)
        mask_use = torch.from_numpy(use).to(dev)
        p = p.masked_scatter(mask_use, torch.sigmoid(2.0 * z_leaf)) if use.any() else p
        okt = torch.from_numpy(ok.astype(np.float32)).to(dev)
        n = okt.sum(2).clamp(min=1.0)
        yt = torch.from_numpy(y).to(dev)
        return _DecisionMeans((p * okt).sum(2) / n, (yt * okt).sum(2) / n, z_leaf, yt[mask_use])

    def centred_loss(pm: torch.Tensor, tm: torch.Tensor) -> torch.Tensor:
        return (((pm - pm.mean(1, keepdim=True)) - (tm - tm.mean(1, keepdim=True))) ** 2).mean()

    def split_half_ceiling(dec: Dict[str, np.ndarray]) -> float:
        """Sign agreement of candidate differences between the even and the odd worlds' results: how
        far the labels themselves agree on which candidate is better, the ceiling for the metric."""
        ok = dec["ok"]
        agree = total = 0
        for half_a, half_b in ((0, 1),):
            wa = np.zeros_like(ok); wa[:, half_a::2] = True
            wb = np.zeros_like(ok); wb[:, half_b::2] = True
            ta = (dec["y"] * (ok & wa)[:, None, :]).sum(2) / np.maximum((ok & wa).sum(1), 1)[:, None]
            tb = (dec["y"] * (ok & wb)[:, None, :]).sum(2) / np.maximum((ok & wb).sum(1), 1)[:, None]
            da = ta[:, :, None] - ta[:, None, :]
            db = tb[:, :, None] - tb[:, None, :]
            both = (da != 0) & (db != 0)
            agree += int(((np.sign(da) == np.sign(db)) & both).sum())
            total += int(both.sum())
        return agree / max(1, total)

    dec_ceiling = split_half_ceiling(dec_hold) if dec_hold["row"].shape[0] else float("nan")

    @torch.no_grad()
    def evaluate() -> Dict[str, float]:
        model.eval()
        aux.eval()
        zs, vps = [], []
        for lo in range(0, x_hold.shape[0], 8192):
            z, vp, _ = heads(codes.decode(torch.from_numpy(x_hold[lo:lo + 8192]).to(dev)))
            zs.append(z)
            vps.append(vp)
        z = torch.cat(zs)
        vp = torch.cat(vps)
        y = torch.from_numpy(y_hold["score"]).to(dev)
        yv = torch.from_numpy(y_hold["vp"]).to(dev)
        wh = torch.from_numpy(w_hold).to(dev)
        bce_rows = F.binary_cross_entropy_with_logits(2.0 * z, y, reduction="none")
        bce = bce_rows.mean().item()
        # The floor: the labels' own entropy -- a perfect fit of the 32-world means reaches it. KL is
        # what is left above it, comparable across slices whose labels are more or less decided.
        yc = y.clamp(1e-6, 1 - 1e-6)
        ent_rows = -(yc * yc.log() + (1 - yc) * (1 - yc).log())
        floor = ent_rows.mean().item()
        kl_rows = bce_rows - ent_rows
        out = {"hold_bce": bce, "hold_bce_floor": floor, "hold_vp_mse": F.mse_loss(vp, yv).item(),
               "hold_kl": kl_rows.mean().item(), "hold_kl_leafw": ((kl_rows * wh).sum() / wh.sum()).item(),
               "hold_vp_mse_leafw": (((vp - yv) ** 2 * wh).sum() / wh.sum()).item(),
               "hold_score_mse": F.mse_loss(torch.sigmoid(2.0 * z), y).item(),
               "hold_score_var": float(y.var().item())}
        if dec_hold["row"].shape[0]:
            n = min(dec_hold["row"].shape[0], 3000)
            pms, tms = [], []
            for lo in range(0, n, 64):
                means = decision_means(dec_hold, np.arange(lo, min(n, lo + 64)))
                pms.append(means.pred)
                tms.append(means.target)
            pm, tm = torch.cat(pms), torch.cat(tms)
            dp = pm[:, :, None] - pm[:, None, :]
            dt = tm[:, :, None] - tm[:, None, :]
            decided = dt != 0
            out["hold_dec_sign_agree"] = ((dp.sign() == dt.sign()) & decided).sum().item() / max(1, decided.sum().item())
            out["hold_dec_sign_ceiling"] = dec_ceiling
            out["hold_dec_centred_mse"] = centred_loss(pm, tm).item()
            cp, ct = (pm - pm.mean(1, keepdim=True)).flatten(), (tm - tm.mean(1, keepdim=True)).flatten()
            out["hold_dec_corr"] = float(torch.corrcoef(torch.stack([cp, ct]))[0, 1].item())
        if g_hold.shape[0]:
            n = min(g_hold.shape[0], 4000)
            zz = []
            for lo in range(0, n, 512):
                zg, _, _ = heads(leaf_obs(g_hold[lo:min(n, lo + 512)]))
                zz.append(zg.view(-1, g_hold.shape[1]))
            zg = torch.cat(zz)
            yg = torch.from_numpy(gy_hold[:n]).to(dev)
            out["hold_sibling_paired_mse"] = paired(zg, yg).item()
            dz = zg[:, :, None] - zg[:, None, :]
            dy = yg[:, :, None] - yg[:, None, :]
            decided = dy != 0
            out["hold_sibling_pair_acc"] = ((dz.sign() == dy.sign()) & decided).sum().item() / max(1, decided.sum().item())
        model.train()
        aux.train()
        return out

    run_dir = _resolve_run_dir(output_dir, run_name, "valuefit", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    meta: Dict[str, Any] = dict(cli or {})
    if not cfg.reinit:
        meta["resumed_from"] = cfg.init      # the lineage: a fine-tune branches from its RL network
    meta |= {"run_name": run_name, "mode": "value-fit", "description": description, "seed": cfg.seed,
            "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain", "--", "ai", "tools", "bindings", "engine")),
            "engine": engine, "dataset": manifest["dataset"], "dataset_dir": str(ds),
            "dataset_files": manifest["files"],
            "siblings_dataset": sib_manifest["dataset"] if sib_manifest else None,
            "siblings_files": sib_manifest["files"] if sib_manifest else None, "init": cfg.init, "config": asdict(cfg),
            "ladder_config": lcfg, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (pathlib.Path(run_dir) / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    log = open(os.path.join(run_dir, "value_fit_metrics.jsonl"), "a", encoding="utf-8")

    first = evaluate()
    first.update(step=0, epoch=0.0)
    print("  step 0 (the initial network):", {k: round(v, 4) for k, v in first.items()}, flush=True)
    log.write(json.dumps(first) + "\n")
    log.flush()
    best = math.inf
    best_step = 0
    bad = 0
    step = 0
    perm = rng.permutation(x_train.shape[0])
    cursor = 0
    t_fit = time.time()
    while step < max_steps:
        if cursor + cfg.batch > perm.size:
            perm = rng.permutation(x_train.shape[0])
            cursor = 0
        idx = np.sort(perm[cursor:cursor + cfg.batch])
        cursor += cfg.batch
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        obs = codes.decode(torch.from_numpy(x_train[idx]).to(dev, non_blocking=True))
        ti = torch.from_numpy(idx).to(dev)
        z, vp, a = heads(obs)
        wr = w_dev[ti]
        wsum = wr.sum().clamp(min=1e-6)
        loss_win = (F.binary_cross_entropy_with_logits(2.0 * z, y_dev["score"][ti], reduction="none") * wr).sum() / wsum
        loss_vp = (((vp - y_dev["vp"][ti]) ** 2) * wr).sum() / wsum
        w = y_dev["aux_w"][ti] * wr[:, None]
        loss_aux = (w * (a - y_dev["aux"][ti]) ** 2).sum() / w.sum().clamp(min=1e-6)
        loss = loss_win + cfg.vp_coef * loss_vp + cfg.aux_coef * loss_aux
        loss_sib = torch.zeros((), device=dev)
        loss_leaf = torch.zeros((), device=dev)
        if cfg.sibling_loss == "decision" and dec_train["row"].shape[0]:
            di = rng.integers(0, dec_train["row"].shape[0], size=cfg.sibling_decisions)
            means = decision_means(dec_train, di)
            pm, tm, z_leaf, y_leaf = means.pred, means.target, means.leaf_logits, means.leaf_results
            loss_sib = centred_loss(pm, tm)
            loss = loss + cfg.sibling_coef * loss_sib
            if cfg.leaf_coef > 0 and z_leaf.numel():
                loss_leaf = F.binary_cross_entropy_with_logits(2.0 * z_leaf, y_leaf)
                loss = loss + cfg.leaf_coef * loss_leaf
        elif g_train.shape[0] and cfg.sibling_coef > 0:
            gi = rng.integers(0, g_train.shape[0], size=cfg.sibling_groups)
            zg, _, _ = heads(leaf_obs(g_train[gi]))
            loss_sib = paired(zg.view(len(gi), -1), torch.from_numpy(gy_train[gi]).to(dev))
            loss = loss + cfg.sibling_coef * loss_sib
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if step % cfg.eval_every == 0 or step == max_steps:
            ev = evaluate()
            score = (ev["hold_kl_leafw"] + cfg.vp_coef * ev["hold_vp_mse_leafw"] if cfg.leaf_weights
                     else ev["hold_bce"] + cfg.vp_coef * ev["hold_vp_mse"])
            ev.update(step=step, epoch=round(step / steps_per_epoch, 2), lr=lr_at(step),
                      train_win=loss_win.item(), train_vp=loss_vp.item(), train_aux=loss_aux.item(),
                      train_sib=loss_sib.item(), train_leaf=loss_leaf.item(), seconds=round(time.time() - t_fit, 1))
            log.write(json.dumps(ev) + "\n")
            log.flush()
            better = score < best - 1e-5
            print(f"  step {step} epoch {ev['epoch']}: held-out KL {ev['hold_kl']:.4f} (leaf-weighted "
                  f"{ev['hold_kl_leafw']:.4f}) vp {ev['hold_vp_mse']:.4f} decision sign agreement "
                  f"{ev.get('hold_dec_sign_agree', float('nan')):.3f} (ceiling {ev.get('hold_dec_sign_ceiling', float('nan')):.3f})"
                  f"{'  *' if better else ''}", flush=True)
            if better:
                best, best_step, bad = score, step, 0
                torch.save(model.state_dict(), os.path.join(run_dir, "value_fit_best.pt"))
                torch.save(aux.state_dict(), os.path.join(run_dir, "value_aux_best.pt"))
            else:
                bad += 1
                if bad >= cfg.patience:
                    print(f"  early stop: no better held-out loss in {bad} evaluations", flush=True)
                    break
    meta.update(finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), best_step=best_step,
                steps=step, best_holdout=best, initial=first)
    (pathlib.Path(run_dir) / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    log.close()
    print(f"value fit done: best held-out at step {best_step} -> {run_dir}/value_fit_best.pt")
    return run_dir
