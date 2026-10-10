#!/usr/bin/env python3
"""Can the network's own trunk rank sibling moves better than its critic does, given better labels?

The search bank found search evaluation-limited, and its data put the bare critic's ranking of two
sibling moves at 61% right against 2,048-pair playouts (55% at card choice; 256-simulation search
over 64 worlds, 74%). Backgammon programs improved by training their evaluation on rollout-labelled
positions. Before training anything, this asks whether that is possible here:

    label  fresh self-play of the model (seeds disjoint from the bank's); at a sampled share of its
           decisions with two or more legal moves, the network's top --k moves played out in paired
           raw-network continuations (ai/eval/paired_playouts.compare: one redeal and one set of
           dice per pair, shared by every move). Each row: the position, the moves, each move's mean
           score for the mover, and each move's paired difference from the network's own move.
    fit    a value head on the FROZEN trunk features h of each child state (the network's own
           representation, unchanged), trained on those labels -- an MLP, and the critic's own head
           fine-tuned from its weights -- and tested on the bank's positions against their
           2,048-pair playouts: how often each picks the better of two siblings the playouts clearly
           separate, beside the untouched critic.

If a head on frozen h ranks siblings well above the critic, the information is in the trunk and the
critic's training signal is what limits it -- a value-target problem. If it does not, the trunk does
not carry it -- a representation problem.

    PYTHONPATH=.:build/release .venv/bin/python tools/value_probe.py label --model m.pt --games 50 \\
        --rate 0.025 --k 3 --pairs 256 --part 1/20 --out label-1.jsonl.gz
    ... fit --model m.pt --labels label-*.jsonl.gz --bank bank.jsonl.gz \\
        --bank-playouts full/playouts.jsonl.gz --out probe_report.md
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np


def _read(paths: Sequence[str]) -> Iterator[Dict[str, Any]]:
    for p in paths:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def _write(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def position_seed(rid: str, seed: int) -> int:
    """As tools/scripts/bank_playouts.position_seed."""
    return (int(rid, 16) ^ (seed * 0x9E3779B97F4A7C15)) & 0x7FFF_FFFF_FFFF_FFFF


# --- label --------------------------------------------------------------------------------------

def label(model: str, games: int, rate: float, k: int, pairs: int, part: str, seed: int,
          temperature: float, out: str) -> int:
    import ts_engine as ts
    from ai.eval.paired_playouts import compare, paired_diff
    from bindings.action_encoder import ActionEncoder
    from tools.lib.corpus_driver import load_policy, position_token, selfplay

    pk, pn = (int(x) for x in part.split("/"))
    logits_fn, feats = load_policy(model)
    rng = random.Random(seed * 7919 + pk)

    class Sampler:
        def __init__(self) -> None:
            self.positions: List[Tuple[str, Dict[str, Any]]] = []

        def observe(self, st: Any, action: int) -> None:
            mask = np.asarray(ActionEncoder.get_legal_mask(st))
            if int(mask.sum()) < 2 or rng.random() >= rate:
                return
            c = st.ctx()
            mover = c.decision_player if c.decision_player != ts.Player.NONE else st.phasing_player
            self.positions.append((position_token(st), {
                "decision_type": str(c.decision_type).split(".")[-1], "turn": int(st.turn),
                "side": "US" if mover == ts.Player.US else "USSR"}))

    t0 = time.time()
    base = 5_000_000 + 100_000 * pk + 1_000 * seed       # the bank's games were seeds 700,000+
    trackers = selfplay(logits_fn, feats, games, base, min(games, 50), temperature, Sampler)
    sampled = [p for t in trackers for p in t.positions]
    print(f"{len(sampled)} positions from {games} games in {time.time() - t0:.0f}s", file=sys.stderr)

    from tools.lib.corpus_driver import state_from_token
    states = [state_from_token(tok) for tok, _ in sampled]
    from ai.search.pimcts import acting_player
    obs = np.stack([np.asarray(ts.extract_observation_features(s, acting_player(s), feats), dtype=np.float32)
                    for s in states])
    masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8) for s in states])
    lg = np.where(masks.astype(bool), np.asarray(logits_fn(obs, masks)), -np.inf)
    rows, positions = [], []
    for (tok, info), s, z in zip(sampled, states, lg):
        top = [int(a) for a in np.argsort(-z)[:k] if np.isfinite(z[a])]
        rid = hashlib.sha256(tok.encode()).hexdigest()[:16]
        rows.append({"id": rid, "pos": tok, "raw": top[0], "moves": top, **info})
        positions.append((s, top))

    def act(o: np.ndarray, m: np.ndarray) -> np.ndarray:
        return np.asarray(logits_fn(o, m)).argmax(axis=1).astype(np.int32)

    t1 = time.time()
    scores = compare(positions, act, pairs, [position_seed(r["id"], seed) for r in rows])
    for r, sc in zip(rows, scores):
        r["pairs"] = pairs
        r["score"] = {str(a): round(sum(v) / len(v), 5) for a, v in sc.items()}
        r["diff_vs_raw"] = {str(a): [round(x, 5) for x in paired_diff(v, sc[r["raw"]])]
                            for a, v in sc.items() if a != r["raw"]}
    _write(out, rows)
    with open(out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"stage": "label", "model": os.path.basename(model), "model_sha256": _sha256(model),
                   "games": games, "rate": rate, "k": k, "pairs": pairs, "part": part, "seed": seed,
                   "temperature": temperature, "game_seed_base": base, "positions": len(rows),
                   "selfplay_s": round(t1 - t0, 1), "playout_s": round(time.time() - t1, 1)}, f, indent=1)
    print(f"{len(rows)} positions labelled, {pairs} pairs, playouts {time.time() - t1:.0f}s", file=sys.stderr)
    return 0


# --- fit ----------------------------------------------------------------------------------------

def child_features(model: Any, feats: int, pos: str, moves: Sequence[int], worlds: int,
                   seed: int, perspective: str = "acting") -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Per move: the trunk features h of its child in `worlds` redealt worlds (the mover's unseen
    cards and the dice drawn afresh, as the paired playouts draw them), and the sign that turns the
    value for the observing player into one for the mover. `perspective` "acting" observes the child
    as the player to act there (what a search's leaf reads: after a headline choice that is the
    opponent, who cannot see it), "mover" as the player who chose (who can)."""
    import torch
    import ts_engine as ts
    from ai.eval.paired_playouts import apply_move, decider, pair_start
    from ai.search.pimcts import acting_player
    from tools.lib.corpus_driver import state_from_token

    st = state_from_token(pos)
    mover = decider(st)
    rows, keys, signs = [], [], []
    for w in range(worlds):
        base = pair_start(st, w, seed)
        for a in moves:
            ch = apply_move(base, int(a))
            if ts.Engine.is_terminal(ch):
                continue
            who = acting_player(ch) if perspective == "acting" else mover
            rows.append(np.asarray(ts.extract_observation_features(ch, who, feats), dtype=np.float32))
            keys.append(int(a))
            signs.append(1.0 if who == mover else -1.0)
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    if not rows:
        return out
    with torch.no_grad():
        h = model.extract_features(torch.from_numpy(np.stack(rows))).float().numpy()
    for a in set(keys):
        idx = [i for i, x in enumerate(keys) if x == a]
        out[a] = (h[idx], np.asarray([signs[i] for i in idx], dtype=np.float32))
    return out


def sibling_accuracy(pred: Dict[str, Dict[int, float]], truth: Dict[str, Dict[int, Tuple[float, float]]],
                     raw: Dict[str, int], min_z: float = 2.0, min_gap: float = 0.01) -> Tuple[float, int]:
    """Share of (move, raw) sibling pairs the playouts clearly separate (|diff| > min_z SE and >
    min_gap) on which `pred` ranks the two the same way."""
    hit = n = 0
    for pid, diffs in truth.items():
        for a, (m, se) in diffs.items():
            if abs(m) <= min_z * se or abs(m) <= min_gap:
                continue
            if pid not in pred or a not in pred[pid] or raw[pid] not in pred[pid]:
                continue
            n += 1
            hit += int((pred[pid][a] - pred[pid][raw[pid]] > 0) == (m > 0))
    return (hit / n if n else float("nan")), n


def _groups(net: Any, feats: int, rows: Sequence[Dict[str, Any]], worlds: int, seed: int,
            perspective: str) -> Tuple[Any, Any, Any, Any, List[int]]:
    """Per (position, world): the children of the position's moves (the network's own move first),
    as features H (G x K x d), signs S and the mover-value labels L (G x K), a mask M, and each
    group's position index. A child that ends the game is left out (masked)."""
    import torch
    K = max(len(r["moves"]) for r in rows)
    width = 0
    Hs, Ss, Ls, Ms, P = [], [], [], [], []
    for pi, r in enumerate(rows):
        cf = child_features(net, feats, r["pos"], r["moves"], worlds, position_seed(r["id"], seed), perspective)
        n_w = max((len(v[1]) for v in cf.values()), default=0)
        if cf and not width:
            width = int(next(iter(cf.values()))[0].shape[1])
        for w in range(n_w):
            h = np.zeros((K, width), dtype=np.float32)
            sg = np.zeros(K, dtype=np.float32)
            lab = np.zeros(K, dtype=np.float32)
            m = np.zeros(K, dtype=np.float32)
            for j, a in enumerate(r["moves"]):
                if a not in cf or len(cf[a][1]) <= w:
                    continue
                h[j] = cf[a][0][w]
                sg[j] = cf[a][1][w]
                lab[j] = 2.0 * float(r["score"][str(a)]) - 1.0
                m[j] = 1.0
            if m[0] == 0:
                continue                                  # the network's own move is the reference
            Hs.append(h); Ss.append(sg); Ls.append(lab); Ms.append(m); P.append(pi)
    return (torch.from_numpy(np.stack(Hs)), torch.from_numpy(np.stack(Ss)), torch.from_numpy(np.stack(Ls)),
            torch.from_numpy(np.stack(Ms)), P)


def _ranking(pred: np.ndarray, rows: Sequence[Dict[str, Any]], P: List[int]) -> Dict[str, float]:
    """Held-out ranking of each move against the network's own: predicted differences (averaged over
    worlds) against the labels' paired differences -- accuracy on the pairs the labels separate
    (> 2 SE and > 1 point), and the correlation over every pair, also disattenuated by the labels'
    own reliability."""
    by_pos: Dict[int, List[np.ndarray]] = {}
    for g, pi in enumerate(P):
        by_pos.setdefault(pi, []).append(pred[g])
    hit = n = 0
    xs, ys, ses = [], [], []
    for pi, preds in by_pos.items():
        r = rows[pi]
        mean = np.nanmean(np.stack(preds), axis=0)
        for j, a in enumerate(r["moves"][1:], start=1):
            if str(a) not in r["diff_vs_raw"] or not np.isfinite(mean[j]):
                continue
            m, se = r["diff_vs_raw"][str(a)]
            d = float(mean[j] - mean[0])
            xs.append(d); ys.append(2.0 * m); ses.append(2.0 * se)
            if abs(m) > 2 * se and abs(m) > 0.01:
                n += 1
                hit += int((d > 0) == (m > 0))
    x, y = np.asarray(xs), np.asarray(ys)
    corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")
    rel = 1.0 - float(np.mean(np.asarray(ses) ** 2)) / float(np.var(y)) if len(y) > 2 else float("nan")
    return {"accuracy": hit / n if n else float("nan"), "clear_pairs": n, "pairs": len(x),
            "corr": corr, "label_reliability": rel,
            "corr_disattenuated": corr / math.sqrt(rel) if rel > 0 else float("nan")}


def fit(model_path: str, labels: Sequence[str], bank: str, bank_playouts: str, worlds: int,
        test_worlds: int, epochs: int, out: str, seed: int = 0, test_limit: int = 0,
        perspective: str = "acting", holdout: float = 0.2, pair_weight: float = 10.0) -> int:
    import copy
    import torch
    import torch.nn as nn
    from tools.lib.player_agent import NeuralAgent
    from bindings.ts_env import model_obs_features

    torch.manual_seed(seed)
    net = NeuralAgent.from_checkpoint(model_path, device="cpu").model.eval()
    feats = int(model_obs_features(net))
    head = getattr(net, "val_win_head")

    lab = [r for r in _read(labels) if len(r["moves"]) >= 2]
    rng = random.Random(seed)
    rng.shuffle(lab)
    n_test = int(len(lab) * holdout)
    test_rows, train_rows = lab[:n_test], lab[n_test:]
    t0 = time.time()
    with torch.no_grad():
        Htr, Str, Ltr, Mtr, Ptr = _groups(net, feats, train_rows, worlds, 11, perspective)
        Hte, Ste, Lte, Mte, Pte = _groups(net, feats, test_rows, worlds, 11, perspective)
    print(f"{len(train_rows)} train / {len(test_rows)} held-out positions; {len(Ptr)} / {len(Pte)} "
          f"groups; features in {time.time() - t0:.0f}s", file=sys.stderr)

    def mover_values(fn: Any, H: Any, S: Any, M: Any) -> Any:
        g, k, d = H.shape
        v = fn(H.reshape(g * k, d)).reshape(g, k) * S
        return torch.where(M > 0, v, torch.full_like(v, float("nan")))

    def heldout(fn: Any) -> Dict[str, float]:
        with torch.no_grad():
            pred = mover_values(fn, Hte, Ste, Mte).numpy()
        return _ranking(pred, test_rows, Pte)

    # The bank: adversarial for any critic-like judge (its positions are where critic-driven search
    # disagreed with the network), reported second.
    bank_rows = {str(r["id"]): r for r in _read([bank])}
    truth: Dict[str, Dict[int, Tuple[float, float]]] = {}
    test_moves: Dict[str, List[int]] = {}
    raw: Dict[str, int] = {}
    for p in _read([bank_playouts]):
        pid = str(p["id"])
        if pid not in bank_rows:
            continue
        sc = {int(a): v for a, v in p["scores"].items()}
        r0 = int(bank_rows[pid]["raw"])
        if r0 not in sc:
            continue
        raw[pid] = r0
        test_moves[pid] = sorted(sc)
        d: Dict[int, Tuple[float, float]] = {}
        for a, v in sc.items():
            if a == r0:
                continue
            diff = [x - y for x, y in zip(v, sc[r0])]
            m = sum(diff) / len(diff)
            se = math.sqrt(sum((x - m) ** 2 for x in diff) / (len(diff) - 1) / len(diff))
            d[a] = (m, se)
        truth[pid] = d
        if test_limit and len(truth) >= test_limit:
            break
    bank_feats = {pid: child_features(net, feats, bank_rows[pid]["pos"], test_moves[pid], test_worlds,
                                      position_seed(pid, 13), perspective) for pid in truth}

    def on_bank(fn: Any) -> float:
        out_: Dict[str, Dict[int, float]] = {}
        with torch.no_grad():
            for pid, cf in bank_feats.items():
                out_[pid] = {a: float((fn(torch.from_numpy(h)).reshape(-1) * torch.from_numpy(sg)).mean())
                             for a, (h, sg) in cf.items()}
        return sibling_accuracy(out_, truth, raw)[0]

    results: Dict[str, Dict[str, float]] = {}
    results["critic (untouched)"] = {**heldout(head), "bank": on_bank(head)}

    def train(module: nn.Module, name: str, lr: float, w_abs: float, w_pair: float,
              keep: Optional[List[int]] = None) -> Dict[str, float]:
        opt = torch.optim.Adam(module.parameters(), lr=lr)
        idx_all = np.arange(len(Ptr)) if keep is None else np.asarray(keep)
        for ep in range(epochs):
            idx = idx_all.copy()
            np.random.default_rng(seed + ep).shuffle(idx)
            for lo in range(0, len(idx), 512):
                b = torch.from_numpy(idx[lo:lo + 512])
                H, S, L, M = Htr[b], Str[b], Ltr[b], Mtr[b]
                g, k, dd = H.shape
                v = module(H.reshape(g * k, dd)).reshape(g, k) * S
                l_abs = (((v - L) ** 2) * M).sum() / M.sum()
                dv, dl = v[:, 1:] - v[:, :1], L[:, 1:] - L[:, :1]
                mm = M[:, 1:] * M[:, :1]
                l_pair = (((dv - dl) ** 2) * mm).sum() / mm.sum().clamp(min=1.0)
                loss = w_abs * l_abs + w_pair * l_pair
                opt.zero_grad()
                loss.backward()
                opt.step()
            r_ = heldout(module)
            print(f"  {name} epoch {ep + 1}: held-out accuracy {r_['accuracy']:.1%} ({r_['clear_pairs']}), "
                  f"corr {r_['corr']:+.3f}", file=sys.stderr, flush=True)
        return heldout(module)

    def mlp() -> nn.Module:
        return nn.Sequential(nn.Linear(Htr.shape[2], 256), nn.GELU(), nn.Linear(256, 1), nn.Tanh())

    tuned = copy.deepcopy(head)
    for p_ in tuned.parameters():
        p_.requires_grad_(True)
    results["critic head fine-tuned (abs + pair)"] = {
        **train(tuned, "tuned", 3e-4, 1.0, pair_weight), "bank": on_bank(tuned)}
    m1 = mlp()
    results["fresh MLP on frozen h (abs + pair)"] = {**train(m1, "mlp", 1e-3, 1.0, pair_weight), "bank": on_bank(m1)}
    m2 = mlp()
    results["fresh MLP on frozen h (pair only)"] = {**train(m2, "mlp-pair", 1e-3, 0.0, 1.0), "bank": on_bank(m2)}
    curve = {}
    for frac in (0.25, 0.5):
        keep_pos = set(range(int(len(train_rows) * frac)))
        keep = [g for g, pi in enumerate(Ptr) if pi in keep_pos]
        curve[f"{len(keep_pos)} positions"] = train(mlp(), f"curve{frac}", 1e-3, 1.0, pair_weight, keep)["accuracy"]
    curve[f"{len(train_rows)} positions"] = results["fresh MLP on frozen h (abs + pair)"]["accuracy"]

    lines = ["# Value probe: can the frozen trunk rank sibling moves better than the critic?", "",
             f"* model `{os.path.basename(model_path)}`; {len(lab)} rollout-labelled positions of fresh "
             f"self-play (network's top {max(len(r['moves']) for r in lab)} moves, "
             f"{lab[0]['pairs']} paired continuations each), {len(train_rows)} to train, {len(test_rows)} held "
             f"out; children observed from the {perspective} player's side, {worlds} worlds each.",
             "* held-out accuracy: pairs (a move against the network's own) the labels separate by > 2 SE "
             "and > 1 point, ranked the same way; corr: predicted against labelled difference over every "
             "pair, and disattenuated by the labels' reliability.",
             "* bank: the search bank's 2,048-pair pairs -- adversarial for critic-like judges (chosen where "
             "critic-driven search disagreed with the network).", "",
             "| value | held-out accuracy | clear pairs | corr | corr disattenuated | bank accuracy |",
             "|:---|---:|---:|---:|---:|---:|"]
    for k_, v in results.items():
        lines.append(f"| {k_} | {v['accuracy']:.1%} | {v['clear_pairs']} | {v['corr']:+.3f} | "
                     f"{v['corr_disattenuated']:+.3f} | {v['bank']:.1%} |")
    rel = results["critic (untouched)"]["label_reliability"]
    lines += ["", f"Label reliability (paired differences): {rel:.2f}.",
              "Learning curve (fresh MLP, abs + pair): " + ", ".join(f"{k_} {v:.1%}" for k_, v in curve.items()), ""]
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "curve": curve}, f, indent=1)
    torch.save({"tuned_head": tuned.state_dict(), "mlp": m1.state_dict()}, os.path.splitext(out)[0] + ".heads.pt")
    print("\n".join(lines))
    return 0


def tune(model_path: str, labels: Sequence[str], worlds: int, epochs: int, pair_weight: float,
         out: str, seed: int = 0) -> int:
    """The critic head fine-tuned on every labelled position (absolute + paired-difference loss, the
    form `fit` validated on held-out positions), written as a full checkpoint: the base network with
    only `val_win_head` replaced, so its policy is untouched and a search over it differs only in
    the values it reads."""
    import copy
    import torch
    from tools.lib.player_agent import NeuralAgent
    from bindings.ts_env import model_obs_features

    torch.manual_seed(seed)
    net = NeuralAgent.from_checkpoint(model_path, device="cpu").model.eval()
    feats = int(model_obs_features(net))
    rows = [r for r in _read(labels) if len(r["moves"]) >= 2]
    with torch.no_grad():
        H, S, L, M, _P = _groups(net, feats, rows, worlds, 11, "acting")
    if net.val_win_head is None:
        raise ValueError(f"{model_path} has no scalar win head to tune")
    head = copy.deepcopy(net.val_win_head)
    for p_ in head.parameters():
        p_.requires_grad_(True)
    opt = torch.optim.Adam(head.parameters(), lr=3e-4)
    for ep in range(epochs):
        idx = np.arange(len(H))
        np.random.default_rng(seed + ep).shuffle(idx)
        for lo in range(0, len(idx), 512):
            b = torch.from_numpy(idx[lo:lo + 512])
            g, k, d = H[b].shape
            v = head(H[b].reshape(g * k, d)).reshape(g, k) * S[b]
            l_abs = (((v - L[b]) ** 2) * M[b]).sum() / M[b].sum()
            mm = M[b][:, 1:] * M[b][:, :1]
            l_pair = ((((v[:, 1:] - v[:, :1]) - (L[b][:, 1:] - L[b][:, :1])) ** 2) * mm).sum() / mm.sum().clamp(min=1.0)
            loss = l_abs + pair_weight * l_pair
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"  epoch {ep + 1}: loss {float(loss):.4f}", file=sys.stderr, flush=True)
    sd = torch.load(model_path, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    for k, v in head.state_dict().items():
        sd[f"val_win_head.{k}"] = v.detach().clone()
    torch.save(sd, out)
    with open(out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"base": os.path.basename(model_path), "base_sha256": _sha256(model_path),
                   "positions": len(rows), "groups": len(H), "worlds": worlds, "epochs": epochs,
                   "pair_weight": pair_weight, "replaced": "val_win_head"}, f, indent=1)
    print(f"wrote {out} ({len(rows)} positions, {len(H)} groups)", file=sys.stderr)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lab = sub.add_parser("label")
    lab.add_argument("--model", required=True)
    lab.add_argument("--games", type=int, default=50)
    lab.add_argument("--rate", type=float, default=0.025)
    lab.add_argument("--k", type=int, default=3)
    lab.add_argument("--pairs", type=int, default=256)
    lab.add_argument("--part", default="1/1")
    lab.add_argument("--seed", type=int, default=0)
    lab.add_argument("--temperature", type=float, default=0.1)
    lab.add_argument("--out", required=True)
    ft = sub.add_parser("fit")
    ft.add_argument("--model", required=True)
    ft.add_argument("--labels", nargs="+", required=True)
    ft.add_argument("--bank", required=True)
    ft.add_argument("--bank-playouts", required=True)
    ft.add_argument("--worlds", type=int, default=4)
    ft.add_argument("--test-worlds", type=int, default=16)
    ft.add_argument("--epochs", type=int, default=8)
    ft.add_argument("--test-limit", type=int, default=0, help="first N bank positions only (smoke runs)")
    ft.add_argument("--perspective", choices=["acting", "mover"], default="acting")
    ft.add_argument("--holdout", type=float, default=0.2)
    ft.add_argument("--pair-weight", type=float, default=10.0)
    ft.add_argument("--out", required=True)
    tu = sub.add_parser("tune")
    tu.add_argument("--model", required=True)
    tu.add_argument("--labels", nargs="+", required=True)
    tu.add_argument("--worlds", type=int, default=4)
    tu.add_argument("--epochs", type=int, default=8)
    tu.add_argument("--pair-weight", type=float, default=10.0)
    tu.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "tune":
        return tune(a.model, a.labels, a.worlds, a.epochs, a.pair_weight, a.out)
    if a.cmd == "label":
        return label(a.model, a.games, a.rate, a.k, a.pairs, a.part, a.seed, a.temperature, a.out)
    return fit(a.model, a.labels, a.bank, a.bank_playouts, a.worlds, a.test_worlds, a.epochs, a.out,
               test_limit=a.test_limit, perspective=a.perspective, holdout=a.holdout,
               pair_weight=a.pair_weight)


if __name__ == "__main__":
    sys.exit(main())
