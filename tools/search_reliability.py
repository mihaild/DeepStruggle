#!/usr/bin/env python3
"""Is a Gumbel root's departure from the network real? Searchers compared on the search bank by
reproducibility and by paired playouts, at a stated budget.

The search resolves a move's dice once, when it expands it, and searches one draw of the hidden
cards per halving phase, so a bigger budget deepens the tree under one roll without averaging over
rolls: two k=16 @1,024 roots reproduce each other's departures only ~47% of the time on the bank.
`gumbel_worlds` (`BatchedMCTSConfig`) spreads the same evaluations over more independent draws of
the cards and the dice. This measures whether that makes the departures reproducible and better.

    search    every bank position searched by each --spec, --seeds times on independent streams;
              each search's pick and the network evaluations it spent.
    playouts  each position's raw move and every pick, played out in paired continuations by the
              raw network (ai/eval/paired_playouts.compare: one redeal and one set of dice per pair,
              shared by every move), so all searchers are judged on the same games.
    report    per searcher: departure rate; reproducibility (another seed picks the same departure);
              the playout gain of its pick over raw, per departure and per game, population-weighted
              by the bank's inclusion weights; precision (departures that beat raw); time; and each
              searcher against the first, paired by position.

    PYTHONPATH=.:build/release .venv/bin/python tools/search_reliability.py search \\
        --bank bank.jsonl.gz --model model.pt --spec w1=256:8:0.2:all:1 --spec w16=256:8:0.2:all:16 \\
        --seeds 4 --part 1/20 --out search-1.jsonl.gz
    ... playouts --bank bank.jsonl.gz --search search-*.jsonl.gz --model model.pt --pairs 512 \\
        --part 1/20 --out playouts-1.jsonl.gz
    ... report --bank bank.jsonl.gz --search search-*.jsonl.gz --playouts playouts-*.jsonl.gz \\
        --out report.md

A --spec is NAME=SIMS:K:FPU:FILTER:WORLDS, the fields of a `gumbel:` agent spec after its checkpoint.
The bank is the search-disagreement bank's playout input (fork release `search-bank-playouts-20261007`,
2,300 raw-play positions of `E7line_swa_4720-4800M`, each with its population weight `weight` and its
inclusion probability `pl_incl`); its model is the one to search with.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

#: The search bank's annotate stage sampled one decision in 8 of 1,500 games, so its population
#: weights sum to the decisions of 1,500 / 8 games: a weighted sum times this is per game.
PER_GAME = 8.0 / 1500.0


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


def _part(rows: List[Dict[str, Any]], part: str) -> List[Dict[str, Any]]:
    k, n = (int(x) for x in part.split("/"))
    return [r for i, r in enumerate(sorted(rows, key=lambda r: str(r["id"]))) if i % n == k - 1]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_spec(text: str) -> Tuple[str, str]:
    """NAME=SIMS:K:FPU:FILTER:WORLDS (a Gumbel root: the fields after a gumbel: spec's checkpoint)
    or NAME=rollout:K:WORLDS:HORIZON:RULE (ai/search/rollout_root.py) -> (name, fields)."""
    name, _, fields = text.partition("=")
    ok = (len(fields.split(":")) == 5 and not fields.startswith("rollout:")) or \
        (fields.startswith("rollout:") and len(fields.split(":")) == 5)
    if not name or not ok:
        raise argparse.ArgumentTypeError(
            f"--spec must be NAME=SIMS:K:FPU:FILTER:WORLDS or NAME=rollout:K:WORLDS:HORIZON:RULE, not {text!r}")
    return name, fields


def chunk_seed(spec: str, seed: int, ids: Sequence[str]) -> int:
    """One search call's streams: its searcher, its seed index and the positions it holds."""
    h = hashlib.sha256(f"{spec}|{seed}|{','.join(ids)}".encode()).digest()
    return int.from_bytes(h[:4], "little")


def search(bank: str, model: str, specs: Sequence[Tuple[str, str]], seeds: int, part: str,
           chunk: int, out: str) -> int:
    import torch
    from ai.search.batched_mcts import BatchedMCTS
    from tools.lib.corpus_driver import state_from_token
    from tools.lib.player_agent import NeuralAgent, search_spec_config
    from bindings.action_encoder import ActionEncoder

    rows = _part(list(_read([bank])), part)
    net = NeuralAgent.from_checkpoint(model, device="cpu").model.eval()
    torch.set_grad_enabled(False)
    from ai.search.rollout_root import RolloutRoot, rollout_spec_config
    searchers: Dict[str, Tuple[Any, str]] = {}
    for name, fields in specs:
        if fields.startswith("rollout:"):
            _path, rcfg, label = rollout_spec_config(f"rollout:{model}:{fields[len('rollout:'):]}")
            searchers[name] = (RolloutRoot(net, rcfg), label)
        else:
            _path, cfg, label = search_spec_config(f"gumbel:{model}:{fields}")
            searchers[name] = (BatchedMCTS(net, device="cpu", config=cfg), label)
    res = {str(r["id"]): {"id": r["id"], "raw": int(r["raw"]), "picks": {n: [] for n, _ in specs},
                          "evals": {n: [] for n, _ in specs}} for r in rows}
    secs = {n: 0.0 for n, _ in specs}
    t0 = time.time()
    for lo in range(0, len(rows), chunk):
        batch = rows[lo:lo + chunk]
        ids = [str(r["id"]) for r in batch]
        for name, _fields in specs:
            mcts, _label = searchers[name]
            for s in range(seeds):
                states = [state_from_token(r["pos"]) for r in batch]
                mcts.reseed(chunk_seed(name, s, ids))
                t = time.time()
                if isinstance(mcts, RolloutRoot):
                    picks = mcts.choose(states)
                    evals = [mcts.rows / len(batch)] * len(batch)     # rows, shared by the batch
                else:
                    picks = mcts.best_actions(states)
                    evals = [float(sum(gs["n"].values())) for gs in mcts.gumbel_stats]
                secs[name] += time.time() - t
                for r, a, e in zip(batch, picks, evals):
                    if not ActionEncoder.get_legal_mask(state_from_token(r["pos"]))[int(a)]:
                        raise RuntimeError(f"{name} picked an illegal move at {r['id']}")
                    res[str(r["id"])]["picks"][name].append(int(a))
                    res[str(r["id"])]["evals"][name].append(e)
        print(f"  {min(lo + chunk, len(rows))}/{len(rows)} positions | {time.time() - t0:.0f}s",
              file=sys.stderr, flush=True)
    _write(out, list(res.values()))
    meta = {"stage": "search", "bank": os.path.basename(bank), "model": os.path.basename(model),
            "model_sha256": _sha256(model), "specs": {n: f for n, f in specs},
            "labels": {n: searchers[n][1] for n, _ in specs}, "seeds": seeds, "part": part,
            "positions": len(rows), "search_seconds": secs, "wall_s": round(time.time() - t0, 1)}
    with open(out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return 0


def position_seed(rid: str, seed: int) -> int:
    """As tools/scripts/bank_playouts.position_seed: the row id mixed with the run's seed."""
    return (int(rid, 16) ^ (seed * 0x9E3779B97F4A7C15)) & 0x7FFF_FFFF_FFFF_FFFF


def playouts(bank: str, search_paths: Sequence[str], model: str, pairs: int, seed: int, part: str,
             out: str) -> int:
    from ai.eval.paired_playouts import compare, paired_diff
    from tools.lib.corpus_driver import load_policy, state_from_token

    picks = {str(r["id"]): r for r in _read(search_paths)}
    rows = [r for r in _part(list(_read([bank])), part) if str(r["id"]) in picks]
    logits_fn, _ = load_policy(model)

    def act(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return np.asarray(logits_fn(obs, masks)).argmax(axis=1).astype(np.int32)

    positions, ids = [], []
    for r in rows:
        moves = {int(r["raw"])}
        for ps in picks[str(r["id"])]["picks"].values():
            moves.update(int(a) for a in ps)
        positions.append((state_from_token(r["pos"]), sorted(moves)))
        ids.append(str(r["id"]))
    t0 = time.time()
    scores = compare(positions, act, pairs, [position_seed(i, seed) for i in ids])
    res = []
    for r, sc in zip(rows, scores):
        raw = int(r["raw"])
        res.append({"id": r["id"], "pairs": pairs,
                    "score": {str(a): round(sum(v) / len(v), 5) for a, v in sc.items()},
                    "diff_vs_raw": {str(a): [round(x, 5) for x in paired_diff(v, sc[raw])]
                                    for a, v in sc.items() if a != raw}})
    _write(out, res)
    with open(out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"stage": "playouts", "model": os.path.basename(model), "pairs": pairs,
                   "seed": seed, "part": part, "positions": len(res), "continuation": "raw greedy",
                   "wall_s": round(time.time() - t0, 1)}, f, indent=1)
    print(f"{len(res)} positions, {pairs} pairs, in {time.time() - t0:.0f}s", file=sys.stderr)
    return 0


def _wmean(xs: Sequence[float], ws: Sequence[float]) -> Tuple[float, float]:
    """Weighted mean and its standard error (independent units)."""
    sw = sum(ws)
    if sw <= 0:
        return float("nan"), float("nan")
    m = sum(x * w for x, w in zip(xs, ws)) / sw
    var = sum((w * (x - m)) ** 2 for x, w in zip(xs, ws)) / sw ** 2
    return m, math.sqrt(var)


def summarize(bank_rows: Dict[str, Dict[str, Any]], search_rows: Dict[str, Dict[str, Any]],
              play_rows: Dict[str, Dict[str, Any]], names: Sequence[str]) -> Dict[str, Any]:
    """Every figure the report prints, per searcher (see the module docstring)."""
    out: Dict[str, Any] = {}
    common = [i for i in search_rows if i in play_rows and i in bank_rows]
    w_of = {i: float(bank_rows[i]["weight"]) / float(bank_rows[i]["pl_incl"]) for i in common}
    gain_by: Dict[str, Dict[str, float]] = {}
    for n in names:
        dep_rate, rep_num, rep_den, same_num, same_den = [], 0.0, 0.0, 0.0, 0.0
        gains, gw, dep_gain, dep_w, prec_pos, prec_conf, prec_harm, per_pos = [], [], [], [], [], [], [], {}
        for i in common:
            s, p, w = search_rows[i], play_rows[i], w_of[i]
            raw, ps = int(s["raw"]), [int(a) for a in s["picks"][n]]
            sc = {int(a): v for a, v in p["score"].items()}
            dv = {int(a): v for a, v in p["diff_vs_raw"].items()}
            deps = [a for a in ps if a != raw]
            dep_rate.append(len(deps) / len(ps))
            for x in range(len(ps)):
                for y in range(len(ps)):
                    if x == y:
                        continue
                    same_den += w
                    same_num += w * (ps[x] == ps[y])
                    if ps[x] != raw:
                        rep_den += w
                        rep_num += w * (ps[y] == ps[x])
            g = sum((dv[a][0] if a in dv else sc[a] - sc[raw]) if a != raw else 0.0 for a in ps) / len(ps)
            gains.append(g)
            gw.append(w)
            per_pos[i] = g
            for a in deps:
                m, se = dv[a]
                dep_gain.append(m)
                dep_w.append(w / len(ps))
                prec_pos.append(float(m > 0))
                prec_conf.append(float(m > 2 * se))
                prec_harm.append(float(m < -2 * se))
        gm, gse = _wmean(gains, gw)
        out[n] = {"positions": len(common),
                  "departure_rate": _wmean(dep_rate, gw)[0],
                  "reproducibility": rep_num / rep_den if rep_den else float("nan"),
                  "same_move": same_num / same_den if same_den else float("nan"),
                  "gain_per_decision": gm, "gain_per_decision_se": gse,
                  "gain_per_game": gm * sum(gw) * PER_GAME, "gain_per_game_se": gse * sum(gw) * PER_GAME,
                  "gain_per_departure": _wmean(dep_gain, dep_w)[0],
                  "precision_positive": _wmean(prec_pos, dep_w)[0],
                  "confirmed_better": _wmean(prec_conf, dep_w)[0],
                  "confirmed_worse": _wmean(prec_harm, dep_w)[0]}
        gain_by[n] = per_pos
    base = names[0]
    for n in names[1:]:
        d = [gain_by[n][i] - gain_by[base][i] for i in common]
        m, se = _wmean(d, [w_of[i] for i in common])
        sw = sum(w_of[i] for i in common)
        out[n]["vs_first_per_game"] = m * sw * PER_GAME
        out[n]["vs_first_per_game_se"] = se * sw * PER_GAME
    return out


def report(bank: str, search_paths: Sequence[str], play_paths: Sequence[str], out: str) -> int:
    bank_rows = {str(r["id"]): r for r in _read([bank])}
    # Several runs merge by position: their searchers side by side, their playouts' moves pooled.
    # Valid because a position's pairs depend only on its id and the playout seed, so the raw move
    # scores the same in every run that shares the seed (checked below).
    search_rows: Dict[str, Dict[str, Any]] = {}
    for r in _read(search_paths):
        cur = search_rows.setdefault(str(r["id"]), {"id": r["id"], "raw": r["raw"], "picks": {}, "evals": {}})
        cur["picks"].update(r["picks"])
        cur["evals"].update(r["evals"])
    play_rows: Dict[str, Dict[str, Any]] = {}
    for r in _read(play_paths):
        cur = play_rows.get(str(r["id"]))
        if cur is None:
            play_rows[str(r["id"])] = {"id": r["id"], "pairs": r["pairs"], "score": dict(r["score"]),
                                       "diff_vs_raw": dict(r["diff_vs_raw"])}
            continue
        raw = str(search_rows[str(r["id"])]["raw"]) if str(r["id"]) in search_rows else None
        # The same pairs, but the network's greedy play can flip a near tie when its batch differs
        # (CPU outputs depend slightly on batch composition): measured, the raw move scores the same in
        # 99.3% of positions and at most 0.002 apart. Gains are each run's own paired differences.
        if raw is not None and abs(cur["score"][raw] - r["score"][raw]) > 0.01:
            raise ValueError(f"{r['id']}: the raw move scored differently across runs -- not the same games")
        cur["score"].update(r["score"])
        cur["diff_vs_raw"].update(r["diff_vs_raw"])
    metas = [json.load(open(p + ".meta.json")) for p in search_paths if os.path.exists(p + ".meta.json")]
    names: List[str] = []
    for m in metas:
        names += [n for n in m["specs"] if n not in names]
    if not names:
        names = sorted(next(iter(search_rows.values()))["picks"])
    res = summarize(bank_rows, search_rows, play_rows, names)
    secs = {n: sum(m["search_seconds"].get(n, 0.0) for m in metas) for n in names} if metas else {}
    nsearch = {n: sum(m["positions"] * m["seeds"] for m in metas if n in m["specs"]) for n in names} if metas else {}
    evals = {n: float(np.mean([e for r in search_rows.values() for e in r["evals"][n]])) for n in names}
    for n in names:
        res[n]["evals_per_search"] = evals[n]
        res[n]["seconds_per_search"] = secs[n] / nsearch[n] if nsearch.get(n) else float("nan")
    lines = ["# Gumbel roots on the search bank: reproducibility and playout gain", ""]
    if metas:
        m0 = metas[0]
        labels = {n: l for m in metas for n, l in m["labels"].items()}
        lines += [f"* model `{m0['model']}` (sha256 `{m0['model_sha256'][:12]}…`), {m0['seeds']} seeds per "
                  f"searcher, {len(search_rows)} positions; specs "
                  + ", ".join(f"`{n}` = {labels[n]}" for n in names), ""]
    pr = next(iter(play_rows.values()))
    lines += [f"Paired raw-network continuations, {pr['pairs']} pairs per move; every figure population-"
              "weighted by the bank's inclusion weights. Gain is the pick's playout score minus raw's, "
              "in win probability (per game: points of a game summed over its decisions).", "",
              "| searcher | evals / search | s / search | departs | reproduced by another seed | same move | gain per game (pts) | vs first (pts / game) | gain per departure (pts) | departures beating raw | confirmed better / worse |",
              "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|"]
    for n in names:
        r = res[n]
        vs = (f"{100 * r['vs_first_per_game']:+.1f} ± {100 * r['vs_first_per_game_se']:.1f}"
              if "vs_first_per_game" in r else "--")
        lines.append(f"| `{n}` | {r['evals_per_search']:.0f} | {r['seconds_per_search']:.3f} | "
                     f"{r['departure_rate']:.1%} | {r['reproducibility']:.1%} | {r['same_move']:.1%} | "
                     f"{100 * r['gain_per_game']:+.1f} ± {100 * r['gain_per_game_se']:.1f} | {vs} | "
                     f"{100 * r['gain_per_departure']:+.2f} | {r['precision_positive']:.1%} | "
                     f"{r['confirmed_better']:.1%} / {r['confirmed_worse']:.1%} |")
    lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("\n".join(lines))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--bank", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--spec", type=parse_spec, action="append", required=True)
    s.add_argument("--seeds", type=int, default=4)
    s.add_argument("--part", default="1/1")
    s.add_argument("--chunk", type=int, default=32)
    s.add_argument("--out", required=True)
    p = sub.add_parser("playouts")
    p.add_argument("--bank", required=True)
    p.add_argument("--search", nargs="+", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--pairs", type=int, default=512)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--part", default="1/1")
    p.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--bank", required=True)
    r.add_argument("--search", nargs="+", required=True)
    r.add_argument("--playouts", nargs="+", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "search":
        return search(a.bank, a.model, a.spec, a.seeds, a.part, a.chunk, a.out)
    if a.cmd == "playouts":
        return playouts(a.bank, a.search, a.model, a.pairs, a.seed, a.part, a.out)
    return report(a.bank, a.search, a.playouts, a.out)


if __name__ == "__main__":
    sys.exit(main())
