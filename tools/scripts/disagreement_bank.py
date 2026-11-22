#!/usr/bin/env python3
"""Positions where a human, the raw network and its search disagree, for review in the workbench.

Every human decision of the chosen kinds in the ts-replayer corpus, as the human faced it, is put
to two deciders:

* the **network** -- the policy's favourite move (greedy), with its probability of each of the
  others' moves and its v_win for the mover;
* the **searcher** -- a `load_agent` spec that decides positions in a batch, by default the Gumbel
  root (`gumbel:<pt>:32:4`: honest, determinized search from the mover's side, so it does not see the
  hidden cards the corpus's solved hands put in the state).

A position is kept when the three moves are not all the same, labelled by the pattern of agreement:

* `human-alone` -- network and search agree, the human differs;
* `network-alone` -- human and search agree, the network differs (search corrects the network);
* `search-alone` -- human and network agree, search differs;
* `all-differ`.

Kinds: `headline` (the headline card), `card` (the card that opens an action round), `mode` (event /
Ops / space for the card just chosen). These are the decisions whose order is not an artifact --
the Influence points of one play, where a different order is the same play, are left out.

Each row carries the position as the workbench's `pos=` token. Output is JSONL.gz, one row per kept
position, plus a summary of how often each kind agrees.

    PYTHONPATH=.:build/release python tools/scripts/disagreement_bank.py \\
        --net newest.onnx --search "gumbel:newest.pt:32:4" --part 1/1 --out bank.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import torch
import ts_engine as ts

from ai.eval.paired_playouts import decider
from bindings.action_encoder import ActionEncoder
from tools.lib.corpus_driver import (feed_corpus_game, load_policy, position_token, read_corpus_game,
                                     require_e4_view, rules_json)
from tools.lib.corpus_paths import distinct_corpus_files
from tools.lib.player_agent import load_agent
from tools.lib.position_bank import move_name
from tools.lib.engine_fingerprint import require_fresh

KINDS = ("headline", "card", "mode")
PATTERNS = ("human-alone", "network-alone", "search-alone", "all-differ")
_CARDS = {int(c["id"]): c for c in rules_json("cards.json")}


def kind_of(st: ts.GameState) -> Optional[str]:
    ctx = st.ctx()
    if int(ctx.resolving_card) != 0:
        return None
    if st.current_phase == ts.Phase.HEADLINE and ctx.decision_type == ts.DecisionType.SELECT_CARD:
        return "headline"
    if st.current_phase != ts.Phase.ACTION_ROUND:
        return None
    if (ctx.decision_type == ts.DecisionType.SELECT_CARD and ctx.decision_player == st.phasing_player
            and int(ctx.pending_op_card) == 0):
        return "card"
    if ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE:
        return "mode"
    return None


def pattern(h: int, n: int, s: int) -> Optional[str]:
    if h == n == s:
        return None
    if n == s:
        return "human-alone"
    if h == s:
        return "network-alone"
    if h == n:
        return "search-alone"
    return "all-differ"


class Collector:
    """The human's decisions of the chosen kinds, with two or more legal moves."""

    def __init__(self, kinds: Sequence[str]) -> None:
        self.kinds = set(kinds)
        self.seen: List[Tuple[ts.GameState, str, int]] = []

    def observe(self, st: ts.GameState, a: int) -> None:
        k = kind_of(st)
        if k is not None and k in self.kinds and int(np.asarray(ActionEncoder.get_legal_mask(st)).sum()) >= 2:
            self.seen.append((st, k, a))


def _softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    z = np.where(m, logits, -np.inf)
    z = z - z[m].max()
    p = np.exp(z)
    return p / p.sum()



def scan(paths: Sequence[str], net_spec: str, search_spec: str, kinds: Sequence[str],
         chunk: int, on_game: Optional[Any] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """Every game's kept rows and agreement counts. `on_game(replay_id, rows, counts)` is called as
    each game finishes, so a long run can be written out -- and resumed -- game by game. The
    searcher is reseeded with each game's replay id, so a game's rows do not depend on which part
    it fell in or whether the run was resumed before it."""
    net_fn, features = load_policy(net_spec)
    searcher: Any = load_agent(search_spec, device="cuda" if torch.cuda.is_available() else "cpu")
    require_e4_view(search_spec, searcher)
    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Dict[str, int]] = {k: {"decisions": 0, **{p: 0 for p in PATTERNS}} for k in kinds}
    for gi, path in enumerate(paths):
        replay_id = int(read_corpus_game(path).get("replay_id", 0))
        col = Collector(kinds)
        if feed_corpus_game(path, col) == "skipped" or not col.seen:
            if on_game is not None:
                on_game(replay_id, [], {})
            continue
        first = len(rows)
        counts: Dict[str, Dict[str, int]] = {k: {"decisions": 0, **{p: 0 for p in PATTERNS}} for k in kinds}
        states = [s for s, _, _ in col.seen]
        obs = np.stack([np.asarray(ts.extract_observation_features(s, decider(s), features), dtype=np.float32)
                        for s in states])
        masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8) for s in states])
        logits = net_fn(obs, masks)
        v_win = _v_win(net_spec, obs, masks)
        if hasattr(searcher, "reseed"):
            searcher.reseed(replay_id)
        picks: List[int] = []
        for i in range(0, len(states), chunk):
            # Clones (fix verification S-9.6, 2026-11-22): a searcher seats its decider on the states it is given, in
            # place, which moved position_token(st) and the rows' fields read from `st` below.
            picks += [int(a) for a in searcher.select_actions_batch([s.clone() for s in states[i:i + chunk]])]
        for i, ((st, kind, human), s_act) in enumerate(zip(col.seen, picks)):
            p = _softmax(logits[i], masks[i])
            n_act = int(np.argmax(np.where(masks[i].astype(bool), p, -1.0)))
            summary[kind]["decisions"] += 1
            counts[kind]["decisions"] += 1
            pat = pattern(human, n_act, s_act)
            if pat is None:
                continue
            summary[kind][pat] += 1
            counts[kind][pat] += 1
            ctx = st.ctx()
            card = int(ctx.pending_op_card) if kind == "mode" else 0
            rows.append({
                "game": replay_id, "kind": kind, "pattern": pat,
                "side": "US" if decider(st) == ts.Player.US else "USSR",
                "turn": int(st.turn), "ar": int(st.action_round), "defcon": int(st.defcon),
                "vp": int(st.victory_points), "card": str(_CARDS[card]["name"]) if card else "",
                "human": move_name(st, human), "network": move_name(st, n_act), "search": move_name(st, s_act),
                "p_human": round(float(p[human]), 4), "p_network": round(float(p[n_act]), 4),
                "p_search": round(float(p[s_act]), 4),
                "v_win": None if v_win is None else round(float(v_win[i]), 4),
                "pos": position_token(st),
            })
        if on_game is not None:
            on_game(replay_id, rows[first:], counts)
        print(f"[{gi + 1}/{len(paths)}] replay {replay_id}: {len(col.seen)} decisions, {len(rows)} kept so far",
              file=sys.stderr, flush=True)
    return rows, summary


def win_probability(v_win: np.ndarray) -> np.ndarray:
    """The value head regresses the mover's result on [-1, +1] (a loss -1, a win +1), so the win
    probability is (1 + v_win) / 2 -- the workbench's reading (web/ui/src/trace_view.ts) --
    clipped, since a regressed value can stray past the ends."""
    return np.clip((1.0 + np.asarray(v_win, dtype=np.float64)) / 2.0, 0.0, 1.0)


def _v_win(net_spec: str, obs: np.ndarray, masks: np.ndarray) -> Optional[np.ndarray]:
    """The network's win probability for the mover, when the net is an ONNX export (which names it)."""
    if not net_spec.endswith(".onnx"):
        return None
    sess = _SESSIONS.get(net_spec)
    if sess is None:
        sess = _SESSIONS[net_spec] = ort.InferenceSession(net_spec, providers=["CPUExecutionProvider"])
    out = sess.run(["v_win"], {"obs": obs, "mask": masks})[0]
    return win_probability(np.asarray(out).reshape(-1))


_SESSIONS: Dict[str, Any] = {}

#: the review page's columns, in order (web page reads `fields`, then each row as a list)
FIELDS = ("id", "game", "kind", "pattern", "side", "turn", "ar", "defcon", "vp", "card", "human", "network",
          "search", "p_human", "p_network", "p_search", "v_win", "pos",
          # bank_clarity.py: the stronger search's move and whose it is, each move's search value and
          # SE for the mover, the value-best move(s), the margin in SEs, and whether the two agree
          "strong", "strong_is", "q", "best", "z", "strong_agrees",
          # bank_playouts.py: each move's playout score, the paired differences, the playout-best move(s)
          "pl_pairs", "pl_score", "pl_diff", "pl_best")


def row_id(r: Dict[str, Any]) -> str:
    """A stable id: the position and the kind of decision asked there, so a rebuilt bank keeps
    the verdicts already given."""
    return hashlib.sha1((r["kind"] + ":" + r["pos"]).encode()).hexdigest()[:16]


PART_ROWS = 8000


def pack(banks: Sequence[str], out_dir: str, human_alone_max_p: float,
         clarity: Sequence[str] = (), playouts: Sequence[str] = (), exclude: Optional[str] = None) -> int:
    """Merge banks into the review page's data, deduplicated by position: one file per pattern
    (`bank-<pattern>.json`, so the page can load the large human-alone set only when asked) and a
    `manifest.json` of their sizes."""
    os.makedirs(out_dir, exist_ok=True)
    skip = set(json.load(open(exclude))) if exclude else set()
    seen: Dict[str, Dict[str, Any]] = {}
    total = 0
    for path in banks:
        with gzip.open(path, "rt") as f:
            for line in f:
                r = json.loads(line)
                total += 1
                if r["pattern"] == "human-alone" and r["p_human"] >= human_alone_max_p:
                    continue
                if row_id(r) in skip:
                    continue
                seen.setdefault(row_id(r), r)
    def read(paths: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for path in paths:
            with gzip.open(path, "rt") as f:
                for line in f:
                    d = json.loads(line)
                    out[d["id"]] = d
        return out

    cl, pl = read(clarity), read(playouts)
    for rid, r in seen.items():
        c, p = cl.get(rid, {}), pl.get(rid, {})
        r.update({"strong": c.get("strong"), "strong_is": c.get("strong_is"), "q": c.get("q"),
                  "best": c.get("best"), "z": c.get("z"), "strong_agrees": c.get("strong_agrees"),
                  "pl_pairs": p.get("pairs"), "pl_score": p.get("score"), "pl_diff": p.get("diff"),
                  "pl_best": p.get("best")})
    manifest: Dict[str, Any] = {"fields": list(FIELDS), "files": {},
                                "clarity": len(cl), "playouts": len(pl)}
    for pat in PATTERNS:
        rows = [[rid if f == "id" else r[f] for f in FIELDS] for rid, r in seen.items() if r["pattern"] == pat]
        # A published file holds at most 16 MB; a row is about 1.1 KB, so parts of PART_ROWS rows.
        names: List[str] = []
        size = 0
        for i in range(0, max(1, len(rows)), PART_ROWS):
            name = f"bank-{pat}-{i // PART_ROWS + 1}.json"
            json.dump({"fields": list(FIELDS), "rows": rows[i:i + PART_ROWS]},
                      open(os.path.join(out_dir, name), "w"), separators=(",", ":"))
            names.append(name)
            size += os.path.getsize(os.path.join(out_dir, name))
        manifest["files"][pat] = {"parts": names, "rows": len(rows), "bytes": size}
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    print(f"{len(seen)} of {total} rows packed into {out_dir}: "
          + ", ".join(f"{p} {v['rows']} ({v['bytes'] / 1e6:.1f} MB)" for p, v in manifest["files"].items()),
          file=sys.stderr)
    return 0

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", default=None, help="the raw network: .onnx or .pt")
    ap.add_argument("--search", default=None, help="a load_agent spec, e.g. gumbel:<ckpt.pt>:32:4")
    ap.add_argument("--pack", nargs="+", default=None, metavar="BANK",
                    help="instead of scanning: merge these banks into --out as the review page's data file")
    ap.add_argument("--clarity", nargs="*", default=[], help="--pack: bank_clarity.py outputs to merge in")
    ap.add_argument("--exclude", default=None, help="--pack: a JSON list of row ids to leave out")
    ap.add_argument("--playouts", nargs="*", default=[], help="--pack: bank_playouts.py outputs to merge in")
    ap.add_argument("--human-alone-max-p", type=float, default=1.0,
                    help="--pack keeps a human-alone row only when the network gave the human's move less")
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
    ap.add_argument("--part", default="1/1", help="k/N: this part's share of the corpus's distinct games")
    ap.add_argument("--games", type=int, default=0, help="at most this many games of the share (0 = all)")
    ap.add_argument("--chunk", type=int, default=128, help="positions per search call")
    ap.add_argument("--out", required=True, help="JSONL.gz of the kept positions, appended game by game "
                                                 "(with --pack: a directory for the review page's data files)")
    ap.add_argument("--summary", default=None, help="JSON of agreement counts per kind")
    ap.add_argument("--resume", action="store_true",
                    help="skip the games --out already holds (their counts are kept in <out>.games.jsonl)")
    a = ap.parse_args(argv)
    require_fresh()
    if a.pack:
        return pack(a.pack, a.out, a.human_alone_max_p, a.clarity, a.playouts, a.exclude)
    if not a.net or not a.search:
        ap.error("--net and --search are required unless --pack is given")
    k, n = (int(x) for x in a.part.split("/"))
    files, _ = distinct_corpus_files()
    paths = [str(p) for i, p in enumerate(sorted(files)) if i % n == k - 1][: a.games or None]
    ledger = a.out + ".games.jsonl"
    done: Dict[int, Dict[str, Dict[str, int]]] = {}
    if a.resume and os.path.exists(ledger):
        for line in open(ledger):
            g = json.loads(line)
            done[int(g["game"])] = g["counts"]
    elif os.path.exists(a.out) or os.path.exists(ledger):
        raise SystemExit(f"{a.out} exists: pass --resume to continue it, or remove it")

    def replay_id(path: str) -> int:
        return int(read_corpus_game(path).get("replay_id", 0))

    todo = [p for p in paths if replay_id(p) not in done]

    def on_game(game: int, rows: List[Dict[str, Any]], counts: Dict[str, Dict[str, int]]) -> None:
        # A game's rows, then its ledger line: a crash between the two re-runs the game, and the
        # duplicate rows it leaves are dropped by --pack, which keys rows by position.
        with gzip.open(a.out, "at") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(ledger, "a") as f:
            f.write(json.dumps({"game": game, "counts": counts}) + "\n")
        done[game] = counts

    t0 = time.time()
    scan(todo, a.net, a.search, a.kinds, a.chunk, on_game=on_game)
    summary: Dict[str, Dict[str, int]] = {}
    for counts in done.values():
        for kind, c in counts.items():
            for key, v in c.items():
                summary.setdefault(kind, {}).setdefault(key, 0)
                summary[kind][key] += v
    if a.summary:
        json.dump({"games": len(paths), "part": a.part, "net": a.net, "search": a.search, "kinds": summary},
                  open(a.summary, "w"), indent=1)
    print(f"{len(todo)} games scanned in {time.time() - t0:.0f}s ({len(paths) - len(todo)} already done): {summary}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
