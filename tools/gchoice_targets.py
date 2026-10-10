#!/usr/bin/env python3
"""Arms and checks for one offline round of Gumbel-choice expert iteration.

`tools/generate_search_targets.py --target gchoice` plays the raw network against itself and,
at a sampled share of its decisions, records a noise-free Gumbel root's choice at each budget
beside the network's argmax and distribution (`gchoice`). This tool turns one such set of games
into the arms of a fine-tune, which all train on the same positions, and checks a fine-tuned
checkpoint on held-out games: did it move where the search departs, and nowhere else?

**Arms** (`arm --form`), each rewriting every searched record's `search_pi`:

* `departures` -- a one-hot on the root's choice at `--budget` where it departs from the
  network's argmax, the network's own distribution where they agree. The bank found the better
  move inside the network's top 8 almost always and search evaluation-limited
  (research/log/E7_search_disagreement_bank.md), so the search's own pick is the target.
* `gated` -- as `departures`, but only where the root's own value margin, Q(choice) - Q(argmax)
  in win probability, is at least `--min-margin`; the network's own distribution elsewhere.
* `consensus` -- as `departures`, but only where the root at `--agree-with`, another recorded
  budget searched on its own random streams, departs to the same move: a departure that survives
  two independent searches rather than one root's resolution of a near tie.
* `soft` -- one KL-regularised step from the network's policy toward the teacher's values, where it
  recorded values for its candidates: pi'(a) proportional to pi(a) exp((Q(a) - Q(argmax)) / tau),
  Q the mover's value in [-1, 1], the other moves keeping pi(a) -- a nudge in proportion to the
  evidence rather than a flip to the teacher's pick (`--tau`).
* `own` -- the network's own distribution at every searched position: no signal, the same
  optimiser steps. Its distance from the base is the fine-tune's drift, the floor every other
  arm is read against.

At its own distribution the cross-entropy's gradient is zero, so the agreement positions hold
the policy where it is rather than sharpening it.

    tools/gchoice_targets.py arm --input part-*.jsonl.gz --form departures --budget 256 \\
        --out departures.jsonl.gz
    tools/train.py --mode distill --warmup-checkpoint base.pt --distill-dataset departures.jsonl.gz ...
    tools/gchoice_targets.py check --input heldout.jsonl.gz --base base.pt --model distilled.pt \\
        --budget 256 --out check.json

`check` replays the held-out games itself (the loader drops a game's later records at a desync,
which would misalign them) and reports, at departures, how often the checkpoint's argmax is now
the search's choice and how its log-probability moved, split by the root's margin; at agreements,
how often its argmax changed and the KL from the base; and the mean entropy, both checkpoints.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

FORMS = ("departures", "gated", "consensus", "soft", "own")

#: Margin bins (win probability) the check splits departures by.
MARGIN_BINS: Tuple[Tuple[float, float], ...] = ((-1.0, 0.0), (0.0, 0.02), (0.02, 0.04),
                                                (0.04, 0.08), (0.08, 1.0))


def margin(rec: Dict[str, Any], budget: str) -> Optional[float]:
    """Q(choice) - Q(argmax) at `budget`, in win probability (the root's Q is the mover's value
    in [-1, 1]); None where either was never evaluated."""
    g = rec["g"][budget]
    qc, qr = g["q"].get(str(g["c"])), g["q"].get(str(rec["raw"]))
    if qc is None or qr is None:
        return None
    return (float(qc) - float(qr)) / 2.0


def soft_target(rec: Dict[str, Any], budget: str, tau: float) -> Dict[str, List[float]]:
    """pi'(a) proportional to pi(a) exp((Q(a) - Q(argmax)) / tau) over the moves with a value, pi(a)
    elsewhere (`soft`)."""
    own = rec["pi"]
    q = {int(a): float(v) for a, v in rec["g"][budget]["q"].items() if v is not None}
    raw = int(rec["raw"])
    if raw not in q or len(q) < 2:
        return {"a": list(own["a"]), "v": list(own["v"])}
    p = {int(a): float(v) for a, v in zip(own["a"], own["v"])}
    for a in q:
        p.setdefault(a, 1e-4)
    w = {a: pv * (math.exp((q[a] - q[raw]) / tau) if a in q else 1.0) for a, pv in p.items()}
    tot = sum(w.values())
    acts = sorted(w)
    out: Dict[str, List[float]] = {"a": [a for a in acts], "v": [round(w[a] / tot, 6) for a in acts]}
    return out


def target(rec: Dict[str, Any], form: str, budget: str, min_margin: float,
           agree_with: Optional[str] = None, tau: float = 0.1) -> Dict[str, List[float]]:
    """One searched record's `search_pi` under `form`."""
    if form not in FORMS:
        raise ValueError(f"unknown form {form!r}")
    if form == "soft":
        return soft_target(rec, budget, tau)
    own = rec["pi"]
    c = int(rec["g"][budget]["c"])
    if form == "own" or c == int(rec["raw"]):
        return {"a": list(own["a"]), "v": list(own["v"])}
    if form == "gated":
        m = margin(rec, budget)
        if m is None or m < min_margin:
            return {"a": list(own["a"]), "v": list(own["v"])}
    if form == "consensus":
        if agree_with is None:
            raise ValueError("form 'consensus' needs the budget to agree with")
        if int(rec["g"][agree_with]["c"]) != c:
            return {"a": list(own["a"]), "v": list(own["v"])}
    return {"a": [c], "v": [1.0]}


def _read_games(paths: Sequence[str]) -> Iterator[Dict[str, Any]]:
    for p in paths:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def _source_meta(paths: Sequence[str]) -> List[Dict[str, Any]]:
    out = []
    for p in paths:
        if os.path.exists(p + ".meta.json"):
            with open(p + ".meta.json", encoding="utf-8") as f:
                out.append(json.load(f))
    return out


def arm(inputs: Sequence[str], form: str, budget: str, min_margin: float, out: str,
        agree_with: Optional[str] = None, tau: float = 0.1) -> Dict[str, Any]:
    """Rewrite every searched record of `inputs` under `form` into one dataset at `out`, with a
    sidecar naming the arm and its sources (the trainer prints the searcher from it)."""
    b = str(budget)
    counts = {"games": 0, "searched": 0, "departures": 0, "targeted": 0}
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for game in _read_games(inputs):
            counts["games"] += 1
            for a in game["actions"]:
                rec = a.pop("gchoice", None)
                if rec is None:
                    a.pop("search_pi", None)
                    continue
                counts["searched"] += 1
                counts["departures"] += int(int(rec["g"][b]["c"]) != int(rec["raw"]))
                a["search_pi"] = target(rec, form, b, min_margin,
                                        None if agree_with is None else str(agree_with), tau)
                counts["targeted"] += int(len(a["search_pi"]["a"]) == 1
                                          and a["search_pi"]["a"][0] != int(rec["raw"]))
            f.write(json.dumps(game) + "\n")
    srcs = _source_meta(inputs)
    meta = {"purpose": "gchoice expert-iteration arm (tools/gchoice_targets.py)",
            "form": form, "budget": budget, "min_margin": min_margin if form == "gated" else None,
            "agree_with": agree_with if form == "consensus" else None,
            "tau": tau if form == "soft" else None,
            "searcher": srcs[0].get("searcher") if srcs else None,
            "checkpoint_sha256": srcs[0].get("checkpoint_sha256") if srcs else None,
            "commit": srcs[0].get("commit") if srcs else None,
            "git_dirty": any(bool(s.get("git_dirty")) for s in srcs),
            "sources": [os.path.basename(p) for p in inputs], **counts}
    with open(out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return meta


def replay(paths: Sequence[str]) -> Iterator[Tuple[np.ndarray, np.ndarray, Dict[str, Any], int]]:
    """(observation, mask, gchoice record, decision type) at every searched record of the games,
    replayed through the engine as `WarmupDataset.stream_policy_transitions` does."""
    import ts_engine as ts
    for game in _read_games(paths):
        st = ts.GameState()
        ts.Engine.init_game(st, game["seed"])
        for a in game["actions"]:
            p_ = (st.ctx().decision_player
                  if st.ctx().decision_player != ts.Player.NONE else st.phasing_player)
            mask = np.array(ts.get_flat_action_mask(st, False), copy=True)
            fa = int(a["flat_action"])
            if mask.sum() == 0 or fa < 0 or fa >= mask.shape[0] or mask[fa] == 0:
                raise RuntimeError(f"game {game.get('game_id')} desynchronised on replay: "
                                   "the engine is not the one that generated it")
            rec = a.get("gchoice")
            if rec is not None:
                yield (np.array(ts.extract_observation(st, p_), copy=True), mask, rec,
                       int(st.ctx().decision_type))
            ts.Engine.step(st, ts.decode_flat_action(st, fa))
            while (not ts.Engine.is_terminal(st) and st.ctx().decision_player == ts.Player.NONE
                   and st.ctx().decision_type == ts.DecisionType.ROLL_DIE):
                ts.Engine.step(st, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))


def _policies(model: Any, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    import torch
    with torch.no_grad():
        lg, _, _ = model(torch.from_numpy(obs).float(), torch.from_numpy(masks))
    lg = lg.float().numpy()
    z = np.where(masks.astype(bool), lg, -np.inf)
    p = np.exp(z - z.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def _entropy(p: np.ndarray) -> np.ndarray:
    return -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)).sum(axis=1)


def check(inputs: Sequence[str], base: str, model: str, budget: str,
          batch: int = 512) -> Dict[str, Any]:
    """Where did `model` move against `base` on the held-out searched positions?"""
    from tools.lib.player_agent import NeuralAgent
    nb = NeuralAgent.from_checkpoint(base, device="cpu").model.eval()
    nm = NeuralAgent.from_checkpoint(model, device="cpu").model.eval()
    b = str(budget)
    dep: List[Dict[str, Any]] = []
    agr: List[Dict[str, float]] = []
    ent_b: List[float] = []
    ent_m: List[float] = []
    same: List[float] = []
    rows: List[Tuple[np.ndarray, np.ndarray, Dict[str, Any], int]] = []

    def flush() -> None:
        if not rows:
            return
        obs = np.stack([r[0] for r in rows])
        masks = np.stack([r[1] for r in rows])
        pb, pm = _policies(nb, obs, masks), _policies(nm, obs, masks)
        ent_b.extend(_entropy(pb).tolist())
        ent_m.extend(_entropy(pm).tolist())
        for i, (_o, _mk, rec, dt) in enumerate(rows):
            # The generator's argmax: the base's own here, unless a near tie split differently.
            raw, c = int(rec["raw"]), int(rec["g"][b]["c"])
            same.append(float(int(np.argmax(pb[i])) == raw))
            if c != raw:
                dep.append({"moved": float(int(np.argmax(pm[i])) == c),
                            "left_raw": float(int(np.argmax(pm[i])) != raw),
                            "dlogp": float(math.log(max(pm[i, c], 1e-12)) - math.log(max(pb[i, c], 1e-12))),
                            "p_base": float(pb[i, c]), "p_new": float(pm[i, c]),
                            "margin": margin(rec, b), "dt": dt})
            else:
                kl = float(np.sum(np.where(pb[i] > 0, pb[i] * (np.log(np.where(pb[i] > 0, pb[i], 1.0))
                                                               - np.log(np.maximum(pm[i], 1e-12))), 0.0)))
                agr.append({"changed": float(int(np.argmax(pm[i])) != raw), "kl": kl})
        rows.clear()

    for r in replay(inputs):
        rows.append(r)
        if len(rows) >= batch:
            flush()
    flush()

    def mean(xs: Sequence[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    by_margin = []
    for lo, hi in MARGIN_BINS:
        sel = [d for d in dep if d["margin"] is not None and lo <= d["margin"] < hi]
        by_margin.append({"bin": [lo, hi], "n": len(sel), "moved": mean([d["moved"] for d in sel]),
                          "dlogp": mean([d["dlogp"] for d in sel])})
    return {"base": os.path.basename(base), "model": os.path.basename(model), "budget": budget,
            "positions": len(dep) + len(agr),
            "departures": {"n": len(dep), "moved": mean([d["moved"] for d in dep]),
                           "left_raw": mean([d["left_raw"] for d in dep]),
                           "dlogp": mean([d["dlogp"] for d in dep]),
                           "p_base": mean([d["p_base"] for d in dep]),
                           "p_new": mean([d["p_new"] for d in dep]),
                           "by_margin": by_margin},
            "agreements": {"n": len(agr), "changed": mean([a["changed"] for a in agr]),
                           "kl": mean([a["kl"] for a in agr])},
            "entropy": {"base": mean(ent_b), "model": mean(ent_m)},
            "base_argmax_is_recorded": mean(same)}


def check_markdown(res: Dict[str, Any]) -> str:
    d, a = res["departures"], res["agreements"]
    lines = [f"### `{res['model']}` against `{res['base']}` (Gumbel @{res['budget']}, "
             f"{res['positions']} held-out searched positions)", "",
             "| | n | argmax now the search's choice | argmax changed | mean Δ log p(choice) | p(choice) base → new | KL(base ‖ new) |",
             "|:---|---:|---:|---:|---:|:---|---:|",
             f"| departures | {d['n']} | {d['moved']:.1%} | {d['left_raw']:.1%} | {d['dlogp']:+.3f} | "
             f"{d['p_base']:.3f} → {d['p_new']:.3f} | |",
             f"| agreements | {a['n']} | | {a['changed']:.2%} | | | {a['kl']:.4f} |", "",
             "| departures by root margin (win prob) | n | argmax now the choice | mean Δ log p |",
             "|:---|---:|---:|---:|"]
    for b in d["by_margin"]:
        lines.append(f"| [{b['bin'][0]:g}, {b['bin'][1]:g}) | {b['n']} | {b['moved']:.1%} | {b['dlogp']:+.3f} |")
    lines += ["", f"Mean policy entropy: base {res['entropy']['base']:.3f}, model {res['entropy']['model']:.3f}.", ""]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("arm", help="rewrite the searched records' targets into one arm's dataset")
    a.add_argument("--input", nargs="+", required=True)
    a.add_argument("--form", choices=FORMS, required=True)
    a.add_argument("--budget", default="256", help="the recorded budget to read: a Gumbel budget or 'rollout'")
    a.add_argument("--min-margin", type=float, default=0.04,
                   help="gated: the root's Q(choice) - Q(argmax), in win probability")
    a.add_argument("--tau", type=float, default=0.1, help="soft: the step's temperature, in [-1, 1] value units")
    a.add_argument("--agree-with", default=None,
                   help="consensus: the other recorded budget whose choice must be the same")
    a.add_argument("--out", required=True)
    c = sub.add_parser("check", help="where a fine-tuned checkpoint moved, on held-out games")
    c.add_argument("--input", nargs="+", required=True)
    c.add_argument("--base", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--budget", default="256")
    c.add_argument("--out", required=True, help="JSON; a Markdown table goes beside it (.md)")
    args = ap.parse_args(argv)
    if args.cmd == "arm":
        print(json.dumps(arm(args.input, args.form, args.budget, args.min_margin, args.out,
                             args.agree_with, args.tau), indent=1))
        return 0
    res = check(args.input, args.base, args.model, args.budget)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    md = check_markdown(res)
    with open(os.path.splitext(args.out)[0] + ".md", "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
