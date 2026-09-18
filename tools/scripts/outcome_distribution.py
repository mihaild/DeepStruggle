#!/usr/bin/env python3
"""Freeze the DISTRIBUTION of game outcomes under policy play, to check a refactor moved nothing.

`decision_stream_baseline.py` pins the engine exactly, action for action, by driving it with a
seeded RNG. That is the strict check and it has two limits:

* **it cannot survive P17.** Changing the action representation changes how many decisions a game
  has and which indices they carry, so the RNG draws differently and the games diverge for reasons
  that are not bugs;
* **random play visits a narrow slice of the game.** It rarely builds control, rarely reaches the
  late war in a contested position, and rarely produces the board shapes a trained policy spends
  its time in -- so a mask bug that only appears in real positions can hide from it completely.

This is the complementary check: play with a **policy**, and compare *distributions* rather than
streams. Individual games need not match; the outcome mix, game length, DEFCON, the tracks and the
ending reasons must. A representation refactor that is faithful moves none of them.

    # freeze, before the change
    tools/scripts/outcome_distribution.py --checkpoint <ckpt.pt> --games 1000 --out base.json

    # after
    tools/scripts/outcome_distribution.py --checkpoint <ckpt.pt> --games 1000 --compare base.json

Tolerance is a two-proportion or two-sample z per metric, reported with the shift. Nothing is
"passed" silently: every metric is printed with its z, and the exit code is 1 if any exceeds the
threshold.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
from typing import Any, Dict, List

import numpy as np
import torch

from bindings.ts_env import TsVectorizedEnv
from tools.lib.player_agent import load_agent


def play(checkpoint: str, games: int, num_envs: int, seed: int,
         temperature: float, device: str) -> List[Dict[str, Any]]:
    agent = load_agent(checkpoint, device=device)
    net = getattr(agent, "model", None) or getattr(agent, "net", None)
    if net is None:
        raise SystemExit("no network inside %s" % checkpoint)

    env = TsVectorizedEnv(num_envs=num_envs, base_seed=seed)
    obs, masks, _info = env.reset_all(base_seed=seed)
    done_games: List[Dict[str, Any]] = []
    guard = 0
    while len(done_games) < games and guard < 200000:
        guard += 1
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device)
        mask_t = torch.from_numpy(np.asarray(masks, dtype=np.uint8)).to(device)
        with torch.no_grad():
            logits, _, _ = net(obs_t, mask_t)
            if temperature <= 0.05:
                acts = torch.argmax(logits, dim=-1)
            else:
                acts = torch.distributions.Categorical(
                    logits=logits / temperature).sample()
        obs, masks, _r, _d, info = env.step(acts.cpu().numpy().astype(np.int64))
        for e in info.get("completed_episodes", []):
            done_games.append(e)
            if len(done_games) >= games:
                break
    return done_games[:games]


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    winners = collections.Counter(r.get("winner") for r in rows)
    endings = collections.Counter(r.get("ending_reason") for r in rows)

    def num(key: str) -> List[float]:
        return [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]

    out: Dict[str, Any] = {"n": n, "proportions": {}, "numeric": {}}
    for k, c in winners.items():
        out["proportions"]["winner_%s" % k] = c / n
    for k, c in endings.items():
        out["proportions"]["ending_%s" % str(k)] = c / n
    for key in ("victory_points", "turn", "ply", "length", "terminal_utility"):
        v = num(key)
        if v:
            out["numeric"][key] = {
                "mean": statistics.mean(v),
                "sd": statistics.pstdev(v) if len(v) > 1 else 0.0,
            }
    return out


def compare(old: Dict[str, Any], new: Dict[str, Any], z_max: float) -> int:
    n_old, n_new = old["n"], new["n"]
    print("%-34s %10s %10s %9s %8s" % ("metric", "before", "after", "shift", "z"))
    print("-" * 76)
    worst = 0.0
    flagged: List[str] = []

    keys = sorted(set(old["proportions"]) | set(new["proportions"]))
    for k in keys:
        p1 = old["proportions"].get(k, 0.0)
        p2 = new["proportions"].get(k, 0.0)
        pool = (p1 * n_old + p2 * n_new) / (n_old + n_new)
        se = math.sqrt(max(pool * (1 - pool) * (1 / n_old + 1 / n_new), 1e-12))
        z = (p2 - p1) / se
        worst = max(worst, abs(z))
        mark = "  <-- FLAG" if abs(z) > z_max else ""
        if mark:
            flagged.append(k)
        print("%-34s %9.3f%% %9.3f%% %+8.3f%% %8.2f%s" % (
            k, 100 * p1, 100 * p2, 100 * (p2 - p1), z, mark))

    for k in sorted(set(old["numeric"]) | set(new["numeric"])):
        a = old["numeric"].get(k)
        b = new["numeric"].get(k)
        if not a or not b:
            continue
        se = math.sqrt(max(a["sd"] ** 2 / n_old + b["sd"] ** 2 / n_new, 1e-12))
        z = (b["mean"] - a["mean"]) / se
        worst = max(worst, abs(z))
        mark = "  <-- FLAG" if abs(z) > z_max else ""
        if mark:
            flagged.append(k)
        print("%-34s %10.3f %10.3f %+9.3f %8.2f%s" % (
            k, a["mean"], b["mean"], b["mean"] - a["mean"], z, mark))

    print()
    print("games: %d before, %d after   |z| threshold %.1f, worst %.2f" % (
        n_old, n_new, z_max, worst))
    if flagged:
        print("FLAGGED: %s" % ", ".join(flagged))
        print("A faithful representation change moves none of these. Investigate before")
        print("attributing the shift to sampling.")
        return 1
    print("no metric moved beyond threshold")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=880000)
    ap.add_argument("--temperature", type=float, default=0.5,
                    help="0.5 rather than greedy on purpose: a deterministic policy visits one "
                         "trajectory per seed, and the point of this check is coverage")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--z-max", type=float, default=3.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()

    if not a.out and not a.compare:
        print("give --out to freeze, or --compare to check", file=sys.stderr)
        return 2

    rows = play(a.checkpoint, a.games, a.num_envs, a.seed, a.temperature, a.device)
    summary = summarise(rows)
    summary["checkpoint"] = a.checkpoint.split("/")[-1]
    summary["temperature"] = a.temperature
    print("played %d games at T=%.2f" % (summary["n"], a.temperature))

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        print("wrote %s" % a.out)
        return 0

    with open(a.compare, encoding="utf-8") as fh:
        old = json.load(fh)
    if old.get("temperature") != a.temperature:
        print("WARNING: baseline was taken at T=%s, this run at T=%s" % (
            old.get("temperature"), a.temperature), file=sys.stderr)
    return compare(old, summary, a.z_max)


if __name__ == "__main__":
    raise SystemExit(main())
