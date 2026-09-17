#!/usr/bin/env python3
"""Two seat specialists play each other; record each one's policy as a distillation target.

The question this exists for: **can one network hold strong play on both seats at once, or does
the representation force a trade-off?** Every collapse measured in this project has been
seat-specific — a US seat that dissolves, a USSR seat that falls 15 points in 5M steps — and none
of them distinguishes "training dynamics went wrong" from "one network cannot represent both".

So take the best available US player and the best available USSR player, have them play each
other, and record the acting teacher's full distribution at each of its own decisions. Distilling
that into a single student asks the capacity question directly: if the student matches both
teachers per seat, the representation was never the constraint.

The state distribution is the one the *combined* policy would visit, because each teacher plays
its own seat throughout -- not the distribution either teacher generates in self-play, which is a
different and less relevant thing to imitate.

Output is the dataset format `--mode distill` already reads: one JSON line per game carrying
`{seed, actions, winner, final_vp}`, with each action record holding

    "search_pi": {"a": [flat action ids], "v": [weights]}

so the target is the teacher's probabilities rather than a searcher's visit counts. The name is
kept because the loader keys on it; the sidecar `.meta.json` records which teacher is which.

    tools/generate_policy_targets.py --us <ckpt.pt> --ussr <ckpt.pt> --total-games 300 \\
        --output-path /workspace/data/datasets/two_seat_targets.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
from typing import Any, Dict, List

import time

import numpy as np
import torch
import ts_engine as ts

from tools.lib.player_agent import NeuralAgent

TOP_K = 12


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()[:10]
    except Exception:
        return "unknown"


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--us", required=True, help="checkpoint that plays, and teaches, the US seat")
    ap.add_argument("--ussr", required=True, help="checkpoint that plays, and teaches, USSR")
    ap.add_argument("--total-games", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=128,
                    help="Games stepped together. One forward pass a seat a step over the whole "
                         "batch, rather than one a game: the first version of this tool stepped "
                         "a single game at a time and ran at 13 s/game.")
    ap.add_argument("--temperature", type=float, default=0.25,
                    help="sampling temperature for the ACTING teachers; some spread is wanted so "
                         "the targets cover more than one line of play")
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--device", default="cuda")
    return ap


def main() -> int:
    ap = _parser()
    a = ap.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    teachers = {
        1: NeuralAgent.from_checkpoint(a.us, device=dev).model,     # ts.Player.US
        -1: NeuralAgent.from_checkpoint(a.ussr, device=dev).model,  # ts.Player.USSR
    }
    for m in teachers.values():
        m.eval()

    os.makedirs(os.path.dirname(os.path.abspath(a.output_path)), exist_ok=True)
    games_done = 0
    targets = 0
    decisions = 0
    outcomes: Dict[str, int] = {"US": 0, "USSR": 0, "DRAW": 0}
    t0 = time.time()

    with gzip.open(a.output_path, "wt", encoding="utf-8") as out:
        while games_done < a.total_games:
            n = min(a.batch_size, a.total_games - games_done)
            base_seed = 500_000 + games_done * 10_007 + 1
            runner = ts.VectorizedBatchRunner(n, base_seed)
            hist: List[Dict[str, Any]] = [
                {"seed": base_seed + i * 10007 + 1, "actions": []} for i in range(n)]
            done = [False] * n

            for _ in range(4000):
                terminals = runner.get_terminals()
                utils = runner.get_terminal_utilities()
                for i in range(n):
                    if not done[i] and terminals[i]:
                        done[i] = True
                        st = runner.get_state(i)
                        w = "USSR" if utils[i] < 0 else ("US" if utils[i] > 0 else "DRAW")
                        outcomes[w] += 1
                        hist[i]["winner"] = w
                        hist[i]["final_vp"] = int(st.victory_points)
                if all(done):
                    break

                active = [i for i in range(n) if not done[i]]
                obs_all = np.array(runner.get_observations(), copy=False)
                masks_all = np.array(runner.get_action_masks(), copy=False)
                players = np.array(runner.get_decision_players(), dtype=np.int8)

                actions = [0] * n
                # One batched pass per seat, on the subset that seat is to move in -- the same
                # shape the opponent pool uses, and the reason this is not one call a game.
                for side in (1, -1):
                    idx = [i for i in active if int(players[i]) == side]
                    if not idx:
                        continue
                    obs_t = torch.from_numpy(obs_all[idx]).float().to(dev)
                    mask_t = torch.from_numpy(masks_all[idx]).to(dev)
                    with torch.no_grad():
                        logits, _, _ = teachers[side](obs_t, mask_t)
                    lg = logits.float()
                    lg = lg.masked_fill(mask_t <= 0, float("-inf"))
                    probs = torch.softmax(lg, dim=-1)

                    t = max(1e-3, a.temperature)
                    sharp = torch.softmax(lg / t, dim=-1)
                    picks = torch.multinomial(sharp, 1).squeeze(1).cpu().numpy()

                    topv, topi = torch.topk(probs, k=min(TOP_K, probs.shape[-1]), dim=-1)
                    topv_np = topv.cpu().numpy()
                    topi_np = topi.cpu().numpy()
                    nlegal = mask_t.sum(dim=-1).cpu().numpy()

                    for sub, i in enumerate(idx):
                        act = int(picks[sub])
                        actions[i] = act
                        rec: Dict[str, Any] = {"flat_action": act}
                        if nlegal[sub] > 1:
                            keep = [(int(c), float(v))
                                    for c, v in zip(topi_np[sub], topv_np[sub]) if v > 0.0]
                            if keep:
                                rec["search_pi"] = {"a": [c for c, _ in keep],
                                                    "v": [v for _, v in keep]}
                                targets += 1
                        hist[i]["actions"].append(rec)
                        decisions += 1

                runner.step_flat_all(actions)

            for i in range(n):
                hist[i].setdefault("winner", "DRAW")
                hist[i].setdefault("final_vp", 0)
                out.write(json.dumps(hist[i]) + "\n")
            games_done += n
            el = time.time() - t0
            print("  %d/%d games | %s targets | %.0fs | %.1f games/s"
                  % (games_done, a.total_games, f"{targets:,}", el, games_done / max(el, 1e-9)),
                  flush=True)

    meta = {
        "purpose": "two-seat capacity probe: can one network hold both teachers?",
        "us_teacher": a.us,
        "ussr_teacher": a.ussr,
        "acting_temperature": a.temperature,
        "top_k": TOP_K,
        "games": games_done,
        "targets": targets,
        "decisions": decisions,
        "outcomes": outcomes,
        "commit": _commit(),
        "engine_obs_size": int(ts.OBS_SIZE),
    }
    with open(a.output_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print("\n%d games, %s targets -> %s" % (games_done, f"{targets:,}", a.output_path))
    print("outcomes: %s" % outcomes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
