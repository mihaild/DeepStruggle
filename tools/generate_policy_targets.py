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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--us", required=True, help="checkpoint that plays, and teaches, the US seat")
    ap.add_argument("--ussr", required=True, help="checkpoint that plays, and teaches, USSR")
    ap.add_argument("--total-games", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.25,
                    help="sampling temperature for the ACTING teachers; some spread is wanted so "
                         "the targets cover more than one line of play")
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    teachers = {
        ts.Player.US: NeuralAgent.from_checkpoint(a.us, device=dev).model,
        ts.Player.USSR: NeuralAgent.from_checkpoint(a.ussr, device=dev).model,
    }
    for m in teachers.values():
        m.eval()

    os.makedirs(os.path.dirname(os.path.abspath(a.output_path)), exist_ok=True)
    games = 0
    targets = 0
    outcomes: Dict[str, int] = {"US": 0, "USSR": 0, "DRAW": 0}

    with gzip.open(a.output_path, "wt", encoding="utf-8") as out:
        for g in range(a.total_games):
            seed = 500_000 + g
            st = ts.GameState()
            ts.Engine.init_game(st, seed)
            actions: List[Dict[str, Any]] = []
            guard = 0
            while not ts.Engine.is_terminal(st) and guard < 4000:
                guard += 1
                ctx = st.ctx()
                pl = ctx.decision_player
                if pl == ts.Player.NONE:
                    if not ts.Engine.try_step(st, ts.MicroAction(ctx.decision_type, 0, 0, 0)):
                        break
                    continue

                obs = np.asarray(ts.extract_observation(st, pl), dtype=np.float32)
                mask = np.asarray(ts.get_flat_action_mask(st), dtype=np.uint8)
                with torch.no_grad():
                    logits, _, _ = teachers[pl](
                        torch.from_numpy(obs).unsqueeze(0).to(dev),
                        torch.from_numpy(mask).unsqueeze(0).to(dev))
                lg = logits[0].float().cpu().numpy()
                lg = np.where(mask > 0, lg, -np.inf)
                lg = lg - np.max(lg)
                p = np.exp(lg)
                p = p / p.sum()

                legal = np.flatnonzero(mask)
                rec: Dict[str, Any] = {}
                if legal.size > 1:
                    order = legal[np.argsort(p[legal])[::-1]][:TOP_K]
                    rec["search_pi"] = {"a": [int(i) for i in order],
                                        "v": [float(p[int(i)]) for i in order]}
                    targets += 1

                # Act by sampling the teacher, so the states seen are the ones the pair reaches.
                t = max(1e-3, a.temperature)
                q = np.power(p, 1.0 / t)
                q = q / q.sum()
                chosen = int(np.random.choice(len(q), p=q))
                rec["flat_action"] = chosen
                actions.append(rec)
                if not ts.Engine.try_step(st, ts.decode_flat_action(st, chosen)):
                    break

            vp = int(st.victory_points)
            winner = "US" if vp > 0 else ("USSR" if vp < 0 else "DRAW")
            outcomes[winner] += 1
            out.write(json.dumps({"seed": seed, "actions": actions,
                                  "winner": winner, "final_vp": vp}) + "\n")
            games += 1
            if games % 25 == 0:
                print(f"  {games}/{a.total_games} games | {targets:,} targets", flush=True)

    meta = {
        "purpose": "two-seat capacity probe: can one network hold both teachers?",
        "us_teacher": a.us,
        "ussr_teacher": a.ussr,
        "acting_temperature": a.temperature,
        "top_k": TOP_K,
        "games": games,
        "targets": targets,
        "outcomes": outcomes,
        "commit": _commit(),
        "engine_obs_size": int(ts.OBS_SIZE),
    }
    with open(a.output_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"\n{games} games, {targets:,} targets -> {a.output_path}")
    print(f"outcomes: {outcomes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
