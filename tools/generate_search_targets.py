#!/usr/bin/env python3
"""Generate expert-iteration targets: the searcher's answer at states the raw policy visits.

P15-X4a step 2. The policy plays its own moves — search never acts — so the state distribution
is the policy's own. That is the whole point: imitating an *off-distribution* source is the
failure this project has measured twice (human injection, −157 to −236 Elo), and distilling
search-on-own-positions is the on-distribution case that evidence does not touch.

Output is the existing dataset format — one JSON line per game, `{seed, actions, winner,
final_vp}`, replayed through the engine by `WarmupDataset` — with one addition: an action record
at a searched decision carries

    "search_pi": {"a": [flat action ids], "v": [visit counts]}

Sparse, so a position costs tens of bytes rather than the 3,824 floats an observation would.
Everything needed to rebuild the observation is already implied by `seed` plus the actions.

The searcher's configuration and the commit that produced it go in a sidecar `<output>.meta.json`
rather than a header line, because every line of the dataset itself has to stay a game.

    tools/generate_search_targets.py --checkpoint <ckpt.pt> --total-games 200 \
        --sims 96 --output-path /workspace/data/datasets/x4a_targets.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, cast

import numpy as np
import torch
import ts_engine as ts

from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSConfig
from tools.lib.data_root import data_path
from tools.lib.player_agent import NeuralAgent


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True,
                                            stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        return True


def generate(checkpoint: str, total_games: int, batch_size: int, sims: int,
             node_filter: str, temperature: float, output_path: str, device_str: str,
             max_steps: int = 4000) -> int:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    agent = NeuralAgent.from_checkpoint(checkpoint, device=device)
    model = agent.model
    model.eval()

    cfg = BatchedMCTSConfig(simulations=sims, temperature=0.0, auto_advance=True,
                            advance_root=False, determinize=True, node_filter=node_filter,
                            subsample=1.0)
    searcher = BatchedMCTS(model, device=device, config=cfg)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    meta: Dict[str, Any] = {
        "purpose": "P15-X4a expert-iteration targets",
        "checkpoint": os.path.abspath(checkpoint),
        "searcher": {"simulations": sims, "determinize": True, "node_filter": node_filter,
                     "advance_root": False, "subsample": 1.0},
        "acting_policy": "raw (search never acts)",
        "temperature": temperature,
        "commit": _commit(),
        "git_dirty": _dirty(),
        "engine_obs_size": int(ts.OBS_SIZE),
    }

    t0 = time.time()
    games_done = 0
    searched_total = 0
    decisions_total = 0
    wins = {"US": 0, "USSR": 0, "DRAW": 0}

    with gzip.open(output_path, "wt", encoding="utf-8") as out:
        while games_done < total_games:
            n = min(batch_size, total_games - games_done)
            base_seed = 7_000_000 + games_done * 10_007 + 1
            runner = ts.VectorizedBatchRunner(n, base_seed)
            hist: List[Dict[str, Any]] = [
                {"game_id": f"x4a_{games_done + i:06d}", "seed": base_seed + i * 10007 + 1,
                 "actions": []} for i in range(n)]
            done = [False] * n

            for _ in range(max_steps):
                terminals = runner.get_terminals()
                utils = runner.get_terminal_utilities()
                for i in range(n):
                    if not done[i] and terminals[i]:
                        done[i] = True
                        st = runner.get_state(i)
                        w = "USSR" if utils[i] < 0 else ("US" if utils[i] > 0 else "DRAW")
                        wins[w] += 1
                        hist[i]["winner"] = w
                        hist[i]["final_vp"] = int(st.victory_points)
                        hist[i]["end_turn"] = int(st.turn)
                if all(done):
                    break

                active = [i for i in range(n) if not done[i]]
                obs_all = np.array(runner.get_observations(), copy=False)
                masks_all = np.array(runner.get_action_masks(), copy=False)

                # Which active positions this configuration searches. The state handle is cloned:
                # a handle held across a step is invalidated, which is a recorded measurement bug.
                to_search = [i for i in active
                             if searcher.should_search(runner.get_state(i))]
                targets: Dict[int, Dict[str, List[float]]] = {}
                if to_search:
                    states = [runner.get_state(i).clone() for i in to_search]
                    res = searcher.run(states)
                    for i, (acts, visits) in zip(to_search, res):
                        if not acts:
                            continue
                        targets[i] = {"a": [int(a) for a in acts],
                                      "v": [float(v) for v in np.asarray(visits)]}
                    searched_total += len(targets)

                # The RAW policy acts, always. Search is a teacher, not a player.
                obs_t = torch.from_numpy(obs_all[active]).float().to(device)
                mask_t = torch.from_numpy(masks_all[active]).to(device)
                with torch.no_grad():
                    act_t, _, _, _, _ = cast(Any, model).sample_action(
                        obs_t, mask_t, temperature=temperature,
                        deterministic=(temperature <= 0.05))
                chosen = [int(a) for a in act_t.cpu().numpy().tolist()]

                actions = [0] * n
                for sub, i in enumerate(active):
                    a = chosen[sub]
                    actions[i] = a
                    rec: Dict[str, Any] = {"flat_action": a}
                    if i in targets:
                        rec["search_pi"] = targets[i]
                    hist[i]["actions"].append(rec)
                    decisions_total += 1
                runner.step_flat_all(actions)

            for i in range(n):
                hist[i].setdefault("winner", "DRAW")
                hist[i].setdefault("final_vp", 0)
                hist[i]["total_steps"] = len(hist[i]["actions"])
                out.write(json.dumps(hist[i]) + "\n")
            games_done += n
            el = time.time() - t0
            print(f"  {games_done}/{total_games} games | {searched_total:,} searched positions "
                  f"| {el:.0f}s | {searched_total / max(el, 1e-9):.1f} targets/s", flush=True)

    meta.update({"games": games_done, "searched_positions": searched_total,
                 "total_decisions": decisions_total, "seconds": round(time.time() - t0, 1),
                 "outcomes": wins})
    with open(output_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"\nwrote {output_path}")
    print(f"  {games_done:,} games, {searched_total:,} searched positions of "
          f"{decisions_total:,} decisions ({searched_total / max(1, decisions_total):.1%})")
    print(f"  outcomes: {wins}")
    print(f"  metadata: {output_path}.meta.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--checkpoint", required=True, help="the policy that plays AND is searched")
    ap.add_argument("--total-games", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--sims", type=int, default=96,
                    help="simulations per searched decision; 96 is the measured saturation point")
    ap.add_argument("--node-filter", default="card_playmode", choices=["card_playmode", "all"])
    ap.add_argument("--temperature", type=float, default=0.2,
                    help="sampling temperature for the ACTING policy; some spread is wanted so "
                         "the targets cover more than one line of play")
    ap.add_argument("--output-path", default=None)
    ap.add_argument("--device", default="cuda")
    return ap


def main() -> int:
    a = build_parser().parse_args()
    out = a.output_path or data_path("datasets", "x4a_search_targets.jsonl.gz")
    return generate(a.checkpoint, a.total_games, a.batch_size, a.sims, a.node_filter,
                    a.temperature, out, a.device)


if __name__ == "__main__":
    sys.exit(main())
