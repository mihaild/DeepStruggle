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

**`--target gchoice`** answers with a noise-free Gumbel root (`ai/search/gumbel_root.py`, the
`gumbel:` tournament searcher) instead of PUCT visit counts, which the target-forms probe found
carry no measurable gain at 32-64 simulations (research/log/E7_search_target_forms.md). Where the
root's choice departs from the network's argmax, `search_pi` is a one-hot on it; where they agree
it is the network's own distribution, so those positions hold the policy where it is. Each such
record also carries `gchoice` -- the network's argmax and distribution, and per budget in
`--gumbel-sims` the root's choice with every candidate's evaluations and mean value -- so
`tools/gchoice_targets.py` can rewrite the target (another budget, a margin gate, the network's
own policy everywhere) without searching again. Decisions with one legal move are not searched.

**`--target rollout`** is the same with the rollout root as the teacher (`--rollout-spec
k:worlds:horizon:rule`): its pick where it departs from the network's move, recorded under the
budget name "rollout" with each candidate's mean rollout value and paired lead.

The searcher's configuration and the commit that produced it go in a sidecar `<output>.meta.json`
rather than a header line, because every line of the dataset itself has to stay a game.

    tools/generate_search_targets.py --checkpoint <ckpt.pt> --total-games 200 \
        --sims 96 --output-path /workspace/data/datasets/x4a_targets.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, cast

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


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True,
                                            stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        return True


#: Probabilities below this are left out of a stored network distribution (the loader
#: renormalises), so an anchor record holds the moves that carry the policy, not all ~80 targets.
PI_FLOOR = 1e-4


def gumbel_teachers(model: Any, device: torch.device, budgets: List[int], k: int, fpu: float,
                    seed: int) -> List[BatchedMCTS]:
    """One noise-free Gumbel root per budget, configured as the `gumbel:` tournament spec is
    (tools/lib/player_agent.search_spec_config), each on its own seeded streams."""
    out = []
    for j, sims in enumerate(budgets):
        cfg = BatchedMCTSConfig(simulations=sims, temperature=0.0, auto_advance=True,
                                advance_root=False, determinize=True, node_filter="all",
                                gumbel_k=k, gumbel_scale=0.0, fpu_reduction=fpu)
        t = BatchedMCTS(model, device=device, config=cfg)
        t.reseed((seed * 7919 + j) % (1 << 32))     # numpy's RandomState takes 32 bits
        out.append(t)
    return out


def gchoice_record(raw: int, pi: np.ndarray, budgets: List[int], choices: List[int],
                   stats: List[Any], value: float) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    """`search_pi` (one-hot on the first budget's choice where it departs from `raw`, the
    network's own distribution `pi` where it agrees) and the `gchoice` record beside it."""
    keep = np.flatnonzero(pi >= PI_FLOOR)
    own: Dict[str, List[float]] = {"a": [int(a) for a in keep], "v": [round(float(pi[a]), 5) for a in keep]}
    rec: Dict[str, Any] = {"raw": int(raw), "pi": own, "value": round(float(value), 4), "g": {}}
    for sims, c, st in zip(budgets, choices, stats):
        rec["g"][str(sims)] = {
            "c": int(c),
            "n": {str(a): float(x) for a, x in st["n"].items()},
            "q": {str(a): (None if x is None else round(float(x), 5)) for a, x in st["q"].items()}}
    first = int(choices[0])
    one_hot: Dict[str, List[float]] = {"a": [first], "v": [1.0]}
    return (own if first == int(raw) else one_hot), rec


def rollout_record(pi: np.ndarray, pick: int, stats: Dict[str, Any],
                   value: float) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    """`search_pi` and the record beside it for a rollout-root teacher (ai/search/rollout_root.py),
    in the gchoice record's shape under the budget name "rollout" -- the root's pick as `c`, each
    candidate's mean rollout value for the mover as `q` -- plus each candidate's paired lead over
    the network's move (`lead`: mean, standard error) and the worlds used."""
    keep = np.flatnonzero(pi >= PI_FLOOR)
    own: Dict[str, List[float]] = {"a": [int(a) for a in keep], "v": [round(float(pi[a]), 5) for a in keep]}
    cands = [int(a) for a in stats.get("candidates", [])]
    raw = cands[0] if cands else int(np.argmax(pi))
    q = {str(a): round(float(v), 5) for a, v in stats.get("q", {}).items()}
    rec: Dict[str, Any] = {"raw": raw, "pi": own, "value": round(float(value), 4),
                           "g": {"rollout": {"c": int(pick), "n": {str(a): float(stats.get("worlds", 0))
                                                                    for a in cands}, "q": q}},
                           "lead": {str(a): [round(float(x), 5) for x in v]
                                    for a, v in stats.get("lead", {}).items()},
                           "worlds": int(stats.get("worlds", 0))}
    one_hot: Dict[str, List[float]] = {"a": [int(pick)], "v": [1.0]}
    return (own if int(pick) == raw else one_hot), rec


def generate(checkpoint: str, total_games: int, batch_size: int, sims: int,
             node_filter: str, temperature: float, output_path: str, device_str: str,
             max_steps: int = 4000, target: str = "visits", gumbel_sims: Optional[List[int]] = None,
             gumbel_k: int = 8, fpu: float = 0.2, subsample: float = 1.0,
             seed_offset: int = 0, rollout_spec: str = "4:32:4:z2") -> int:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    agent = NeuralAgent.from_checkpoint(checkpoint, device=device)
    model = agent.model
    model.eval()

    # Which decisions are searched (node filter, subsample) is this searcher's rule in both modes.
    cfg = BatchedMCTSConfig(simulations=sims, temperature=0.0, auto_advance=True,
                            advance_root=False, determinize=True, node_filter=node_filter,
                            subsample=subsample)
    searcher = BatchedMCTS(model, device=device, config=cfg)
    if seed_offset:
        searcher.reseed((seed_offset + 1) % (1 << 32))
    budgets = list(gumbel_sims or [256])
    teachers = gumbel_teachers(model, device, budgets, gumbel_k, fpu, seed_offset + 2) \
        if target == "gchoice" else []
    rollout_teacher: Any = None
    if target == "rollout":
        from ai.search.rollout_root import RolloutConfig, RolloutRoot
        rk, rw, rh, rrule = rollout_spec.split(":")
        rollout_teacher = RolloutRoot(model, RolloutConfig(k=int(rk), worlds=int(rw), horizon=int(rh),
                                                           rule=rrule), device=device)
        rollout_teacher.reseed((seed_offset + 3) % (1 << 32))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    meta: Dict[str, Any] = {
        "purpose": "P15-X4a expert-iteration targets",
        "checkpoint": os.path.abspath(checkpoint),
        "searcher": ({"simulations": sims, "determinize": True, "node_filter": node_filter,
                      "advance_root": False, "subsample": subsample} if target == "visits" else
                     {"target": "gchoice", "gumbel_sims": budgets, "gumbel_k": gumbel_k,
                      "fpu_reduction": fpu, "determinize": True, "node_filter": node_filter,
                      "subsample": subsample, "search_pi_from": budgets[0]} if target == "gchoice" else
                     {"target": "rollout", "rollout_spec": rollout_spec, "node_filter": node_filter,
                      "subsample": subsample, "search_pi_from": "rollout"}),
        "checkpoint_sha256": _sha256(checkpoint),
        "seed_offset": seed_offset,
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
            base_seed = 7_000_000 + seed_offset + games_done * 10_007 + 1
            runner = ts.VectorizedBatchRunner(n, base_seed)
            hist: List[Dict[str, Any]] = [
                {"game_id": f"x4a_{seed_offset}_{games_done + i:06d}", "seed": base_seed + i * 10007 + 1,
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

                # Which active positions this configuration searches. The states are cloned: a
                # handle held across a step once followed the runner (get_state was a live view, a
                # recorded measurement bug); it is a copy now, and the clone is kept as harmless.
                to_search = [i for i in active
                             if (target == "visits" or int(masks_all[i].sum()) >= 2)
                             and searcher.should_search(runner.get_state(i))]
                targets: Dict[int, Dict[str, List[float]]] = {}
                gchoice: Dict[int, Dict[str, Any]] = {}
                if to_search and target in ("gchoice", "rollout"):
                    states = [runner.get_state(i).clone() for i in to_search]
                    with torch.no_grad():
                        lg, val, _ = cast(Any, model)(
                            torch.from_numpy(obs_all[to_search]).float().to(device),
                            torch.from_numpy(masks_all[to_search]).to(device))
                    lg_np = lg.float().cpu().numpy()
                    val_np = val.float().reshape(-1).cpu().numpy()
                    per_budget = []
                    for t in teachers:
                        per_budget.append((t.best_actions(states), list(t.gumbel_stats)))
                    r_picks: List[int] = []
                    r_stats: List[Dict[str, Any]] = []
                    if rollout_teacher is not None:
                        r_picks = rollout_teacher.choose(states)
                        r_stats = list(rollout_teacher.last_stats)
                    for row, i in enumerate(to_search):
                        legal = masks_all[i].astype(bool)
                        z = np.where(legal, lg_np[row], -np.inf)
                        pi = np.exp(z - z.max())
                        pi /= pi.sum()
                        if rollout_teacher is not None:
                            targets[i], gchoice[i] = rollout_record(pi, r_picks[row], r_stats[row],
                                                                    float(val_np[row]))
                            continue
                        targets[i], gchoice[i] = gchoice_record(
                            int(np.argmax(z)), pi, budgets, [pb[0][row] for pb in per_budget],
                            [pb[1][row] for pb in per_budget], float(val_np[row]))
                    searched_total += len(targets)
                elif to_search:
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
                    if i in gchoice:
                        rec["gchoice"] = gchoice[i]
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
    ap.add_argument("--target", default="visits", choices=["visits", "gchoice", "rollout"],
                    help="visits: PUCT visit counts (P15-X4a). gchoice: a one-hot on a noise-free "
                         "Gumbel root's choice where it departs from the network's argmax, the "
                         "network's own distribution where it agrees (see the module docstring)")
    ap.add_argument("--gumbel-sims", type=int, nargs="+", default=[256],
                    help="gchoice: the Gumbel root's evaluation budgets, each searched at the same "
                         "positions; search_pi follows the first")
    ap.add_argument("--gumbel-k", type=int, default=8, help="gchoice: candidates per root")
    ap.add_argument("--rollout-spec", default="4:32:4:z2",
                    help="rollout: the rollout root's k:worlds:horizon:rule (ai/search/rollout_root.py); "
                         "records under the budget name 'rollout' in the gchoice record's shape")
    ap.add_argument("--fpu", type=float, default=0.2,
                    help="gchoice: first-play urgency reduction (the gumbel: spec's default)")
    ap.add_argument("--subsample", type=float, default=1.0,
                    help="fraction of the node filter's decisions searched")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="shifts every game seed and the searchers' streams, so shards made on "
                         "separate machines are disjoint games")
    return ap


def main() -> int:
    a = build_parser().parse_args()
    out = a.output_path or data_path("datasets", "x4a_search_targets.jsonl.gz")
    return generate(a.checkpoint, a.total_games, a.batch_size, a.sims, a.node_filter,
                    a.temperature, out, a.device, target=a.target, gumbel_sims=a.gumbel_sims,
                    gumbel_k=a.gumbel_k, fpu=a.fpu, subsample=a.subsample,
                    seed_offset=a.seed_offset, rollout_spec=a.rollout_spec)


if __name__ == "__main__":
    sys.exit(main())
