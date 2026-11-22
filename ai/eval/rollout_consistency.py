"""Why doesn't the rollout root distil? (1) is its verdict consistent across seeds, (2) where do its departures
(and their measured leads) lie. Positions: greedy self-play of the network, every decision with >= 2 legal moves
sampled at rate p (all node types, as the training targets were; `--entry`: card choices only, the entry
decisions). Each position searched R times, reseeded -- at every world count of `--worlds` (P36b: the
target's signal-to-noise against the number of worlds, with `--keyed --stop earlier --leaf own`).
    PYTHONPATH=.:build/release python -m ai.eval.rollout_consistency <checkpoint> <games> <p> <repeats>
        [--worlds 16 64 128] [--keyed | --keyed2 | --keyedsym] [--stop earlier] [--leaf own] [--entry] [--positions 3000]

Findings: research/log/E7_search_depth_and_value.md section 6; research/plans/P36_*.md (P36b).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np
import torch

import ts_engine as ts
from ai.search.pimcts import acting_player
from ai.search.rollout_root import RolloutConfig, RolloutTeacher
from bindings.ts_env import model_obs_features
from tools.lib.player_agent import NeuralAgent, load_agent


def collect(model: Any, dev: torch.device, games: int, p: float, rng: np.random.Generator, entry: bool = False,
            limit: int = 0) -> List[Any]:
    feats = model_obs_features(model)
    runner = ts.VectorizedBatchRunner(games, 515)
    if feats:
        runner.set_obs_features([feats] * games, [feats] * games)
    done = np.zeros(games, dtype=bool)
    out: List[Any] = []
    for _ in range(6000):
        done |= np.asarray(runner.get_terminals(), dtype=bool)
        if done.all():
            break
        obs = torch.from_numpy(np.asarray(runner.get_observations(), dtype=np.float32)).to(dev)
        masks = np.asarray(runner.get_action_masks())
        with torch.no_grad():
            lg = model(obs, torch.from_numpy(masks).to(dev))[0].float().cpu().numpy()
        acts = np.where(masks > 0, lg, -np.inf).argmax(axis=1)
        acts[done] = 0
        for i in np.flatnonzero(~done & (masks.sum(axis=1) >= 2)):
            if rng.random() < p:
                st = runner.get_state(int(i))
                if entry and int(st.ctx().decision_type) != int(ts.DecisionType.SELECT_CARD):
                    continue
                out.append(st.clone())
        runner.step_flat_all(acts.tolist(), True)
    if limit and len(out) > limit:
        # a uniform subset of whole games' positions (stopping early would keep only the early turns)
        out = [out[i] for i in sorted(rng.choice(len(out), size=limit, replace=False))]
    return out


def describe(st: Any) -> Dict[str, str]:
    ctx = st.ctx()
    dt = ts.DecisionType(int(ctx.decision_type)).name
    if dt == "POINT_NODE":
        dt += "/COUP" if ctx.op_mode == ts.OpMode.COUP else ("/REALIGN" if ctx.op_mode == ts.OpMode.REALIGN else "/INFLUENCE")
    card = int(ctx.pending_op_card) or int(ctx.resolving_card)
    phase = "HEADLINE" if st.current_phase == ts.Phase.HEADLINE else ("SETUP" if st.current_phase == ts.Phase.SETUP else "AR")
    t = int(st.turn)
    era = "early(1-3)" if t <= 3 else ("mid(4-7)" if t <= 7 else "late(8-10)")
    side = "US" if acting_player(st) == ts.Player.US else "USSR"
    return {"type": dt, "phase": phase, "era": era, "defcon": f"DEFCON {int(st.defcon)}", "side": side,
            "card": (str(ts.CardData.get_card_info(card)["name"]) if 1 <= card <= 110 else "-")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("games", type=int)
    ap.add_argument("p", type=float, help="sampling rate of the decisions")
    ap.add_argument("repeats", type=int)
    ap.add_argument("--worlds", type=int, nargs="+", default=[16])
    ap.add_argument("--keyed", action="store_true", help="keyed chance (bindings/keyed_chance.hpp)")
    ap.add_argument("--keyed2", action="store_true",
                    help="keyed chance with the player fields narrowed (keyed_chance.hpp's keyed2); implies --keyed")
    ap.add_argument("--keyedsym", action="store_true",
                    help="keyed chance with both sides packed as keyed packs the USSR (keyed_chance.hpp's keyedsym); "
                         "implies --keyed, not with --keyed2")
    ap.add_argument("--stop", default="boundaries")
    ap.add_argument("--leaf", default="mover")
    ap.add_argument("--entry", action="store_true", help="card choices only (the entry decisions)")
    ap.add_argument("--positions", type=int, default=0, help="stop collecting at this many positions")
    a = ap.parse_args()
    if a.keyed2 and a.keyedsym:
        ap.error("--keyed2 and --keyedsym are exclusive")
    path, games, p, R = a.checkpoint, a.games, a.p, a.repeats
    agent = load_agent(path, device="cuda")
    if not isinstance(agent, NeuralAgent):
        raise SystemExit(f"{path} is not a network")
    model = agent.model.eval()
    dev = next(model.parameters()).device
    rng = np.random.default_rng(7)
    states = collect(model, dev, games, p, rng, a.entry, a.positions)
    print(f"{len(states)} positions from {games} greedy games{' (entry decisions)' if a.entry else ''}")
    for worlds in a.worlds:
        chance = "keyed2" if a.keyed2 else "keyedsym" if a.keyedsym else "keyed" if a.keyed else "shared-stream"
        print(f"\n======== {worlds} worlds ({chance} chance, stop={a.stop}, leaf={a.leaf})")
        report(model, dev, states, R, RolloutConfig(k=4, worlds=worlds, horizon=4, rule="z2", stop=a.stop,
                                                     leaf_side=a.leaf, keyed=a.keyed, keyed2=a.keyed2,
                                                     keyedsym=a.keyedsym))


def report(model: Any, dev: torch.device, states: List[Any], R: int, cfg: RolloutConfig) -> None:
    teacher = RolloutTeacher(model, cfg, device=dev)
    chunk_n = max(32, 8192 // cfg.worlds)
    runs: List[List[Dict[str, Any]]] = []
    for r in range(R):
        teacher.root.reseed(1000 + 17 * r)
        res: List[Dict[str, Any]] = []
        for lo in range(0, len(states), chunk_n):
            chunk = states[lo:lo + chunk_n]
            tg = teacher.improved_policies(chunk)
            for st, stats, (acts, probs) in zip(chunk, teacher.root.last_stats, tg):
                cands = stats["candidates"]
                base = cands[0] if cands else None
                pick = base
                gain = 0.0
                se = 0.0
                if stats["worlds"] > 0:
                    leads = stats["lead"]
                    best = max(cands, key=lambda a: stats["q"][a])
                    m, s = leads[best]
                    if s > 0 and m / s >= 2.0 and best != base:
                        pick, gain, se = best, m, s
                res.append({"pick": pick, "base": base, "lead": gain, "se": se,
                            "acts": acts, "probs": probs, "searched": stats["worlds"] > 0})
        runs.append(res)
    # (1) consistency
    n = len(states)
    searched = [i for i in range(n) if runs[0][i]["searched"]]
    dep = [[runs[r][i]["pick"] != runs[r][i]["base"] for i in range(n)] for r in range(R)]
    print(f"\n(1) consistency over {R} reseeded searches, {len(searched)} searched positions")
    for r in range(R):
        print(f"  run {r}: departs from the network's move in {100*np.mean([dep[r][i] for i in searched]):.1f}%")
    pairs = [(a, b) for a in range(R) for b in range(R) if a < b]
    for a, b in pairs:
        da = [i for i in searched if dep[a][i]]
        same_pick = np.mean([runs[a][i]["pick"] == runs[b][i]["pick"] for i in da]) if da else float("nan")
        also_dep = np.mean([dep[b][i] for i in da]) if da else float("nan")
        tv: List[float] = []
        for i in searched:
            pa = dict(zip(runs[a][i]["acts"], runs[a][i]["probs"])); pb = dict(zip(runs[b][i]["acts"], runs[b][i]["probs"]))
            keys = set(pa) | set(pb)
            tv.append(0.5 * sum(abs(pa.get(k, 0) - pb.get(k, 0)) for k in keys))
        top = np.mean([max(zip(runs[a][i]["probs"], runs[a][i]["acts"]))[1] == max(zip(runs[b][i]["probs"], runs[b][i]["acts"]))[1]
                       for i in searched])
        print(f"  runs {a} vs {b}: of run {a}'s departures, run {b} departs too in {100*also_dep:.0f}% and picks the same move in "
              f"{100*same_pick:.0f}%; targets' total variation {np.mean(tv):.3f} (median {np.median(tv):.3f}); "
              f"target top move agrees {100*top:.0f}%")
    # (2) where the departures lie (run 0), with leads
    print(f"\n(2) where run 0's departures lie (share of positions, share of departures, mean lead of the departure)")
    d0 = [i for i in searched if dep[0][i]]
    descs = [describe(states[i]) for i in range(n)]
    for key in ("type", "phase", "era", "defcon", "side"):
        allc = Counter(descs[i][key] for i in searched)
        depc = Counter(descs[i][key] for i in d0)
        lead: Dict[str, List[float]] = defaultdict(list)
        for i in d0:
            lead[descs[i][key]].append(runs[0][i]["lead"])
        print(f"  by {key}:")
        for k, c in allc.most_common():
            print(f"    {k:22s} positions {100*c/len(searched):5.1f}%  departs {100*depc[k]/max(1,c):5.1f}% of them  "
                  f"share of departures {100*depc[k]/max(1,len(d0)):5.1f}%  mean lead {np.mean(lead[k]) if lead[k] else 0:+.3f}")
    cardc = Counter(descs[i]["card"] for i in d0)
    print("  departures by card (top 15):", ", ".join(f"{k} {v}" for k, v in cardc.most_common(15)))
    stable = [i for i in d0 if all(runs[r][i]["pick"] == runs[0][i]["pick"] for r in range(R))]
    print(f"\n  departures stable in all {R} runs: {len(stable)} of {len(d0)} ({100 * len(stable) / max(1, len(d0)):.1f}%); "
          f"their mean lead "
          f"{np.mean([runs[0][i]['lead'] for i in stable]) if stable else 0:+.3f} vs the unstable "
          f"{np.mean([runs[0][i]['lead'] for i in d0 if i not in set(stable)]) if len(stable) < len(d0) else 0:+.3f}")
    stc = Counter(descs[i]["type"] for i in stable)
    print("  stable departures by decision type:", ", ".join(f"{k} {v}" for k, v in stc.most_common()))


if __name__ == "__main__":
    main()
