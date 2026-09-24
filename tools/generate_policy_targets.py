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

**`--merged-view` (P23): an E4 policy translated into the E4.1 view.** The games are played in the
merged-influence view, and every target is the E4 teachers' policy translated exactly into it
(`tools/lib/merged_targets.py`): at an op-choice node, "influence, first point in X" gets
P_E4(influence) * P_E4(X | after the commit), and everything else keeps its E4 probability. The
acting teacher samples from that translated policy, so the games are the ones the translated
policy plays. Distilling the dataset (`--mode distill`, which replays it in the merged view because
the sidecar says so) gives an E4.1 network that starts from the E4 policy instead of from scratch.
A raw E4 network loaded into the E4.1 view puts all of its op-choice mass on influence
(`tools/scripts/merged_view_warmstart.py`), which is why a warm start needs this translation.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import time

import numpy as np
import torch
import ts_engine as ts

from tools.lib.merged_targets import (after_influence_commit, factorised_policy, is_merged_op_choice,
                                      post_commit_policy)
from tools.lib.player_agent import NeuralAgent

TOP_K = 12
#: In the merged view an op-choice node has ~50 options and the translated policy can spread over
#: many of them, so a top-12 cut would reshape the target. Keep everything above a floor instead.
MERGED_TOP_K = 96
MERGED_FLOOR = 1e-5


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
    ap.add_argument("--merged-view", action="store_true", default=False,
                    help="P23: play in the E4.1 merged-influence view and record the E4 teachers' "
                         "policy translated exactly into it (see the module docstring)")
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--device", default="cuda")
    return ap


def _dump_bad_rows(runner: Any, idx: List[int], bad: np.ndarray, mask_t: torch.Tensor,
                   probs: torch.Tensor, output_path: str) -> None:
    """Positions the teacher cannot sample at (an empty mask, or a non-finite policy): write them
    out with everything needed to reproduce them, and stop. Never skip or patch them -- a position
    with no legal action, or a network that outputs NaN, is a bug to find, not a row to drop."""
    rows = []
    for sub in np.flatnonzero(bad):
        st = runner.get_state(idx[int(sub)])
        ctx = st.ctx()
        rows.append({
            "env": int(idx[int(sub)]),
            "decision_type": ts.DecisionType(int(ctx.decision_type)).name,
            "decision_player": int(ctx.decision_player),
            "turn": int(st.turn), "action_round": int(st.action_round),
            "mask_sum": int(mask_t[int(sub)].sum()),
            "e4_mask_sum": int(np.asarray(ts.Engine.get_flat_action_mask(st, False)).sum()),
            "merged_mask_sum": int(np.asarray(ts.Engine.get_flat_action_mask(st, True)).sum()),
            "probs_finite": bool(torch.isfinite(probs[int(sub)]).all()),
            "is_terminal": bool(ts.Engine.is_terminal(st)),
            "obs_finite": bool(np.isfinite(np.asarray(
                ts.extract_observation(st, ctx.decision_player))).all()),
            "state": st.to_save_dict(),
        })
    path = output_path + ".bad_rows.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, default=str)
    raise RuntimeError(f"{len(rows)} position(s) with no finite policy to sample from; written to {path}: "
                       + "; ".join(f"{r['decision_type']} player {r['decision_player']} turn {r['turn']} "
                                   f"AR {r['action_round']} mask {r['mask_sum']} (E4 {r['e4_mask_sum']}, "
                                   f"E4.1 {r['merged_mask_sum']}) obs_finite={r['obs_finite']} "
                                   f"probs_finite={r['probs_finite']} "
                                   f"terminal={r['is_terminal']}" for r in rows))


def _probs(model: Any, obs: torch.Tensor, mask: np.ndarray, dev: torch.device) -> np.ndarray:
    m = torch.from_numpy(np.asarray(mask)).to(dev)
    with torch.no_grad():
        logits, _, _ = model(obs, m)
    return torch.softmax(logits.float().masked_fill(m <= 0, float("-inf")), dim=-1).cpu().numpy()


def _merged_targets(model: Any, runner: Any, idx: List[int], obs_t: torch.Tensor,
                    merged_masks: np.ndarray, dev: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """The teacher's E4 policy at each position in `idx`, translated into the E4.1 view.

    Returns (targets over the E4.1 legal set, the E4.1 masks). Where the views agree the target is
    the E4 policy itself, and the two masks must be identical there -- checked, because a
    difference would mean the translation is being applied at a node it does not describe."""
    states = [runner.get_state(i) for i in idx]
    merged = np.asarray(merged_masks[idx])
    e4 = np.stack([np.asarray(ts.Engine.get_flat_action_mask(st, False)) for st in states])
    p_e4 = _probs(model, obs_t, e4, dev)
    out = p_e4.astype(np.float64)
    ops = [j for j, st in enumerate(states) if is_merged_op_choice(st)]
    same = [j for j in range(len(states)) if j not in set(ops)]
    if same and not np.array_equal(e4[same], merged[same]):
        raise RuntimeError("the E4 and E4.1 masks differ at a node that is not a merged op choice")
    if ops:
        posts = [after_influence_commit(states[j]) for j in ops]

        def _eval(ts_states: List[Any]) -> np.ndarray:
            obs = torch.from_numpy(np.stack([
                np.asarray(ts.extract_observation(t, t.ctx().decision_player), np.float32)
                for t in ts_states])).to(dev)
            masks = np.stack([np.asarray(ts.Engine.get_flat_action_mask(t, False)) for t in ts_states])
            return _probs(model, obs, masks, dev)

        p_post = post_commit_policy(posts, _eval)
        out[ops] = factorised_policy(p_e4[ops], p_post, merged[ops])
    return out.astype(np.float32), merged


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
            if a.merged_view:
                runner.set_merged_influence([True] * n, [True] * n)
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
                    if a.merged_view:
                        probs_np, merged_np = _merged_targets(teachers[side], runner, idx,
                                                              obs_t, masks_all, dev)
                        probs = torch.from_numpy(probs_np).to(dev)
                        mask_t = torch.from_numpy(merged_np).to(dev)
                        lg = torch.log(probs.clamp_min(1e-30)).masked_fill(mask_t <= 0, float("-inf"))
                        k = MERGED_TOP_K
                    else:
                        mask_t = torch.from_numpy(masks_all[idx]).to(dev)
                        with torch.no_grad():
                            logits, _, _ = teachers[side](obs_t, mask_t)
                        lg = logits.float()
                        lg = lg.masked_fill(mask_t <= 0, float("-inf"))
                        probs = torch.softmax(lg, dim=-1)
                        k = TOP_K

                    t = max(1e-3, a.temperature)
                    sharp = torch.softmax(lg / t, dim=-1)
                    bad = (~torch.isfinite(sharp).all(dim=-1)) | (mask_t.sum(dim=-1) == 0)
                    if bool(bad.any()):
                        _dump_bad_rows(runner, idx, bad.cpu().numpy(), mask_t, probs, a.output_path)
                    picks = torch.multinomial(sharp, 1).squeeze(1).cpu().numpy()

                    topv, topi = torch.topk(probs, k=min(k, probs.shape[-1]), dim=-1)
                    topv_np = topv.cpu().numpy()
                    topi_np = topi.cpu().numpy()
                    nlegal = mask_t.sum(dim=-1).cpu().numpy()
                    floor = MERGED_FLOOR if a.merged_view else 0.0

                    for sub, i in enumerate(idx):
                        act = int(picks[sub])
                        actions[i] = act
                        rec: Dict[str, Any] = {"flat_action": act}
                        if nlegal[sub] > 1:
                            keep = [(int(c), float(v))
                                    for c, v in zip(topi_np[sub], topv_np[sub]) if v > floor]
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
        # Read by --mode distill: the dataset must be replayed in the view it was played in.
        "merged_influence": bool(a.merged_view),
    }
    with open(a.output_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print("\n%d games, %s targets -> %s" % (games_done, f"{targets:,}", a.output_path))
    print("outcomes: %s" % outcomes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
