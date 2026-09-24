#!/usr/bin/env python3
"""How far an E4-trained network's merged-view policy is from the E4 policy it implies (P23).

Resuming an E4 checkpoint in the E4.1 view asks it, at every op-choice node, for NODE-slot
logits it was never trained to produce there. Its own E4 policy already implies a sensible
merged policy, by factorising through the step the merge removes:

    P_fact(X)            = P_E4(OPS_INFLUENCE | s) * P_E4(X | s')          s' = s after the commit
    P_fact(OPS_INFLUENCE) = P_E4(OPS_INFLUENCE | s) * P_E4(CONFIRM_DONE | s')
    P_fact(a)            = P_E4(a | s)                                      every other action

This reports how far the raw merged-view output is from that, at op-choice nodes of the model's
own E4 self-play: KL(P_fact || P_raw), total variation, the influence mass each assigns, and top-1
agreement. It decides whether a warm start needs a short distillation toward P_fact first.

    PYTHONPATH=.:build/release python tools/scripts/merged_view_warmstart.py <snapshot.pt> [--games 64]
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import ts_engine as ts  # noqa: E402

from bindings.ts_env import TsVectorizedEnv  # noqa: E402
from tools.lib.player_agent import NeuralAgent, resolve_device  # noqa: E402

from tools.lib.merged_targets import (INFL, NODE, OP_CHOICE, factorised_policy,  # noqa: E402
                                      post_commit_policy)


def collect(model: Any, games: int, device: torch.device) -> List[ts.GameState]:
    """Op-choice states with influence legal, from the model's own E4 self-play."""
    env = TsVectorizedEnv(num_envs=games, base_seed=424_242)
    obs, masks, _ = env.reset_all()
    done = np.zeros(games, dtype=bool)
    out: List[ts.GameState] = []
    for _ in range(20_000):
        if done.all():
            break
        for i in range(games):
            if done[i]:
                continue
            s = env.runner.get_state(i)
            if s.ctx().decision_type in OP_CHOICE and masks[i][INFL]:
                out.append(s.clone())
        with torch.no_grad():
            a, *_ = model.sample_action(torch.from_numpy(np.asarray(obs, np.float32)).to(device),
                                        torch.from_numpy(np.asarray(masks)).to(device), temperature=1.0)
        obs, masks, _, dones, _ = env.step(a.cpu().numpy())
        done |= np.asarray(dones) > 0.5
    return out


def policy(model: Any, states: List[ts.GameState], masks: np.ndarray,
           device: torch.device) -> np.ndarray:
    obs = np.stack([np.asarray(ts.extract_observation(s, s.ctx().decision_player), np.float32)
                    for s in states])
    with torch.no_grad():
        logits, _, _ = model(torch.from_numpy(obs).to(device), torch.from_numpy(masks).to(device))
        return F.softmax(logits.float(), dim=-1).cpu().numpy()


def compare(model: Any, states: List[ts.GameState], device: torch.device,
            student: Any = None) -> Tuple[np.ndarray, ...]:
    e4_masks = np.stack([np.asarray(ts.Engine.get_flat_action_mask(s)) for s in states])
    mv_masks = np.stack([np.asarray(ts.Engine.get_flat_action_mask(s, True)) for s in states])
    afters: List[ts.GameState] = []
    for s in states:
        t = s.clone()
        ts.Engine.step_flat(t, INFL, False)
        afters.append(t)
    post_masks = np.stack([np.asarray(ts.Engine.get_flat_action_mask(t)) for t in afters])

    p_e4 = policy(model, states, e4_masks, device)
    p_post = post_commit_policy(
        afters, lambda sts: policy(model, sts, np.stack([np.asarray(ts.Engine.get_flat_action_mask(t))
                                                         for t in sts]), device))
    p_raw = policy(student if student is not None else model, states, mv_masks, device)

    p_fact = factorised_policy(p_e4, p_post, mv_masks)
    node = NODE

    eps = 1e-12
    legal = mv_masks > 0
    kl = np.where(legal, p_fact * (np.log(p_fact + eps) - np.log(p_raw + eps)), 0.0).sum(axis=1)
    tv = 0.5 * np.abs(p_fact - p_raw).sum(axis=1)
    infl_fact = p_fact[:, node].sum(axis=1) + p_fact[:, INFL]
    infl_raw = p_raw[:, node].sum(axis=1) + p_raw[:, INFL]
    top1 = p_fact.argmax(axis=1) == p_raw.argmax(axis=1)
    return kl, tv, infl_fact, infl_raw, top1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--student", default=None,
                    help="an E4.1 checkpoint to score against the teacher's translated policy "
                         "(e.g. one distilled from it), instead of the teacher's own raw output")
    args = ap.parse_args()
    device = resolve_device(args.device)
    agent = NeuralAgent.from_checkpoint(args.checkpoint, device=device)
    model = agent.model
    states = collect(model, args.games, device)
    student = (NeuralAgent.from_checkpoint(args.student, device=device).model
               if args.student else None)
    kl, tv, inf_f, inf_r, top1 = compare(model, states, device, student)
    who = f"student {os.path.basename(os.path.dirname(os.path.abspath(args.student)))}" if args.student else "raw"
    print(f"{agent.name} ({who}): {len(states)} op-choice nodes with influence legal, from {args.games} games")
    print(f"  KL(P_fact || P_model) mean {kl.mean():.3f}  median {np.median(kl):.3f}  p90 {np.quantile(kl, 0.9):.3f}")
    print(f"  total variation       mean {tv.mean():.3f}  median {np.median(tv):.3f}")
    print(f"  influence mass        implied {inf_f.mean():.3f}  model {inf_r.mean():.3f}")
    print(f"  top-1 agreement       {100 * top1.mean():.1f}%")


if __name__ == "__main__":
    main()
