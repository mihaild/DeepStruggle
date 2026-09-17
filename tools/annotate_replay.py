#!/usr/bin/env python3
"""Ask a checkpoint what it thought of a game, and write the answers onto the replay.

The inline trace (`tools/lib/self_play.py`) records what the policy that *played* believed. This
records what *a* policy believes about a game it may not have played, which is a different and
also useful question: what does the 320M snapshot make of the 80M snapshot's turn 4, and where do
they disagree? It is also how a replay recorded before the trace existed gets one.

The method is `ai/eval/replay_critic.py`'s: re-drive the game from its seed by applying the
logged actions, and read the model at every step. That rests entirely on the reconstruction being
the same game the replay recorded -- a drifted one still returns numbers and they still look like
results -- so `verify()` runs at **every** step and the first mismatch aborts. A rebuilt engine
can change the decision stream with no Python change (invariant 13), which is exactly how a
reconstruction drifts, so the engine fingerprint goes into the file next to the numbers.

    PYTHONPATH=.:build/release .venv/bin/python tools/annotate_replay.py \
        --replay data/replays/s160_1.tslog.json --model data/checkpoints/run/final.pt --print
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import ts_engine as ts
from ai.eval.policy_readout import (
    TRACE_P_FLOOR,
    TRACE_TOP_K,
    read_critic,
    read_policy,
    unasked_readout,
)
from ai.eval.replay_critic import load_actions, verify
from bindings.action_encoder import ActionEncoder
from bindings.ts_env import check_obs_width
from tools.lib.engine_fingerprint import fingerprint as engine_fingerprint
from tools.lib.game_step import drain_chance
from tools.lib.player_agent import NeuralAgent, load_agent
from tools.lib.self_play import checkpoint_digest
from web.server.replay_types import ReplayCriticDict, ReplayPolicyDict, ReplayTraceMetaDict


class ReconstructionDiverged(RuntimeError):
    """The re-driven game stopped being the game the replay recorded."""


def annotate(
    replay_path: str,
    model: Any,
    top_k: int = TRACE_TOP_K,
    p_floor: float = TRACE_P_FLOOR,
    critic_every: str = "step",
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Re-drive the replay and attach a policy/critic block to each of its steps.

    Returns the whole replay document, with the blocks added in place.
    """
    with open(replay_path, encoding="utf-8") as f:
        doc: Dict[str, Any] = json.load(f)

    seed, steps = load_actions(replay_path)
    if not steps:
        raise ValueError(f"{replay_path} has no steps")

    state = ts.GameState()
    ts.Engine.init_game(state, seed)
    drain_chance(state, context="annotate_replay chance node")

    by_index = {int(s["step_index"]): s for s in doc["steps"]}

    for rec in steps:
        if limit is not None and rec["step_index"] > limit:
            break
        flat = int(rec["flat"])
        if flat < 0:
            raise ReconstructionDiverged(
                f"step {rec['step_index']} has no flat_action_idx, so the game cannot be "
                f"re-driven from here")

        # The policy is read BEFORE the action, on the node the move was chosen at, and the mask
        # is the engine's own -- not the recorded one, which a replay does not carry.
        mask_np = np.asarray(ActionEncoder.get_legal_mask(state)).reshape(1, -1)
        n_legal = int(mask_np.sum())
        if not mask_np[0, flat]:
            raise ReconstructionDiverged(
                f"step {rec['step_index']}: the replay plays action {flat} "
                f"({ActionEncoder.get_action_name(state, flat)}) but the re-driven game does not "
                f"allow it. The reconstruction is not this game any more.")

        policy: ReplayPolicyDict
        if n_legal <= 1:
            policy = unasked_readout(flat, state)
        else:
            obs_np = np.asarray(
                ts.extract_observation(state, _acting(state)), dtype=np.float32).reshape(1, -1)
            device = next(model.parameters()).device
            _, policy = read_policy(
                model,
                torch.from_numpy(obs_np).float().to(device),
                torch.from_numpy(mask_np).to(device),
                temperature=1.0,
                deterministic=True,   # nothing is being played; do not touch the torch RNG
                state=state,
                top_k=top_k,
                p_floor=p_floor,
                source="annotated",
                include=flat,
            )
            # Nothing was sampled here: the action is the one the replay recorded, so the
            # "chosen" fields must describe it and not the model's own preference. `include`
            # guarantees it is listed even when it falls below the floor -- which is precisely
            # the case worth reading.
            p_recorded = next(e["p"] for e in policy["top"] if e["idx"] == flat)
            policy["chosen_idx"] = flat
            policy["p_chosen"] = p_recorded
            policy["p_chosen_sampled"] = p_recorded

        ts.Engine.step_flat(state, flat)
        drain_chance(state, context="annotate_replay chance node")

        problem = verify(state, rec)
        if problem is not None:
            raise ReconstructionDiverged(
                f"step {rec['step_index']}: the re-driven game no longer matches the replay "
                f"({problem}). The numbers this run would produce belong to a different game. "
                f"Most likely the engine has been rebuilt since the replay was recorded -- "
                f"check tools/scripts/check_engine_fresh.sh and the replay's own fingerprint.")

        critic: Optional[ReplayCriticDict] = None
        if critic_every == "step" or (critic_every == "decision" and n_legal > 1):
            critic = read_critic(model, state)

        target = by_index.get(int(rec["step_index"]))
        if target is None:
            raise ReconstructionDiverged(
                f"step {rec['step_index']} is in the action list but not in the document")
        target["policy"] = policy
        if critic is not None:
            target["critic"] = critic

    return doc


def _acting(state: ts.GameState) -> ts.Player:
    ctx = state.ctx()
    return ctx.decision_player if ctx.decision_player != ts.Player.NONE else state.phasing_player


def _print_table(doc: Dict[str, Any], rows: int) -> None:
    """One line per step: what was played, how much the model liked it, what it cost."""
    print(f"{'#':>5} {'T':>2} {'AR':>2} {'side':>4} {'p':>6} {'pmax':>6} {'H':>5} "
          f"{'v_win':>6} {'dv':>6}  action")
    print("-" * 110)
    prev: Optional[float] = None
    shown = 0
    for step in doc["steps"]:
        pol = step.get("policy") or {}
        cri = step.get("critic") or {}
        v = cri.get("v_win_us")
        dv = None if (v is None or prev is None) else v - prev
        if v is not None:
            prev = v
        if pol.get("source") in ("forced",) and dv is not None and abs(dv) < 1e-4:
            continue   # a settled step that moved nothing says nothing
        print(f"{step['step_index']:>5} {step.get('turn', 0):>2} {step.get('ar', 0):>2} "
              f"{str(step.get('player', '')):>4} "
              f"{_f(pol.get('p_chosen')):>6} {_f(pol.get('p_max')):>6} "
              f"{_f(pol.get('entropy'), 2):>5} {_f(v, 3):>6} {_f(dv, 3):>6}  "
              f"{str(step.get('description', ''))[:52]}")
        shown += 1
        if shown >= rows:
            print(f"... {len(doc['steps']) - shown} more steps")
            break


def _f(x: Optional[float], nd: int = 3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Annotate a .tslog.json replay with a checkpoint's policy and critic.")
    parser.add_argument("--replay", required=True, help="Path to the .tslog.json to annotate")
    parser.add_argument("--model", required=True, help="Checkpoint (.pt) to ask")
    parser.add_argument("--out", default=None,
                        help="Where to write (default: alongside, with .annotated.tslog.json)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trace-top-k", type=int, default=TRACE_TOP_K,
                        help="Legal actions listed per node; 0 (the default) lists them all")
    parser.add_argument("--trace-p-floor", type=float, default=TRACE_P_FLOOR)
    parser.add_argument("--trace-critic-every", default="step",
                        choices=("step", "decision", "off"))
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many steps (for a quick look)")
    parser.add_argument("--print", dest="do_print", action="store_true",
                        help="Print the per-step table")
    parser.add_argument("--print-rows", type=int, default=60)
    args = parser.parse_args(argv)

    agent = load_agent(args.model, device=args.device)
    if not isinstance(agent, NeuralAgent):
        print(f"--model must be a checkpoint, not {args.model!r}", file=sys.stderr)
        return 2
    model = agent.model
    model.eval()
    check_obs_width(model)

    doc = annotate(args.replay, model, top_k=args.trace_top_k, p_floor=args.trace_p_floor,
                   critic_every=args.trace_critic_every, limit=args.limit)

    trace_meta: ReplayTraceMetaDict = {
        "mode": "annotated",
        "critic_model": os.path.basename(args.model),
        "checkpoint_sha256_12": checkpoint_digest(args.model),
        "arch": type(model).__name__,
        "temperature": 1.0,
        "top_k": int(args.trace_top_k),
        "p_floor": float(args.trace_p_floor),
        "engine_fingerprint": engine_fingerprint(),
    }
    doc.setdefault("metadata", {})["trace"] = trace_meta

    out = args.out
    if out is None:
        base = args.replay
        for suffix in (".tslog.json", ".json"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        out = f"{base}.annotated.tslog.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"annotated {len(doc['steps'])} steps -> {out}")

    if args.do_print:
        _print_table(doc, args.print_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
