#!/usr/bin/env python3
"""Behavioural test suite CLI: scores a checkpoint on named strategic decisions.

Win rate says an agent is better; this says what it understands. Each claim is a
constructed position with a known-correct answer, checked against the masked policy
distribution, reported as a pass rate that moves for a legible reason.

    PYTHONPATH=.:build/release .venv/bin/python tools/behavioral_test.py \
        --models data/checkpoints/run_x/snapshot_final.pt heuristic --device cuda
"""

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import ts_engine as ts  # noqa: E402
from ai.eval.behavioral_suite import PolicyFn, SuiteResult, run_suite  # noqa: E402
from ai.eval.claims import build_claims  # noqa: E402
from bindings.ts_env import legal_mask, observe  # noqa: E402
from tools.lib.player_agent import NeuralAgent, PlayerAgent, load_agent  # noqa: E402

BASELINES = ("random", "heuristic", "heuristic_v2")


def bot_policy(agent: PlayerAgent) -> PolicyFn:
    """A baseline bot's choice as a one-hot. A deterministic bot has no distribution, and a
    one-hot keeps NeverArgmax and Prefer meaningful. A bot that raises is a bot defect: it
    propagates (the suite records it against the claim) rather than being scored as a uniform
    policy it never was."""

    def fn(state: "ts.GameState", obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        del obs
        idx = int(agent.select_action(state, state.ctx().decision_player, temperature=0.0))
        if not (0 <= idx < mask.shape[0]) or not mask[idx]:
            raise ValueError(f"{agent.name} chose flat action {idx}, not legal here")
        probs = np.zeros(mask.shape[0], dtype=np.float64)
        probs[idx] = 1.0
        return probs

    return fn


def network_policy(agent: NeuralAgent) -> PolicyFn:
    """A network's masked distribution, read in its own observation feature set.

    The observation is rebuilt from the state with `bindings.ts_env.observe` in the network's
    set rather than taken from the suite, whose row is the base layout: a HEADLINE_REVEAL network
    (every E8 checkpoint) reads different card rows at a box-4 headline, at the same width. The
    claims are written in the E4 action view, so a merged-view network is refused at load."""
    device = next(agent.model.parameters()).device

    def fn(state: "ts.GameState", obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        del obs, mask   # both rebuilt in the network's own set: its observation, and its rules' mask
        o = torch.from_numpy(observe(state, state.ctx().decision_player, agent.obs_features))
        m = torch.from_numpy(legal_mask(state.clone(), False, agent.obs_features))
        with torch.no_grad():
            logits, _, _ = agent.model(o.unsqueeze(0).to(device), m.unsqueeze(0).to(device))
            return F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    return fn


def load_policy(spec: str, device: str) -> PolicyFn:
    """Loads a checkpoint, or a baseline bot, as a (state, obs, mask) -> probabilities fn."""
    if spec in BASELINES:
        return bot_policy(load_agent(spec, device=device))
    agent = NeuralAgent.from_checkpoint(spec, device=device)
    if agent.merged_influence:
        raise ValueError(f"{spec} decides in the merged-influence (E4.1) view; the claims are "
                         f"written in E4 action slots and cannot score it")
    return network_policy(agent)


def main() -> None:
    ap = argparse.ArgumentParser(description="Behavioural test suite for Twilight Struggle policies")
    ap.add_argument("--models", nargs="+", required=True,
                    help="Checkpoint paths, or the baselines 'random' / 'heuristic'")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--output-json", type=str, default=None)
    ap.add_argument("--quiet", action="store_true", help="Print only the pass-rate summary")
    args = ap.parse_args()

    tests = build_claims()
    all_results: Dict[str, SuiteResult] = {}
    for spec in args.models:
        name = os.path.basename(spec).replace(".pt", "") if spec.endswith(".pt") else spec
        print(f"\n=== {name} ({len(tests)} claims) ===")
        # A model that does not load stops the run with its error (and a non-zero exit) rather
        # than leaving a summary that silently lacks it.
        policy = load_policy(spec, args.device)
        res = run_suite(policy, tests)
        all_results[name] = res
        print(res.format_report() if not args.quiet else
              f"pass rate: {res.pass_rate * 100:.1f}%")

    if len(all_results) > 1:
        print("\n=== summary ===")
        for name, res in all_results.items():
            tiers = " ".join(f"{t}:{r * 100:.0f}%" for t, r in res.by_tier().items())
            print(f"  {name:44s} {res.pass_rate * 100:5.1f}%   {tiers}")

    if args.output_json:
        payload = {
            name: {
                "pass_rate": res.pass_rate,
                "by_tier": res.by_tier(),
                "results": [
                    {"claim_id": r.claim_id, "tier": r.tier, "passed": r.passed,
                     "detail": r.detail, "error": r.error}
                    for r in res.results
                ],
            }
            for name, res in all_results.items()
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.output_json}")


if __name__ == "__main__":
    main()
