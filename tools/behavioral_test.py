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
from typing import Dict, List, Optional

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from ai.eval.behavioral_suite import PolicyFn, SuiteResult, run_suite, torch_policy  # noqa: E402
from ai.eval.claims import build_claims  # noqa: E402


def load_policy(spec: str, device: str) -> PolicyFn:
    """Loads a checkpoint, or a baseline bot, as a (state, obs, mask) -> probabilities fn."""
    import ts_engine as ts
    from tools.lib.player_agent import NeuralAgent, load_agent

    if spec in ("random", "heuristic"):
        agent = load_agent(spec, device=device)

        def bot_policy(state: "ts.GameState", obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
            # A deterministic bot has no distribution, so represent its choice as a
            # one-hot. That keeps NeverArgmax and Prefer meaningful rather than scoring
            # the baseline against a uniform stand-in.
            del obs
            probs = np.zeros_like(mask, dtype=np.float64)
            try:
                idx = int(agent.select_action(state, state.ctx().decision_player, temperature=0.0))
            except Exception:
                legal = np.flatnonzero(mask)
                return (mask.astype(np.float64) / len(legal)) if len(legal) else probs
            if 0 <= idx < probs.shape[0]:
                probs[idx] = 1.0
            return probs

        return bot_policy

    net = NeuralAgent.from_checkpoint(spec, device=device).model
    return torch_policy(net, device=device)


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
        try:
            policy = load_policy(spec, args.device)
        except Exception as exc:
            print(f"  could not load: {exc}")
            continue
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
