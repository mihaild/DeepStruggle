#!/usr/bin/env python3
"""Compare advantage estimators on ONE shared rollout, before spending a run on one.

Run this before launching an arm that changes how advantages are computed. It is minutes of GPU
against the hours an arm costs, and it is the check that would have caught E3-22-28.

Why a *shared* rollout. Each arm's `adv_std_raw` is measured on its own trajectories from its own
policy, so comparing two runs' logged values cannot separate "this estimator is noisier" from
"this policy reached noisier states". Computing every estimator over the same buffer can.

Why *advantage* spread and not return accuracy. Per-player GAE beat the default on every offline
return metric -- RMSE 0.551 -> 0.481, outcome correlation 0.861 -> 0.901, exact telescoping at
lambda 1 -- and lost by 520 Elo at 80M steps, because its advantages carried 28% more variance on
identical data and the advantage is what the policy gradient consumes. A better offline return
estimate is not a better training signal. See
research/findings/training/value_bootstrap_perspective.md.

    tools/scripts/advantage_variance_probe.py <checkpoint.pt> [--envs 64] [--steps 512]
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

#: (label, compute_gae kwargs). Add a row when adding an estimator.
ESTIMATORS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("interleaved (default)", {}),
    ("per-player", {"per_player_gae": True}),
    ("same-perspective", {"same_perspective_bootstrap": True}),
)


def probe(checkpoint: str, num_envs: int, steps: int, device: str,
          seed: int, gae_lambda: float) -> int:
    from ai.training.nash_pg import NashPGTrainer
    from bindings.ts_env import TsVectorizedEnv
    from tools.lib.player_agent import NeuralAgent

    torch.manual_seed(seed)
    np.random.seed(seed)

    # NeuralAgent rather than load_agent: only the neural agent carries `.model`, and the
    # probe needs the network itself to drive a rollout.
    net = NeuralAgent.from_checkpoint(checkpoint, device=device).model
    env = TsVectorizedEnv(num_envs=num_envs, base_seed=seed)
    trainer = NashPGTrainer(env=env, active_net=net, num_envs=num_envs,
                            buffer_size=steps, device=device)
    trainer.collect_rollouts()
    buf = trainer.buffer

    last_players = buf.players[-1].to(torch.int8)
    zeros = torch.zeros(num_envs, device=buf.device)
    no_done = torch.zeros(num_envs, dtype=torch.bool, device=buf.device)

    print(f"{checkpoint}\n{num_envs} envs x {steps} steps, lambda={gae_lambda}, seed={seed}\n")

    rows: List[Tuple[str, float, np.ndarray]] = []
    skipped: List[str] = []
    for label, kwargs in ESTIMATORS:
        try:
            buf.compute_gae(last_v_win=zeros, last_v_vp=zeros, last_dones=no_done,
                            last_players=last_players, gamma=1.0,
                            gae_lambda=gae_lambda, **kwargs)
        except (TypeError, ValueError) as exc:
            # Some estimators need data captured at collection time -- same_perspective_bootstrap
            # needs next_values_own, and refuses to fall back rather than measure nothing. Skip
            # it here rather than fake it; that guard is correct.
            skipped.append(f"  {label}: {str(exc).split('.')[0]}")
            continue
        raw = float(buf.raw_advantage_std)
        ret = buf.returns_win.detach().cpu().numpy().ravel()
        rows.append((label, raw, ret.copy()))

    if not rows:
        print("no estimator produced a result")
        return 1

    header = (f"{'estimator':>24} | {'adv SD':>9} | {'vs default':>11} | "
              f"{'return SD':>10} | {'|ret|max':>9}")
    print(header)
    print("-" * len(header))
    base_sd = rows[0][1]
    for label, raw, ret in rows:
        ratio = raw / base_sd if base_sd else float("nan")
        flag = "  <-- noisier" if ratio > 1.05 else ""
        print(f"{label:>24} | {raw:>9.4f} | {ratio:>10.2f}x | {ret.std():>10.4f} | "
              f"{np.abs(ret).max():>9.4f}{flag}")

    print()
    for label, _raw, ret in rows[1:]:
        corr = float(np.corrcoef(rows[0][2], ret)[0, 1])
        print(f"  return-target correlation with the default, {label}: {corr:.4f}")
    if skipped:
        print("\nnot measurable on this rollout (each needs data captured at collection time):")
        for line in skipped:
            print(line)
    print("\nA ratio above ~1.1 is a warning: that much extra gradient noise compounds over a run.")
    print("Sane return targets do NOT clear an estimator -- per-player GAE had both, and lost.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("checkpoint", help="checkpoint to roll out with (any reasonable policy)")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--gae-lambda", type=float, default=0.98)
    return ap


def main() -> int:
    a = build_parser().parse_args()
    return probe(a.checkpoint, a.envs, a.steps, a.device, a.seed, a.gae_lambda)


if __name__ == "__main__":
    sys.exit(main())
