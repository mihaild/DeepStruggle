#!/usr/bin/env python3
"""Average the weights of several checkpoints into one, uniformly.

Two uses, named differently in the research record:

* **SWA** -- the snapshots of *one run* along its trajectory, e.g. its last 80M, one every 10M
  (P28 step 1, `research/log/P28_step1_swa_and_search.md`). If the late snapshots swing around a plateau
  because of step-size noise rather than because they play differently, their average plays at
  least as well as the best of them. Every arm is rated on its snapshots and on this. Written as
  `<run>/swa_<from>-<to>M.pt`.
* **Model soup** -- separately trained branches of one trained state, e.g. E6-06/07/08-44, all
  continued from E6-04-44@560M (`research/log/model_soups_2026-09-30.md`).

The result is a bare state dict, the format of `snapshot_*.pt`, so every tournament and probe loads
it as it loads a snapshot. Floating-point tensors are averaged; integer buffers (a BatchNorm's
`num_batches_tracked`) are taken from the last checkpoint given. Checkpoints of different
architectures refuse to mix: keys and shapes must match exactly. A hyperspherical trunk's weight rows
(`--ladder-sphere-blocks`) are projected back to unit norm after averaging, as the live weights are
after every step (the mean of unit rows is shorter than unit).

    PYTHONPATH=.:build/release python tools/scripts/average_weights.py \\
        --snapshots <run>/snapshot_480..steps.pt <run>/snapshot_490..steps.pt ... \\
        --output <run>/swa_480-560M.pt
"""

from __future__ import annotations

import argparse
import hashlib
import os
from typing import Dict, List, Optional, Sequence

import torch

from ai.models.hyperspherical import project_sphere_state


def average(paths: Sequence[str]) -> Dict[str, torch.Tensor]:
    """The uniform average of the state dicts at `paths`."""
    if not paths:
        raise ValueError("no snapshots to average")
    states: List[Dict[str, torch.Tensor]] = [
        torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    keys = list(states[0].keys())
    for p, s in zip(paths[1:], states[1:]):
        if list(s.keys()) != keys:
            raise ValueError(f"{p} has different parameters from {paths[0]}; not one architecture")
        for k in keys:
            if s[k].shape != states[0][k].shape:
                raise ValueError(f"{p}: {k} is {tuple(s[k].shape)}, not {tuple(states[0][k].shape)}")
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        if states[0][k].is_floating_point():
            out[k] = torch.stack([s[k].float() for s in states]).mean(0).to(states[0][k].dtype)
        else:
            out[k] = states[-1][k].clone()
    # Fix verification V-7.4 (2026-11-22): the sphere trunk's rows are unit vectors; their Euclidean mean is not.
    project_sphere_state(out)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshots", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.output):
        raise SystemExit(f"{a.output} exists; refusing to overwrite an average that may already be rated")
    torch.save(average(a.snapshots), a.output)
    digest = hashlib.sha256(open(a.output, "rb").read()).hexdigest()
    print(f"average of {len(a.snapshots)} snapshots -> {a.output} (sha256 {digest[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
