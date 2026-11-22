"""P28 step 2: a learning-rate schedule and an exponential moving average of the weights.

Until P28 the learning rate was a constant 3e-4 with no averaging, and late snapshots of a run swing
by about 8 points against the same opponent (P28, text in git at e5e8a7c). What each lever did:
`research/log/P28_step2a_lr_schedule.md` (a stepped rate does not cut the spread) and
`research/log/P28_step2b_ema_weights.md` (EMA beats raw snapshots).
Both levers here are off by default: `constant` returns the base rate unchanged, and an EMA time
constant of 0 builds no average.

Both are measured in environment steps, like every other budget in the trainer, never iterations
or wall-clock.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from ai.models.hyperspherical import project_sphere

LR_SCHEDULES = ("constant", "step", "cosine")


def warmup_factor(steps: int, warmup_steps: int) -> float:
    """P39 arm 1: the share of the scheduled rate at `steps` under a linear warmup over the run's first
    `warmup_steps` env steps, counted from step 0 so a resume mid-ramp continues it and a branch from a
    trained state (past the ramp) is untouched. 0 steps is no warmup."""
    if warmup_steps < 0:
        raise ValueError(f"--lr-warmup-steps must be non-negative, got {warmup_steps}")
    if warmup_steps == 0 or steps >= warmup_steps:
        return 1.0
    return max(steps, 0) / warmup_steps


def scheduled_lr(steps: int, base_lr: float, kind: str = "constant", start: int = 0,
                 every: int = 60_000_000, values: Sequence[float] = (1e-4, 3e-5),
                 span: int = 200_000_000, lr_min: float = 3e-5) -> float:
    """The learning rate at `steps`.

    * `constant`: `base_lr` throughout.
    * `step`: `base_lr` until `start + every`, then `values[0]` until `start + 2 * every`, and so
      on; the last value holds from there on.
    * `cosine`: from `base_lr` at `start` down to `lr_min` at `start + span`, half a cosine, and
      `lr_min` after.

    Before `start` every schedule is `base_lr`, so a schedule switched on at a resume point leaves
    the steps before it as they were trained.
    """
    if kind not in LR_SCHEDULES:
        raise ValueError(f"unknown --lr-schedule {kind!r}; one of {', '.join(LR_SCHEDULES)}")
    if kind == "constant" or steps < start:
        return float(base_lr)
    done = steps - start
    if kind == "step":
        if every <= 0:
            raise ValueError(f"--lr-schedule-every must be positive, got {every}")
        k = done // every
        if k == 0 or not values:
            return float(base_lr)
        return float(values[min(k, len(values)) - 1])
    if span <= 0:
        raise ValueError(f"--lr-schedule-span must be positive, got {span}")
    frac = min(1.0, done / span)
    return float(lr_min + 0.5 * (base_lr - lr_min) * (1.0 + math.cos(math.pi * frac)))


class WeightEMA:
    """An exponential moving average of a model's weights, by environment steps.

    After a training iteration that advanced the run by `d` steps, every floating-point parameter
    and buffer moves toward the live weights by `1 - exp(-d / tau)`, so the average's memory is
    `tau` steps whatever the iteration size. Integer buffers are copied. The average is a full
    module (a deep copy of the live one), so it is saved, evaluated and pooled exactly as the
    live model is.
    """

    def __init__(self, model: nn.Module, tau_steps: float,
                 state: Optional[Dict[str, Any]] = None) -> None:
        if tau_steps <= 0:
            raise ValueError(f"--ema-weights must be positive to build an average, got {tau_steps}")
        self.tau = float(tau_steps)
        self.model = copy.deepcopy(model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        if state is not None:
            self.model.load_state_dict(state)

    @torch.no_grad()
    def update(self, model: nn.Module, d_steps: float) -> float:
        """Move toward `model` by one iteration of `d_steps` steps; returns the weight used."""
        if d_steps <= 0:
            return 0.0
        alpha = 1.0 - math.exp(-float(d_steps) / self.tau)
        live = model.state_dict()
        for k, v in self.model.state_dict().items():
            src = live[k]
            if v.is_floating_point():
                v.lerp_(src.to(v.dtype), alpha)
            else:
                v.copy_(src)
        # A sphere trunk's rows are unit-norm; a Euclidean average of two unit rows is shorter (audit M-6), so the
        # average is projected back as the live weights are after each step. A no-op without a sphere trunk.
        project_sphere(self.model)
        return alpha

    def state_dict(self) -> Dict[str, Any]:
        return self.model.state_dict()
