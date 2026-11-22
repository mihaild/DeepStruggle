"""The guard that keeps side computations off the training loop's random streams (its own module so that modules
the trainer imports, such as the start pool, can use it without an import cycle)."""

from __future__ import annotations

import functools
import random
from typing import Any, Callable, TypeVar, cast

import numpy as np
import torch

_F = TypeVar("_F", bound=Callable[..., Any])


def preserves_training_rng(fn: _F) -> _F:
    """Run `fn` without moving any global random stream the training loop draws from.

    Snapshot evaluation samples actions with torch's global generators (and may touch numpy's and
    Python's), and those are the same streams the rollout samples from. Left alone, every
    evaluation shifts the rest of the run onto a different trajectory -- training is chaotic at
    1e-9 (research/log/P25_stress_bench.md) -- so how often a run is evaluated would change what it
    learns. Restoring the streams afterwards makes the evaluation interval a reporting setting
    only: the same run evaluated every 5M or every 10M trains bit for bit the same.
    """
    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        cpu = torch.get_rng_state()
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        np_state = np.random.get_state()
        py_state = random.getstate()
        try:
            return fn(*args, **kwargs)
        finally:
            torch.set_rng_state(cpu)
            if cuda is not None:
                torch.cuda.set_rng_state_all(cuda)
            np.random.set_state(np_state)
            random.setstate(py_state)
    return cast(_F, wrapped)
