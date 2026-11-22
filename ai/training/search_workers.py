"""Rollout-root search targets priced in worker processes, beside the rollout that queues them.

`--search-workers N` with `--search-rollout-horizon`: the decisions the trainer queues for search
targets during a rollout are streamed, in chunks as they accumulate, to N processes that each hold
their own copy of the **frozen** teacher and a `RolloutTeacher`; at the rollout's end the trainer
sends what is left and collects every answer before GAE. The targets therefore land on the same
buffer rows, in the same iteration, as when the trainer searched them itself -- R39's mechanics are
unchanged, only the time spent waiting goes. Positions travel as engine saves (the round trip is
exact; `tests/training/test_turn_credit.py`), answers as (queue index, legal actions, probabilities).
"""

from __future__ import annotations

import io
import os
import multiprocessing as mp
import queue as _queue
import traceback
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

import ts_engine as ts
from ai.search.rollout_root import RolloutConfig, RolloutTeacher

#: (queue index, legal actions, target probabilities)
Answer = Tuple[int, List[int], np.ndarray]


@dataclass(frozen=True)
class WorkerRolloutConfig:
    """The `RolloutConfig` a worker process builds its root from (picklable, sent at spawn): the whole of it,
    so a worker searches exactly what the trainer would. It used to carry six fields and silently drop the
    rest -- keyed, keyed2, aligned, box4, oracle, cuda_graphs (audit S-9.4, 2026-11-22)."""
    config: RolloutConfig

    @classmethod
    def of(cls, config: RolloutConfig) -> WorkerRolloutConfig:
        return cls(replace(config))

    def build(self, **overrides: Any) -> RolloutConfig:
        """The config, with `overrides` (a worker's own seed) replaced."""
        return replace(self.config, **overrides)


def _worker_main(jobs: Any, results: Any, model_bytes: bytes, cfg: WorkerRolloutConfig,
                 device: str, seed: int) -> None:
    torch.set_num_threads(2)
    parent = os.getppid()
    dev = torch.device(device)
    net = torch.load(io.BytesIO(model_bytes), map_location=dev, weights_only=False).eval()
    teacher = RolloutTeacher(net, cfg.build(), device=dev, subsample=1.0, seed=seed)
    while True:
        try:
            msg = jobs.get(timeout=30.0)
        except _queue.Empty:
            if os.getppid() != parent:          # the trainer died without closing us: do not linger
                return
            continue
        if msg is None:
            return
        rid, items = msg
        try:
            states = [ts.state_from_save_json(js) for _idx, js in items]
            out = teacher.improved_policies(states)
            results.put(("ok", rid, [(idx, acts, probs) for (idx, _js), (acts, probs) in zip(items, out)]))
        except Exception:                            # surfaced in the trainer, not swallowed here
            results.put(("error", rid, traceback.format_exc()))


class SearchWorkers:
    """A pool of rollout-root teachers in their own processes, fed during the rollout."""

    def __init__(self, n: int, teacher_net: Any, config: RolloutConfig, device: Any, chunk: int = 16,
                 seed: int = 13579) -> None:
        if n < 1:
            raise ValueError("--search-workers must be >= 1")
        self.n, self.chunk = int(n), int(chunk)
        ctx = mp.get_context("spawn")
        buf = io.BytesIO()
        torch.save(teacher_net, buf)
        self._results = ctx.Queue()
        self._jobs: List[Any] = []
        self._procs: List[Any] = []
        cfg = WorkerRolloutConfig.of(config)
        for w in range(self.n):
            q = ctx.Queue()
            p = ctx.Process(target=_worker_main, args=(q, self._results, buf.getvalue(), cfg, str(device), seed + w),
                            daemon=True)
            p.start()
            self._jobs.append(q)
            self._procs.append(p)
        self._rid = 0
        self._sent = 0           # queue entries of this rollout already sent
        self._batches = 0        # batches of this rollout in flight
        self._next = 0           # round-robin pointer

    def begin(self) -> None:
        """A new rollout: answers of any earlier one are never mixed in (`collect` checks the id)."""
        self._rid += 1
        self._sent = 0
        self._batches = 0

    def feed(self, queue: Sequence[Tuple[int, int, Any]], final: bool = False) -> None:
        """Send the queue entries not sent yet, in chunks (the tail too when `final`)."""
        while len(queue) - self._sent >= self.chunk or (final and len(queue) > self._sent):
            hi = min(len(queue), self._sent + self.chunk)
            items = [(j, queue[j][2].to_save_json()) for j in range(self._sent, hi)]
            self._jobs[self._next].put((self._rid, items))
            self._next = (self._next + 1) % self.n
            self._sent = hi
            self._batches += 1

    def collect(self) -> Dict[int, Tuple[List[int], np.ndarray]]:
        """Every answer of this rollout, by queue index; raises on a worker error or death."""
        got: Dict[int, Tuple[List[int], np.ndarray]] = {}
        need = self._batches
        while need > 0:
            try:
                kind, rid, payload = self._results.get(timeout=10.0)
            except _queue.Empty:
                dead = [p.pid for p in self._procs if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"search worker(s) {dead} died without answering")
                continue
            if kind != "ok":
                raise RuntimeError(f"search worker failed:\n{payload}")
            if rid != self._rid:
                continue
            for idx, acts, probs in payload:
                got[int(idx)] = (list(acts), np.asarray(probs))
            need -= 1
        return got

    def close(self) -> None:
        for q in self._jobs:
            q.put(None)
        for p in self._procs:
            p.join(timeout=10)
        self._procs, self._jobs = [], []
