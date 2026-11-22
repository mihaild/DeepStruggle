"""Whole games played by the rollout root on both sides, in worker processes, as value targets for the critic.

`--searched-games-workers N` (P33, owner 2026-10-17; the R54 family's "searched environments"): the critic
is what the search keeps (P33a) and it is at chance where the searched player errs (P33c); the policy-side
ways of passing the search into the network (search targets, distillation) were thin or null. Here the
searched player does not label the network's games -- it plays its own. Each worker holds the trainer's
frozen reference network (refreshed with pi_ref) and a `RolloutRoot`, plays a batch of games with the root
choosing every move of both sides, and streams back each finished game's decisions -- the decider's
observation and the game's result from the decider's side. The trainer keeps them in a replay store and
trains only the value head on them (`--searched-value-coef`): no policy loss, the moves being the root's.
They are the searched player's value function, which is what the root reads at its leaves.
"""

from __future__ import annotations

import io
import multiprocessing as mp
import os
import queue as _queue
import traceback
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

import ts_engine as ts
from ai.search.rollout_root import RolloutConfig, RolloutRoot
from ai.training.search_workers import WorkerRolloutConfig
from bindings.ts_env import model_obs_features

#: (observations (n, width) float32, targets (n,) float32 in [-1, 1], games finished)
Rows = Tuple[np.ndarray, np.ndarray, int]


def _worker_main(control: Any, results: Any, model_bytes: bytes, cfg: WorkerRolloutConfig,
                 envs: int, device: str, seed: int) -> None:
    torch.set_num_threads(2)
    parent = os.getppid()
    dev = torch.device(device)
    try:
        net = torch.load(io.BytesIO(model_bytes), map_location=dev, weights_only=False).eval()
        root = RolloutRoot(net, cfg.build(seed=seed), device=dev)
        feats = model_obs_features(net)
        runner = ts.VectorizedBatchRunner(envs, seed)
        if feats:
            runner.set_obs_features([feats] * envs, [feats] * envs)
        games: List[List[Tuple[np.ndarray, int]]] = [[] for _ in range(envs)]
        next_seed = seed * 1_000_003
        while True:
            if os.getppid() != parent:                 # the trainer died without closing us
                return
            try:
                msg = control.get_nowait()
                if msg is None:
                    return
                net.load_state_dict(torch.load(io.BytesIO(msg), map_location=dev, weights_only=True))
            except _queue.Empty:
                pass
            term = np.asarray(runner.get_terminals(), dtype=bool)
            if term.any():
                utils = np.asarray(runner.get_terminal_utilities(), dtype=np.float32)
                obs_out: List[np.ndarray] = []
                tgt_out: List[float] = []
                for i in np.flatnonzero(term):
                    for o, side in games[i]:
                        obs_out.append(o)
                        tgt_out.append(float(utils[i]) * side)
                    games[i] = []
                    next_seed += 1
                    runner.reset_game(int(i), next_seed)
                if obs_out:
                    results.put(("ok", (np.stack(obs_out).astype(np.float32), np.asarray(tgt_out, np.float32),
                                        int(term.sum()))))
            obs = np.asarray(runner.get_observations(), dtype=np.float32)
            players = np.asarray(runner.get_decision_players())
            states = [runner.get_state(i) for i in range(envs)]
            picks = root.choose(states)
            for i in range(envs):
                games[i].append((obs[i].copy(), 1 if int(players[i]) == 1 else -1))
            runner.step_flat_all([int(a) for a in picks], True)
    except Exception:                                  # surfaced in the trainer, not swallowed here
        results.put(("error", traceback.format_exc()))


class SearchedGames:
    """Worker processes playing whole searched games; `drain` returns the decisions finished so far."""

    def __init__(self, n: int, net: Any, config: RolloutConfig, device: Any, envs: int = 32,
                 seed: int = 24680) -> None:
        if n < 1:
            raise ValueError("--searched-games-workers must be >= 1")
        ctx = mp.get_context("spawn")
        buf = io.BytesIO()
        torch.save(net, buf)
        self._results = ctx.Queue()
        self._control: List[Any] = []
        self._procs: List[Any] = []
        cfg = WorkerRolloutConfig.of(config)
        for w in range(int(n)):
            q = ctx.Queue()
            p = ctx.Process(target=_worker_main, args=(q, self._results, buf.getvalue(), cfg, int(envs), str(device),
                                                       seed + 7919 * w), daemon=True)
            p.start()
            self._control.append(q)
            self._procs.append(p)
        self.games = 0

    def refresh(self, net: Any) -> None:
        """New weights for every worker (pi_ref moved); each takes them between two moves."""
        buf = io.BytesIO()
        torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, buf)
        for q in self._control:
            q.put(buf.getvalue())

    def drain(self) -> Optional[Rows]:
        """Every decision of the games finished since the last call, or None; raises on a worker error."""
        obs: List[np.ndarray] = []
        tgt: List[np.ndarray] = []
        games = 0
        while True:
            try:
                kind, payload = self._results.get_nowait()
            except _queue.Empty:
                break
            if kind != "ok":
                raise RuntimeError(f"searched-games worker failed:\n{payload}")
            o, t, g = payload
            obs.append(o)
            tgt.append(t)
            games += g
        if not obs:
            dead = [p.pid for p in self._procs if not p.is_alive()]
            if dead:
                raise RuntimeError(f"searched-games worker(s) {dead} died")
            return None
        self.games += games
        return np.concatenate(obs), np.concatenate(tgt), games

    def close(self) -> None:
        for q in self._control:
            q.put(None)
        for p in self._procs:
            p.join(timeout=10)
        self._procs, self._control = [], []
