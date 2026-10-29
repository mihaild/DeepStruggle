"""P40 Arm S: the rollout root as the OPPONENT in a share of training games (owner, 2026-10-29).

The root is the punisher, not the teacher: no search target, no root-played learner moves. In a fixed
block of envs (the last `share` of them, disjoint from the opponent pool's mixed envs, which are the
first) the learner plays one side, drawn per episode, and the other side is a lagged snapshot of the
learner -- refreshed every `refresh_every` steps -- playing greedily, with the rollout root choosing its
move at its searched decisions: entry decisions (a headline card or an action round's card play;
`scope="entry"`) or, as `rollout:` plays in a tournament, every decision with two or more legal moves
(`scope="all"`). Those rows are the environment's (learner = 0), as a pool opponent's are.

The root costs ~35 ms a call whatever it searches, plus well under 1 ms a position, so one call per
step would cost ~20x a plain step. A searched game therefore WAITS: at a searched decision its env is
queued and held in place (`TsVectorizedEnv.step(hold=)`), the other games step on, and once `batch`
are queued or the oldest has waited `max_wait` steps the root decides them all in one call and they
play on. A held step is stored as a row (the buffer is dense in steps), marked `held`: it repeats the
next row's state with reward 0, so GAE passes through it and no loss reads it (`RolloutBuffer.held`).
"""

from __future__ import annotations

import copy
import random
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch

import ts_engine as ts

from ai.search.rollout_root import RolloutRoot, is_entry_decision, rollout_spec_config

SCOPES = ("entry", "all")


class SearchedOpponent:
    def __init__(self, learner: torch.nn.Module, spec: str, device: Any, num_envs: int, share: float,
                 pool_mixed: int, refresh_every: int, batch: int, max_wait: int, scope: str = "entry",
                 seed: int = 13579) -> None:
        """`spec` is a rollout spec tail without its checkpoint (e.g. "4:4:1:z2:earlier:own:keyed:aligned:box4"),
        read by the same parser as every other player. `pool_mixed` is the number of envs the opponent
        pool plays mixed (its first ones): the searched block must not overlap them."""
        if scope not in SCOPES:
            raise ValueError(f"--searched-opponent-scope must be one of {SCOPES}, got {scope!r}")
        if not 0.0 < share < 1.0:
            raise ValueError(f"--searched-opponent-share must be in (0, 1), got {share}")
        if batch < 1 or max_wait < 0 or refresh_every <= 0:
            raise ValueError("the searched opponent needs batch >= 1, max_wait >= 0 and a positive refresh cadence")
        self.spec = spec
        _, self.config, self.label = rollout_spec_config(f"rollout:-:{spec}")
        if self.config.belief or self.config.oracle:
            raise ValueError("the searched opponent plays from its own information: no belief or oracle worlds")
        n = int(round(num_envs * share))
        if n < 1:
            raise ValueError(f"--searched-opponent-share {share} of {num_envs} envs is no env")
        if pool_mixed + n > num_envs:
            raise ValueError(f"the searched block ({n} envs) overlaps the opponent pool's {pool_mixed} mixed envs "
                             f"of {num_envs}")
        self.num_envs = int(num_envs)
        self.envs = np.arange(num_envs - n, num_envs)
        self.is_searched = np.zeros(num_envs, dtype=bool)
        self.is_searched[self.envs] = True
        self.scope = scope
        self.refresh_every = int(refresh_every)
        self.batch = int(batch)
        self.max_wait = int(max_wait)
        self.device = device
        self._rng = random.Random(seed)
        #: The learner's side in each searched env (+1 US, -1 USSR), drawn per episode.
        self.side = np.ones(num_envs, dtype=np.int8)
        for e in self.envs:
            self.side[int(e)] = self._draw_side()
        self.snapshot = copy.deepcopy(learner).to(device).eval()
        for p in self.snapshot.parameters():
            p.requires_grad_(False)
        self.root = RolloutRoot(self.snapshot, self.config, device=device)
        self.root.reseed(seed + 1)
        self.next_refresh_step: int | None = None
        #: env -> steps it has waited so far.
        self.queue: Dict[int, int] = {}
        self._zero_counters()
        #: The learner's results against the searched opponent over the run, per learner side.
        self.wins = {1: 0.0, -1: 0.0}
        self.games = {1: 0.0, -1: 0.0}

    def _draw_side(self) -> int:
        return 1 if self._rng.random() < 0.5 else -1

    def _zero_counters(self) -> None:
        self.n_calls = 0
        self.n_searched = 0
        self.n_departures = 0
        self.n_held = 0
        self.n_opp_rows = 0
        self.root_seconds = 0.0

    # ---- who moves --------------------------------------------------------------------------------
    def learner_acts(self, decision_players: np.ndarray, learner_np: np.ndarray) -> np.ndarray:
        """`learner_np` (the pool's answer) with the searched envs' answer in place: the learner moves
        there only on its own side."""
        out = np.array(learner_np, dtype=bool, copy=True)
        dp = np.asarray(decision_players, dtype=np.int8)
        out[self.envs] = dp[self.envs] == self.side[self.envs]
        return out

    def opponent_rows(self, learner_np: np.ndarray) -> np.ndarray:
        """The env indices where the searched opponent is to move."""
        return np.flatnonzero(self.is_searched & ~np.asarray(learner_np, dtype=bool))

    # ---- the snapshot -----------------------------------------------------------------------------
    def maybe_refresh(self, learner: torch.nn.Module, steps: int) -> bool:
        """Copy the learner into the snapshot when its schedule says so (in place, so the root's and
        the rollout graphs' network object stays the same). The first call always refreshes."""
        if self.next_refresh_step is not None and steps < self.next_refresh_step:
            return False
        self.snapshot.load_state_dict(learner.state_dict())
        self.next_refresh_step = (int(steps) // self.refresh_every + 1) * self.refresh_every
        return True

    # ---- acting -----------------------------------------------------------------------------------
    def _searched_decision(self, st: ts.GameState, mask: np.ndarray) -> bool:
        if int(np.count_nonzero(mask)) < 2:
            return False
        return self.scope == "all" or is_entry_decision(st)

    def plan(self, runner: Any, masks: np.ndarray, opp_idx: np.ndarray,
             greedy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """This step's searched-opponent moves: (envs the root plays now, its moves there, the hold mask
        over all envs). `greedy[e]` is the snapshot's greedy move at env e, played at the opponent's
        other decisions by the caller and counted here as the root's departures."""
        self.n_opp_rows += int(len(opp_idx))
        for raw in opp_idx:
            e = int(raw)
            if e in self.queue:
                continue
            st = runner.get_state(e)
            if ts.Engine.is_terminal(st):
                continue
            if self._searched_decision(st, masks[e]):
                self.queue[e] = 0
        to_move = set(int(i) for i in opp_idx)
        for e in self.queue:
            if e not in to_move:
                raise RuntimeError(f"env {e} is queued for the searched opponent but its opponent is not to move: "
                                   f"a held game moved")
        hold = np.zeros(self.num_envs, dtype=bool)
        if not self.queue:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), hold
        if len(self.queue) >= self.batch or max(self.queue.values()) >= self.max_wait:
            envs = np.array(sorted(self.queue), dtype=np.int64)
            states = [runner.get_state(int(e)) for e in envs]
            t0 = time.perf_counter()
            picks = np.array(self.root.choose(states), dtype=np.int64)
            self.root_seconds += time.perf_counter() - t0
            for e, a in zip(envs, picks):
                if not masks[int(e)][int(a)]:
                    raise RuntimeError(f"the rollout root returned the illegal action {int(a)} in env {int(e)}")
            self.n_calls += 1
            self.n_searched += int(len(envs))
            self.n_departures += int(np.count_nonzero(picks != np.asarray(greedy)[envs]))
            self.queue.clear()
            return envs, picks, hold
        for e in self.queue:
            hold[e] = True
            self.queue[e] += 1
        self.n_held += int(hold.sum())
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), hold

    # ---- results ----------------------------------------------------------------------------------
    def on_episode_end(self, env_idx: int, victory_points: float) -> None:
        """Record the learner's result in a searched env and redraw its side; a no-op elsewhere."""
        if not self.is_searched[env_idx]:
            return
        side = int(self.side[env_idx])
        vp = float(victory_points)
        self.games[side] += 1.0
        self.wins[side] += 0.5 if vp == 0.0 else (1.0 if (vp > 0) == (side == 1) else 0.0)
        self.side[env_idx] = self._draw_side()

    def stats(self) -> Dict[str, float]:
        """This rollout's counters (then zeroed) and the run's results, as training_metrics.jsonl keys."""
        g = self.games[1] + self.games[-1]
        out = {
            "searched_opp_calls": float(self.n_calls),
            "searched_opp_positions": float(self.n_searched),
            "searched_opp_batch_mean": float(self.n_searched / max(1, self.n_calls)),
            "searched_opp_departure_rate": float(self.n_departures / max(1, self.n_searched)),
            "searched_opp_held_rows": float(self.n_held),
            "searched_opp_rows": float(self.n_opp_rows),
            "searched_opp_root_seconds": float(self.root_seconds),
            "searched_opp_games": float(g),
            "searched_opp_learner_win_rate": float((self.wins[1] + self.wins[-1]) / g) if g else float("nan"),
            "searched_opp_learner_win_rate_us": (float(self.wins[1] / self.games[1]) if self.games[1]
                                                 else float("nan")),
            "searched_opp_learner_win_rate_ussr": (float(self.wins[-1] / self.games[-1]) if self.games[-1]
                                                   else float("nan")),
        }
        self._zero_counters()
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {"wins": {str(k): v for k, v in self.wins.items()},
                "games": {str(k): v for k, v in self.games.items()}}

    def restore(self, state: Dict[str, Any] | None) -> None:
        """The run's results so far. The snapshot is refreshed from the learner at the first rollout
        after a resume (the lagged copy is not saved), and queued games start over with the envs."""
        if not state:
            return
        self.wins = {int(k): float(v) for k, v in state.get("wins", {}).items()} or self.wins
        self.games = {int(k): float(v) for k, v in state.get("games", {}).items()} or self.games
