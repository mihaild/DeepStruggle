"""The critic at every moment of a side's turn, not only where it decides (owner, 2026-11-18).

The critic is trained only on the positions where its side is to decide, so its value of anything
else -- the position a move leaves, or one waiting on a die -- is an extrapolation. A player that
picks the move whose resulting position the critic likes best (`CriticArgmaxAgent`) reads exactly
those positions. This module adds them to the value loss:

* **after the decision**: the state the engine is in right after the side's action is applied,
  before anything else happens;
* **before a die roll**: each chance node the engine then reaches on the way to the next decision
  (the same state as the previous one when the roll comes at once);
* **after a die roll**: the state each roll leaves.

Each is observed from the deciding side's perspective. Positions that are that side's own next
decision are skipped (the ordinary value loss already has them), and so are finished games. The
target is the pre-deal critic's (P33): the lambda-return of the same side's next decision in the
same env, or the game's result if it ends first; a row whose next decision falls outside the
rollout, or is an opponent-pool member's, is dropped.

The batch runner records the moments itself as it steps (`set_capture_afterstates`,
`bindings/ts_bindings.cpp`, 2026-11-20): it walks its own game one node at a time
(`Engine.auto_advance_step(state, 1)`, what its bulk advance repeats), in its parallel loop, so the
decision stream is unchanged and nothing is replayed. `walk_afterstates` below is the reference walk
on a copy, which `tests/training/test_afterstate_critic.py` holds the runner's rows to; a walk longer
than MAX_WALK nodes is dropped and counted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
import ts_engine as ts

#: The kinds of moment, in the order a turn meets them.
AFTER_DECISION, BEFORE_ROLL, AFTER_ROLL = 0, 1, 2
KIND_NAMES = ("after_decision", "before_roll", "after_roll")
#: A walk longer than this is cut (events chain a few forced steps and rolls; none chain hundreds).
MAX_WALK = 64


@dataclass
class AfterstateRows:
    """One rollout step's captured moments until the update: the buffer step, and per row the env,
    the side (+1 US, -1 USSR), the kind, and the observation from that side."""
    step: int
    envs: np.ndarray
    sides: np.ndarray
    kinds: np.ndarray
    obs: torch.Tensor


def _is_chance(state: Any) -> bool:
    ctx = state.ctx()
    return ctx.decision_player == ts.Player.NONE and ctx.decision_type == ts.DecisionType.ROLL_DIE


def walk_afterstates(state: Any, action: int, side: Any, merged: bool = False, macro: bool = False,
                     forced: bool = True) -> Tuple[List[Tuple[int, Any]], Any]:
    """The moments of `side`'s turn after it plays flat `action` in `state` (not modified): a list of
    (kind, state copy), and the state the walk ends in -- where the runner would stop next. `forced` is
    the env's settle mode: True steps through single-choice decisions as well as chance nodes
    (SettleMode.FORCED, the default), False through chance nodes only."""
    st = state.clone()
    ts.Engine.step_flat(st, int(action), False, bool(merged), bool(macro))
    out: List[Tuple[int, Any]] = []

    def keep(kind: int, s: Any) -> None:
        if ts.Engine.is_terminal(s):
            return
        ctx = s.ctx()
        # The side's own next decision is an ordinary value-loss row already.
        if ctx.decision_player == side and not _is_chance(s) and (not forced or _settled(s)):
            return
        out.append((kind, s.clone()))

    keep(AFTER_DECISION, st)
    fresh = True                     # nothing has happened since the last state kept
    for _ in range(MAX_WALK):
        if ts.Engine.is_terminal(st):
            break
        if _is_chance(st):
            if not fresh:
                keep(BEFORE_ROLL, st)
            elif out and out[-1][0] == AFTER_DECISION:
                # The decision's own state is this roll's "before": record it under both readings
                # once, as before-roll (the more specific).
                out[-1] = (BEFORE_ROLL, out[-1][1])
            ts.Engine.auto_advance_step(st, 1)
            keep(AFTER_ROLL, st)
            fresh = True
            continue
        if not forced or ts.Engine.auto_advance_step(st, 1) == 0:
            break                    # a decision the runner presents: the walk is over
        fresh = False
    return out, st


def _settled(state: Any) -> bool:
    """True if the runner would stop here: a decision with more than one legal move."""
    probe = state.clone()
    return ts.Engine.auto_advance_step(probe, 1) == 0


class AfterstateCapture:
    """Samples a fraction of the learner's decisions each step; the runner files their moments.

    `before()` runs before the env step (it picks the envs), `after()` right after it (it reads the rows)."""

    def __init__(self, frac: float, features: int, width: int, forced: bool = True, seed: int = 0) -> None:
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"--afterstate-frac must be in (0, 1], got {frac}")
        self.frac = float(frac)
        self.features = int(features)
        self.width = int(width)
        self.forced = bool(forced)
        self.rng = np.random.default_rng(seed)
        self._on: Optional[Any] = None
        self.reset_counters()

    def reset_counters(self) -> None:
        self.decisions = 0
        self.rows_by_kind = [0, 0, 0]
        self.mismatched = 0

    def before(self, runner: Any, learner: np.ndarray, deciders: np.ndarray) -> None:
        """Pick the learner decisions of this step whose moments the runner records as it applies them. `deciders`
        is +1 / -1 / 0 per env."""
        if self._on is not runner:
            runner.set_capture_afterstates(True)
            self._on = runner
        pick = (np.asarray(learner, dtype=bool) & (np.asarray(deciders) != 0)
                & (self.rng.random(len(deciders)) < self.frac))
        runner.set_afterstate_envs(np.ascontiguousarray(pick, dtype=np.uint8))
        self.decisions += int(pick.sum())

    def after(self, runner: Any, step: int, dones: np.ndarray, device: Any) -> Optional[AfterstateRows]:
        """The moments the runner recorded this step, filed under `step`, on `device`. `dones` is unused: a finished
        game's walk ended in its result, which is the target, and the runner kept it before the reset. (Copying each
        step's rows here is faster than one rollout-sized copy in build_targets: 81k against 69k steps/s.)"""
        envs, sides, kinds, obs, cut = runner.get_afterstates()
        self.mismatched += int(cut)
        kinds = np.asarray(kinds, dtype=np.int8)
        if kinds.size == 0:
            return None
        self.rows_by_kind = [n + int(c) for n, c in zip(self.rows_by_kind, np.bincount(kinds, minlength=3))]
        o = np.asarray(obs)
        # The buffer's width: rows are zero-padded to the widest set in a mixed batch.
        o = o[:, :self.width] if o.shape[1] >= self.width else np.pad(o, ((0, 0), (0, self.width - o.shape[1])))
        return AfterstateRows(step, np.asarray(envs, dtype=np.int64), np.asarray(sides, dtype=np.int8), kinds,
                              torch.from_numpy(np.ascontiguousarray(o, dtype=np.float32)).to(device))

    def metrics(self) -> dict:
        out = {"afterstate/decisions": float(self.decisions), "afterstate/mismatched": float(self.mismatched)}
        for k, name in enumerate(KIND_NAMES):
            out[f"afterstate/rows_{name}"] = float(self.rows_by_kind[k])
        self.reset_counters()
        return out


def next_decision_target(t: int, e: int, side: int, players: np.ndarray, dones: np.ndarray,
                         rewards: np.ndarray, rets: np.ndarray, learner: np.ndarray) -> Optional[float]:
    """The value target of a position `side` reaches between buffer steps t and t+1 in env e: the
    lambda-return of `side`'s next decision in that env (a learner row), or the game's result if it
    ends first; None if neither falls inside the buffer or the next decision is a pool member's.
    The pre-deal critic's rule (P33), shared."""
    horizon = players.shape[0]
    for d in range(t, horizon):
        if dones[d, e]:
            return float(np.clip(rewards[d, e] * players[d, e] * side, -1.0, 1.0))
        if d + 1 < horizon and players[d + 1, e] == side:
            return float(rets[d + 1, e]) if learner[d + 1, e] else None
    return None


def _next_index(flags: np.ndarray, strict: bool) -> np.ndarray:
    """Per (t, e): the first d (> t if strict, else >= t) with flags[d, e], or the horizon."""
    horizon, n = flags.shape
    out = np.empty((horizon, n), dtype=np.int64)
    cur = np.full(n, horizon, dtype=np.int64)
    for t in range(horizon - 1, -1, -1):
        if strict:
            out[t] = cur
            cur = np.where(flags[t], t, cur)
        else:
            cur = np.where(flags[t], t, cur)
            out[t] = cur
    return out


def build_targets(rows: Sequence[AfterstateRows], players: np.ndarray, dones: np.ndarray, rewards: np.ndarray,
                  rets: np.ndarray, learner: np.ndarray, device: Any
                  ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """(observations, targets, kinds) of the rows that have a target, or Nones: `next_decision_target` for every
    row at once."""
    rows = [r for r in rows if len(r.envs)]
    if not rows:
        return None, None, None
    horizon = players.shape[0]
    t = np.concatenate([np.full(len(r.envs), r.step, dtype=np.int64) for r in rows])
    e = np.concatenate([r.envs for r in rows]).astype(np.int64)
    side = np.concatenate([r.sides for r in rows]).astype(np.int64)
    kinds = np.concatenate([r.kinds for r in rows]).astype(np.int64)
    done_at = _next_index(np.asarray(dones) > 0.5, strict=False)[t, e]
    dec_at = np.where(side == 1, _next_index(np.asarray(players) == 1, strict=True)[t, e],
                      _next_index(np.asarray(players) == -1, strict=True)[t, e])
    ends = (done_at < dec_at) & (done_at < horizon)
    decides = ~ends & (dec_at < horizon)
    dn, dd = np.minimum(done_at, horizon - 1), np.minimum(dec_at, horizon - 1)
    result = np.clip(np.asarray(rewards)[dn, e] * np.asarray(players)[dn, e] * side, -1.0, 1.0)
    keep = ends | (decides & np.asarray(learner, dtype=bool)[dd, e])
    target = np.where(ends, result, np.asarray(rets)[dd, e]).astype(np.float32)
    if not keep.any():
        return None, None, None
    idx = torch.from_numpy(np.flatnonzero(keep)).to(rows[0].obs.device)
    obs = torch.cat([r.obs for r in rows]).index_select(0, idx)
    return (obs, torch.from_numpy(target[keep]).to(device), torch.from_numpy(kinds[keep]).to(device))
