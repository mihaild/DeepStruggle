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
from typing import Any, Callable, List, Optional, Sequence, Tuple

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
    #: P56, with roll faces: the distinct outcomes of each before-roll row's die (the runner's roll_outcomes) --
    #: the row each belongs to, its probability, whether it ends the game and with what result (the decider's
    #: side), its observation.
    face_row: Optional[np.ndarray] = None
    face_prob: Optional[np.ndarray] = None
    face_term: Optional[np.ndarray] = None
    face_res: Optional[np.ndarray] = None
    face_obs: Optional[torch.Tensor] = None
    #: P56: each row's value read at rollout time, and a before-roll row's rolls' average (NaN elsewhere).
    values: Optional[torch.Tensor] = None
    avg: Optional[torch.Tensor] = None


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

    def __init__(self, frac: float, features: int, width: int, forced: bool = True, seed: int = 0,
                 roll_faces: bool = False) -> None:
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"--afterstate-frac must be in (0, 1], got {frac}")
        self.frac = float(frac)
        #: P56: also capture each before-roll row's distinct die outcomes (rows' face_* fields).
        self.roll_faces = bool(roll_faces)
        self.features = int(features)
        self.width = int(width)
        self.forced = bool(forced)
        self.rng = np.random.default_rng(seed)
        self._on: Optional[Any] = None
        # On CUDA, two pinned host buffers the runner gathers into in turn, each copied to the device without
        # blocking; a buffer is reused only once its copy (recorded by the event) has finished.
        self._pinned: List[Optional[torch.Tensor]] = [None, None]
        self._copied: List[Optional[Any]] = [None, None]
        self._slot = 0
        self.reset_counters()

    def reset_counters(self) -> None:
        self.decisions = 0
        self.rows_by_kind = [0, 0, 0]
        self.mismatched = 0

    def before(self, runner: Any, learner: np.ndarray, deciders: np.ndarray) -> None:
        """Pick the learner decisions of this step whose moments the runner records as it applies them. `deciders`
        is +1 / -1 / 0 per env."""
        if self._on is not runner:
            runner.set_capture_afterstates(True, self.roll_faces)
            self._on = runner
        pick = (np.asarray(learner, dtype=bool) & (np.asarray(deciders) != 0)
                & (self.rng.random(len(deciders)) < self.frac))
        runner.set_afterstate_envs(np.ascontiguousarray(pick, dtype=np.uint8))
        self.decisions += int(pick.sum())

    def after(self, runner: Any, step: int, dones: np.ndarray, device: Any) -> Optional[AfterstateRows]:
        """The moments the runner recorded this step, filed under `step`, on `device`. `dones` is unused: a finished
        game's walk ended in its result, which is the target, and the runner kept it before the reset. (Copying each
        step's rows here is faster than one rollout-sized copy in build_targets: 81k against 69k steps/s.)"""
        if torch.device(device).type == "cuda":
            return self._after_pinned(runner, step, torch.device(device))
        envs, sides, kinds, obs, cut = runner.get_afterstates()
        self.mismatched += int(cut)
        kinds = np.asarray(kinds, dtype=np.int8)
        if kinds.size == 0:
            return None
        self.rows_by_kind = [n + int(c) for n, c in zip(self.rows_by_kind, np.bincount(kinds, minlength=3))]
        o = np.asarray(obs)
        # The buffer's width: rows are zero-padded to the widest set in a mixed batch.
        o = o[:, :self.width] if o.shape[1] >= self.width else np.pad(o, ((0, 0), (0, self.width - o.shape[1])))
        return self._with_faces(runner, AfterstateRows(
            step, np.asarray(envs, dtype=np.int64), np.asarray(sides, dtype=np.int8), kinds,
            torch.from_numpy(np.ascontiguousarray(o, dtype=np.float32)).to(device)), device)

    def _after_pinned(self, runner: Any, step: int, device: torch.device) -> Optional[AfterstateRows]:
        """after() on CUDA: the runner gathers the rows in parallel into a pinned buffer, copied to the device without
        blocking (0.6 ms a step at every decision with a fresh array and a pageable copy). The same rows."""
        k = self._slot
        self._slot ^= 1
        done = self._copied[k]
        if done is not None:
            done.synchronize()
        width = int(np.asarray(runner.get_observations()).shape[1])
        buf = self._pinned[k]
        if buf is None or buf.shape[1] != width:
            buf = torch.empty((1024, width), dtype=torch.float32).pin_memory()
        envs, sides, kinds, n, cut = runner.get_afterstates_into(buf.numpy())
        if n > buf.shape[0]:
            buf = torch.empty((2 * n, width), dtype=torch.float32).pin_memory()
            envs, sides, kinds, n, cut = runner.get_afterstates_into(buf.numpy())
        self._pinned[k] = buf
        self.mismatched += int(cut)
        kinds = np.asarray(kinds, dtype=np.int8)
        if n == 0:
            return None
        self.rows_by_kind = [m + int(c) for m, c in zip(self.rows_by_kind, np.bincount(kinds, minlength=3))]
        obs = buf[:n].to(device, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._copied[k] = event
        # The buffer's width: rows are zero-padded to the widest set in a mixed batch.
        if width > self.width:
            obs = obs[:, :self.width].contiguous()
        elif width < self.width:
            obs = torch.nn.functional.pad(obs, (0, self.width - width))
        return self._with_faces(runner, AfterstateRows(step, np.asarray(envs, dtype=np.int64),
                                                       np.asarray(sides, dtype=np.int8), kinds, obs), device)

    def _with_faces(self, runner: Any, rows: AfterstateRows, device: Any) -> AfterstateRows:
        """P56: the step's roll faces attached to its rows (at the buffer's width)."""
        if not self.roll_faces:
            return rows
        fr, fp, ft, fres, fo = runner.get_afterstate_faces()
        fo = np.asarray(fo)
        fo = fo[:, :self.width] if fo.shape[1] >= self.width else np.pad(fo, ((0, 0), (0, self.width - fo.shape[1])))
        rows.face_row = np.asarray(fr, dtype=np.int64)
        rows.face_prob = np.asarray(fp, dtype=np.float32)
        rows.face_term = np.asarray(ft, dtype=bool)
        rows.face_res = np.asarray(fres, dtype=np.float32)
        rows.face_obs = torch.from_numpy(np.ascontiguousarray(fo, dtype=np.float32)).to(device)
        return rows

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


def next_decision_targets(t: np.ndarray, e: np.ndarray, side: np.ndarray, players: np.ndarray, dones: np.ndarray,
                          rewards: np.ndarray, rets: np.ndarray, learner: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """`next_decision_target` for many rows at once: per row (buffer step t, env e, side +1 / -1) the target and
    whether it has one. Shared by the afterstate rows and the pre-deal critic's (nash_pg._build_predeal_targets)."""
    horizon = players.shape[0]
    done_at = _next_index(np.asarray(dones) > 0.5, strict=False)[t, e]
    dec_at = np.where(side == 1, _next_index(np.asarray(players) == 1, strict=True)[t, e],
                      _next_index(np.asarray(players) == -1, strict=True)[t, e])
    ends = (done_at < dec_at) & (done_at < horizon)
    decides = ~ends & (dec_at < horizon)
    dn, dd = np.minimum(done_at, horizon - 1), np.minimum(dec_at, horizon - 1)
    result = np.clip(np.asarray(rewards)[dn, e] * np.asarray(players)[dn, e] * side, -1.0, 1.0)
    keep = ends | (decides & np.asarray(learner, dtype=bool)[dd, e])
    return np.where(ends, result, np.asarray(rets)[dd, e]).astype(np.float32), keep


def read_node_values(rows: AfterstateRows, value_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
    """P56: fill rows.values (every row's value from its side, read by `value_fn` -- the rollout-time network) and
    rows.avg (a before-roll row's probability-weighted value over its die's outcomes, a finished game counting its
    result; NaN for a row without outcomes)."""
    n = int(rows.obs.shape[0])
    obs = rows.obs if rows.face_obs is None or rows.face_obs.shape[0] == 0 else torch.cat([rows.obs, rows.face_obs])
    with torch.no_grad():
        v = value_fn(obs).reshape(-1).float()
    rows.values = v[:n]
    avg = torch.full((n,), float("nan"), device=v.device)
    if rows.face_obs is not None and rows.face_obs.shape[0]:
        assert rows.face_row is not None and rows.face_prob is not None and rows.face_term is not None
        assert rows.face_res is not None
        dev = v.device
        fv = torch.where(torch.from_numpy(rows.face_term).to(dev), torch.from_numpy(rows.face_res).to(dev), v[n:])
        idx = torch.from_numpy(rows.face_row).to(dev)
        acc = torch.zeros(n, device=dev).index_add_(0, idx, torch.from_numpy(rows.face_prob).to(dev) * fv)
        has = torch.zeros(n, dtype=torch.bool, device=dev).index_fill_(0, idx, True)
        avg = torch.where(has, acc, avg)
    rows.avg = avg


def node_tensors(rows: Sequence[AfterstateRows], horizon: int, num_envs: int, device: Any
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """P56: (U, U_post), each (horizon, num_envs), for compute_gae's afterstate node (RolloutBuffer._gae_per_player).
    A decision's walk is its env's rows in its step. U is the value of the walk's first moment: the rolls' average
    when that moment is a roll's (the move is followed by a die), else the moment's own value. U_post is U, except
    after a die: the value after the roll that actually came (the next row), so the roll's own luck is left out of
    the advantage -- NaN when that state is no row (the side's own next decision, or the end of the game), where the
    step after the roll is zero. NaN in U: no afterstate, the plain step."""
    u = torch.full((horizon, num_envs), float("nan"), device=device)
    post = torch.full((horizon, num_envs), float("nan"), device=device)
    for r in rows:
        n = len(r.envs)
        if n == 0 or r.values is None:
            continue
        envs = np.asarray(r.envs)
        first = np.flatnonzero(np.r_[True, envs[1:] != envs[:-1]])
        kinds = np.asarray(r.kinds)
        vals = r.values
        avg = r.avg if r.avg is not None else torch.full((n,), float("nan"), device=vals.device)
        f_t = torch.from_numpy(first).to(vals.device)
        is_roll = torch.from_numpy(kinds[first] == BEFORE_ROLL).to(vals.device) & torch.isfinite(avg.index_select(0, f_t))
        nxt = np.minimum(first + 1, n - 1)
        after_ok = (first + 1 < n) & (envs[nxt] == envs[first]) & (kinds[nxt] == AFTER_ROLL)
        u_f = torch.where(is_roll, avg.index_select(0, f_t), vals.index_select(0, f_t))
        nan = torch.full_like(u_f, float("nan"))
        after_v = torch.where(torch.from_numpy(after_ok).to(vals.device),
                              vals.index_select(0, torch.from_numpy(nxt).to(vals.device)), nan)
        post_f = torch.where(is_roll, after_v, u_f)
        e_t = torch.from_numpy(envs[first]).to(device)
        u[r.step].index_copy_(0, e_t, u_f.to(device))
        post[r.step].index_copy_(0, e_t, post_f.to(device))
    return u, post


def build_targets(rows: Sequence[AfterstateRows], players: np.ndarray, dones: np.ndarray, rewards: np.ndarray,
                  rets: np.ndarray, learner: np.ndarray, device: Any
                  ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """(observations, targets, kinds) of the rows that have a target, or Nones."""
    rows = [r for r in rows if len(r.envs)]
    if not rows:
        return None, None, None
    t = np.concatenate([np.full(len(r.envs), r.step, dtype=np.int64) for r in rows])
    e = np.concatenate([r.envs for r in rows]).astype(np.int64)
    side = np.concatenate([r.sides for r in rows]).astype(np.int64)
    kinds = np.concatenate([r.kinds for r in rows]).astype(np.int64)
    target, keep = next_decision_targets(t, e, side, players, dones, rewards, rets, learner)
    if any(r.avg is not None for r in rows):
        # P56: a before-roll row's target is its die's outcomes' average, read at rollout time.
        ovr = np.concatenate([r.avg.detach().float().cpu().numpy() if r.avg is not None
                              else np.full(len(r.envs), np.nan, dtype=np.float32) for r in rows])
        has = np.isfinite(ovr)
        target = np.where(has, ovr, target).astype(np.float32)
        keep = keep | has
    if not keep.any():
        return None, None, None
    idx = torch.from_numpy(np.flatnonzero(keep)).to(rows[0].obs.device)
    obs = torch.cat([r.obs for r in rows]).index_select(0, idx)
    return (obs, torch.from_numpy(target[keep]).to(device), torch.from_numpy(kinds[keep]).to(device))
