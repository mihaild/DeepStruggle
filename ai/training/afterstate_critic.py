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

The states are reconstructed by replaying the side's action on a copy of the pre-step state and
walking the engine one node at a time (`Engine.auto_advance_step(state, 1)`), the way the runner
does with forced settling. The walk ends at the next real decision, which must be the runner's own
next state; a row whose replay disagrees with the runner is dropped and counted (it would mean the
copy and the runner took different chance draws).
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


def _signature(state: Any) -> Tuple[int, ...]:
    """Enough of a state to tell two walks apart: the RNG and where the game is."""
    ctx = state.ctx()
    return (int(state.rng_state), int(state.turn), int(state.action_round), int(state.victory_points),
            int(state.defcon), int(ctx.decision_type), int(ctx.decision_player), int(state.current_phase))


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
    """Samples a fraction of the learner's decisions each step and files their moments.

    `before()` runs before the env step (it needs the pre-step states and the chosen actions),
    `after()` right after it (it checks each walk against the runner's next state)."""

    def __init__(self, frac: float, features: int, width: int, forced: bool = True, seed: int = 0) -> None:
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"--afterstate-frac must be in (0, 1], got {frac}")
        self.frac = float(frac)
        self.features = int(features)
        self.width = int(width)
        self.forced = bool(forced)
        self.rng = np.random.default_rng(seed)
        self._pending: List[Tuple[int, int, List[Tuple[int, Any]], Tuple[int, ...]]] = []
        self.reset_counters()

    def reset_counters(self) -> None:
        self.decisions = 0
        self.rows_by_kind = [0, 0, 0]
        self.mismatched = 0

    def before(self, runner: Any, learner: np.ndarray, deciders: np.ndarray, actions: np.ndarray,
               merged: Optional[np.ndarray] = None, macro: Optional[np.ndarray] = None) -> None:
        """Walk the sampled learner decisions of this step. `deciders` is +1 / -1 / 0 per env."""
        self._pending = []
        pick = np.flatnonzero(np.asarray(learner, dtype=bool) & (np.asarray(deciders) != 0)
                              & (self.rng.random(len(deciders)) < self.frac))
        for e in pick:
            e = int(e)
            side = ts.Player.US if int(deciders[e]) == 1 else ts.Player.USSR
            st = runner.get_state(e)
            moments, end = walk_afterstates(st, int(actions[e]), side,
                                            bool(merged[e]) if merged is not None else False,
                                            bool(macro[e]) if macro is not None else False, self.forced)
            self.decisions += 1
            if moments:
                self._pending.append((e, int(deciders[e]), moments, _signature(end)))

    def after(self, runner: Any, step: int, dones: np.ndarray, device: Any) -> Optional[AfterstateRows]:
        """The walks of this step that the runner agrees with, observed and filed under `step`."""
        envs: List[int] = []
        sides: List[int] = []
        kinds: List[int] = []
        obs: List[np.ndarray] = []
        for e, side_code, moments, sig in self._pending:
            # A finished game has auto-reset: nothing to compare with, and the moments are kept
            # (the walk ended in the result, which is what the target will be).
            if not bool(dones[e]) and _signature(runner.get_state(e)) != sig:
                self.mismatched += 1
                continue
            side = ts.Player.US if side_code == 1 else ts.Player.USSR
            for kind, s in moments:
                envs.append(e)
                sides.append(side_code)
                kinds.append(kind)
                o = np.asarray(ts.extract_observation_features(s, side, self.features), dtype=np.float32)
                # The buffer's width: rows are zero-padded to the widest set in a mixed batch.
                obs.append(o[:self.width] if o.shape[0] >= self.width else np.pad(o, (0, self.width - o.shape[0])))
                self.rows_by_kind[kind] += 1
        self._pending = []
        if not obs:
            return None
        return AfterstateRows(step, np.asarray(envs, dtype=np.int64), np.asarray(sides, dtype=np.int8),
                              np.asarray(kinds, dtype=np.int8),
                              torch.from_numpy(np.stack(obs)).to(device))

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


def build_targets(rows: Sequence[AfterstateRows], players: np.ndarray, dones: np.ndarray, rewards: np.ndarray,
                  rets: np.ndarray, learner: np.ndarray, device: Any
                  ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """(observations, targets, kinds) of the rows that have a target, or Nones."""
    obs: List[torch.Tensor] = []
    tgt: List[float] = []
    kinds: List[int] = []
    for r in rows:
        for j in range(len(r.envs)):
            target = next_decision_target(r.step, int(r.envs[j]), int(r.sides[j]), players, dones, rewards, rets,
                                          learner)
            if target is not None:
                obs.append(r.obs[j])
                tgt.append(target)
                kinds.append(int(r.kinds[j]))
    if not obs:
        return None, None, None
    return (torch.stack(obs), torch.tensor(tgt, dtype=torch.float32, device=device),
            torch.tensor(kinds, dtype=torch.int64, device=device))
