"""The critic at every moment of a side's turn, not only where it decides (owner, 2026-11-18).

The critic is trained only on the positions where its side is to decide, so its value of anything
else -- the position a move leaves, or one waiting on a die -- is an extrapolation. A player that
picks the move whose resulting position the critic likes best (`CriticArgmaxAgent`) reads exactly
those positions. This module adds them to the value loss:

* **after the decision**: the state the engine is in right after the side's action is applied,
  before anything else happens;
* **before a die roll**: each chance node the engine then reaches on the way to the next decision
  (the same state as the previous one when the roll comes at once: the decision's own row is relabelled,
  and a die right after another is kept again as a before-roll row next to the first roll's after-roll);
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

**The value before a roll is blind to the move that led there** (P56 log section 6, 2026-11-22): the
observation does not show which roll is pending, so a coup in one country and in another are the same
input before the die -- identical for every roll move at 75% of decisions. The before-roll row can only
learn the average over the targets the policy picks. Anything that needs the value of a position waiting
on a die should take its outcomes' exact average (`read_node_values`, the runner's roll faces), as the
afterstate advantage (`--afterstate-advantage`, `node_tensors`) does.

**Never train or read the critic at a state waiting on a die (owner, 2026-11-22).** Under `--afterstate-advantage` the
runner captures with roll faces and walks exactly (`walk_afterstates(exact=True)`, the runner's
walk_afterstates_exact; rows marked `exact`):

1. A state waits on a die when it is at one or reaches one through single-choice steps alone (`die_ahead`,
   `ts.die_ahead`). The before-roll rows are exactly those states, each kept only at its die; no other row waits on
   one. They are nodes of the advantage and are neither read (values NaN) nor trained on (`build_targets` drops
   them; the trainer reports `afterstate_trained_<kind>` beside the capture's `afterstate/rows_<kind>`).
2. Dice that follow each other with no decision between them (an Ops-first coup or realignment on Korean War or
   Arab-Israeli War whose owed Event rolls at once, two rolling headlines) are one compound chance node: one
   before-roll row whose outcomes are the second die's outcomes of each outcome of the first, recursively,
   probabilities multiplied (`ts.roll_outcomes(compound=True)`, bindings/chance_outcomes.hpp; the reference is the
   nested brute force `reference_roll_outcomes`), and whose next row is the state after its LAST die. The states
   between are neither kept nor read, so `node_tensors` drops the luck of every die of the node. A chain longer than
   ts.MAX_ROLL_CHAIN dice or with more than ts.MAX_COMPOUND_OUTCOMES outcomes raises (never truncated). Measured
   2026-11-22: 0.9% of rolls on random play, 0.6% on R169's self-play, at most 41 distinct outcome states (12 after
   merging by observation).
3. Outside training, the critic's-best-move player can read before a die and takes the exact average behind its
   opt-in `critic:...:exact`; search leaves are read only after chance is drained (the P56 log, 2026-11-22).

Without `--afterstate-advantage` (R169's afterstate critic) the first walk and its training are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

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
    #: P56, the exact walk (roll faces on; owner, 2026-11-22): the before-roll rows are exactly the rows that wait on a
    #: die, their outcomes are compound, and they are neither read (values NaN) nor trained on.
    exact: bool = False


def _is_chance(state: Any) -> bool:
    ctx = state.ctx()
    return ctx.decision_player == ts.Player.NONE and ctx.decision_type == ts.DecisionType.ROLL_DIE


def walk_afterstates(state: Any, action: int, side: Any, merged: bool = False, macro: bool = False,
                     forced: bool = True, exact: bool = False) -> Tuple[List[Tuple[int, Any]], Any]:
    """The moments of `side`'s turn after it plays flat `action` in `state` (not modified): a list of
    (kind, state copy), and the state the walk ends in -- where the runner would stop next. `forced` is
    the env's settle mode: True steps through single-choice decisions as well as chance nodes
    (SettleMode.FORCED, the default), False through chance nodes only. `exact`: the runner's walk with roll faces
    (`_walk_exact`)."""
    st = state.clone()
    ts.Engine.step_flat(st, int(action), False, bool(merged), bool(macro))
    if exact:
        return _walk_exact(st, side, forced)
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
            elif out and out[-1][0] == AFTER_ROLL and _rolls_a_die(st):
                # A die right after another: the previous roll's after-state is this one's before-state too, kept
                # again as before-roll so node_tensors pairs it as a later roll and drops its luck (fix verification
                # V-4, 2026-11-22). The after-roll row stays (the previous roll's result); turn cleanup rolls nothing.
                keep(BEFORE_ROLL, st)
            ts.Engine.auto_advance_step(st, 1)
            keep(AFTER_ROLL, st)
            fresh = True
            continue
        if not forced or ts.Engine.auto_advance_step(st, 1) == 0:
            break                    # a decision the runner presents: the walk is over
        fresh = False
    return out, st


def _rolls_a_die(state: Any) -> bool:
    """A chance node that rolls a die (the runner's roll_outcomes enumerates it); turn cleanup rolls nothing."""
    return _is_chance(state) and state.ctx().pending_roll_type not in (ts.RollType.NONE, ts.RollType.TURN_CLEANUP)


def die_ahead(state: Any, forced: bool = True) -> Optional[Any]:
    """If `state` waits on a die -- it is at one, or (`forced`) reaches one through single-choice decisions alone, the
    steps the runner takes by itself -- a copy of the state at that die; otherwise None. The reference of the
    runner's ts.die_ahead (bindings/chance_outcomes.hpp), which tests hold to it."""
    if ts.Engine.is_terminal(state):
        return None
    if _rolls_a_die(state):
        return state.clone()
    if not forced or _is_chance(state):
        return None
    probe = state.clone()
    for _ in range(MAX_WALK):
        if ts.Engine.auto_advance_step(probe, 1) == 0 or ts.Engine.is_terminal(probe):
            return None
        if _is_chance(probe):
            return probe if _rolls_a_die(probe) else None
    raise RuntimeError(f"die_ahead: more than {MAX_WALK} single-choice steps")


def _walk_exact(st: Any, side: Any, forced: bool) -> Tuple[List[Tuple[int, Any]], Any]:
    """The exact walk (P56 with roll faces; owner, 2026-11-22), from `st` right after the decision: no row waits on a
    die but the before-roll rows, each at its die and standing for the whole compound chance node that starts there
    (`reference_roll_outcomes`): the dice it covers get no row, the states between them are not kept, and the row
    after its last roll is its after-roll. A decision's own state that reaches a die through single-choice steps is
    not kept (the die's row stands for it); one at the turn-end park stays an after-decision row. The runner's
    walk_afterstates_exact."""
    out: List[Tuple[int, Any]] = []

    def keep(kind: int, s: Any) -> None:
        if ts.Engine.is_terminal(s):
            return
        ctx = s.ctx()
        if ctx.decision_player == side and not _is_chance(s) and (not forced or _settled(s)):
            return
        out.append((kind, s.clone()))

    if _rolls_a_die(st):
        keep(BEFORE_ROLL, st)
    elif die_ahead(st, forced) is None:
        keep(AFTER_DECISION, st)
    chained = False                  # the die ahead belongs to a compound node already kept
    for k in range(MAX_WALK):
        if ts.Engine.is_terminal(st):
            break
        if _is_chance(st):
            die = _rolls_a_die(st)
            if die and not chained and k > 0:
                keep(BEFORE_ROLL, st)
            ts.Engine.auto_advance_step(st, 1)
            waits = die_ahead(st, forced) is not None
            chained = waits and die
            if not waits:
                keep(AFTER_ROLL, st)
            continue
        if not forced or ts.Engine.auto_advance_step(st, 1) == 0:
            break
    return out, st


_TWO_DICE = (ts.RollType.REALIGNMENT, ts.RollType.SUMMIT, ts.RollType.OLYMPIC_GAMES)


def reference_roll_outcomes(state: Any, side: Any, features: int = 0, forced: bool = True
                            ) -> List[Tuple[float, bool, float, np.ndarray]]:
    """The compound chance node at `state` (a die roll), by brute force: every face -- all 36 pairs for a realignment,
    the Summit and the Olympic Games (an Olympics tie, re-rolled, is no outcome; the rest renormalised), 6 faces
    otherwise -- stepped with the forced die, and an outcome that waits on another die (`die_ahead`) expanded into that
    die's faces, recursively, probabilities multiplied (float64). Outcomes merged as the runner merges them: equal
    observations from `side`, or the same result where the game ends. A list of (probability, finished, result from
    `side`, observation -- zeros for a finished game). The reference of ts.roll_outcomes(compound=True), whose
    two-dice enumeration by the dice's difference it checks."""
    sign = 1.0 if side == ts.Player.US else -1.0
    finals: List[Tuple[float, Any]] = []

    def expand(g: Any, weight: float, depth: int) -> None:
        if depth >= int(ts.MAX_ROLL_CHAIN):
            raise RuntimeError("reference_roll_outcomes: a chain longer than MAX_ROLL_CHAIN dice")
        rt = g.ctx().pending_roll_type
        if rt in _TWO_DICE:
            faces = [(a, b, 1.0 / 36.0) for a in range(1, 7) for b in range(1, 7)
                     if not (rt == ts.RollType.OLYMPIC_GAMES and a + 2 == b)]
        else:
            faces = [(a, 0, 1.0 / 6.0) for a in range(1, 7)]
        kids: List[Tuple[Any, float]] = []
        for a, b, p in faces:
            c = g.clone()
            try:
                ts.Engine.step(c, ts.MicroAction(ts.DecisionType.ROLL_DIE, a, b, 0), False)
            except RuntimeError:
                continue
            kids.append((c, p))
        kept = sum(p for _, p in kids)
        if not kids:
            raise RuntimeError("reference_roll_outcomes: a die with no outcome the engine accepts")
        for c, p in kids:
            nxt = die_ahead(c, forced)
            if nxt is not None:
                expand(nxt, weight * p / kept, depth + 1)
            else:
                finals.append((weight * p / kept, c))

    if not _rolls_a_die(state):
        return []
    expand(state, 1.0, 0)
    width = int(ts.obs_size_for(features))
    merged: List[List[Any]] = []
    for p, c in finals:
        if ts.Engine.is_terminal(c):
            res = sign * float(ts.Engine.get_terminal_utility(c))
            hit = next((m for m in merged if m[1] and m[2] == res), None)
            if hit is None:
                merged.append([p, True, res, np.zeros(width, dtype=np.float32)])
            else:
                hit[0] += p
            continue
        o = np.asarray(ts.extract_observation_features(c, side, features), dtype=np.float32)
        hit = next((m for m in merged if not m[1] and np.array_equal(m[3], o)), None)
        if hit is None:
            merged.append([p, False, 0.0, o])
        else:
            hit[0] += p
    return [(float(m[0]), bool(m[1]), float(m[2]), m[3]) for m in merged]


def _settled(state: Any) -> bool:
    """True if the runner would stop here: a decision with more than one legal move."""
    probe = state.clone()
    return ts.Engine.auto_advance_step(probe, 1) == 0


class AfterstateCapture:
    """Samples a fraction of the learner's decisions each step; the runner files their moments.

    `before()` runs before the env step (it picks the envs), `after()` right after it (it reads the rows)."""

    def __init__(self, frac: float, features: int, width: int, forced: bool = True, seed: int = 0,
                 roll_faces: bool = False, every_decider: bool = False) -> None:
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"--afterstate-frac must be in (0, 1], got {frac}")
        if every_decider and frac < 1.0:
            raise ValueError("every_decider captures every decision's walk; it needs frac 1.0")
        self.frac = float(frac)
        #: --per-player-gae (fix verification V-5, 2026-11-22): every decision's walk, not only the learner's, so each
        #: side's chain has its own last moment to bootstrap from (build_targets still trains on learner walks only).
        self.every_decider = bool(every_decider)
        #: P56: also capture each before-roll row's distinct die outcomes (rows' face_* fields).
        self.roll_faces = bool(roll_faces)
        self.features = int(features)
        self.width = int(width)
        self.forced = bool(forced)
        self.rng = np.random.default_rng(seed)
        self._on: Optional[Any] = None
        # P56 on CUDA: the roll outcomes' own two pinned buffers, as for the rows below.
        self._fpinned: List[Optional[torch.Tensor]] = [None, None]
        self._fcopied: List[Optional[Any]] = [None, None]
        self._fslot = 0
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
        if self.every_decider:
            pick = np.asarray(deciders) != 0
        else:
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
            buf = torch.empty((4096 if self.roll_faces else 1024, width), dtype=torch.float32).pin_memory()
        faces: Optional[Tuple[Any, Any, Any, Any]] = None
        nf = 0
        if self.roll_faces:
            # P56: the rows and their rolls' outcomes in one gather, one buffer and one copy.
            envs, sides, kinds, n, cut, fr, fp, ft, fres, nf = runner.get_afterstates_with_faces_into(buf.numpy())
            if n + nf > buf.shape[0]:
                buf = torch.empty((2 * (n + nf), width), dtype=torch.float32).pin_memory()
                envs, sides, kinds, n, cut, fr, fp, ft, fres, nf = runner.get_afterstates_with_faces_into(buf.numpy())
            faces = (fr, fp, ft, fres)
        else:
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
        obs = buf[:n + nf].to(device, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._copied[k] = event
        # The buffer's width: rows are zero-padded to the widest set in a mixed batch.
        if width > self.width:
            obs = obs[:, :self.width].contiguous()
        elif width < self.width:
            obs = torch.nn.functional.pad(obs, (0, self.width - width))
        rows = AfterstateRows(step, np.asarray(envs, dtype=np.int64), np.asarray(sides, dtype=np.int8), kinds, obs[:n])
        if faces is not None:
            fr, fp, ft, fres = faces
            rows.face_row = np.asarray(fr, dtype=np.int64)
            rows.face_prob = np.asarray(fp, dtype=np.float32)
            rows.face_term = np.asarray(ft, dtype=bool)
            rows.face_res = np.asarray(fres, dtype=np.float32)
            rows.face_obs = obs[n:]
            rows.exact = True
        return rows

    def _with_faces(self, runner: Any, rows: AfterstateRows, device: Any) -> AfterstateRows:
        """P56: the step's roll faces attached to its rows (at the buffer's width)."""
        if not self.roll_faces:
            return rows
        dev = torch.device(device)
        if dev.type == "cuda":
            # Gathered in parallel into a pinned buffer and copied without blocking, like the rows.
            k = self._fslot
            self._fslot ^= 1
            done = self._fcopied[k]
            if done is not None:
                done.synchronize()
            width = int(runner.obs_width)
            buf = self._fpinned[k]
            if buf is None or buf.shape[1] != width:
                buf = torch.empty((4096, width), dtype=torch.float32).pin_memory()
            fr, fp, ft, fres, n = runner.get_afterstate_faces_into(buf.numpy())
            if n > buf.shape[0]:
                buf = torch.empty((2 * n, width), dtype=torch.float32).pin_memory()
                fr, fp, ft, fres, n = runner.get_afterstate_faces_into(buf.numpy())
            self._fpinned[k] = buf
            fo_t = buf[:n].to(dev, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
            self._fcopied[k] = event
            if width > self.width:
                fo_t = fo_t[:, :self.width].contiguous()
            elif width < self.width:
                fo_t = torch.nn.functional.pad(fo_t, (0, self.width - width))
        else:
            fr, fp, ft, fres, fo = runner.get_afterstate_faces()
            fo = np.asarray(fo)
            fo = fo[:, :self.width] if fo.shape[1] >= self.width else np.pad(fo, ((0, 0), (0, self.width - fo.shape[1])))
            fo_t = torch.from_numpy(np.ascontiguousarray(fo, dtype=np.float32)).to(dev)
        rows.face_row = np.asarray(fr, dtype=np.int64)
        rows.face_prob = np.asarray(fp, dtype=np.float32)
        rows.face_term = np.asarray(ft, dtype=bool)
        rows.face_res = np.asarray(fres, dtype=np.float32)
        rows.face_obs = fo_t
        rows.exact = True
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


def _read_mask(rows: AfterstateRows) -> Optional[np.ndarray]:
    """The rows whose value is read, or None for all: the exact walk's before-roll rows wait on a die, so the critic is
    never asked about them (their value is their outcomes' average)."""
    if not rows.exact:
        return None
    return np.asarray(rows.kinds) != BEFORE_ROLL


def _node_inputs(rows: AfterstateRows) -> List[torch.Tensor]:
    """The observations read for `rows`, in order: the rows read (`_read_mask`), then the outcomes."""
    keep = _read_mask(rows)
    obs = rows.obs if keep is None else rows.obs.index_select(0, torch.from_numpy(np.flatnonzero(keep)).to(rows.obs.device))
    parts = [obs]
    if rows.face_obs is not None and rows.face_obs.shape[0]:
        parts.append(rows.face_obs)
    return parts


def _node_count(rows: AfterstateRows) -> int:
    keep = _read_mask(rows)
    n = int(rows.obs.shape[0]) if keep is None else int(keep.sum())
    return n + (0 if rows.face_obs is None else int(rows.face_obs.shape[0]))


def read_node_values(rows: AfterstateRows, value_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
    """P56: fill rows.values (every row's value from its side, read by `value_fn` -- the rollout-time network; NaN at
    the exact walk's before-roll rows, which wait on a die and are never read) and rows.avg (a before-roll row's
    probability-weighted value over its die's outcomes -- compound in the exact walk --, a finished game counting its
    result; NaN for a row without outcomes)."""
    parts = _node_inputs(rows)
    obs = parts[0] if len(parts) == 1 else torch.cat(parts)
    with torch.no_grad():
        v = value_fn(obs).reshape(-1).float()
    fill_node_values(rows, v)


def fill_node_values(rows: AfterstateRows, v: torch.Tensor) -> None:
    """read_node_values from the values already read: `v` holds the rows' read (`_read_mask`) then the outcomes'
    values, in order. Done on the host (a few numpy operations; on the GPU these were many tiny launches a row);
    rows.values and rows.avg are CPU tensors."""
    n = int(rows.obs.shape[0])
    vn = v.detach().float().cpu().numpy()
    keep = _read_mask(rows)
    if keep is None:
        vals, fv_read = vn[:n], vn[n:]
    else:
        m = int(keep.sum())
        vals = np.full(n, np.nan, dtype=np.float32)
        vals[keep] = vn[:m]
        fv_read = vn[m:]
    rows.values = torch.from_numpy(np.ascontiguousarray(vals, dtype=np.float32))
    avg = np.full(n, np.nan, dtype=np.float32)
    if rows.face_obs is not None and rows.face_obs.shape[0]:
        assert rows.face_row is not None and rows.face_prob is not None and rows.face_term is not None
        assert rows.face_res is not None
        fv = np.where(rows.face_term, rows.face_res, fv_read).astype(np.float32)
        acc = np.bincount(rows.face_row, weights=rows.face_prob * fv, minlength=n)[:n].astype(np.float32)
        has = np.bincount(rows.face_row, minlength=n)[:n] > 0
        avg = np.where(has, acc, avg).astype(np.float32)
    if rows.exact and np.any((np.asarray(rows.kinds) == BEFORE_ROLL) & ~np.isfinite(avg)):
        raise ValueError("the exact walk kept a before-roll row without outcomes: its value would be read blind")
    rows.avg = torch.from_numpy(avg)


def read_all_node_values(rows: Sequence[AfterstateRows], value_fn: Callable[[torch.Tensor], torch.Tensor],
                         chunk: int = 32768) -> None:
    """read_node_values for a whole rollout's rows in a few large batches instead of one small read a step: the
    network does not change during a rollout, so the values are the rollout-time network's either way."""
    group: List[AfterstateRows] = []
    size = 0
    def flush() -> None:
        nonlocal group, size
        if not group:
            return
        parts: List[torch.Tensor] = []
        for r in group:
            parts.extend(_node_inputs(r))
        with torch.no_grad():
            v = value_fn(torch.cat(parts)).reshape(-1).float().cpu()
        at = 0
        for r in group:
            m = _node_count(r)
            fill_node_values(r, v[at:at + m])
            at += m
        group, size = [], 0
    for r in rows:
        group.append(r)
        size += _node_count(r)
        if size >= chunk:
            flush()
    flush()


def node_tensors(rows: Sequence[AfterstateRows], horizon: int, num_envs: int, device: Any
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """P56: (U, U_post, extra, n_target), each (horizon, num_envs), for compute_gae's afterstate node
    (RolloutBuffer._gae_per_player_both). A decision's walk is its env's rows in its step.

    Without a die in the walk, U is the walk's first moment's value and U_post = U. With one, U is the average over
    the outcomes of the walk's FIRST real roll (a before-roll row with outcomes -- a forced step may come before it,
    its value change stays in the move's own term), and every roll's own luck is left out of the advantage: U_post is
    the value after the first roll that actually came, and for each later roll k, extra adds (value after it - its
    average). A roll whose after-state is no row (it led straight to the side's next decision, or the end) counts
    that next node's target as its after-value: NaN in U_post for the first roll, a count in n_target (and -average
    in extra) for a later one. The step is measured from U_post + extra + n_target x target. NaN in U: no afterstate
    (the plain step). (Audit T-1, 2026-11-22: the first version read only the walk's first moment, so a forced step
    before the die kept the roll's luck, and only the first of two rolls was dropped.)

    In the exact walk (rows.exact, 2026-11-22, owner) a roll is a compound chance node: dice that follow each other
    with no decision between them are one before-roll row whose average is over the compound outcomes (probabilities
    multiplied) and whose next row is the state after the LAST die, so U is the exact average and U_post the value
    after every die of the node -- the luck of all of them is out of the advantage, and the states between them are
    never read. Its before-roll rows are never read either (values NaN there): every one has outcomes."""
    u = np.full((horizon, num_envs), np.nan, dtype=np.float32)
    post = np.full((horizon, num_envs), np.nan, dtype=np.float32)
    extra = np.zeros((horizon, num_envs), dtype=np.float32)
    ntgt = np.zeros((horizon, num_envs), dtype=np.float32)
    for r in rows:
        n = len(r.envs)
        if n == 0 or r.values is None:
            continue
        envs = np.asarray(r.envs)
        kinds = np.asarray(r.kinds)
        vals = r.values.detach().float().cpu().numpy()
        avg = r.avg.detach().float().cpu().numpy() if r.avg is not None else np.full(n, np.nan, dtype=np.float32)
        newg = np.r_[True, envs[1:] != envs[:-1]]
        first = np.flatnonzero(newg)
        g = np.cumsum(newg) - 1                          # the walk (group) each row belongs to
        ng = len(first)
        nxt = np.minimum(np.arange(n) + 1, n - 1)
        aft = np.where((np.arange(n) + 1 < n) & (envs[nxt] == envs) & (kinds[nxt] == AFTER_ROLL), vals[nxt], np.nan)
        roll = (kinds == BEFORE_ROLL) & np.isfinite(avg)
        ridx = np.flatnonzero(roll)
        has_roll = np.zeros(ng, dtype=bool)
        u_g = vals[first].astype(np.float32)
        post_g = u_g.copy()
        ex_g = np.zeros(ng, dtype=np.float32)
        nt_g = np.zeros(ng, dtype=np.float32)
        if ridx.size:
            rg = g[ridx]
            firsts = np.r_[True, rg[1:] != rg[:-1]]      # ridx is ascending, so a group's rolls are contiguous
            f_idx = ridx[firsts]
            has_roll[g[f_idx]] = True
            u_g[g[f_idx]] = avg[f_idx]
            post_g[g[f_idx]] = aft[f_idx]                 # NaN: the first roll led straight to the next node
            later = ridx[~firsts]
            if later.size:
                fin = np.isfinite(aft[later])
                contrib = np.where(fin, aft[later] - avg[later], -avg[later]).astype(np.float32)
                ex_g += np.bincount(g[later], weights=contrib, minlength=ng).astype(np.float32)
                nt_g += np.bincount(g[later], weights=(~fin).astype(np.float64), minlength=ng).astype(np.float32)
        e_first = envs[first]
        u[r.step, e_first] = u_g
        post[r.step, e_first] = post_g
        extra[r.step, e_first] = np.where(has_roll, ex_g, 0.0)
        ntgt[r.step, e_first] = np.where(has_roll, nt_g, 0.0)
    return (torch.from_numpy(u).to(device), torch.from_numpy(post).to(device), torch.from_numpy(extra).to(device),
            torch.from_numpy(ntgt).to(device))


def _last_moments(rows: Sequence[AfterstateRows], players: np.ndarray, num_envs: int
                  ) -> Iterator[Tuple[AfterstateRows, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Per step's rows that hold a side's last decision's walk: (rows, the index of each walk's last row, its side
    index q -- 0 US, 1 USSR --, its env, and which of those walks is that side's last decision's)."""
    lastdec = np.full((2, num_envs), -1, dtype=np.int64)
    for qi, pl in ((0, 1), (1, -1)):
        for t in range(players.shape[0]):
            lastdec[qi] = np.where(players[t] == pl, t, lastdec[qi])
    for r in rows:
        if len(r.envs) == 0:
            continue
        envs = np.asarray(r.envs)
        last = np.flatnonzero(np.r_[envs[1:] != envs[:-1], True])       # an env's rows in a step are one walk
        e = envs[last]
        q = np.where(np.asarray(r.sides)[last] > 0, 0, 1)
        sel = lastdec[q, e] == r.step                                    # the walk of that side's last decision
        if sel.any():
            yield r, last, q, e, sel


def last_moment_vp_values(rows: Sequence[AfterstateRows], players: np.ndarray, num_envs: int,
                          vp_fn: Callable[[torch.Tensor], torch.Tensor]) -> np.ndarray:
    """(2, num_envs): `last_moment_values` for the auxiliary VP head -- per env, the US's and the USSR's own v_vp (read
    by `vp_fn`, the side's own network) at the latest captured moment of its last decision, in its own frame; NaN
    where there is none. The bootstrap past the buffer of the per-player returns_vp (2026-11-24,
    RolloutBuffer._vp_returns_per_player) for a side not to move at the end, in place of the negated v_vp of the side
    to move.

    A moment that waits on a die (a row with outcomes) is NaN too: its v_win is the outcomes' average with a finished
    game counting its result, and a finished face has no VP here. It does not arise where the value is needed: a walk
    ends at a die only when the state after it is not kept -- the game over (a done, the chain cut), or the side's own
    next decision, which makes this decision not its last one or the side the one to move at the end. Should it arise,
    the trainer refuses the rollout (`missing_last_moments`) rather than read the critic blind."""
    out = np.full((2, num_envs), np.nan, dtype=np.float32)
    pend_q: List[np.ndarray] = []
    pend_e: List[np.ndarray] = []
    pend_o: List[torch.Tensor] = []
    for r, last, q, e, sel in _last_moments(rows, players, num_envs):
        waits = np.zeros(len(r.envs), dtype=bool)
        if r.face_row is not None and len(r.face_row):
            waits[np.asarray(r.face_row, dtype=np.int64)] = True
        if r.exact:
            waits |= np.asarray(r.kinds) == BEFORE_ROLL
        keep = sel & ~waits[last]
        if not keep.any():
            continue
        pend_q.append(q[keep])
        pend_e.append(e[keep])
        pend_o.append(r.obs.index_select(0, torch.from_numpy(last[keep]).to(r.obs.device)))
    if pend_o:
        with torch.no_grad():
            v = vp_fn(torch.cat(pend_o)).reshape(-1).float().cpu().numpy()
        out[np.concatenate(pend_q), np.concatenate(pend_e)] = v
    return out


def last_moment_values(rows: Sequence[AfterstateRows], players: np.ndarray, num_envs: int,
                       value_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> np.ndarray:
    """(2, num_envs): per env, for the US (row 0) and the USSR (row 1), the value of the latest captured moment of that
    side's last decision in the rollout, in its own frame -- the exact roll average where that moment waits on a die
    -- or NaN where there is none (no decision of that side, or its walk was cut). Audit T-4(2), 2026-11-22: the
    per-player chain of the side NOT to move at the rollout's end bootstraps from its own last trained moment instead
    of the negated value of the side to move (another information set), and never reads the critic at the
    opponent's decision, a position it is not trained on.

    Rows whose values were not read yet (no --afterstate-advantage, which reads them all) are read here with
    `value_fn`, only the few this needs -- at most one a side and env (fix verification V-5, 2026-11-22: reading the
    whole rollout's rows for them was wasted inference)."""
    out = np.full((2, num_envs), np.nan, dtype=np.float32)
    pending: List[Tuple[np.ndarray, np.ndarray, torch.Tensor]] = []
    for r, last, q, e, sel in _last_moments(rows, players, num_envs):
        n = len(r.envs)
        if r.values is None:
            if value_fn is None:
                raise ValueError("last_moment_values: the rows' values were not read and no value_fn was given")
            if r.face_obs is not None and r.face_obs.shape[0]:
                read_node_values(r, value_fn)                            # the roll average needs the outcomes
            else:
                idx = torch.from_numpy(last[sel]).to(r.obs.device)
                pending.append((q[sel], e[sel], r.obs.index_select(0, idx)))
                continue
        assert r.values is not None
        vals = r.values.detach().float().cpu().numpy()
        avg = r.avg.detach().float().cpu().numpy() if r.avg is not None else np.full(n, np.nan, dtype=np.float32)
        v = np.where(np.isfinite(avg[last]), avg[last], vals[last])
        out[q[sel], e[sel]] = v[sel]
    if pending:
        assert value_fn is not None
        with torch.no_grad():
            v_all = value_fn(torch.cat([p[2] for p in pending])).reshape(-1).float().cpu().numpy()
        at = 0
        for q_p, e_p, _o in pending:
            out[q_p, e_p] = v_all[at:at + len(e_p)]
            at += len(e_p)
    return out


def missing_last_moments(last_side: np.ndarray, players: np.ndarray, cuts: np.ndarray,
                         last_players: np.ndarray) -> np.ndarray:
    """(2, num_envs) bool: where per-player GAE needs a side's own last moment (`last_moment_values`) and has none.
    The bootstrap past the buffer is read for a side not to move at the end (`last_players`) whose last decision comes
    after its env's last cut (`cuts`: the dones, plus the turn boundaries under --slice-turn-boundaries -- the rows the
    per-player pass drops the carried state at). Without its own moment it would fall back to the negated value of
    the side to move, the cross-perspective bootstrap per-player GAE exists to remove (fix verification V-5,
    2026-11-22): the trainer refuses that instead."""
    horizon, n = players.shape
    lastcut = np.full(n, -1, dtype=np.int64)
    lastdec = np.full((2, n), -1, dtype=np.int64)
    for t in range(horizon):
        lastcut = np.where(cuts[t], t, lastcut)
        lastdec[0] = np.where(players[t] == 1, t, lastdec[0])
        lastdec[1] = np.where(players[t] == -1, t, lastdec[1])
    lp = np.asarray(last_players)
    need = np.stack([(lastdec[0] > lastcut) & (lp != 1), (lastdec[1] > lastcut) & (lp != -1)])
    return need & ~np.isfinite(last_side)


def learner_seat_rows(t: np.ndarray, e: np.ndarray, side: np.ndarray, learner_seat: np.ndarray) -> np.ndarray:
    """Per row (buffer step t, env e, side +1 / -1): whether `side` is a seat the learner plays in that env at that
    step -- `learner_seat[t, e]` is 0 where it plays both (self-play), else its one seat (--update-learner-rows-only)."""
    seat = np.asarray(learner_seat)[t, e]
    return (seat == 0) | (seat == side)


def build_targets(rows: Sequence[AfterstateRows], players: np.ndarray, dones: np.ndarray, rewards: np.ndarray,
                  rets: np.ndarray, learner: np.ndarray, device: Any, learner_seat: Optional[np.ndarray] = None
                  ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """(observations, targets, kinds) of the rows that have a target, or Nones. With `learner_seat` (per buffer row,
    the learner's seat: --update-learner-rows-only), a row of a seat the learner does not play is dropped too."""
    rows = [r for r in rows if len(r.envs)]
    if not rows:
        return None, None, None
    t = np.concatenate([np.full(len(r.envs), r.step, dtype=np.int64) for r in rows])
    e = np.concatenate([r.envs for r in rows]).astype(np.int64)
    side = np.concatenate([r.sides for r in rows]).astype(np.int64)
    kinds = np.concatenate([r.kinds for r in rows]).astype(np.int64)
    target, keep = next_decision_targets(t, e, side, players, dones, rewards, rets, learner)
    # P56, the exact walk (owner, 2026-11-22): the critic is never trained at a state waiting on a die -- the
    # observation does not show which roll is pending --, so its before-roll rows (exactly the rows that wait on one)
    # stay nodes of the advantage and out of the value loss.
    exact = np.concatenate([np.full(len(r.envs), bool(r.exact)) for r in rows])
    keep = keep & ~(exact & (kinds == BEFORE_ROLL))
    # Only the walks of learner decisions are trained on: --per-player-gae also captures the others' (a pool
    # member's, a forced move's) for their last moments alone (fix verification V-5, 2026-11-22).
    keep = keep & np.asarray(learner, dtype=bool)[t, e]
    if learner_seat is not None:
        keep = keep & learner_seat_rows(t, e, side, learner_seat)
    if not keep.any():
        return None, None, None
    idx = torch.from_numpy(np.flatnonzero(keep)).to(rows[0].obs.device)
    obs = torch.cat([r.obs for r in rows]).index_select(0, idx)
    return (obs, torch.from_numpy(target[keep]).to(device), torch.from_numpy(kinds[keep]).to(device))
