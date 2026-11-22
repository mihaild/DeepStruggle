"""P48b: restart a share of games from positions the run's own games pass through (plans session, 2026-11-18).

A quarter of the environments (--restart-pool-frac) start each game from a pooled position instead of turn 1. The pool
is fed live by the run's whole games: at every turn rollover the pre-deal state -- the state at the last decision
before the turn rolls over, so the next deal has not happened -- is offered to it, and a restart reseeds the state, so
every continuation draws a fresh deal and dice. Restarted games are ordinary self-play on positions the policy itself
reaches: no loss correction, no forced moves.

The rules (P48b, the plans session's; the starred constants are its choices, not measured):

* **Turns and capacity.** Turns 3-10 weighted 5 / 5 / 10 / 10 / 15 / 20 / 20 / 15% (restart draws); about 1,000*
  positions in all, in per-turn buckets sized by those weights. Small on purpose: at ~1,000-1,500 finished
  continuations a million steps a position lives 2-7M steps, so its results come from nearly one policy.
* **Admission, by the critic.** p = the learner's US win probability at the pre-deal state, its two seats' values
  averaged (P48a's test); openness o = 4p(1 - p). Admitted with probability 0.2 + 0.8 o at turns 3-7 and 0.4 + 0.6 o
  at turns 8-10, where P48a found the critic calls 13-16% of open positions decided. The floor is the unfiltered share.
  In a full bucket a newcomer replaces the oldest never-played position (n = 0), and is refused when there is none: a
  position with a result (checked out or put back) keeps its place, so its put-back always fits and it leaves only by
  the put-back rule or its age. An unplayed position carries no information, so swapping it for a newer one is free,
  and with offers several times the use the unplayed stock comes from the newest games. (A bucket that dropped its
  oldest outright evicted every put-back -- it is the oldest -- and each position was played once: the first launch,
  voided. One that refused every newcomer kept unplayed positions from the run's past; the second, voided too.)
* **Check-out.** A drawn position leaves the pool until its continuation ends, so one position never runs in two games
  at once. Draw: a turn by the weights among the stocked buckets, then uniformly in the bucket; nothing stocked gives a
  whole game.
* **Put back.** When a continuation ends: n += 1, m = the mean US score of its continuations, v = the critic's US win
  probability at that restart's first decision (the rollout's own value). m' = (n m + 2 v) / (n + 2), openness
  4 m'(1 - m'), z = min(2, |m - v| / sqrt(max(v (1 - v), 0.05) / n)), and the position goes back with probability
  0.1 + 0.9 openness (0.25 + 0.75 z / 2). Retired regardless at n = 8* or 10M* steps after admission. A lopsided
  position leaves after a game or two if the critic agreed, a few more if it had called the position open.

Replayed on P48a's saved positions, the rule sends 67% of restart games to open positions against 55% for uniform
restarts (58% / 45% at turns 8-10), and about two continuations to each admitted position
(data/reports/p48b_replay.md).

The pool is not saved with the resume state: a resumed run refills it from its own games within a few million steps.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import ts_engine as ts

#: Share of restart draws per turn (P48b).
DEFAULT_TURN_WEIGHTS: Dict[int, float] = {3: 0.05, 4: 0.05, 5: 0.10, 6: 0.10, 7: 0.15, 8: 0.20, 9: 0.20, 10: 0.15}
#: The critic counts as this many results in a position's estimate.
CRITIC_WEIGHT = 2.0
#: Admission: floor + slope x openness, early (turns before LATE_FROM) and late.
ADMIT_EARLY = (0.2, 0.8)
ADMIT_LATE = (0.4, 0.6)
LATE_FROM = 8
#: How many positions wait for the critic's verdict before one batched forward pass.
JUDGE_BATCH = 256
UINT64 = 1 << 64


def parse_turn_weights(text: str) -> Dict[int, float]:
    """'3:5,4:5,...' (any scale) -> {turn: share}, normalised."""
    out: Dict[int, float] = {}
    for item in text.split(","):
        t, w = item.split(":")
        out[int(t)] = float(w)
    total = sum(out.values())
    if total <= 0 or any(w < 0 for w in out.values()):
        raise ValueError(f"turn weights must be non-negative with a positive sum, got {text!r}")
    check_turns(out)
    return {t: w / total for t, w in sorted(out.items())}


def check_turns(weights: Dict[int, float]) -> None:
    """Refuse restart turns below 3 (audit T-8). A turn-2 position is captured pre-deal with ``state.turn == 1``,
    so the env books its continuation as a whole game (``start_turn`` 1): ``complete()`` is never called, the bucket's
    checked-out count grows until it wedges, and the restart's games pollute the whole-game statistics.

    TODO(pool): mark restarts explicitly instead of inferring them from the start turn before turn 2 is allowed, and
    before the restart pool is used again at all (it is unmaintained since P48b)."""
    bad = sorted(t for t in weights if t < 3)
    if bad:
        raise ValueError(f"restart turns must be >= 3 (a turn-2 restart is booked as a whole game and never "
                         f"completes); got turns {bad}")


def put_back_probability(n: int, mean: float, v: float, critic_weight: float = CRITIC_WEIGHT,
                         use_disagreement: bool = True) -> float:
    """P48b's put-back rule after a position's n-th continuation (mean US score `mean`, critic `v`). Without
    `use_disagreement` the term in z is read as 0 -- what the rule would do without it."""
    m_post = (n * mean + critic_weight * v) / (n + critic_weight)
    openness = 4.0 * m_post * (1.0 - m_post)
    z = min(2.0, abs(mean - v) / math.sqrt(max(v * (1.0 - v), 0.05) / n)) if use_disagreement else 0.0
    return 0.1 + 0.9 * openness * (0.25 + 0.75 * z / 2.0)


def admit_probability(p: float, turn: int) -> float:
    o = 4.0 * p * (1.0 - p)
    floor, slope = ADMIT_LATE if turn >= LATE_FROM else ADMIT_EARLY
    return floor + slope * o


@dataclass
class RestartPosition:
    state: Any
    turn: int
    admitted_step: int
    #: The critic's US win probability at admission.
    p_admit: float
    #: Finished continuations and the sum of their US scores.
    n: int = 0
    score_sum: float = 0.0
    #: The critic's US win probability at each continuation's first decision.
    v_values: List[float] = field(default_factory=list)
    #: The step it last entered the bucket (admission or put-back).
    stocked_step: int = 0


class RestartPool:
    """The pool, its admission and put-back, and the hooks the rollout loop calls."""

    def __init__(self, num_envs: int, frac: float = 0.25, capacity: int = 1000,
                 turn_weights: Optional[Dict[int, float]] = None, max_continuations: int = 8,
                 max_age_steps: int = 10_000_000, critic_weight: float = CRITIC_WEIGHT, seed: int = 0) -> None:
        if not 0.0 < frac < 1.0:
            raise ValueError(f"the restart share must be in (0, 1), got {frac}")
        if capacity < 1 or max_continuations < 1 or max_age_steps < 1:
            raise ValueError("capacity, max_continuations and max_age_steps must be positive")
        self.num_envs = int(num_envs)
        self.turn_weights = dict(turn_weights or DEFAULT_TURN_WEIGHTS)
        check_turns(self.turn_weights)
        self.capacity = {t: max(1, int(round(capacity * w))) for t, w in self.turn_weights.items()}
        self.max_continuations = int(max_continuations)
        self.max_age_steps = int(max_age_steps)
        self.critic_weight = float(critic_weight)
        self.rng = random.Random(seed)
        #: The environments that restart (the first `frac` of them), fixed for the run.
        n_restart = max(1, int(round(self.num_envs * frac)))
        self.restart_env = np.zeros(self.num_envs, dtype=bool)
        self.restart_env[:n_restart] = True
        self.buckets: Dict[int, List[RestartPosition]] = {t: [] for t in self.turn_weights}
        #: Positions of each turn checked out and being played: they keep their place in the bucket's capacity.
        self.out: Dict[int, int] = {t: 0 for t in self.turn_weights}
        #: The position each env is playing (checked out), and the one whose game just ended, until its result is read.
        self.playing: List[Optional[RestartPosition]] = [None] * self.num_envs
        self.finished: List[Optional[RestartPosition]] = [None] * self.num_envs
        #: Envs whose next decision is a restart's first: its value is the continuation's v.
        self.awaiting_v = np.zeros(self.num_envs, dtype=bool)
        self.v_now: List[Optional[float]] = [None] * self.num_envs
        #: The finished continuation's v, snapshotted with `finished` when its game ends: the env's next start
        #: (provide(), called inside the step that ended the game) resets v_now before complete() reads the result.
        self.finished_v: List[Optional[float]] = [None] * self.num_envs
        #: Pre-step states of the whole-game envs, and their turns, for capture.
        self._prev: List[Optional[Any]] = [None] * self.num_envs
        self._prev_turn = np.zeros(self.num_envs, dtype=np.int64)
        self._pending: List[RestartPosition] = []
        self.steps = 0
        self._reset_counters()

    def size_now(self) -> int:
        """Positions in the pool now (checked-out ones are not)."""
        return sum(len(b) for b in self.buckets.values())

    # -- counters, read and cleared by metrics() -------------------------------------------------

    def _reset_counters(self) -> None:
        self.offered = 0
        self.admitted = 0
        self.admitted_openness = 0.0
        self.handed_out = 0
        self.no_stock = 0
        self.retired_n: List[int] = []
        self.retired_turn: List[int] = []
        self.retired_excess: List[float] = []
        self.z_values: List[float] = []
        self.dropped_full = 0
        self.replaced_unplayed = 0
        self.dropped_old = 0
        #: Steps a drawn position waited in the bucket since it entered it: unplayed, and put back.
        self.wait_fresh: List[int] = []
        self.wait_back: List[int] = []
        #: Put-backs, and of them those the disagreement term decided (back with it, retired without it).
        self.put_back = 0
        self.put_back_by_z = 0

    # -- drawing ----------------------------------------------------------------------------------

    def provide(self, env_idx: int) -> Optional[Any]:
        """The env's next start (the start provider): a checked-out position reseeded, or None for a whole game."""
        if self.playing[env_idx] is not None:
            # Its game just ended; the result is read after the step (complete()).
            self.finished[env_idx] = self.playing[env_idx]
            self.finished_v[env_idx] = self.v_now[env_idx]
            self.v_now[env_idx] = None
            self.playing[env_idx] = None
        if not self.restart_env[env_idx]:
            return None
        stocked = [t for t, w in self.turn_weights.items() if w > 0 and self.buckets[t]]
        if not stocked:
            self.no_stock += 1
            return None
        turn = self.rng.choices(stocked, weights=[self.turn_weights[t] for t in stocked], k=1)[0]
        bucket = self.buckets[turn]
        at = self.rng.randrange(len(bucket))
        pos = bucket[at]
        bucket[at] = bucket[-1]
        bucket.pop()
        (self.wait_back if pos.n else self.wait_fresh).append(self.steps - pos.stocked_step)
        self.playing[env_idx] = pos
        self.out[turn] += 1
        self.awaiting_v[env_idx] = True
        self.v_now[env_idx] = None
        self.handed_out += 1
        state = pos.state.clone()
        state.rng_state = self.rng.getrandbits(64) % UINT64
        return state

    def first_values(self, v_us: np.ndarray) -> None:
        """Before each step: the US win probability of every env's current decision; an env whose restart has just
        begun takes it as that continuation's v."""
        idx = np.flatnonzero(self.awaiting_v)
        for i in idx:
            self.v_now[i] = float(v_us[i])
        self.awaiting_v[idx] = False

    # -- capture ------------------------------------------------------------------------------------

    def before_step(self, runner: Any, whole_game: np.ndarray) -> None:
        """The pre-step state and turn of every whole-game env: a turn rollover in this step makes it a candidate."""
        for i in np.flatnonzero(whole_game):
            self._prev[i] = runner.get_state(int(i))
        self._prev_turn[:] = np.asarray(runner.get_turns(), dtype=np.int64)

    def after_step(self, runner: Any, whole_game: np.ndarray, dones: np.ndarray, judge: Any) -> None:
        """Offer each whole-game env that crossed into a weighted turn this step; judge the queue in batches."""
        turns = np.asarray(runner.get_turns(), dtype=np.int64)
        crossed = whole_game & (dones < 0.5) & (turns == self._prev_turn + 1)
        for i in np.flatnonzero(crossed):
            t = int(turns[i])
            prev = self._prev[i]
            if t in self.turn_weights and prev is not None:
                self._pending.append(RestartPosition(state=prev, turn=t, admitted_step=self.steps, p_admit=0.5))
        if len(self._pending) >= JUDGE_BATCH:
            self.judge_pending(judge)

    def judge_pending(self, judge: Any) -> None:
        """Admit the queued positions by the critic (`judge`: positions -> US win probabilities)."""
        if not self._pending:
            return
        probs = judge([p.state for p in self._pending])
        # Offered counted here, with the verdicts, so the admission rate's two counts cover the same positions.
        self.offered += len(self._pending)
        for pos, p in zip(self._pending, probs):
            pos.p_admit = float(p)
            if self.rng.random() < admit_probability(pos.p_admit, pos.turn):
                self._admit(pos)
        self._pending = []

    def _admit(self, pos: RestartPosition) -> None:
        bucket = self.buckets[pos.turn]
        if len(bucket) + self.out[pos.turn] >= self.capacity[pos.turn]:
            unplayed = [k for k in range(len(bucket)) if bucket[k].n == 0]
            if not unplayed:
                self.dropped_full += 1
                return
            oldest = min(unplayed, key=lambda k: bucket[k].admitted_step)
            bucket[oldest] = bucket[-1]
            bucket.pop()
            self.replaced_unplayed += 1
        pos.admitted_step = self.steps
        pos.stocked_step = self.steps
        bucket.append(pos)
        self.admitted += 1
        self.admitted_openness += 4.0 * pos.p_admit * (1.0 - pos.p_admit)

    # -- results ----------------------------------------------------------------------------------

    def complete(self, env_idx: int, us_score: float) -> None:
        """A restart's continuation ended with this US score: update its position, then put it back or retire it."""
        pos = self.finished[env_idx]
        self.finished[env_idx] = None
        v = self.finished_v[env_idx]
        self.finished_v[env_idx] = None
        if pos is None:
            return
        self.out[pos.turn] -= 1
        pos.n += 1
        pos.score_sum += us_score
        if v is None:
            v = pos.p_admit
        pos.v_values.append(v)
        mean = pos.score_sum / pos.n
        v_mean = float(np.mean(pos.v_values))
        z = min(2.0, abs(mean - v_mean) / math.sqrt(max(v_mean * (1.0 - v_mean), 0.05) / pos.n))
        self.z_values.append(z)
        too_old = self.steps - pos.admitted_step > self.max_age_steps
        if pos.n < self.max_continuations and not too_old:
            u = self.rng.random()
            if u < put_back_probability(pos.n, mean, v_mean, self.critic_weight):
                self.put_back += 1
                if u >= put_back_probability(pos.n, mean, v_mean, self.critic_weight, use_disagreement=False):
                    self.put_back_by_z += 1
                pos.stocked_step = self.steps
                self.buckets[pos.turn].append(pos)
                return
        if too_old:
            self.dropped_old += 1
        self.retired_n.append(pos.n)
        self.retired_turn.append(pos.turn)
        # |m - v| beyond chance: the critic's error on this position, past what n results can show.
        self.retired_excess.append(abs(mean - v_mean) - math.sqrt(max(v_mean * (1.0 - v_mean), 1e-6) / pos.n))

    # -- logging -----------------------------------------------------------------------------------

    def metrics(self) -> Dict[str, float]:
        out: Dict[str, float] = {f"restart/pool_turn{t}": float(len(b)) for t, b in self.buckets.items()}
        out["restart/pool_size"] = float(sum(len(b) for b in self.buckets.values()))
        out["restart/offered"] = float(self.offered)
        out["restart/admission_rate"] = self.admitted / self.offered if self.offered else 0.0
        out["restart/admitted_openness"] = self.admitted_openness / self.admitted if self.admitted else 0.0
        out["restart/handed_out"] = float(self.handed_out)
        out["restart/no_stock"] = float(self.no_stock)
        out["restart/dropped_full"] = float(self.dropped_full)
        out["restart/dropped_old"] = float(self.dropped_old)
        out["restart/replaced_unplayed"] = float(self.replaced_unplayed)
        out["restart/pool_unplayed"] = float(sum(p.n == 0 for b in self.buckets.values() for p in b))
        if self.wait_fresh:
            out["restart/wait_fresh_median"] = float(np.median(self.wait_fresh))
        if self.wait_back:
            out["restart/wait_back_median"] = float(np.median(self.wait_back))
            out["restart/wait_back_p90"] = float(np.percentile(self.wait_back, 90))
        if self.retired_n:
            out["restart/retired"] = float(len(self.retired_n))
            out["restart/n_at_retirement"] = float(np.mean(self.retired_n))
        if self.put_back:
            out["restart/put_back"] = float(self.put_back)
            out["restart/put_back_by_disagreement"] = self.put_back_by_z / self.put_back
        if self.z_values:
            out["restart/z_mean"] = float(np.mean(self.z_values))
            out["restart/z_at_cap"] = float(np.mean(np.asarray(self.z_values) >= 2.0))
        for t in self.turn_weights:
            ex = [e for e, tt in zip(self.retired_excess, self.retired_turn) if tt == t]
            if ex:
                out[f"restart/critic_excess_turn{t}"] = float(np.mean(ex))
        self._reset_counters()
        return out


def critic_judge(model: Any, features: int, device: Any) -> Any:
    """positions -> the model's US win probability at each, its two seats' values averaged (P48a's test)."""
    def judge(states: Sequence[Any]) -> List[float]:
        masks = np.stack([np.asarray(ts.ActionMask.generate_flat_mask(s), dtype=np.uint8) for s in states])
        m = torch.from_numpy(masks).to(device)
        values = []
        with torch.no_grad():
            for player, sign in ((ts.Player.US, 1.0), (ts.Player.USSR, -1.0)):
                o = np.stack([np.asarray(ts.extract_observation_features(s, player, features), dtype=np.float32)
                              for s in states])
                out = model(torch.from_numpy(o).to(device), m)
                values.append(sign * out[1].float().reshape(-1).cpu().numpy())
        v = 0.5 * (values[0] + values[1])
        return [float((x + 1.0) / 2.0) for x in v]
    return judge
