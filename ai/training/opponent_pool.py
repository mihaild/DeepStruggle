"""Frozen past policies to play a share of the rollout environments against.

The problem this exists for: once one side finds a strategy the other has not answered, outcomes
stop depending on what the trailing side does. The critic then has nothing to discriminate, it
degenerates toward the base rate, and advantages vanish for *both* sides -- measured on E3-17-22
and reproduced independently on E3-15-22. Nothing that re-weights, filters or rescales the
existing signal escapes that, because there is no signal left to rescale.

Opponent *diversity* does escape it, and it is worth being precise about why, because opponent
*prioritisation* does not. Sampling opponents you lose to is pointless in pure self-play: the
current policy already faces the strongest opponent there is, namely itself. Diversity is the
different claim -- against an 80M snapshot the trailing side has games it can actually win, so the
terminal outcome separates its actions again and there is something for the advantage to be
non-zero about.

**`--opponent-pfsp` exists to test that paragraph, and may well refute the flag rather than the
paragraph.** Prioritised fictitious self-play weights the draw by how the learner is doing against
each snapshot. The argument above predicts it should *hurt*: the weak old snapshots that PFSP
deprioritises are exactly the ones supplying winnable games to the learner's trailing side. The
counter-argument is that a snapshot beaten 99% of the time supplies no signal to either side and
its games are wasted. Nobody has measured it, so the weighting defaults to `var` -- x(1-x), which
peaks on *evenly matched* opponents rather than on hard ones -- and `pfsp_uniform_mix` keeps a
floor of uniform probability under every pool member so that prioritisation cannot delete the
diversity the mechanism is built on. Watch `opp_pfsp_entropy`: if it collapses toward 0 the pool
has become all-recent and this is no longer testing prioritisation, it is testing a smaller pool.

**One opponent per rollout iteration.** The pool may hold many snapshots, but a single one is
sampled for each iteration and used by every mixed environment in it. That keeps the step to
exactly two forward passes -- learner and opponent, on complementary subsets of the batch, the
same total FLOPs as the single pass it replaces. Diversity accumulates across iterations instead
of within them. Sampling several per iteration would work too, at more kernel launches on smaller
batches.

**Which means a game is played against about five different snapshots.** An episode does not fit
in an iteration: measured on E3-20-28 the mean episode is 626.5 micro-actions against a
`buffer_size` of 128, so one game faces **4.89** opponents in sequence -- a swap roughly every 1.4
turns, and a random ~40% subset of a capacity-12 pool per game. The learner therefore never meets
a coherent adversary over a whole game; it meets a composite that exists as no policy. That is a
consequence of the two facts above sitting together, it was unremarked until 2026-09-16, and it is
why results have to be attributed by exposure rather than to "the" opponent (`on_episode_end`).
Whether it helps or hurts is untested: it costs coherence and buys within-episode diversity, and
diversity is the mechanism this docstring credits. See
`research/findings/training/pooling.md` §3b.

**A growing pool, kept spread across the run.** The first version took a fixed list of
checkpoints, which measured well (E3-19-22 held self-play imbalance to 9.0pp where its control
reached 71.7pp) but is not a usable recipe: it required eight snapshots of an already-finished
run. A real run has to pool against its own history as that history appears.

Eviction is by *spacing*, not by age. Dropping the oldest would let the pool become all-recent,
and then every opponent carries whatever strategy the run has converged on -- the trailing side
loses the winnable games that made the mechanism work in the first place, and the window quietly
closes. Instead the first and most recent snapshots are always kept and the interior point with
the smallest neighbouring gap is evicted, so the pool stays spread over the whole run.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class OpponentPool:
    """A set of frozen policies, plus the per-environment assignment of who plays whom.

    `frac` of environments are *mixed*: the learner plays one side and a frozen snapshot the
    other. The rest are ordinary self-play, where the learner plays both sides.
    """

    def __init__(self, nets: Sequence[Any], num_envs: int, frac: float = 0.25,
                 seed: int = 0, lock_learner_side: Optional[int] = None,
                 capacity: int = 12, pfsp: bool = False, pfsp_weighting: str = "var",
                 pfsp_uniform_mix: float = 0.25, pfsp_prior: float = 4.0) -> None:
        if not nets:
            raise ValueError("OpponentPool needs at least one frozen network")
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"frac must be in (0, 1], got {frac}")
        if pfsp_weighting not in ("var", "hard"):
            raise ValueError(f"pfsp_weighting must be 'var' or 'hard', got {pfsp_weighting!r}")
        if not 0.0 <= pfsp_uniform_mix < 1.0:
            raise ValueError(f"pfsp_uniform_mix must be in [0, 1), got {pfsp_uniform_mix}")
        if pfsp_prior <= 0.0:
            raise ValueError(f"pfsp_prior must be positive, got {pfsp_prior}")
        self.nets = list(nets)
        self.capacity = capacity
        self.pfsp = bool(pfsp)
        self.pfsp_weighting = pfsp_weighting
        self.pfsp_uniform_mix = float(pfsp_uniform_mix)
        #: Beta prior on each opponent's win rate, in pseudo-games a side. Large enough that an
        #: opponent with two games does not dominate the draw on noise.
        self.pfsp_prior = float(pfsp_prior)
        #: A stable id per pool member. Indices shift under eviction, so results are keyed by id.
        self._next_id = 0
        self.ids: List[int] = []
        #: Weighted learner wins and games against each opponent id. Weighted, because an episode
        #: spans several iterations and therefore several opponents -- see `on_episode_end`.
        self.wins: Dict[int, float] = {}
        self.games: Dict[int, float] = {}
        #: Per environment, how many iterations of the current episode each opponent id played.
        self._exposure: List[Dict[int, float]] = [{} for _ in range(num_envs)]
        #: Step count each net was captured at, parallel to self.nets. Seeds get 0 so that a
        #: pool started from the initial policy keeps that policy as its earliest point.
        self.steps: List[int] = [0] * len(self.nets)
        for n in self.nets:
            n.eval()
            for p in n.parameters():
                p.requires_grad_(False)
        self.num_envs = num_envs
        self.frac = frac
        self.rng = random.Random(seed)
        for _ in self.nets:
            self._register()
        #: None to alternate per episode; +1 or -1 to pin the learner to one side throughout,
        #: which is what the single-frozen-opponent experiment wants.
        self.lock_learner_side = lock_learner_side

        n_mixed = int(round(num_envs * frac))
        #: True where the learner faces a frozen opponent rather than itself.
        self.is_mixed = np.zeros(num_envs, dtype=bool)
        self.is_mixed[:n_mixed] = True
        #: Which side the learner plays in a mixed env (+1 US, -1 USSR); unused elsewhere.
        self.learner_side = np.ones(num_envs, dtype=np.int8)
        for i in range(num_envs):
            self.learner_side[i] = self._draw_side()
        self.current: Any = self.nets[0]
        self.current_id: int = self.ids[0]

    def _register(self) -> int:
        """Give the newest net a stable id and zeroed statistics."""
        oid = self._next_id
        self._next_id += 1
        self.ids.append(oid)
        self.wins[oid] = 0.0
        self.games[oid] = 0.0
        return oid

    def win_rate(self, oid: int) -> float:
        """The learner's smoothed win rate against one opponent.

        Smoothed toward 0.5 with a Beta prior: an opponent with one game must not capture the
        draw because that game happened to be a win.
        """
        w = self.wins.get(oid, 0.0) + self.pfsp_prior
        n = self.games.get(oid, 0.0) + 2.0 * self.pfsp_prior
        return w / n

    def sampling_probs(self) -> List[float]:
        """The draw distribution over the pool, in `self.nets` order.

        Uniform unless `pfsp`. `var` weights x(1-x), peaking where the learner is evenly matched;
        `hard` weights (1-x)^2, peaking where the learner is losing. A uniform floor of
        `pfsp_uniform_mix` is mixed in either way, so no pool member can be driven to zero -- the
        pool's value is its spread, and a prioritiser that empties it has defeated the purpose.
        """
        m = len(self.nets)
        if m == 0:
            return []
        if not self.pfsp:
            return [1.0 / m] * m
        raw: List[float] = []
        for oid in self.ids:
            x = self.win_rate(oid)
            raw.append(x * (1.0 - x) if self.pfsp_weighting == "var" else (1.0 - x) ** 2)
        total = sum(raw)
        if total <= 0.0:
            return [1.0 / m] * m
        u = self.pfsp_uniform_mix
        return [(1.0 - u) * (r / total) + u / m for r in raw]

    def _draw_side(self) -> int:
        if self.lock_learner_side is not None:
            return int(self.lock_learner_side)
        return 1 if self.rng.random() < 0.5 else -1

    def add(self, net: Any, steps: int) -> None:
        """Add a snapshot taken at `steps`, evicting to stay within capacity.

        The net is frozen in place. Callers pass a freshly loaded copy, not the live training
        model -- adding the model under training would give the learner an opponent whose
        weights move with it, which is self-play with extra steps.
        """
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.nets.append(net)
        self.steps.append(int(steps))
        self._register()

        while len(self.nets) > self.capacity:
            # Keep the endpoints; drop the interior snapshot whose neighbours are closest
            # together, which is the one carrying the least information about the run's span.
            order = sorted(range(len(self.steps)), key=lambda i: self.steps[i])
            victim = None
            best_gap = None
            for pos in range(1, len(order) - 1):
                i = order[pos]
                gap = self.steps[order[pos + 1]] - self.steps[order[pos - 1]]
                if best_gap is None or gap < best_gap:
                    best_gap, victim = gap, i
            if victim is None:  # capacity < 3: fall back to dropping the oldest
                victim = order[0]
            self.nets.pop(victim)
            self.steps.pop(victim)
            dead = self.ids.pop(victim)
            self.wins.pop(dead, None)
            self.games.pop(dead, None)
            # Exposure to an evicted opponent is dropped rather than reattributed: the outcome
            # would otherwise be credited to a snapshot that is no longer in the pool.
            for env in self._exposure:
                env.pop(dead, None)

    def start_iteration(self) -> None:
        """Pick the opponent this iteration's mixed environments will face, and record exposure.

        One opponent per iteration, as before; only the *distribution* changes under PFSP. An
        episode outlives an iteration -- roughly 300-600 micro-actions against 128 -- so a single
        game is played against three to five different snapshots in sequence. That is why results
        are attributed by exposure rather than to one opponent: crediting a whole episode to
        whichever snapshot happened to be current when it ended would be wrong most of the time.
        """
        probs = self.sampling_probs()
        if self.pfsp and probs:
            idx = self._weighted_index(probs)
        else:
            idx = self.rng.randrange(len(self.nets))
        self.current = self.nets[idx]
        self.current_id = self.ids[idx]
        for i in range(self.num_envs):
            if self.is_mixed[i]:
                env = self._exposure[i]
                env[self.current_id] = env.get(self.current_id, 0.0) + 1.0

    def _weighted_index(self, probs: Sequence[float]) -> int:
        r = self.rng.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r < acc:
                return i
        return len(probs) - 1

    def on_episode_end(self, env_index: int,
                       victory_points: Optional[float] = None) -> None:
        """Record the result against whoever played it, then re-draw the learner's side.

        The runaway is bidirectional -- one arm ran away in the US's favour -- so an experiment
        that only ever trains the learner as one side would answer a narrower question than the
        one being asked.

        `victory_points` is the engine's signed VP, positive for the US. The learner won iff its
        sign matches the side the learner was playing; zero is a draw and counts as half. The
        result is split across the opponents this episode was actually played against, in
        proportion to how many iterations each was current for. Passing None records nothing,
        which is what a caller with no outcome to report should do.
        """
        if victory_points is not None and self.is_mixed[env_index]:
            exposure = self._exposure[env_index]
            total = sum(exposure.values())
            if total > 0.0:
                side = int(self.learner_side[env_index])
                vp = float(victory_points)
                score = 0.5 if vp == 0.0 else (1.0 if (vp > 0) == (side == 1) else 0.0)
                for oid, share in exposure.items():
                    if oid not in self.games:      # evicted mid-episode
                        continue
                    w = share / total
                    self.games[oid] += w
                    self.wins[oid] += w * score
        self._exposure[env_index] = {}
        self.learner_side[env_index] = self._draw_side()

    def learner_acts(self, decision_players: np.ndarray) -> np.ndarray:
        """Per environment: is the learner the one to move?

        True everywhere in self-play environments, and in mixed ones only when the side to move
        is the learner's.
        """
        dp = np.asarray(decision_players, dtype=np.int8)
        return ~self.is_mixed | (dp == self.learner_side)

    def stats(self) -> Dict[str, float]:
        """Pool diagnostics, including the ones that say whether the pool is doing anything.

        Until 2026-09-16 this returned only size, span and the mixed fraction -- that the pool
        exists and is used, never how the learner was faring against it. There was therefore no
        in-run instrument for whether pooling works, and the only live external opponents
        (HeuristicBot, RandomBot) saturate above 89% by 120M. `opp_win_rate_mean` and its spread
        are that missing instrument, and PFSP needs them anyway.
        """
        out: Dict[str, float] = {
            "opp_frac_mixed": float(self.is_mixed.mean()),
            "opp_pool_size": float(len(self.nets)),
            "opp_pool_span_m": float((max(self.steps) - min(self.steps)) / 1e6)
            if self.steps else 0.0,
        }
        rates = [self.win_rate(oid) for oid in self.ids]
        played = [self.games.get(oid, 0.0) for oid in self.ids]
        if rates:
            out["opp_win_rate_mean"] = float(sum(rates) / len(rates))
            out["opp_win_rate_min"] = float(min(rates))
            out["opp_win_rate_max"] = float(max(rates))
            out["opp_games_recorded"] = float(sum(played))
        probs = self.sampling_probs()
        if probs:
            # Entropy in nats, normalised by log(pool size): 1.0 is uniform, 0.0 is a single
            # opponent. The number to watch under PFSP -- a collapse means the prioritiser has
            # emptied the pool of the diversity the mechanism depends on.
            ent = -sum(q * math.log(q) for q in probs if q > 0.0)
            out["opp_pfsp_entropy"] = float(ent / math.log(len(probs))) if len(probs) > 1 else 1.0
            out["opp_pfsp_max_prob"] = float(max(probs))
        return out


def load_pool(paths: Sequence[str], device: Any) -> List[Any]:
    """Load frozen networks from checkpoint paths, refusing a layout mismatch rather than
    letting a retired-layout checkpoint read the wrong floats."""
    from bindings.ts_env import check_obs_width
    from tools.lib.player_agent import NeuralAgent

    nets = []
    for path in paths:
        agent = NeuralAgent.from_checkpoint(path, device=device)
        check_obs_width(agent.model)
        nets.append(agent.model)
    return nets
