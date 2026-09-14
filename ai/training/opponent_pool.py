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

**One opponent per rollout iteration.** The pool may hold many snapshots, but a single one is
sampled for each iteration and used by every mixed environment in it. That keeps the step to
exactly two forward passes -- learner and opponent, on complementary subsets of the batch, the
same total FLOPs as the single pass it replaces. Diversity accumulates across iterations instead
of within them. Sampling several per iteration would work too, at more kernel launches on smaller
batches.

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
                 capacity: int = 12) -> None:
        if not nets:
            raise ValueError("OpponentPool needs at least one frozen network")
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"frac must be in (0, 1], got {frac}")
        self.nets = list(nets)
        self.capacity = capacity
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

    def start_iteration(self) -> None:
        """Pick the opponent this iteration's mixed environments will face."""
        self.current = self.rng.choice(self.nets)

    def on_episode_end(self, env_index: int) -> None:
        """Re-draw the learner's side so it gets both roles against the pool over time.

        The runaway is bidirectional -- one arm ran away in the US's favour -- so an experiment
        that only ever trains the learner as one side would answer a narrower question than the
        one being asked.
        """
        self.learner_side[env_index] = self._draw_side()

    def learner_acts(self, decision_players: np.ndarray) -> np.ndarray:
        """Per environment: is the learner the one to move?

        True everywhere in self-play environments, and in mixed ones only when the side to move
        is the learner's.
        """
        dp = np.asarray(decision_players, dtype=np.int8)
        return ~self.is_mixed | (dp == self.learner_side)

    def stats(self) -> Dict[str, float]:
        return {
            "opp_frac_mixed": float(self.is_mixed.mean()),
            "opp_pool_size": float(len(self.nets)),
            "opp_pool_span_m": float((max(self.steps) - min(self.steps)) / 1e6)
            if self.steps else 0.0,
        }


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
