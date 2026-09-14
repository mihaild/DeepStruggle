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
                 seed: int = 0, lock_learner_side: Optional[int] = None) -> None:
        if not nets:
            raise ValueError("OpponentPool needs at least one frozen network")
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"frac must be in (0, 1], got {frac}")
        self.nets = list(nets)
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
