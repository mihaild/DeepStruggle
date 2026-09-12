"""A pool of mid-game positions to start training rollouts from.

Self-play from turn 1 reaches the late game rarely and badly: only about a fifth of games
reach turn 10, and eight of the 29 battlegrounds sit untouched from turn 8 onwards -- always
the same ones, always those the opening setup did not seed. The agent fights where it was
placed and has not learned to create access anywhere else. Starting a share of rollouts
from saved mid-game positions raises the sample rate on exactly those states.

Two properties make this safe rather than a way to memorise a stale distribution:

* Positions are captured at the **last decision before a turn rolls over**, so the new
  turn's cards have not been dealt yet. Reseeding rng_state before resuming makes the
  engine deal fresh -- verified as three distinct hands for both sides from one captured
  position. Without it, N rollouts from one position would be N copies of one future.
* The pool is **tagged by the snapshot that produced it** and refreshed incrementally, so
  it tracks the policy instead of preserving a policy that no longer exists.

A position is only worth resuming from if it is undecided, which is judged on the balance
between the sides rather than either side's absolute holdings -- see
ai.eval.position_diagnostics.is_salvageable. A decided position contributes no gradient at
all: with terminal-only reward every action there returns the same value.
"""

from __future__ import annotations

import collections
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ts_engine as ts

from ai.eval.position_diagnostics import is_salvageable

# Share of environments started from each turn. Turn 1 keeps the majority: start sampling
# shifts the distribution the policy is optimised for away from the real game, and the real
# game is what it is graded on. Turn 10 gets its own slice because the last turn is a
# distinct skill -- no further cards are dealt, so everything held has to be spent.
DEFAULT_TURN_MIX: Dict[int, float] = {1: 0.50, 4: 0.15, 6: 0.15, 8: 0.10, 10: 0.10}

UINT64 = 1 << 64


@dataclass
class PooledPosition:
    state: ts.GameState
    turn: int
    victory_points: int
    region_net: int
    generation: int


@dataclass
class HarvestStats:
    """What a harvest actually yielded, per turn. Recorded so pool yield is visible.

    Yield falling is not by itself bad news. If the agent learns to punish early mistakes,
    games decide sooner and fewer late positions stay salvageable -- that is improvement,
    not regression. These numbers size the harvest; they do not score the policy.
    """
    episodes: int = 0
    reached: Dict[int, int] = field(default_factory=lambda: collections.defaultdict(int))
    salvageable: Dict[int, int] = field(default_factory=lambda: collections.defaultdict(int))

    def as_metrics(self) -> Dict[str, float]:
        out: Dict[str, float] = {"pool/harvest_episodes": float(self.episodes)}
        for turn in sorted(self.reached):
            n = self.reached[turn]
            out[f"pool/reached_turn{turn}"] = n / self.episodes if self.episodes else 0.0
            out[f"pool/salvageable_given_reached_turn{turn}"] = (
                self.salvageable[turn] / n if n else 0.0
            )
        return out


class StartPositionPool:
    """Mid-game positions bucketed by turn, refreshed as the policy moves on."""

    def __init__(
        self,
        turns: Sequence[int] = (4, 6, 8, 10),
        capacity_per_turn: int = 512,
        max_generations: int = 3,
        seed: int = 0,
    ):
        self.turns = tuple(turns)
        self.capacity_per_turn = capacity_per_turn
        self.max_generations = max_generations
        self.generation = 0
        self.rng = random.Random(seed)
        self.buckets: Dict[int, List[PooledPosition]] = {t: [] for t in self.turns}

    # -- state ---------------------------------------------------------------------------

    def size(self, turn: Optional[int] = None) -> int:
        if turn is not None:
            return len(self.buckets.get(turn, []))
        return sum(len(v) for v in self.buckets.values())

    def is_ready(self, turn: int, minimum: int = 1) -> bool:
        return len(self.buckets.get(turn, [])) >= minimum

    def _admit(self, pos: PooledPosition) -> None:
        bucket = self.buckets.setdefault(pos.turn, [])
        bucket.append(pos)
        if len(bucket) > self.capacity_per_turn:
            # Drop oldest generations first, then oldest within a generation.
            bucket.sort(key=lambda p: p.generation)
            del bucket[: len(bucket) - self.capacity_per_turn]

    def retire_stale(self) -> int:
        """Drop positions older than max_generations. Returns how many went."""
        cutoff = self.generation - self.max_generations
        dropped = 0
        for turn, bucket in self.buckets.items():
            keep = [p for p in bucket if p.generation > cutoff]
            dropped += len(bucket) - len(keep)
            self.buckets[turn] = keep
        return dropped

    # -- sampling ------------------------------------------------------------------------

    def sample(self, turn: int, rng: Optional[random.Random] = None) -> Optional[ts.GameState]:
        """A fresh copy of a pooled position, reseeded so its future is its own.

        The clone matters as much as the reseed: handing out the stored object would let
        the training loop step it, destroying the pooled position.
        """
        bucket = self.buckets.get(turn)
        if not bucket:
            return None
        r = rng or self.rng
        state = r.choice(bucket).state.clone()
        state.rng_state = r.getrandbits(64) % UINT64
        return state

    def assign_starts(
        self,
        num_envs: int,
        turn_mix: Optional[Dict[int, float]] = None,
        rng: Optional[random.Random] = None,
    ) -> List[Optional[int]]:
        """Which turn each env should start from; None means the real opening.

        Turns with an empty bucket fall back to turn 1 rather than stalling, so a run can
        begin before the pool has been filled.
        """
        mix = turn_mix or DEFAULT_TURN_MIX
        r = rng or self.rng
        turns = list(mix)
        weights = [mix[t] for t in turns]
        out: List[Optional[int]] = []
        for _ in range(num_envs):
            t = r.choices(turns, weights=weights, k=1)[0]
            if t == 1 or not self.is_ready(t):
                out.append(None)
            else:
                out.append(t)
        return out

    # -- harvesting ----------------------------------------------------------------------

    def harvest(
        self,
        model: Any,
        num_envs: int = 256,
        num_episodes: int = 400,
        base_seed: int = 913_000,
        temperature: float = 1.0,
        max_iters: int = 20_000,
    ) -> HarvestStats:
        """Play games and keep the salvageable turn-boundary positions they pass through.

        Runs at a higher temperature than evaluation: the pool wants variety, and positions
        drawn from a near-greedy policy would all look alike.

        A candidate is stored when the *previous* decision of turn N-1 is still current, so
        the new hand has not been dealt. Capturing after the deal would fix the cards, and
        every rollout from that position would replay the same draw.
        """
        import numpy as np
        import torch

        from bindings.ts_env import TsVectorizedEnv

        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        self.generation += 1
        stats = HarvestStats()

        env = TsVectorizedEnv(num_envs=num_envs, base_seed=base_seed + self.generation)
        obs, masks, _ = env.reset_all()

        wanted = set(self.turns)
        # The state as it was just before each env's most recent action, which is the only
        # moment a turn-boundary crossing can be captured pre-deal.
        previous: List[Optional[ts.GameState]] = [None] * num_envs
        previous_turn: List[int] = [0] * num_envs
        counted: List[set] = [set() for _ in range(num_envs)]

        try:
            for _ in range(max_iters):
                if stats.episodes >= num_episodes:
                    break

                for i in range(num_envs):
                    state = env.runner.get_state(i)
                    if ts.Engine.is_terminal(state):
                        continue
                    previous[i] = state.clone()
                    previous_turn[i] = int(state.turn)

                obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device)
                mask_t = torch.from_numpy(np.asarray(masks)).to(device)
                with torch.no_grad():
                    actions, _, _, _, _ = model.sample_action(
                        obs_t, mask_t, temperature=temperature)
                obs, masks, _, dones, _ = env.step(actions.cpu().numpy())

                for i in range(num_envs):
                    if dones[i]:
                        stats.episodes += 1
                        previous[i], previous_turn[i], counted[i] = None, 0, set()
                        continue
                    state = env.runner.get_state(i)
                    new_turn = int(state.turn)
                    prev = previous[i]
                    if prev is None or new_turn != previous_turn[i] + 1:
                        continue
                    if new_turn in counted[i]:
                        continue
                    counted[i].add(new_turn)
                    if new_turn not in wanted:
                        continue
                    stats.reached[new_turn] += 1
                    # prev is the pre-deal state; judge salvageability on the position the
                    # rollout would actually resume into.
                    if not is_salvageable(state):
                        continue
                    stats.salvageable[new_turn] += 1
                    us, ussr = _region_net(state)
                    self._admit(PooledPosition(
                        state=prev, turn=new_turn, victory_points=int(state.victory_points),
                        region_net=us - ussr, generation=self.generation,
                    ))
        finally:
            if was_training:
                model.train()

        self.retire_stale()
        return stats


def _region_net(state: ts.GameState) -> Tuple[int, int]:
    from ai.eval.position_diagnostics import region_score_sums
    return region_score_sums(state)
