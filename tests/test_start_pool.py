"""Mid-game start positions: capture, reseeding, sampling, and injection into envs.

The whole point is to raise the sample rate on states self-play reaches rarely and plays
badly -- only about a fifth of games reach turn 10, and the same eight battlegrounds sit
untouched from turn 8 on. Two properties have to hold or the idea backfires:

* one stored position must yield *different* futures, or N rollouts from it are N copies
  of one deal and one set of dice;
* the stored position must survive being played from, or the pool erodes as it is used.
"""

import collections

import pytest
import ts_engine as ts

from ai.training.start_pool import (DEFAULT_TURN_MIX, PooledPosition,
                                    StartPositionPool)


def _position(turn: int, generation: int = 1) -> PooledPosition:
    state = ts.GameState()
    ts.Engine.init_game(state, 1000 + turn + generation)
    state.turn = turn - 1          # stored pre-deal, one turn behind its target
    return PooledPosition(state=state, turn=turn, victory_points=0,
                          region_net=0, generation=generation)


def test_turn_mix_sums_to_one_and_keeps_half_on_the_real_opening() -> None:
    """Start sampling pulls the optimised distribution away from the real game."""
    assert abs(sum(DEFAULT_TURN_MIX.values()) - 1.0) < 1e-9
    assert DEFAULT_TURN_MIX[1] == 0.50


def test_sample_returns_a_clone_with_a_fresh_seed() -> None:
    """Handing out the stored object would let training step it to death."""
    pool = StartPositionPool(turns=(6,), seed=1)
    pool._admit(_position(6))
    stored = pool.buckets[6][0].state
    original_turn = int(stored.turn)

    a = pool.sample(6)
    b = pool.sample(6)
    assert a is not None and b is not None
    assert a.rng_state != b.rng_state, "two draws must not share a future"

    a.turn = 99  # scribble on the copy
    assert int(pool.buckets[6][0].state.turn) == original_turn, "pool was mutated"


def test_sample_of_an_empty_bucket_is_none() -> None:
    assert StartPositionPool(turns=(8,)).sample(8) is None


def test_assign_starts_falls_back_to_turn_one_when_a_bucket_is_empty() -> None:
    """A run must be able to begin before the pool has been filled."""
    pool = StartPositionPool(turns=(4, 6), seed=2)
    assignment = pool.assign_starts(200, DEFAULT_TURN_MIX)
    assert len(assignment) == 200
    assert all(t is None for t in assignment), "empty pool must yield only real openings"

    for _ in range(50):
        pool._admit(_position(6))
    assignment = pool.assign_starts(400, DEFAULT_TURN_MIX)
    used = collections.Counter(t for t in assignment if t is not None)
    assert set(used) == {6}, f"only turn 6 is stocked, got {dict(used)}"
    # 15% of envs ask for turn 6; the rest fall back. Wide bounds -- this is sampling.
    assert 0.05 < used[6] / 400 < 0.30


def test_capacity_and_staleness_are_enforced() -> None:
    """A pool that keeps everything becomes a museum of a policy that no longer exists."""
    pool = StartPositionPool(turns=(4,), capacity_per_turn=10, max_generations=2)
    for gen in (1, 2, 3):
        pool.generation = gen
        for _ in range(8):
            pool._admit(_position(4, generation=gen))
    assert pool.size(4) == 10, "capacity not enforced"

    pool.generation = 5
    pool.retire_stale()
    assert all(p.generation > 3 for p in pool.buckets[4]), "stale generations kept"


def test_harvest_stores_pre_deal_positions() -> None:
    """Captured one turn *before* the target, so resuming deals fresh cards.

    Capturing after the deal would fix the hand, and every rollout from that position
    would replay the same draw.
    """
    torch = pytest.importorskip("torch")
    from ai.models.coldwar_net_v2 import create_coldwar_net_v2

    pool = StartPositionPool(turns=(2, 3), capacity_per_turn=32, seed=5)
    stats = pool.harvest(create_coldwar_net_v2("cpu"), num_envs=16, num_episodes=12)

    assert stats.episodes > 0
    assert pool.size() > 0, "harvest produced nothing"
    for turn, bucket in pool.buckets.items():
        for pos in bucket:
            assert int(pos.state.turn) == turn - 1, (
                f"position targeting turn {turn} was stored at turn {int(pos.state.turn)}; "
                f"it must be the pre-deal state one turn earlier"
            )


def test_env_starts_from_an_injected_position() -> None:
    """The start_provider hook must actually place the env in the pooled position."""
    from bindings.ts_env import TsVectorizedEnv

    pool = StartPositionPool(turns=(6,), seed=9)
    pool._admit(_position(6))

    env = TsVectorizedEnv(num_envs=4, base_seed=321,
                          start_provider=lambda i: pool.sample(6) if i % 2 == 0 else None)
    env.reset_all()

    turns = [int(env.runner.get_state(i).turn) for i in range(4)]
    assert turns[0] == 5 and turns[2] == 5, f"injection did not take: {turns}"
    assert turns[1] == 1 and turns[3] == 1, f"None must leave the real opening: {turns}"
