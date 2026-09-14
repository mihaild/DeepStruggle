"""Guards for frozen-opponent sampling.

The property that matters most is the first one: with no pool configured, nothing about training
may change. Everything else here is a mechanism this codebase has not had before, so each piece
is pinned against an answer worked out independently of the code.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from ai.training.opponent_pool import OpponentPool
from ai.training.rollout_buffer import RolloutBuffer


def _buf(**kw) -> RolloutBuffer:
    return RolloutBuffer(buffer_size=4, num_envs=3, obs_dim=8, action_dim=212,
                         device="cpu", **kw)


def test_learner_mask_defaults_to_all_ones() -> None:
    """Without a pool the mask must be inert, or every existing run changes meaning."""
    b = _buf()
    assert torch.all(b.learner == 1.0)
    b.add(obs=torch.zeros(3, 8), masks=torch.ones(3, 212, dtype=torch.uint8),
          actions=torch.zeros(3, dtype=torch.long), log_probs=torch.zeros(3),
          rewards=np.zeros(3, dtype=np.float32), dones=torch.zeros(3),
          values_win=torch.zeros(3), values_vp=torch.zeros(3),
          players=torch.ones(3, dtype=torch.int8))
    assert torch.all(b.learner[0] == 1.0)


def test_learner_mask_round_trips_through_get_batches() -> None:
    b = _buf()
    for t in range(4):
        b.add(obs=torch.zeros(3, 8), masks=torch.ones(3, 212, dtype=torch.uint8),
              actions=torch.zeros(3, dtype=torch.long), log_probs=torch.zeros(3),
              rewards=np.zeros(3, dtype=np.float32), dones=torch.zeros(3),
              values_win=torch.zeros(3), values_vp=torch.zeros(3),
              players=torch.ones(3, dtype=torch.int8),
              learner=np.array([1.0, 0.0, 1.0], dtype=np.float32))
    batch = next(b.get_batches(batch_size=12))
    assert len(batch) == 9, "the learner mask is the ninth element"
    assert batch[-1].sum().item() == pytest.approx(8.0), "2 of 3 envs x 4 steps"


def test_pool_marks_the_configured_fraction_mixed() -> None:
    nets = [create_coldwar_net_v2()]
    pool = OpponentPool(nets, num_envs=100, frac=0.25, seed=1)
    assert pool.is_mixed.sum() == 25
    assert pool.stats()["opp_frac_mixed"] == pytest.approx(0.25)


def test_self_play_envs_always_count_as_learner() -> None:
    """In a non-mixed env the learner plays both sides, so every transition is its own."""
    pool = OpponentPool([create_coldwar_net_v2()], num_envs=8, frac=0.5, seed=2)
    for dp in (np.ones(8, dtype=np.int8), -np.ones(8, dtype=np.int8)):
        acts = pool.learner_acts(dp)
        assert acts[~pool.is_mixed].all(), "self-play envs must always be the learner's"


def test_mixed_envs_alternate_by_side() -> None:
    pool = OpponentPool([create_coldwar_net_v2()], num_envs=8, frac=1.0, seed=3)
    us_to_move = pool.learner_acts(np.ones(8, dtype=np.int8))
    ussr_to_move = pool.learner_acts(-np.ones(8, dtype=np.int8))
    # Every mixed env is the learner's on exactly one of the two, never both or neither.
    assert np.all(us_to_move != ussr_to_move)
    assert np.array_equal(us_to_move, pool.learner_side == 1)


def test_locked_side_never_alternates() -> None:
    """The frozen-opponent experiment pins the learner to US throughout."""
    pool = OpponentPool([create_coldwar_net_v2()], num_envs=16, frac=1.0, seed=4,
                        lock_learner_side=1)
    assert np.all(pool.learner_side == 1)
    for i in range(16):
        pool.on_episode_end(i)
    assert np.all(pool.learner_side == 1), "an episode boundary must not unpin it"
    assert pool.learner_acts(np.ones(16, dtype=np.int8)).all()
    assert not pool.learner_acts(-np.ones(16, dtype=np.int8)).any()


def test_pool_networks_are_frozen() -> None:
    net = create_coldwar_net_v2()
    assert any(p.requires_grad for p in net.parameters())
    OpponentPool([net], num_envs=4, frac=0.5)
    assert not any(p.requires_grad for p in net.parameters())


def test_start_iteration_picks_from_the_pool() -> None:
    nets = [create_coldwar_net_v2() for _ in range(3)]
    pool = OpponentPool(nets, num_envs=4, frac=0.5, seed=7)
    seen = set()
    for _ in range(60):
        pool.start_iteration()
        seen.add(id(pool.current))
    assert seen <= {id(n) for n in nets}
    assert len(seen) > 1, "a multi-snapshot pool should not collapse to one opponent"


def test_rejects_empty_pool_and_bad_fraction() -> None:
    with pytest.raises(ValueError):
        OpponentPool([], num_envs=4)
    with pytest.raises(ValueError):
        OpponentPool([create_coldwar_net_v2()], num_envs=4, frac=0.0)
    with pytest.raises(ValueError):
        OpponentPool([create_coldwar_net_v2()], num_envs=4, frac=1.5)


def _trainer(num_envs: int = 16, pool=None):
    from ai.training.nash_pg import NashPGTrainer
    from bindings.ts_env import TsVectorizedEnv

    env = TsVectorizedEnv(num_envs=num_envs, base_seed=1234)
    tr = NashPGTrainer(active_net=create_coldwar_net_v2(), env=env, num_envs=num_envs,
                       buffer_size=8, device="cpu")
    tr.opponent_pool = pool
    return tr


def test_rollout_without_a_pool_marks_every_transition_as_the_learner() -> None:
    tr = _trainer()
    tr.collect_rollouts()
    assert torch.all(tr.buffer.learner == 1.0)


def test_rollout_with_a_pool_marks_only_the_learner_side() -> None:
    """The mask must actually track who moved, not default to all-ones under a pool."""
    tr = _trainer()
    tr.opponent_pool = OpponentPool([create_coldwar_net_v2()], num_envs=16, frac=0.5, seed=11)
    tr.collect_rollouts()
    frac = float(tr.buffer.learner.mean())
    # Half the envs are self-play (all learner) and half are mixed (roughly half learner),
    # so the overall share should sit well inside (0.5, 1.0) rather than at either end.
    assert 0.55 < frac < 0.95, f"learner share {frac:.3f} looks like the mask is not being set"


def test_opponent_actions_come_from_the_opponent_network() -> None:
    """Route-checking by construction: an opponent that can only pick one action must show up."""
    import ts_engine as ts

    from bindings.ts_env import TsVectorizedEnv
    from ai.training.nash_pg import NashPGTrainer

    class OnlyFirstLegal(torch.nn.Module):
        """Stands in for a frozen policy: always the lowest-indexed legal action."""

        def __init__(self) -> None:
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, obs, mask=None):
            n = obs.shape[0]
            logits = torch.full((n, 212), -1e9)
            if mask is not None:
                first = torch.argmax(mask.float(), dim=-1)
                logits[torch.arange(n), first] = 10.0
            return logits, torch.zeros(n, 1), torch.zeros(n, 1)

    env = TsVectorizedEnv(num_envs=8, base_seed=99)
    tr = NashPGTrainer(active_net=create_coldwar_net_v2(), env=env, num_envs=8,
                       buffer_size=6, device="cpu")
    tr.opponent_pool = OpponentPool([OnlyFirstLegal()], num_envs=8, frac=1.0, seed=5,
                                    lock_learner_side=1)
    tr.collect_rollouts()

    learner = tr.buffer.learner.bool()
    opp_actions = tr.buffer.actions[~learner]
    opp_masks = tr.buffer.masks.reshape(-1, 212)[(~learner).reshape(-1)]
    if opp_actions.numel() == 0:
        pytest.skip("no opponent transitions in this rollout")
    expected = torch.argmax(opp_masks.float(), dim=-1)
    assert torch.equal(opp_actions.reshape(-1), expected), (
        "opponent transitions did not come from the opponent network")


def test_masked_policy_loss_ignores_opponent_transitions() -> None:
    """The gradient must be identical to training on the learner's subset alone."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    net = torch.nn.Linear(6, 5)
    obs = torch.randn(20, 6)
    act = torch.randint(0, 5, (20,))
    adv = torch.randn(20)
    learner = (torch.arange(20) % 2 == 0).float()

    def grad_of(rows):
        net.zero_grad(set_to_none=True)
        lp = F.log_softmax(net(obs[rows]), dim=-1).gather(1, act[rows].unsqueeze(1)).squeeze(1)
        (-(lp * adv[rows])).mean().backward()
        return torch.cat([p.grad.reshape(-1) for p in net.parameters()
                          if p.grad is not None]).clone()

    subset = grad_of(torch.nonzero(learner > 0.5).squeeze(1))

    net.zero_grad(set_to_none=True)
    lp_all = F.log_softmax(net(obs), dim=-1).gather(1, act.unsqueeze(1)).squeeze(1)
    keep = learner > 0.5
    (-(lp_all * adv))[keep].mean().backward()
    masked = torch.cat([p.grad.reshape(-1) for p in net.parameters()
                        if p.grad is not None]).clone()

    assert torch.allclose(subset, masked, atol=1e-6)


def test_env_mask_array_is_a_reused_buffer() -> None:
    """The hazard the copy in collect_rollouts exists for.

    `env.step()` hands back views into buffers the runner reuses, so a tensor made with
    `torch.from_numpy` aliases memory the next step overwrites. On CUDA `.to(device)` copies and
    this never bites; on CPU it is a no-op, and the buffer then stores the post-step observation
    against the pre-step action.
    """
    from bindings.ts_env import TsVectorizedEnv

    env = TsVectorizedEnv(num_envs=4, base_seed=7)
    _, masks, _ = env.reset_all()
    acts = np.array([int(np.flatnonzero(np.asarray(masks)[i])[0]) for i in range(4)],
                    dtype=np.int64)
    _, masks2, _, _, _ = env.step(acts)
    assert np.shares_memory(np.asarray(masks), np.asarray(masks2)), (
        "env stopped reusing its mask buffer -- the copy in collect_rollouts can be revisited")


def test_buffer_stores_the_pre_step_mask() -> None:
    """End to end: the stored mask must be the one the stored action was chosen from."""
    tr = _trainer(num_envs=8)
    tr.collect_rollouts()
    for t in range(tr.buffer.buffer_size):
        for e in range(8):
            a = int(tr.buffer.actions[t, e])
            assert tr.buffer.masks[t, e, a] > 0, (
                f"action {a} illegal under the mask stored at t={t}, env={e} -- "
                "the buffer is storing a post-step mask")


# --- growing pool -----------------------------------------------------------------------------

def _tiny():
    """A stand-in net: add() only needs .eval() and .parameters()."""
    return torch.nn.Linear(2, 2)


def test_add_grows_the_pool_until_capacity() -> None:
    pool = OpponentPool([_tiny()], num_envs=4, frac=0.5, capacity=4)
    assert pool.stats()["opp_pool_size"] == 1
    for steps in (10_000_000, 20_000_000, 30_000_000):
        pool.add(_tiny(), steps)
    assert pool.stats()["opp_pool_size"] == 4
    assert pool.steps == [0, 10_000_000, 20_000_000, 30_000_000]


def test_eviction_keeps_the_endpoints() -> None:
    """The whole point: dropping the oldest would make the pool all-recent, and then every
    opponent carries the strategy the run converged on -- the window closes."""
    pool = OpponentPool([_tiny()], num_envs=4, frac=0.5, capacity=4)
    for steps in range(10_000_000, 110_000_000, 10_000_000):
        pool.add(_tiny(), steps)
    assert min(pool.steps) == 0, "the earliest snapshot must survive"
    assert max(pool.steps) == 100_000_000, "the most recent must survive"
    assert len(pool.steps) == 4


def test_eviction_keeps_the_pool_spread_not_clustered() -> None:
    pool = OpponentPool([_tiny()], num_envs=4, frac=0.5, capacity=5)
    for steps in range(5_000_000, 205_000_000, 5_000_000):
        pool.add(_tiny(), steps)
    kept = sorted(pool.steps)
    gaps = [b - a for a, b in zip(kept, kept[1:])]
    span = kept[-1] - kept[0]
    # Evenly spread over 5 points means each gap is about a quarter of the span. Allow slack,
    # but a pool that had collapsed to the recent end would fail this badly.
    assert max(gaps) < 0.6 * span, f"pool is clustered: {kept}"
    assert kept[0] == 0 and kept[-1] == 200_000_000


def test_added_nets_are_frozen_and_pool_never_exceeds_capacity() -> None:
    pool = OpponentPool([_tiny()], num_envs=4, frac=0.5, capacity=3)
    for steps in (1_000_000, 2_000_000, 3_000_000, 4_000_000):
        net = _tiny()
        assert any(p.requires_grad for p in net.parameters())
        pool.add(net, steps)
        assert not any(p.requires_grad for p in net.parameters())
        assert len(pool.nets) <= 3
    assert len(pool.nets) == len(pool.steps)


def test_capacity_below_three_still_makes_progress() -> None:
    """The spacing rule needs an interior point; with capacity 2 there is none."""
    pool = OpponentPool([_tiny()], num_envs=4, frac=0.5, capacity=2)
    for steps in (1_000_000, 2_000_000, 3_000_000):
        pool.add(_tiny(), steps)
    assert len(pool.nets) == 2
    assert max(pool.steps) == 3_000_000, "the newest must always be kept"


def test_rejects_bad_capacity() -> None:
    with pytest.raises(ValueError):
        OpponentPool([_tiny()], num_envs=4, frac=0.5, capacity=0)


# --- game statistics must describe self-play only ---------------------------------------------

def test_selfplay_mask_is_all_true_without_a_pool() -> None:
    tr = _trainer(num_envs=8)
    tr.collect_rollouts()
    assert tr._selfplay_mask.all(), "no pool means every environment is self-play"


def test_selfplay_mask_matches_the_pool() -> None:
    tr = _trainer(num_envs=8)
    tr.opponent_pool = OpponentPool([create_coldwar_net_v2()], num_envs=8, frac=0.5, seed=3)
    tr.collect_rollouts()
    assert np.array_equal(tr._selfplay_mask, ~tr.opponent_pool.is_mixed)
    assert tr._selfplay_mask.sum() == 4


def test_episode_stats_exclude_pool_games() -> None:
    """Win rate and ending mix describe how the policy plays. Games against a frozen opponent
    are a different question, and mixing them in makes a pooled run incomparable with one that
    has no pool at all."""
    tr = _trainer(num_envs=8)
    tr.opponent_pool = OpponentPool([create_coldwar_net_v2()], num_envs=8, frac=0.5, seed=3)
    metrics = tr.collect_rollouts()
    mixed = set(np.flatnonzero(tr.opponent_pool.is_mixed).tolist())
    reported = {int(e["env_idx"]) for e in metrics["completed_episodes"]}
    assert not (reported & mixed), f"pool games leaked into episode stats: {reported & mixed}"


def test_critic_tracker_only_resolves_selfplay_envs() -> None:
    """Otherwise critic/base_rate reports the pool's difficulty rather than the policy's own
    balance, and AUC is measured against the wrong baseline."""
    from ai.training.critic_tracker import CriticTracker

    tr = _trainer(num_envs=8)
    tr.opponent_pool = OpponentPool([create_coldwar_net_v2()], num_envs=8, frac=0.5, seed=3)
    seen = []
    real_resolve = CriticTracker.resolve

    def spy(self, env_index, us_won):
        if us_won is not None:
            seen.append(env_index)
        return real_resolve(self, env_index, us_won)

    CriticTracker.resolve = spy
    try:
        for _ in range(3):
            tr.collect_rollouts()
    finally:
        CriticTracker.resolve = real_resolve

    mixed = set(np.flatnonzero(tr.opponent_pool.is_mixed).tolist())
    assert not (set(seen) & mixed), f"critic resolved pool games: {set(seen) & mixed}"
