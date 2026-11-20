"""The per-region exploiter's reward (ai/rewards/region_reward.py): scorings of its region paid by their VP, final scoring's
share, a small win bonus, -1 for a loss, written zero-sum from the US's side."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import ts_engine as ts

from ai.rewards.region_reward import RegionTargetedReward, final_share_us, reached_final_scoring
from bindings.ts_env import TsVectorizedEnv


def _finished(seed: int) -> Any:
    env = TsVectorizedEnv(num_envs=1, base_seed=seed, auto_reset=False)
    _, masks, _ = env.reset_all()
    rng = np.random.default_rng(seed)
    for _ in range(6000):
        st = env.runner.get_state(0)
        if ts.Engine.is_terminal(st):
            return st
        _, masks, _, _, _ = env.step(np.array([int(rng.choice(np.flatnonzero(np.asarray(masks)[0])))]))
    raise AssertionError("no finished game")


def _game(r: RegionTargetedReward, scorings_us: list[float], u: float, st: Any) -> float:
    """The learner's total reward over a game with these region scorings (+ US) and terminal utility `u`."""
    total = 0.0
    for vp in scorings_us:
        r.pending_us[:] = [vp]
        total += float(r.compute_step_rewards(np.array([r.learner_side]), np.array([False]), np.zeros(1), np.zeros(1),
                                              np.zeros(1))[0])
    total += float(r.compute_step_rewards(np.array([r.learner_side]), np.array([True]), np.array([u]), np.zeros(1),
                                          np.zeros(1), [st])[0])
    return total


def test_a_win_pays_the_region_in_full_a_loss_a_small_share() -> None:
    st = _finished(11)
    share = final_share_us(st, "asia") if reached_final_scoring(st) else 0.0
    for side in (1, -1):
        def make() -> RegionTargetedReward:
            return RegionTargetedReward("asia", learner_side=side, scale=10.0, num_envs=1)
        vp = 5.0 * side            # +5 for the learner from an Asia scoring
        x = float(np.clip((vp + share) * side / 10.0, -1.0, 2.0))
        assert _game(make(), [vp], float(side), st) == pytest.approx(0.2 + 0.8 * x, abs=1e-6)
        assert _game(make(), [vp], -float(side), st) == pytest.approx(-1.0 + 0.2 * 0.8 * x, abs=1e-6)
        assert _game(make(), [vp], 0.0, st) == pytest.approx(0.2 * 0.8 * x, abs=1e-6)


def test_the_loss_share_is_paid_at_the_scoring_and_the_term_is_clipped() -> None:
    r = RegionTargetedReward("asia", learner_side=-1, scale=10.0, num_envs=2)
    r.pending_us[:] = [-5.0, 0.0]          # env 0: Asia scored 5 for the USSR learner
    rew = r.compute_step_rewards(np.array([-1, 1]), np.array([False, False]), np.zeros(2), np.zeros(2), np.zeros(2))
    assert rew[0] == pytest.approx(0.2 * 0.8 * 0.5) and rew[1] == 0.0      # the USSR mover's reward
    r.pending_us[:] = [-40.0, 0.0]         # past the cap (2): only up to it is paid
    rew = r.compute_step_rewards(np.array([-1, 1]), np.array([False, False]), np.zeros(2), np.zeros(2), np.zeros(2))
    assert rew[0] == pytest.approx(0.2 * 0.8 * (2.0 - 0.5))


def test_every_win_pays_more_than_every_loss() -> None:
    st = _finished(11)
    worst_win = _game(RegionTargetedReward("africa", 1, 5.0, 1), [-100.0], 1.0, st)
    best_loss = _game(RegionTargetedReward("africa", 1, 5.0, 1), [100.0], -1.0, st)
    assert worst_win == pytest.approx(0.2 - 0.8) and best_loss == pytest.approx(-1.0 + 0.2 * 0.8 * 2.0)
    assert worst_win > best_loss
    with pytest.raises(ValueError):
        RegionTargetedReward("africa", 1, 5.0, 1, loss_share=0.5)    # best loss -0.2 > worst win -0.6


def test_an_attached_env_pays_each_scoring_of_the_region_the_engine_records_at_its_step() -> None:
    r = RegionTargetedReward("asia", learner_side=1, scale=10.0, num_envs=64, weight=0.8, win_bonus=0.2)
    env = TsVectorizedEnv(num_envs=64, base_seed=7, reward_calculator=r)
    r.attach(env.runner)
    _, masks, _ = env.reset_all()
    rng = np.random.default_rng(3)
    paid = 0
    for _ in range(4000):
        movers = np.asarray(env.runner.get_decision_players(), dtype=np.float64)
        acts = np.array([int(rng.choice(np.flatnonzero(m))) for m in np.asarray(masks)])
        _, masks, rew, dones, _ = env.step(acts)
        rec = np.asarray(env.runner.get_step_track_vp())
        asia = rec[:, [1, 38]].sum(axis=1)
        for e in np.flatnonzero(~np.asarray(dones, dtype=bool)):
            # The US learner's loss share of the scoring, as the mover's reward (within the clip).
            before, after = r.scored_vp[e] - asia[e], r.scored_vp[e]
            if -10.0 <= before <= 20.0 and -10.0 <= after <= 20.0:
                assert rew[e] == pytest.approx(0.2 * 0.8 * asia[e] / 10.0 * movers[e], abs=1e-6), (e, asia[e], rew[e])
            paid += int(asia[e] != 0)
    assert paid > 0, "no Asia scoring in 4,000 random steps of 64 games"


def test_it_needs_a_seat_or_both_scales_a_region_and_a_scale() -> None:
    with pytest.raises(ValueError):
        RegionTargetedReward("asia", 0, 10.0, 1)
    with pytest.raises(ValueError):
        RegionTargetedReward("antarctica", 1, 10.0, 1)
    with pytest.raises(ValueError):
        RegionTargetedReward("asia", 1, 0.0, 1)


def test_both_seats_read_each_games_seat_and_its_scale() -> None:
    """One exploiter playing both seats (owner, 2026-11-19): env 0 is its US game, env 1 its USSR game; each is paid from
    its own seat with that seat's scale, and the record of who scored is the same Asia scoring (+5 for the US)."""
    r = RegionTargetedReward("asia", learner_side=0, scale=10.0, num_envs=2, scale_ussr=5.0)
    sides = np.array([1, -1], dtype=np.int8)
    r.attach_sides(lambda: sides)
    r.pending_us[:] = [5.0, 5.0]
    rew = r.compute_step_rewards(np.array([1, -1]), np.array([False, False]), np.zeros(2), np.zeros(2), np.zeros(2))
    # US exploiter: +0.16 * 0.5 as the US mover; USSR exploiter: -0.16 * 1.0 for it, as the USSR mover's reward.
    assert rew[0] == pytest.approx(0.2 * 0.8 * 0.5) and rew[1] == pytest.approx(-0.2 * 0.8 * 1.0)
    assert r.scored_vp.tolist() == [5.0, -5.0]


def test_both_seats_need_the_pool_and_both_scales() -> None:
    with pytest.raises(ValueError):
        RegionTargetedReward("asia", 0, 10.0, 1)
    r = RegionTargetedReward("asia", 0, 10.0, 1, scale_ussr=8.0)
    with pytest.raises(RuntimeError):
        r.compute_step_rewards(np.array([1]), np.array([False]), np.zeros(1), np.zeros(1), np.zeros(1))


def test_a_europe_control_win_pays_at_least_the_full_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Owner, 2026-11-20: the Europe exploiter is paid at least 1 for winning by Europe Control, whatever its VP. Random
    games almost never end that way, so a finished game is read as one (the ending reason is the only input)."""
    st = _finished(11)
    monkeypatch.setattr(ts, "game_ending_reason", lambda state: "Europe Control")
    for side in (1, -1):
        r = RegionTargetedReward("europe", learner_side=side, scale=10.0, num_envs=1)
        assert _game(r, [-30.0 * side], float(side), st) == pytest.approx(1.0, abs=1e-6)   # x at the floor, then the win
        r = RegionTargetedReward("europe", learner_side=side, scale=10.0, num_envs=1)
        assert _game(r, [30.0 * side], float(side), st) == pytest.approx(0.2 + 0.8 * 2.0, abs=1e-6)  # above 1 stays
        r2 = RegionTargetedReward("asia", learner_side=side, scale=10.0, num_envs=1)
        assert _game(r2, [-30.0 * side], float(side), st) == pytest.approx(0.2 - 0.8, abs=1e-6)    # only Europe's own
