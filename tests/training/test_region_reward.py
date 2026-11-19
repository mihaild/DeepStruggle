"""The per-region exploiter's reward (ai/rewards/region_reward.py): scorings of its region paid by their VP, final scoring's
share, a small win bonus, -1 for a loss, written zero-sum from the US's side."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import ts_engine as ts

from ai.rewards.region_reward import (RegionTargetedReward, final_share_us, reached_final_scoring,
                                      scoring_value_us, selected_scoring_card)
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


def test_a_scoring_of_the_region_pays_its_vp_from_the_learners_side_at_that_step() -> None:
    r = RegionTargetedReward("asia", learner_side=-1, scale=10.0, num_envs=2, weight=0.8, win_bonus=0.2)
    r.pending_us[:] = [5.0, 0.0]          # env 0: Asia scored +5 for the US
    rew = r.compute_step_rewards(np.array([1, -1]), np.array([False, False]), np.zeros(2), np.zeros(2), np.zeros(2))
    # The USSR learner loses 0.8 * 5 / 10 = 0.4; the US mover's reward is +0.4, a USSR mover's would be -0.4.
    assert rew[0] == pytest.approx(0.4) and rew[1] == 0.0
    assert r.pending_us.sum() == 0.0, "paid once"


def test_the_end_pays_the_outcome_and_final_scorings_share() -> None:
    st = _finished(11)
    u = float(ts.Engine.get_terminal_utility(st))
    share = final_share_us(st, "europe") if reached_final_scoring(st) else 0.0
    r = RegionTargetedReward("europe", learner_side=1, scale=10.0, num_envs=1)
    rew = r.compute_step_rewards(np.array([1]), np.array([True]), np.array([u]), np.zeros(1), np.zeros(1), [st])
    outcome = 0.2 if u > 0 else -1.0 if u < 0 else 0.0
    assert rew[0] == pytest.approx(outcome + 0.8 * share / 10.0, abs=1e-6)


def test_won_mode_holds_the_region_part_until_a_win() -> None:
    r = RegionTargetedReward("africa", learner_side=1, scale=5.0, num_envs=1, pay="won")
    r.pending_us[:] = [4.0]
    assert r.compute_step_rewards(np.array([1]), np.array([False]), np.zeros(1), np.zeros(1), np.zeros(1))[0] == 0.0
    assert r.held[0] == pytest.approx(0.8 * 4.0 / 5.0)


def test_scoring_cards_are_seen_and_valued_on_the_board() -> None:
    env = TsVectorizedEnv(num_envs=64, base_seed=7)
    _, masks, _ = env.reset_all()
    rng = np.random.default_rng(3)
    seen = 0
    for _ in range(4000):
        dp = np.asarray(env.runner.get_decision_players())
        acts = np.array([int(rng.choice(np.flatnonzero(m))) for m in np.asarray(masks)])
        for e in np.flatnonzero(dp != 0):
            st = env.runner.get_state(int(e))
            card = selected_scoring_card(st, int(acts[e]))
            if card:
                v = scoring_value_us(st, card)
                assert float(v).is_integer() and abs(v) <= 20, (card, v)
                seen += 1
        _, masks, _, _, _ = env.step(acts)
    assert seen > 0, "no scoring card played in 4,000 random steps"


def test_it_needs_a_locked_seat_a_region_and_a_scale() -> None:
    with pytest.raises(ValueError):
        RegionTargetedReward("asia", 0, 10.0, 1)
    with pytest.raises(ValueError):
        RegionTargetedReward("antarctica", 1, 10.0, 1)
    with pytest.raises(ValueError):
        RegionTargetedReward("asia", 1, 0.0, 1)
