"""The exploiter that pursues one way of winning (--target-endings): its payoff, and the zero-sum rewards GAE reads."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pytest
import ts_engine as ts

from ai.rewards.reward_calculator import EndingTargetedReward
from bindings.ts_env import TsVectorizedEnv


def _finished_game(seed: int) -> Any:
    env = TsVectorizedEnv(num_envs=1, base_seed=seed, auto_reset=False)
    obs, masks, _ = env.reset_all()
    rng = np.random.default_rng(seed)
    for _ in range(5000):
        st = env.runner.get_state(0)
        if ts.Engine.is_terminal(st):
            return st
        a = int(rng.choice(np.flatnonzero(np.asarray(masks)[0])))
        obs, masks, _, dones, _ = env.step(np.array([a]))
    raise AssertionError("no finished game")


def test_the_learner_is_paid_only_for_its_target_ending() -> None:
    games = [_finished_game(s) for s in range(40)]
    seen = set()
    for st in games:
        u = float(ts.Engine.get_terminal_utility(st))
        reason = str(ts.game_ending_reason(st))
        for side in (1, -1):
            for target in ("20vp", "europe_control", "defcon", "wargames"):
                r = EndingTargetedReward([target], side)
                p = r.payoff(u, st)
                if u == 0:
                    assert p == 0.0
                elif (u > 0) != (side > 0):
                    assert p == -1.0, "any loss is -1"
                else:
                    family = {"20 VP": "20vp", "Europe Control": "europe_control", "wargames": "wargames",
                              "DEFCON 1 (own decision)": "defcon", "DEFCON 1 (opponent decision)": "defcon"}.get(reason)
                    assert p == (1.0 if family == target else 0.0), (reason, target)
                    seen.add((family, target, p))
    assert any(p == 1.0 for _, _, p in seen) and any(p == 0.0 for _, _, p in seen)


def test_rewards_are_the_learners_payoff_in_each_movers_frame() -> None:
    st = _finished_game(3)
    u = float(ts.Engine.get_terminal_utility(st))
    r = EndingTargetedReward(["20vp", "europe_control", "defcon", "wargames", "final_scoring"], -1)
    pay = r.payoff(u, st)                     # every ending targeted: the plain win/loss from the USSR's seat
    assert pay == (0.0 if u == 0 else -float(np.sign(u)))
    rew = r.compute_step_rewards(np.array([1, -1]), np.array([True, True]), np.array([u, u]), np.zeros(2), np.zeros(2),
                                 [st, st])
    assert rew[1] == pytest.approx(pay) and rew[0] == pytest.approx(-pay), "zero-sum from the US's side"


def test_it_needs_a_seat_and_known_endings() -> None:
    with pytest.raises(ValueError):
        EndingTargetedReward(["20vp"], 2)
    with pytest.raises(RuntimeError):          # both seats (0) without the pool's seats attached
        EndingTargetedReward(["20vp"], 0).compute_step_rewards(np.array([1]), np.array([True]), np.array([1.0]),
                                                               np.zeros(1), np.zeros(1), [None])
    with pytest.raises(ValueError):
        EndingTargetedReward(["checkmate"], 1)
    no_state: Optional[Any] = None
    with pytest.raises(ValueError):
        EndingTargetedReward(["20vp"], 1).payoff(1.0, no_state)


def test_both_seats_read_each_games_seat_from_the_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Owner, 2026-11-20: a both-seat Europe Control exploiter -- env 0 its US game, env 1 its USSR game, both won by
    the US by Europe Control: +1 for the US exploiter, -1 for the USSR one (as the US mover's rewards: +1 and +1)."""
    r = EndingTargetedReward(["europe_control"], 0)
    sides = np.array([1, -1], dtype=np.int8)
    r.attach_sides(lambda: sides)
    st = ts.GameState()
    monkeypatch.setattr(ts, "game_ending_reason", lambda state: "Europe Control")
    rew = r.compute_step_rewards(np.array([1, 1]), np.array([True, True]), np.array([1.0, 1.0]),
                                 np.zeros(2), np.zeros(2), [st, st])
    assert rew[0] == 1.0          # US exploiter won by Europe Control
    assert rew[1] == 1.0          # the USSR exploiter lost: -1 for it, +1 for the US mover
