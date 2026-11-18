"""P48b's restart pool (ai/training/restart_pool.py): the plans session's rules, live capture from whole games, check-out,
and whole-game statistics kept apart from restarts."""
from __future__ import annotations

import math
from typing import Any, List, Sequence

import numpy as np
import pytest
import ts_engine as ts

from ai.training.generic_trainer import summarize_completed_episodes
from ai.training.restart_pool import (DEFAULT_TURN_WEIGHTS, RestartPool, RestartPosition, admit_probability,
                                      parse_turn_weights, put_back_probability)
from bindings.ts_env import TsVectorizedEnv


def _state(turn: int) -> Any:
    st = ts.GameState()
    ts.Engine.init_game(st, 4242 + turn)
    st.turn = turn - 1
    return st


def _stocked(num_envs: int = 8, frac: float = 0.5, turn: int = 8, n: int = 3, **kw: Any) -> RestartPool:
    pool = RestartPool(num_envs, frac=frac, turn_weights={turn: 1.0}, seed=1, **kw)
    for _ in range(n):
        pool._admit(RestartPosition(state=_state(turn), turn=turn, admitted_step=0, p_admit=0.5))
    return pool


def test_the_put_back_rule_is_the_plans() -> None:
    n, m, v = 3, 2 / 3, 0.4
    mp = (n * m + 2 * v) / (n + 2)
    z = min(2.0, abs(m - v) / math.sqrt(max(v * (1 - v), 0.05) / n))
    assert put_back_probability(n, m, v) == pytest.approx(0.1 + 0.9 * 4 * mp * (1 - mp) * (0.25 + 0.75 * z / 2))
    # Decided and agreed: the floor; open and disputed: near certain.
    assert put_back_probability(4, 1.0, 0.98) < 0.15
    assert put_back_probability(4, 1.0, 0.5) > 0.5
    assert put_back_probability(4, 1.0, 0.5, use_disagreement=False) < put_back_probability(4, 1.0, 0.5)


def test_admission_is_by_openness_with_a_floor_and_a_higher_one_late() -> None:
    assert admit_probability(0.5, 5) == pytest.approx(1.0)
    assert admit_probability(1.0, 5) == pytest.approx(0.2)
    assert admit_probability(1.0, 9) == pytest.approx(0.4)
    assert admit_probability(0.5, 9) == pytest.approx(1.0)


def test_turn_weights_parse_and_default() -> None:
    w = parse_turn_weights("8:2,9:1,10:1")
    assert w == {8: 0.5, 9: 0.25, 10: 0.25}
    assert sum(DEFAULT_TURN_WEIGHTS.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        parse_turn_weights("1:1")
    pool = RestartPool(512, frac=0.25)
    assert pool.restart_env.sum() == 128 and sum(pool.capacity.values()) == pytest.approx(1000, abs=4)


def test_a_drawn_position_is_checked_out_and_reseeded() -> None:
    pool = _stocked(n=1)
    assert pool.provide(5) is None, "env 5 is not a restart env"
    a = pool.provide(0)
    assert a is not None and pool.size_now() == 0, "checked out while played"
    assert pool.provide(1) is None and pool.no_stock == 1, "one position never runs in two games"
    stored = pool.playing[0]
    assert stored is not None and a.rng_state != stored.state.rng_state, "a fresh deal and dice"


def test_a_finished_continuation_goes_back_or_retires_by_the_rule() -> None:
    pool = _stocked(n=1, max_continuations=2)
    pool.provide(0)
    pool.first_values(np.full(8, 0.5))
    pool.provide(0)                      # the game ended: env 0 asks for its next start
    pool.complete(0, 1.0)                # n = 1, open and disputed: very likely back
    pos_n = [p.n for b in pool.buckets.values() for p in b] + [p.n for p in pool.playing if p is not None]
    assert 1 in pos_n
    # At n = max_continuations it retires whatever the rule says.
    pool.provide(1)
    pool.first_values(np.full(8, 0.5))
    pool.provide(1)
    pool.complete(1, 0.0)
    m = pool.metrics()
    assert m["restart/retired"] >= 1 and m["restart/n_at_retirement"] >= 1


def test_an_old_position_is_retired() -> None:
    pool = _stocked(n=1, max_age_steps=10)
    pool.provide(0)
    pool.first_values(np.full(8, 0.5))
    pool.steps = 100
    pool.provide(0)
    pool.complete(0, 0.5)
    assert pool.metrics()["restart/dropped_old"] == 1.0


def test_live_capture_offers_pre_deal_states_of_whole_games() -> None:
    env = TsVectorizedEnv(num_envs=16, base_seed=3)
    pool = RestartPool(16, frac=0.25, turn_weights={t: 1.0 for t in range(2, 11)}, seed=2)
    env.start_provider = pool.provide
    obs, masks, _ = env.reset_all()
    rng = np.random.default_rng(0)

    def judge(states: Sequence[Any]) -> List[float]:
        return [0.5] * len(states)       # every position open: admitted with probability 1

    for _ in range(3000):
        whole = env.env_start_turn == 1
        pool.before_step(env.runner, whole)
        m = np.asarray(masks)
        actions = np.array([rng.choice(np.flatnonzero(row)) for row in m])
        obs, masks, _, dones, info = env.step(actions)
        pool.after_step(env.runner, whole, dones, judge)
    pool.judge_pending(judge)
    held = [p for b in pool.buckets.values() for p in b]
    assert held, "no position was captured"
    for p in held:
        assert int(p.state.turn) == p.turn - 1, "stored pre-deal, one turn before its target"


def test_whole_game_statistics_leave_restarts_out() -> None:
    whole = [{"turn": 9, "winner": "US", "ending_reason": "20vp", "start_turn": 1} for _ in range(4)]
    restarted = [{"turn": 10, "winner": "USSR", "ending_reason": "final_scoring", "start_turn": 9} for _ in range(4)]
    with_restarts = summarize_completed_episodes(whole + restarted)
    alone = summarize_completed_episodes(whole)
    shared = [k for k in alone if k in with_restarts and not k.startswith("_")]
    assert shared and all(with_restarts[k] == alone[k] for k in shared if "_start" not in k)
    assert any(k.endswith("_restart") for k in with_restarts)


def test_a_full_bucket_refuses_newcomers_and_keeps_a_put_back_its_place() -> None:
    pool = RestartPool(8, frac=0.5, turn_weights={8: 1.0}, capacity=2, seed=3)
    for _ in range(3):
        pool._admit(RestartPosition(state=_state(8), turn=8, admitted_step=0, p_admit=0.5))
    assert pool.size_now() == 2 and pool.dropped_full == 1, "a full bucket refuses, it does not evict"
    pool.provide(0)
    pool._admit(RestartPosition(state=_state(8), turn=8, admitted_step=0, p_admit=0.5))
    assert pool.size_now() == 1 and pool.dropped_full == 2, "the checked-out position keeps its place"
    pool.first_values(np.full(8, 0.5))
    pool.provide(0)                      # its game ended
    pool.rng.random = lambda: 0.0        # type: ignore[method-assign]  # the rule says: put back
    pool.complete(0, 1.0)
    # Env 0's next start checked out the other position, so one is back in the bucket and one is out.
    assert pool.size_now() == 1 and pool.out[8] == 1 and pool.put_back == 1, "the put-back fits in its own place"
