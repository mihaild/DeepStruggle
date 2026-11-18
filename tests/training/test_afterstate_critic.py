"""The critic at every moment of a side's turn (ai/training/afterstate_critic.py) and the player that reads
it after each legal move (ai/search/critic_argmax.py)."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import ts_engine as ts

from ai.search.critic_argmax import CriticArgmaxAgent
from ai.training.afterstate_critic import (AFTER_DECISION, AFTER_ROLL, BEFORE_ROLL, AfterstateCapture,
                                           AfterstateRows, build_targets, next_decision_target, walk_afterstates)
from bindings.ts_env import TsVectorizedEnv


def _play(steps: int, frac: float = 1.0) -> Any:
    env = TsVectorizedEnv(num_envs=16, base_seed=5)
    obs, masks, _ = env.reset_all()
    cap = AfterstateCapture(frac, 0, obs.shape[1], forced=bool(env._auto_advance), seed=3)
    rng = np.random.default_rng(1)
    rows = []
    for t in range(steps):
        acts = np.array([rng.choice(np.flatnonzero(m)) for m in np.asarray(masks)])
        cap.before(env.runner, np.ones(16, bool), np.asarray(env.runner.get_decision_players()), acts)
        obs, masks, _, dones, _ = env.step(acts)
        r = cap.after(env.runner, t, np.asarray(dones) > 0.5, "cpu")
        if r is not None:
            rows.append(r)
    return cap, rows, obs.shape[1]


def test_every_walk_ends_where_the_runner_does_and_all_three_kinds_occur() -> None:
    cap, rows, width = _play(1500)
    m = cap.metrics()
    assert m["afterstate/mismatched"] == 0.0, "a replayed walk took a different path from the runner"
    for name in ("after_decision", "before_roll", "after_roll"):
        assert m[f"afterstate/rows_{name}"] > 0, name
    assert all(r.obs.shape[1] == width for r in rows), "rows are the buffer's width"


def test_a_coup_records_the_roll_and_its_result_from_the_couper() -> None:
    # Play until some decision's walk has a before- and an after-roll moment, and check what each is.
    env = TsVectorizedEnv(num_envs=8, base_seed=9)
    obs, masks, _ = env.reset_all()
    rng = np.random.default_rng(2)
    for _ in range(4000):
        dp = np.asarray(env.runner.get_decision_players())
        acts = np.array([rng.choice(np.flatnonzero(m)) for m in np.asarray(masks)])
        for e in range(8):
            if dp[e] == 0:
                continue
            side = ts.Player.US if dp[e] == 1 else ts.Player.USSR
            moments, _ = walk_afterstates(env.runner.get_state(e), int(acts[e]), side)
            kinds = [k for k, _ in moments]
            if BEFORE_ROLL in kinds and AFTER_ROLL in kinds:
                before = moments[kinds.index(BEFORE_ROLL)][1]
                after = moments[kinds.index(AFTER_ROLL)][1]
                assert before.ctx().decision_type == ts.DecisionType.ROLL_DIE
                assert before.ctx().decision_player == ts.Player.NONE
                assert after.ctx().decision_type != ts.DecisionType.ROLL_DIE or \
                    int(after.rng_state) != int(before.rng_state), "the roll was taken"
                return
        obs, masks, _, _, _ = env.step(acts)
    raise AssertionError("no decision with a die roll in 4,000 steps")


def test_the_target_is_the_sides_next_decision_or_the_result() -> None:
    # Buffer of 4 steps, 1 env: US decides at 0, USSR at 1, US at 2 (return 0.3), the game ends at 3.
    players = np.array([[1], [-1], [1], [-1]], dtype=np.float32)
    dones = np.array([[False], [False], [False], [True]])
    rewards = np.array([[0.0], [0.0], [0.0], [1.0]], dtype=np.float32)   # the USSR's last move wins for it
    rets = np.array([[0.1], [0.2], [0.3], [0.9]], dtype=np.float32)
    learner = np.ones((4, 1), dtype=bool)
    assert next_decision_target(0, 0, 1, players, dones, rewards, rets, learner) == np.float32(0.3)
    assert next_decision_target(2, 0, 1, players, dones, rewards, rets, learner) == -1.0, "the game ended first"
    learner[2, 0] = False
    assert next_decision_target(0, 0, 1, players, dones, rewards, rets, learner) is None, "a pool member's row"
    rows = [AfterstateRows(0, np.array([0]), np.array([1], np.int8), np.array([AFTER_DECISION], np.int8),
                           torch.zeros(1, 3))]
    obs, tgt, kinds = build_targets(rows, players, dones, rewards, rets, np.ones((4, 1), bool), "cpu")
    assert obs is not None and tgt is not None and kinds is not None and float(tgt[0]) == np.float32(0.3)


class _Critic(nn.Module):
    """A stand-in network: its value is a fixed function of the observation, the policy uniform."""

    def __init__(self, weights: torch.Tensor) -> None:
        super().__init__()
        self.w = nn.Parameter(weights, requires_grad=False)
        self.register_buffer("obs_features", torch.tensor(0))

    def forward(self, obs: torch.Tensor, mask: Any) -> Any:
        v = torch.tanh(obs @ self.w).unsqueeze(-1)
        n = 220 if mask is None else mask.shape[-1]
        return torch.zeros(obs.shape[0], n), v, v


def test_the_critic_player_takes_the_move_its_critic_likes_best() -> None:
    env = TsVectorizedEnv(num_envs=4, base_seed=21)
    env.reset_all()
    torch.manual_seed(0)
    net = _Critic(torch.randn(int(ts.OBS_SIZE)) * 0.05)
    agent = CriticArgmaxAgent(net, "c", worlds=1, seed=4)
    states = [env.runner.get_state(i) for i in range(4)]
    picks = agent.select_actions_batch(states)
    for st, a in zip(states, picks):
        legal = np.flatnonzero(np.asarray(ts.ActionMask.generate_flat_mask(st), dtype=bool))
        assert a in legal
    # With one world and its seed fixed, re-reading every move by hand gives the same choice.
    agent2 = CriticArgmaxAgent(net, "c", worlds=1, seed=4)
    st = states[0]
    me = st.ctx().decision_player
    legal = np.flatnonzero(np.asarray(ts.ActionMask.generate_flat_mask(st), dtype=bool))
    if len(legal) > 1:
        world = ts.determinize(st, me, agent2._seed(), False, True)
        vals = []
        for a in legal:
            s2 = world.clone()
            s2.rng_state = agent2._seed()
            ts.Engine.step_flat(s2, int(a), False)
            o = torch.from_numpy(np.asarray(ts.extract_observation_features(s2, me, 0), dtype=np.float32))
            vals.append(float(torch.tanh(o @ net.w)))
        assert picks[0] == int(legal[int(np.argmax(vals))])
