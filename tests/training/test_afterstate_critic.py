"""The critic at every moment of a side's turn (ai/training/afterstate_critic.py) and the player that reads
it after each legal move (ai/search/critic_argmax.py)."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import ts_engine as ts

from ai.models.ladder_net import create_ladder_net
from ai.search.critic_argmax import CriticArgmaxAgent
from ai.training.nash_pg import NashPGTrainer
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
        cap.before(env.runner, np.ones(16, bool), np.asarray(env.runner.get_decision_players()))
        obs, masks, _, dones, _ = env.step(acts)
        r = cap.after(env.runner, t, np.asarray(dones) > 0.5, "cpu")
        if r is not None:
            rows.append(r)
    return cap, rows, obs.shape[1]


def test_all_three_kinds_occur_and_no_walk_is_cut() -> None:
    cap, rows, width = _play(1500)
    m = cap.metrics()
    assert m["afterstate/mismatched"] == 0.0, "a walk ran past MAX_WALK nodes"
    for name in ("after_decision", "before_roll", "after_roll"):
        assert m[f"afterstate/rows_{name}"] > 0, name
    assert all(r.obs.shape[1] == width for r in rows), "rows are the buffer's width"


def test_the_runner_records_the_reference_walks_rows() -> None:
    """The batch runner's capture (set_capture_afterstates) against walk_afterstates on a copy of each pre-step state:
    the same envs, sides, kinds and observations, bit for bit, in both settle modes."""
    for auto in (True, False):
        env = TsVectorizedEnv(num_envs=16, base_seed=11)
        obs, masks, _ = env.reset_all()
        env._auto_advance = auto
        width = obs.shape[1]
        env.runner.set_capture_afterstates(True)
        rng = np.random.default_rng(4)
        compared = 0
        for _ in range(600):
            acts = np.array([rng.choice(np.flatnonzero(m)) for m in np.asarray(masks)])
            dp = np.asarray(env.runner.get_decision_players())
            pick = (rng.random(16) < 0.6) & (dp != 0)
            want_e, want_side, want_kind, want_obs = [], [], [], []
            for e in np.flatnonzero(pick):
                side = ts.Player.US if dp[e] == 1 else ts.Player.USSR
                moments, _ = walk_afterstates(env.runner.get_state(int(e)), int(acts[e]), side, forced=auto)
                for kind, st in moments:
                    o = np.asarray(ts.extract_observation_features(st, side, 0), dtype=np.float32)
                    want_e.append(int(e))
                    want_side.append(int(dp[e]))
                    want_kind.append(kind)
                    want_obs.append(np.pad(o, (0, width - o.shape[0])))
            env.runner.set_afterstate_envs(pick.astype(np.uint8))
            obs, masks, _, _, _ = env.step(acts)
            envs, sides, kinds, got_obs, cut = env.runner.get_afterstates()
            assert cut == 0
            # The gather into a caller's buffer (the trainer's pinned one): the same rows; too small, nothing written.
            small = np.full((max(0, len(kinds) - 1), width), 7.0, dtype=np.float32)
            assert env.runner.get_afterstates_into(small)[3] == len(kinds)
            assert (small == 7.0).all()
            out = np.zeros((len(kinds) + 3, width), dtype=np.float32)
            e2, s2, k2, n2, c2 = env.runner.get_afterstates_into(out)
            assert (n2, c2) == (len(kinds), cut) and np.array_equal(np.asarray(e2), np.asarray(envs))
            assert np.array_equal(np.asarray(s2), np.asarray(sides)) and np.array_equal(np.asarray(k2), np.asarray(kinds))
            assert np.array_equal(out[:n2], np.asarray(got_obs)) and not out[n2:].any()
            assert list(np.asarray(envs)) == want_e and list(np.asarray(sides)) == want_side
            assert list(np.asarray(kinds)) == want_kind
            if want_obs:
                assert np.array_equal(np.asarray(got_obs), np.stack(want_obs))
            compared += len(want_e)
        assert compared > 500, compared


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


def test_the_targets_of_every_row_follow_the_scalar_rule() -> None:
    """build_targets (numpy, all rows at once) against next_decision_target row by row, on random buffers."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        horizon, n = int(rng.integers(2, 30)), int(rng.integers(1, 6))
        players = rng.choice([-1.0, 1.0], size=(horizon, n)).astype(np.float32)
        dones = rng.random((horizon, n)) < 0.1
        rewards = rng.normal(size=(horizon, n)).astype(np.float32)
        rets = rng.normal(size=(horizon, n)).astype(np.float32)
        learner = rng.random((horizon, n)) < 0.8
        rows, want = [], []
        for t in range(horizon):
            k = int(rng.integers(0, 4))
            envs = rng.integers(0, n, size=k)
            sides = rng.choice([-1, 1], size=k).astype(np.int8)
            kinds = rng.integers(0, 3, size=k).astype(np.int8)
            obs = torch.from_numpy(rng.normal(size=(k, 3)).astype(np.float32))
            rows.append(AfterstateRows(t, envs, sides, kinds, obs))
            for j in range(k):
                tg = next_decision_target(t, int(envs[j]), int(sides[j]), players, dones, rewards, rets, learner)
                # Only a learner decision's walk is trained on (fix verification V-5: --per-player-gae captures
                # every decision's walk for its last moment alone).
                if tg is not None and learner[t, int(envs[j])]:
                    want.append((obs[j], tg, int(kinds[j])))
        got_obs, got_tgt, got_kind = build_targets(rows, players, dones, rewards, rets, learner, "cpu")
        if not want:
            assert got_obs is None
            continue
        assert got_obs is not None and got_tgt is not None and got_kind is not None
        assert torch.equal(got_obs, torch.stack([w[0] for w in want]))
        assert np.allclose(got_tgt.numpy(), np.array([w[1] for w in want], dtype=np.float32))
        assert got_kind.tolist() == [w[2] for w in want]


def test_the_value_only_pass_is_the_forwards_value() -> None:
    """forward_value (the extra critic rows' pass: no policy heads) gives forward()'s v_win bit for bit, on the
    A8 configuration with a mixed batch of real observations."""
    torch.manual_seed(0)
    net = create_ladder_net("cpu", input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=256,
                            card_self_attention=False, cross_attention=False, per_entity_heads=64, head_context=True,
                            head_static=True, head_entities="country", head_center=True, identity_dim=0,
                            drop_static=True, hidden_dim=480, num_res_blocks=0, num_attn_heads=4, card_lookup=False,
                            card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0,
                            categorical_value=False, branch_head=True, play_mode_head=True).eval()
    cap, rows, _ = _play(200)
    obs = torch.cat([r.obs for r in rows])[:256]
    with torch.no_grad():
        assert torch.equal(net.forward_value(obs), net(obs, None)[1])


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


def test_random_setup_forces_only_the_learners_setup_and_marks_it_environment() -> None:
    env = TsVectorizedEnv(num_envs=8, base_seed=13)
    obs, masks, _ = env.reset_all()
    trainer = NashPGTrainer.__new__(NashPGTrainer)
    trainer.num_envs = 8
    trainer.env = env
    trainer.setup_random_frac = 1.0
    trainer._rs_rng = np.random.default_rng(0)
    trainer._rs_game = np.array([True, True, True, True, False, False, False, False])
    trainer.setup_random_forced = 0
    learner = np.array([True, False, True, False, True, False, True, False])
    acts = torch.zeros(8, dtype=torch.int64)
    forced = trainer._random_setup_apply(acts, np.asarray(masks), learner)
    assert list(np.flatnonzero(forced)) == [0, 2], "drawn games, learner rows, at setup -- nothing else"
    for i in (0, 2):
        assert np.asarray(masks)[i][int(acts[i])], "a legal placement"
