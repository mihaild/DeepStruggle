"""P56 (--afterstate-advantage): the die's outcomes the runner enumerates (forced faces, identical results merged),
and the advantage chain with the afterstate node -- equal to per-player GAE at lambda_opp 1 with the luck kept,
the realised return at lambda 1, and short of it by exactly the dropped luck terms."""

from __future__ import annotations

import math

import numpy as np
import torch
import ts_engine as ts

from ai.models.ladder_net import LadderNet, create_ladder_net
from ai.training.afterstate_critic import (AFTER_ROLL, BEFORE_ROLL, read_all_node_values, read_node_values,
                                           walk_afterstates)
from ai.training.nash_pg import NashPGTrainer, _value_only
from ai.training.rollout_buffer import RolloutBuffer
from bindings.ts_env import TsVectorizedEnv

OBS, ACT, N, T = 8, 4, 3, 14


def test_the_runner_enumerates_every_outcome_of_each_roll() -> None:
    """Each before-roll row's outcomes sum to probability 1, the roll that came is one of them, and a die
    whose faces give the same board is one outcome (fewer outcomes than faces somewhere)."""
    env = TsVectorizedEnv(num_envs=32, base_seed=5)
    obs, masks, _ = env.reset_all()
    r = env.runner
    r.set_capture_afterstates(True, True)
    rng = np.random.default_rng(1)
    rows = merged = hits = checked = 0
    for _ in range(700):
        acts = np.array([rng.choice(np.flatnonzero(m)) if m.any() else 0 for m in np.asarray(masks)])
        r.set_afterstate_envs((np.asarray(r.get_decision_players()) != 0).astype(np.uint8))
        obs, masks, _, _, _ = env.step(acts)
        envs, _, kinds, aobs, _ = r.get_afterstates()
        own, prob, term, _, fobs = r.get_afterstate_faces()
        envs, kinds, aobs, own, prob, fobs = (np.asarray(x) for x in (envs, kinds, aobs, own, prob, fobs))
        for j in np.flatnonzero(kinds == BEFORE_ROLL):
            sel = np.flatnonzero(own == j)
            if not sel.size:
                continue
            rows += 1
            merged += sel.size < 6
            assert abs(float(prob[sel].sum()) - 1.0) < 1e-5
            if j + 1 < len(kinds) and kinds[j + 1] == AFTER_ROLL and envs[j + 1] == envs[j]:
                checked += 1
                hits += any(np.array_equal(fobs[i], aobs[j + 1]) for i in sel)
    assert rows > 200 and merged > 0
    # A step that draws again after the die (~0.6% of coups and realignments) can land elsewhere than the face's
    # outcome read from the state's own stream; every other roll that came is one of the outcomes.
    assert hits >= 0.99 * checked, (hits, checked)


def test_coup_outcomes_merge_and_sum_to_one() -> None:
    """roll_outcomes on real coups (the reference walk's before-roll states): distinct outcomes summing to 1, and
    a coup whose faces leave the same board (a failure at several rolls) is fewer outcomes than faces."""
    env = TsVectorizedEnv(num_envs=16, base_seed=9)
    obs, masks, _ = env.reset_all()
    rng = np.random.default_rng(2)
    coups = merged = 0
    for _ in range(1500):
        acts = np.array([rng.choice(np.flatnonzero(m)) if m.any() else 0 for m in np.asarray(masks)])
        dp = np.asarray(env.runner.get_decision_players())
        for e in range(16):
            if dp[e] == 0:
                continue
            side = ts.Player.US if dp[e] == 1 else ts.Player.USSR
            for kind, st in walk_afterstates(env.runner.get_state(e), int(acts[e]), side)[0]:
                if kind != BEFORE_ROLL or st.ctx().pending_roll_type != ts.RollType.COUP:
                    continue
                o, p, term, _ = ts.roll_outcomes(st, side, 0)
                o = np.asarray(o)
                assert abs(sum(p) - 1.0) < 1e-5 and 1 <= len(p) <= 6
                live = [i for i in range(len(p)) if not term[i]]
                assert len({o[i].tobytes() for i in live}) == len(live)
                coups += 1
                merged += len(p) < 6
        obs, masks, _, _, _ = env.step(acts)
        if coups >= 40:
            break
    assert coups > 0 and merged > 0


def _buffer(values: np.ndarray, players: list, rewards: list, dones: list) -> RolloutBuffer:
    buf = RolloutBuffer(buffer_size=T, num_envs=N, obs_dim=OBS, action_dim=ACT, device="cpu")
    for t in range(T):
        buf.add(obs=torch.zeros(N, OBS), masks=torch.ones(N, ACT, dtype=torch.uint8),
                actions=torch.zeros(N, dtype=torch.long), log_probs=torch.zeros(N), rewards=rewards[t],
                dones=torch.tensor(dones[t]), values_win=torch.tensor(values[t]), values_vp=torch.zeros(N),
                players=torch.tensor(players[t], dtype=torch.int8), next_values_own=torch.zeros(N))
    return buf


def _game(seed: int):
    rng = np.random.default_rng(seed)
    players = [[int(x) for x in rng.choice([1, -1], size=N)] for _ in range(T)]
    rewards = [np.zeros(N, dtype=np.float32) for _ in range(T)]
    rewards[T - 1] = np.array([1.0, -1.0, 1.0], dtype=np.float32)
    dones = [[0.0] * N for _ in range(T)]
    dones[T - 1] = [1.0] * N
    values = rng.uniform(-0.6, 0.6, size=(T, N)).astype(np.float32)
    u = torch.from_numpy(rng.uniform(-0.6, 0.6, size=(T, N)).astype(np.float32))
    u[torch.from_numpy(rng.random((T, N)) < 0.3)] = float("nan")     # decisions with no afterstate
    return players, rewards, dones, values, u


def _pp(buf: RolloutBuffer, lam: float, node=None) -> torch.Tensor:
    return buf._gae_per_player(1.0, lam, torch.zeros(N), torch.ones(N, dtype=torch.int8), node=node)[0]


def test_the_node_at_lambda_opp_one_is_per_player_gae() -> None:
    players, rewards, dones, values, u = _game(0)
    buf = _buffer(values, players, rewards, dones)
    plain = _pp(buf, 0.9)
    node = _pp(buf, 0.9, (u, u.clone(), torch.ones(N)))
    assert torch.allclose(plain, node, atol=1e-6)
    # Smaller lambda_opp changes only rows with a node, and is not the plain chain.
    small = _pp(buf, 0.9, (u, u.clone(), torch.full((N,), 0.5)))
    assert not torch.allclose(plain, small)


def test_the_target_is_the_return_short_by_exactly_the_dropped_luck() -> None:
    """lambda 1, lambda_opp 1: A + V is each side's realised return with the luck kept (post = U), and short of it
    by the sum of the side's own later luck terms (post - U) when they are dropped."""
    players, rewards, dones, values, u = _game(1)
    buf = _buffer(values, players, rewards, dones)
    ret = _pp(buf, 1.0, (u, u.clone(), torch.ones(N))) + torch.from_numpy(values)
    realised = _pp(buf, 1.0) + torch.from_numpy(values)
    assert torch.allclose(ret, realised, atol=1e-6)
    luck = torch.from_numpy(np.random.default_rng(4).uniform(-0.3, 0.3, size=(T, N)).astype(np.float32))
    post = torch.where(torch.isfinite(u), u + luck, u)
    dropped = _pp(buf, 1.0, (u, post, torch.ones(N))) + torch.from_numpy(values)
    for e in range(N):
        for t in range(T):
            q = players[t][e]
            later = sum(float(luck[k, e]) for k in range(t, T)
                        if players[k][e] == q and math.isfinite(float(u[k, e])))
            assert abs(float(realised[t, e] - dropped[t, e]) - later) < 1e-5


def _net() -> LadderNet:
    torch.manual_seed(0)
    return create_ladder_net(torch.device("cpu"), input_mode="grouped", aggregation="flatten", entity_dim=16,
                             entity_proj_dim=64, card_self_attention=False, cross_attention=False,
                             per_entity_heads=16, head_context=True, head_static=True, head_entities="country",
                             head_center=True, identity_dim=0, drop_static=True, hidden_dim=64, num_res_blocks=0,
                             num_attn_heads=4, card_lookup=False, card_lookup_heads=0, card_lookup_dim=0,
                             card_lookup_identity_dim=0, categorical_value=False)


def test_a_rollout_takes_the_node_and_trains() -> None:
    t = NashPGTrainer(active_net=_net(), env=TsVectorizedEnv(num_envs=8, base_seed=7), num_envs=8, buffer_size=96,
                      lr=3e-4, eta=0.1, ref_update_freq=500, cuda_graphs=False, device=torch.device("cpu"),
                      afterstate_value_coef=1.0, afterstate_frac=1.0, afterstate_advantage=True)
    t.collect_rollouts()
    rows = t._as_rows
    assert rows and all(r.values is not None and r.avg is not None for r in rows)
    rolls = [int(torch.isfinite(r.avg).sum()) for r in rows if r.avg is not None]
    assert sum(rolls) > 0, "no before-roll row got its die's average"
    m = t.train_step()
    assert math.isfinite(m["afterstate_value_loss_before_roll"]) and math.isfinite(m["policy_loss"])


def test_the_gathered_outcomes_are_the_listed_ones() -> None:
    """get_afterstate_faces_into (the pinned path's parallel gather) writes exactly get_afterstate_faces' rows."""
    env = TsVectorizedEnv(num_envs=32, base_seed=11)
    obs, masks, _ = env.reset_all()
    r = env.runner
    r.set_capture_afterstates(True, True)
    rng = np.random.default_rng(5)
    seen = 0
    for _ in range(200):
        acts = np.array([rng.choice(np.flatnonzero(m)) if m.any() else 0 for m in np.asarray(masks)])
        r.set_afterstate_envs((np.asarray(r.get_decision_players()) != 0).astype(np.uint8))
        obs, masks, _, _, _ = env.step(acts)
        own, prob, term, res, fobs = (np.asarray(x) for x in r.get_afterstate_faces())
        buf = np.full((max(1, len(prob)) + 3, obs.shape[1]), -1.0, dtype=np.float32)
        own2, prob2, term2, res2, n = r.get_afterstate_faces_into(buf)
        assert n == len(prob)
        for a, b in ((own, own2), (prob, prob2), (term, term2), (res, res2)):
            assert np.array_equal(a, np.asarray(b))
        assert np.array_equal(buf[:n], fobs)
        seen += n
    assert seen > 0
    small = np.zeros((0, obs.shape[1]), dtype=np.float32)
    assert r.get_afterstate_faces_into(small)[4] >= 0


def test_one_batched_read_equals_a_read_a_step() -> None:
    """read_all_node_values (a few large batches at the rollout's end) fills the same values as read_node_values."""
    t = NashPGTrainer(active_net=_net(), env=TsVectorizedEnv(num_envs=8, base_seed=3), num_envs=8, buffer_size=64,
                      lr=3e-4, eta=0.1, ref_update_freq=500, cuda_graphs=False, device=torch.device("cpu"),
                      afterstate_value_coef=1.0, afterstate_frac=1.0, afterstate_advantage=True)
    t.collect_rollouts()
    rows = t._as_rows
    batched = [(r.values.clone(), r.avg.clone()) for r in rows if r.values is not None and r.avg is not None]
    fn = lambda o: _value_only(t.active_net, o)   # noqa: E731
    for r in rows:
        read_node_values(r, fn)
    one = [(r.values, r.avg) for r in rows if r.values is not None and r.avg is not None]
    assert len(batched) == len(one) == len(rows)
    read_all_node_values(rows, fn, chunk=50)
    for (v1, a1), r in zip(one, rows):
        assert r.values is not None and r.avg is not None
        assert torch.allclose(v1, r.values, atol=1e-5)
        assert torch.allclose(a1, r.avg, atol=1e-5, equal_nan=True)
    for (v1, a1), (v2, a2) in zip(one, batched):
        assert torch.allclose(v1, v2, atol=1e-5) and torch.allclose(a1, a2, atol=1e-5, equal_nan=True)
