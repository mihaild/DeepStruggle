"""P56 (--afterstate-advantage): the die's outcomes the runner enumerates (forced faces, identical results merged),
and the advantage chain with the afterstate node -- equal to per-player GAE at lambda_opp 1 with the luck kept,
the realised return at lambda 1, and short of it by exactly the dropped luck terms."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import ts_engine as ts

from ai.models.ladder_net import LadderNet, create_ladder_net
from ai.rewards.reward_calculator import ShapedZeroSumReward, VPPotentialShaping, ZeroSumTerminalReward
from ai.training.afterstate_critic import (AFTER_ROLL, BEFORE_ROLL, AfterstateCapture, AfterstateRows, build_targets,
                                           last_moment_values, missing_last_moments, node_tensors,
                                           read_all_node_values, read_node_values, walk_afterstates)
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
                      afterstate_value_coef=1.0, afterstate_frac=1.0, afterstate_advantage=True,
                      per_player_gae=True, gae_lambda=0.96)
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
                      afterstate_value_coef=1.0, afterstate_frac=1.0, afterstate_advantage=True,
                      per_player_gae=True, gae_lambda=0.96)
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


def test_the_node_needs_per_player_returns() -> None:
    with pytest.raises(ValueError, match="per-player-gae"):
        NashPGTrainer(active_net=_net(), env=TsVectorizedEnv(num_envs=4, base_seed=1), num_envs=4, buffer_size=16,
                      lr=3e-4, eta=0.1, ref_update_freq=500, cuda_graphs=False, device=torch.device("cpu"),
                      afterstate_value_coef=1.0, afterstate_frac=1.0, afterstate_advantage=True)


def test_the_merged_gather_is_the_two_gathers() -> None:
    """get_afterstates_with_faces_into: the rows in out[:n] and the outcomes in out[n:], equal to the two calls."""
    env = TsVectorizedEnv(num_envs=32, base_seed=13)
    obs, masks, _ = env.reset_all()
    r = env.runner
    r.set_capture_afterstates(True, True)
    rng = np.random.default_rng(6)
    seen = 0
    for _ in range(150):
        acts = np.array([rng.choice(np.flatnonzero(m)) if m.any() else 0 for m in np.asarray(masks)])
        r.set_afterstate_envs((np.asarray(r.get_decision_players()) != 0).astype(np.uint8))
        obs, masks, _, _, _ = env.step(acts)
        e0, s0, k0, a0, cut = r.get_afterstates()
        envs, sides, kinds, aobs = (np.asarray(x) for x in (e0, s0, k0, a0))
        own, prob, term, res, fobs = (np.asarray(x) for x in r.get_afterstate_faces())
        buf = np.zeros((len(kinds) + len(prob) + 2, obs.shape[1]), dtype=np.float32)
        e2, s2, k2, n, c2, o2, p2, t2, r2, nf = r.get_afterstates_with_faces_into(buf)
        assert n == len(kinds) and nf == len(prob) and c2 == cut
        for a, b in ((envs, e2), (sides, s2), (kinds, k2), (own, o2), (prob, p2), (term, t2), (res, r2)):
            assert np.array_equal(a, np.asarray(b))
        assert np.array_equal(buf[:n], aobs) and np.array_equal(buf[n:n + nf], fobs)
        seen += nf
    assert seen > 0


def _gae_numpy(players, dones, rewards, values, last_v, gamma, lam, node, last_ussr=None):
    """The numpy per-player loop ts.gae_per_player replaced (RolloutBuffer._gae_per_player_both before it moved to C++)."""
    T, N = players.shape
    adv = np.zeros((T, N), np.float32); ret = np.zeros((T, N), np.float32); nadv = np.zeros((T, N), np.float32)
    lv = {1: last_v, -1: (-last_v if last_ussr is None else last_ussr)}
    st = {q: {"A": np.zeros(N, np.float32), "An": np.zeros(N, np.float32), "V": lv[q].astype(np.float32),
              "R": np.zeros(N, np.float32), "gap": np.zeros(N, np.float32), "has": np.ones(N, bool)} for q in (1, -1)}
    g, lm = np.float32(gamma), np.float32(lam)
    for t in range(T - 1, -1, -1):
        if dones[t].any():
            for q in (1, -1):
                for k in ("A", "An", "V", "R", "gap"):
                    st[q][k] = np.where(dones[t], np.float32(0), st[q][k])
                st[q]["has"] = st[q]["has"] & ~dones[t]
        r_t = rewards[t] * players[t].astype(np.float32)
        v_t = values[t]
        for q in (1, -1):
            c = st[q]
            c["R"] = c["R"] + r_t * np.float32(q)
            c["gap"] = c["gap"] + np.float32(1)
            mv = players[t] == q
            disc = np.power(g, c["gap"]).astype(np.float32)
            carry = c["has"].astype(np.float32)
            target = c["R"] + disc * c["V"] * carry
            a = target - v_t + disc * lm * c["A"] * carry
            an = target - v_t + disc * lm * c["An"] * carry
            if node is not None:
                u, post, lo = node[0][t], node[1][t], node[2]
                ex, nt = (node[3][t], node[4][t]) if len(node) > 3 else (0.0, 0.0)
                p2 = np.where(np.isfinite(post), post, target) + ex + nt * target
                an = np.where(np.isfinite(u), (u - v_t) + lo * ((target - p2) + disc * lm * c["An"] * carry), an)
            adv[t] = np.where(mv, a, adv[t]); ret[t] = np.where(mv, a + v_t, ret[t]); nadv[t] = np.where(mv, an, nadv[t])
            c["A"] = np.where(mv, a, c["A"]); c["An"] = np.where(mv, an, c["An"])
            c["V"] = np.where(mv, v_t, c["V"]); c["R"] = np.where(mv, np.float32(0), c["R"])
            c["gap"] = np.where(mv, np.float32(0), c["gap"]); c["has"] = c["has"] | mv
    return adv, ret, nadv


def test_the_cpp_per_player_pass_is_the_numpy_loop() -> None:
    T2, N2 = 96, 40
    for seed in range(6):
        rng = np.random.default_rng(seed)
        players = rng.choice([1, -1, 1], size=(T2, N2)).astype(np.int8)
        players[rng.random((T2, N2)) < 0.05] = 0
        dones = rng.random((T2, N2)) < 0.03
        rewards = np.where(dones, rng.choice([-1.0, 1.0], size=(T2, N2)), 0).astype(np.float32)
        values = rng.uniform(-0.8, 0.8, (T2, N2)).astype(np.float32)
        last_v = rng.uniform(-0.5, 0.5, N2).astype(np.float32)
        u = rng.uniform(-0.6, 0.6, (T2, N2)).astype(np.float32)
        u[rng.random((T2, N2)) < 0.4] = np.nan
        post = (u + rng.uniform(-0.2, 0.2, (T2, N2))).astype(np.float32)
        post[rng.random((T2, N2)) < 0.1] = np.nan
        lo = rng.choice([1.0, 0.8, 0.5], N2).astype(np.float32)
        for gamma in (1.0, 0.999):
            ex = np.where(rng.random((T2, N2)) < 0.1, rng.uniform(-0.2, 0.2, (T2, N2)), 0).astype(np.float32)
            nt = (rng.random((T2, N2)) < 0.05).astype(np.float32)
            last_r = rng.uniform(-0.5, 0.5, N2).astype(np.float32)
            want = _gae_numpy(players, dones, rewards, values, last_v, gamma, 0.96, (u, post, lo, ex, nt), last_r)
            got = [np.zeros((T2, N2), np.float32) for _ in range(3)]
            ts.gae_per_player(players, dones.astype(np.uint8), rewards, values, last_v, last_r, gamma, 0.96, u, post, lo,
                              ex, nt, *got)
            for a, b in zip(got, want):
                assert np.allclose(a, b, atol=2e-6)


def _rows(step, envs, kinds, vals, avgs, sides=None):
    r = AfterstateRows(step, np.asarray(envs, dtype=np.int64),
                       np.asarray(sides if sides is not None else [1] * len(envs), dtype=np.int8),
                       np.asarray(kinds, dtype=np.int8), torch.zeros(len(envs), 4))
    r.values = torch.tensor(vals, dtype=torch.float32)
    r.avg = torch.tensor(avgs, dtype=torch.float32)
    return r


def test_the_node_drops_every_rolls_luck() -> None:
    """Audit T-1: a forced step before the die (AFTER_DECISION, BEFORE_ROLL, AFTER_ROLL) takes the roll's average as U
    and the actual after-value as U_post; a second roll adds its own (after - average); a later roll that leads
    straight to the next node counts the target; a walk with no die keeps U = U_post = its first value."""
    nan = float("nan")
    AD, BR, AR = 0, BEFORE_ROLL, AFTER_ROLL
    rows = [_rows(0, [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3],
                  [AD, BR, AR, BR, AR, BR, AR, BR, AR, BR, AD],
                  [0.1, 0.2, 0.5, 0.0, 0.3, 0.4, 0.9, 0.0, 0.6, 0.7, 0.25],
                  [nan, 0.35, nan, 0.1, nan, 0.5, nan, 0.2, nan, 0.45, nan])]
    u, post, ex, nt = node_tensors(rows, 2, 5, torch.device("cpu"))
    # env 0: forced step then one roll
    assert abs(float(u[0, 0]) - 0.35) < 1e-6 and abs(float(post[0, 0]) - 0.5) < 1e-6 and float(ex[0, 0]) == 0.0
    # env 1: two rolls -- the second's luck 0.9 - 0.5
    assert abs(float(u[0, 1]) - 0.1) < 1e-6 and abs(float(post[0, 1]) - 0.3) < 1e-6
    assert abs(float(ex[0, 1]) - 0.4) < 1e-6 and float(nt[0, 1]) == 0.0
    # env 2: the second roll led straight to the next node: -average and one target
    assert abs(float(ex[0, 2]) + 0.45) < 1e-6 and float(nt[0, 2]) == 1.0
    # env 3: no die
    assert abs(float(u[0, 3]) - 0.25) < 1e-6 and abs(float(post[0, 3]) - 0.25) < 1e-6
    # env 4: no decision captured
    assert math.isnan(float(u[0, 4]))


def test_a_die_right_after_another_gets_its_own_before_roll_row() -> None:
    """Fix verification V-4: a die rolled right after another (an Ops-first coup or realignment on Korean War or
    Arab-Israeli War, then the war's roll) has a before-roll row of its own -- the first roll's after-state kept again
    -- with that die's outcomes, in the runner and in the reference walk alike, and node_tensors drops both rolls'
    luck: U is the first roll's average, U_post the value after it, and the second roll adds (after - average)."""
    ne = 32
    env = TsVectorizedEnv(num_envs=ne, base_seed=7)
    obs, masks, _ = env.reset_all()
    r = env.runner
    cap = AfterstateCapture(1.0, 0, obs.shape[1], forced=bool(env._auto_advance), seed=3, roll_faces=True)
    rng = np.random.default_rng(3)
    proj = torch.from_numpy(np.random.default_rng(0).normal(size=obs.shape[1]).astype(np.float32)) * 0.01
    found = 0
    for _ in range(4000):
        acts = np.array([rng.choice(np.flatnonzero(m)) for m in np.asarray(masks)])
        dp = np.asarray(r.get_decision_players())
        walks = {}
        for e in np.flatnonzero(dp != 0):
            side = ts.Player.US if dp[e] == 1 else ts.Player.USSR
            walks[int(e)] = (side, walk_afterstates(r.get_state(int(e)), int(acts[e]), side)[0])
        cap.before(r, np.ones(ne, bool), dp)
        obs, masks, _, _, _ = env.step(acts)
        rows = cap.after(r, 0, np.zeros(ne, bool), "cpu")
        if rows is None:
            continue
        read_node_values(rows, lambda o: o @ proj)
        assert rows.values is not None and rows.avg is not None and rows.face_row is not None
        assert rows.face_prob is not None
        vals, avg = rows.values.numpy(), rows.avg.numpy()
        u, post, ex, nt = node_tensors([rows], 1, ne, torch.device("cpu"))
        for e, (side, moments) in walks.items():
            sel = np.flatnonzero(rows.envs == e)
            kinds = list(rows.kinds[sel])
            assert kinds == [k for k, _ in moments], "the runner and the reference walk keep the same rows"
            pairs = [i for i in range(1, len(sel)) if kinds[i - 1] == AFTER_ROLL and kinds[i] == BEFORE_ROLL]
            if not pairs:
                continue
            i = pairs[0]
            a, b = sel[i - 1], sel[i]
            st = moments[i][1]
            assert st.ctx().pending_roll_type not in (ts.RollType.NONE, ts.RollType.TURN_CLEANUP)
            assert torch.equal(rows.obs[a], rows.obs[b]), "the first roll's after-state is the second's before-state"
            faces = np.flatnonzero(rows.face_row == b)
            assert faces.size and abs(float(rows.face_prob[faces].sum()) - 1.0) < 1e-5, "the second roll has outcomes"
            _, p_ref, _, _ = ts.roll_outcomes(st, side, 0)
            assert np.allclose(sorted(p_ref), sorted(rows.face_prob[faces]), atol=1e-6)
            rolls = [j for j in range(len(sel)) if kinds[j] == BEFORE_ROLL and np.isfinite(avg[sel[j]])]
            if len(rolls) != 2 or rolls[1] != i:
                continue                      # the walk's first two rolls are the pair
            f = sel[rolls[0]]
            assert abs(float(u[0, e]) - float(avg[f])) < 1e-6
            assert abs(float(post[0, e]) - float(vals[f + 1])) < 1e-6
            if i + 1 < len(sel) and kinds[i + 1] == AFTER_ROLL:
                assert abs(float(ex[0, e]) - float(vals[b + 1] - avg[b])) < 1e-5 and float(nt[0, e]) == 0.0
            else:
                assert abs(float(ex[0, e]) + float(avg[b])) < 1e-5 and float(nt[0, e]) == 1.0
            found += 1
        if found >= 3:
            break
    assert found >= 3, found


def test_the_side_not_to_move_bootstraps_from_its_own_last_moment() -> None:
    """Audit T-4(2): per env and side, the latest captured moment of that side's last decision (the roll average where it
    waits on a die), NaN where that decision has no captured walk."""
    nan = float("nan")
    players = np.array([[1, -1], [-1, 1], [1, 1]], dtype=np.int8)        # T=3, N=2
    rows = [_rows(0, [0, 0, 1], [0, BEFORE_ROLL, 0], [0.1, 0.2, 0.3], [nan, 0.25, nan], sides=[1, 1, -1]),
            _rows(1, [0, 1], [0, 0], [0.4, 0.5], [nan, nan], sides=[-1, 1]),
            _rows(2, [0], [0], [0.6], [nan], sides=[1])]                 # env 1's step-2 decision has no walk
    out = last_moment_values(rows, players, 2)
    assert abs(out[0, 0] - 0.6) < 1e-6          # US env 0: its last decision is step 2
    assert abs(out[1, 0] - 0.4) < 1e-6          # USSR env 0: step 1
    assert math.isnan(out[0, 1])                 # US env 1: last decision step 2 has no walk
    assert abs(out[1, 1] - 0.3) < 1e-6          # USSR env 1: step 0


def test_the_last_moments_read_only_the_rows_they_need() -> None:
    """Fix verification V-5: without --afterstate-advantage the rows' values are unread; last_moment_values reads the
    last row of each side's last walk alone (one batch), and gives what reading every row would."""
    players = np.array([[1, -1], [-1, 1], [1, 1]], dtype=np.int8)
    def fresh() -> list:
        out = []
        for step, envs, sides in ((0, [0, 0, 1], [1, 1, -1]), (1, [0, 1], [-1, 1]), (2, [0], [1])):
            n = len(envs)
            obs = torch.arange(n * 4, dtype=torch.float32).reshape(n, 4) + 100.0 * step
            out.append(AfterstateRows(step, np.asarray(envs, dtype=np.int64), np.asarray(sides, dtype=np.int8),
                                      np.zeros(n, dtype=np.int8), obs))
        return out
    seen: list = []
    def fn(o: torch.Tensor) -> torch.Tensor:
        seen.append(int(o.shape[0]))
        return o[:, 0] / 1000.0
    lazy = last_moment_values(fresh(), players, 2, fn)
    assert seen == [3]                             # US env 0 (step 2), USSR env 0 (step 1), USSR env 1 (step 0)
    rows = fresh()
    read_all_node_values(rows, fn)
    full = last_moment_values(rows, players, 2)
    assert np.allclose(lazy, full, equal_nan=True)
    assert math.isnan(lazy[0, 1])                  # US env 1: its step-2 decision has no walk
    with pytest.raises(ValueError, match="value_fn"):
        last_moment_values(fresh(), players, 2)


def test_a_side_without_its_own_last_moment_is_caught() -> None:
    """Fix verification V-5: a side not to move at the end whose last decision is after the env's last cut and has no
    moment would fall back to the negated value of the side to move; missing_last_moments flags exactly those."""
    nan = float("nan")
    players = np.array([[1, 1, -1], [-1, -1, 1], [1, 1, -1]], dtype=np.int8)     # T=3, N=3
    cuts = np.zeros((3, 3), dtype=bool)
    cuts[2, 1] = True                               # env 1's game ended at its last row: nothing carries
    last_players = np.array([-1, -1, 1])
    last = np.array([[nan, nan, nan], [0.2, nan, nan]], dtype=np.float32)
    miss = missing_last_moments(last, players, cuts, last_players)
    assert miss[0, 0]                              # env 0: the US (not to move) has no moment
    assert not miss[1, 0]                          # env 0: the USSR moves at the end, no bootstrap of its own
    assert not miss[0, 1] and not miss[1, 1]       # env 1: cut at its last row
    assert not miss[0, 2] and miss[1, 2]           # env 2: the US moves; the USSR (step 2) has none


def test_only_learner_walks_are_trained_on() -> None:
    """--per-player-gae captures every decision's walk; build_targets keeps the learner decisions' rows alone (a
    before-roll row's average target included)."""
    players = np.array([[1, -1], [-1, 1], [1, -1]], dtype=np.int8)
    dones = np.zeros((3, 2), dtype=bool)
    rewards = np.zeros((3, 2), dtype=np.float32)
    rets = np.full((3, 2), 0.5, dtype=np.float32)
    learner = np.array([[True, False], [True, True], [True, True]])
    rows = [_rows(0, [0, 1, 1], [0, 0, BEFORE_ROLL], [0.1, 0.2, 0.3], [float("nan"), float("nan"), 0.4],
                  sides=[1, -1, -1])]
    obs, tgt, kinds = build_targets(rows, players, dones, rewards, rets, learner, torch.device("cpu"))
    assert obs is not None and tgt is not None and int(tgt.shape[0]) == 1   # env 1's step-0 walk is a pool member's


def _ppg(**kw) -> NashPGTrainer:
    args = dict(active_net=_net(), env=TsVectorizedEnv(num_envs=4, base_seed=1), num_envs=4, buffer_size=16, lr=3e-4,
                eta=0.1, ref_update_freq=500, cuda_graphs=False, device=torch.device("cpu"), per_player_gae=True,
                afterstate_value_coef=1.0, afterstate_frac=1.0)
    args.update(kw)
    return NashPGTrainer(**args)


@pytest.mark.parametrize("kw, needle", [
    (dict(afterstate_value_coef=0.0), "afterstate critic at every decision"),
    (dict(afterstate_frac=0.5), "afterstate critic at every decision"),
    (dict(gamma=0.99), "--gamma 1.0"),
    (dict(env=TsVectorizedEnv(num_envs=4, base_seed=1, reward_calculator=ShapedZeroSumReward())), "terminal-only"),
    (dict(env=TsVectorizedEnv(num_envs=4, base_seed=1,
                              reward_calculator=VPPotentialShaping(ZeroSumTerminalReward(), 0.05))), "terminal-only"),
])
def test_per_player_gae_refuses_what_its_last_bootstrap_cannot_take(kw, needle: str) -> None:
    """Fix verification V-5: no own last moment to bootstrap from (no capture at every decision), rewards after that
    moment counted twice (a dense reward), or the bootstrap discounted over the steps after it (gamma < 1)."""
    with pytest.raises(ValueError, match=needle):
        _ppg(**kw)


def test_per_player_gae_refuses_a_dense_reward_set_later() -> None:
    t = _ppg()
    with pytest.raises(ValueError, match="terminal-only"):
        t.set_reward_calculator(ShapedZeroSumReward())
    t.set_reward_calculator(ZeroSumTerminalReward())


def test_a_per_player_rollout_bootstraps_every_side_from_its_own_moment() -> None:
    """--per-player-gae with the afterstate critic and no --afterstate-advantage: every side not to move at the end has
    its own moment (the rollout would raise otherwise), and the update trains."""
    t = _ppg(buffer_size=64, env=TsVectorizedEnv(num_envs=8, base_seed=9), num_envs=8)
    t.collect_rollouts()
    assert t.afterstate is not None and t.afterstate.every_decider
    m = t.train_step()
    assert math.isfinite(m["policy_loss"])
