"""--per-player-gae takes the auxiliary VP head's target along each side's own chain (2026-11-24, P56 plan, T1).

The owner's principle for T1: no target or advantage may combine critic values from different sides. The interleaved
VP trace (`RolloutBuffer._gae_backward`) bootstraps the rollout's end from the v_vp of the side to move, negated for
the other side, so each side's VP head was partly trained on the other side's VP estimate -- with per-side networks
(A39) a cross-network leak. Under --per-player-gae `_vp_returns_per_player` (ts.vp_returns_per_player) keeps the
trace's step (0.1 x the mover's normalised VP + 0.9 x the next return; the VP itself at a done) along the mover's own
decisions, each side bootstrapping past the buffer from its own VP estimate (`last_moment_vp_values`).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

from ai.models.ladder_net import LadderNet, create_ladder_net
from ai.training.afterstate_critic import AFTER_DECISION, AFTER_ROLL, BEFORE_ROLL, AfterstateRows, \
    last_moment_vp_values
from ai.training.nash_pg import NashPGTrainer
from ai.training.rollout_buffer import RolloutBuffer
from bindings.ts_env import TsVectorizedEnv

NUM_ENVS, HORIZON = 8, 160


def _net() -> LadderNet:
    torch.manual_seed(0)
    return create_ladder_net(torch.device("cpu"), input_mode="grouped", aggregation="flatten", entity_dim=16,
                             entity_proj_dim=64, card_self_attention=False, cross_attention=False,
                             per_entity_heads=16, head_context=True, head_static=True, head_entities="country",
                             head_center=True, identity_dim=0, drop_static=True, hidden_dim=64, num_res_blocks=0,
                             num_attn_heads=4, card_lookup=False, card_lookup_heads=0, card_lookup_dim=0,
                             card_lookup_identity_dim=0, categorical_value=False)


class _Spy:
    """Records compute_gae's keyword arguments, the bootstraps the trainer hands the buffer."""

    def __init__(self, buf: RolloutBuffer) -> None:
        self.kw: Dict[str, Any] = {}
        self._orig = buf.compute_gae

        def spy(**kw: Any) -> None:
            self.kw = dict(kw)
            self._orig(**kw)
        buf.compute_gae = spy   # type: ignore[method-assign]


_ROLLOUT: List[Tuple[RolloutBuffer, Dict[str, Any]]] = []


def _real_rollout() -> Tuple[RolloutBuffer, Dict[str, Any]]:
    """One real --per-player-gae rollout (an untrained network's games, so some end inside the buffer) and the
    arguments the trainer gave compute_gae. Built once: every test below reads it and recomputes on its own."""
    if not _ROLLOUT:
        t = NashPGTrainer(active_net=_net(), env=TsVectorizedEnv(num_envs=NUM_ENVS, base_seed=9), num_envs=NUM_ENVS,
                          buffer_size=HORIZON, lr=3e-4, eta=0.1, ref_update_freq=500, cuda_graphs=False,
                          device=torch.device("cpu"), per_player_gae=True, afterstate_value_coef=1.0,
                          afterstate_frac=1.0)
        spy = _Spy(t.buffer)
        t.collect_rollouts()
        _ROLLOUT.append((t.buffer, spy.kw))
    return _ROLLOUT[0]


def _rerun(buf: RolloutBuffer, kw: Dict[str, Any], **over: Any) -> np.ndarray:
    args = dict(kw)
    args.update(over)
    buf.compute_gae(**args)
    return buf.returns_vp.clone().numpy()


def _reference(players: np.ndarray, dones: np.ndarray, cuts: np.ndarray, vps: np.ndarray, boot: np.ndarray
               ) -> np.ndarray:
    """Plain Python, forward scans: a decision's next VP return is its mover's next own decision's return, or -- if
    a cut row comes first -- the game's final VP in the mover's frame (a done) or nothing (a turn cut), or past the
    buffer the mover's own bootstrap (`boot[q, e]`, q 0 US / 1 USSR). The step is the interleaved trace's."""
    T, N = players.shape
    out = np.full((T, N), np.nan)
    for e in range(N):
        for t in reversed(range(T)):
            p = int(players[t, e])
            vpn = float(vps[t, e]) * p / 20.0
            if dones[t, e]:
                out[t, e] = vpn
                continue
            if cuts[t, e]:
                nxt = 0.0
            else:
                nxt = float(boot[0 if p == 1 else 1, e])
                for k in range(t + 1, T):
                    if int(players[k, e]) == p:
                        nxt = float(out[k, e])
                        break
                    if cuts[k, e]:
                        nxt = float(vps[k, e]) * p / 20.0 if dones[k, e] else 0.0
                        break
            out[t, e] = 0.1 * vpn + 0.9 * nxt
    return out


def _boot(kw: Dict[str, Any]) -> np.ndarray:
    lp = kw["last_players"].numpy()
    lv = kw["last_v_vp"].numpy()
    own = np.asarray(kw["last_side_vp"], dtype=np.float64)
    return np.stack([np.where(lp == 1, lv, own[0]), np.where(lp == -1, lv, own[1])])


@pytest.mark.parametrize("sliced", [False, True])
def test_the_per_player_vp_return_is_the_plain_python_one(sliced: bool) -> None:
    buf, kw = _real_rollout()
    dones = buf.dones.numpy() > 0.5
    players = buf.players.numpy()
    assert dones.sum() >= 2, "the rollout must hold terminal rows"
    got = _rerun(buf, kw, slice_turn_boundaries=sliced)
    cuts = buf.per_player_cuts(sliced)
    want = _reference(players, dones, cuts, buf.vps.numpy(), _boot(kw))
    np.testing.assert_allclose(got, want, rtol=0, atol=2e-6)
    # Every kind of row is covered: a done's own row, a decision of the side NOT ending the game whose next is the
    # final VP, and a chain bootstrapping past the buffer from the other side's own (not the side to move's) estimate.
    lp = kw["last_players"].numpy()
    other_boot = False
    other_done = False
    for e in range(NUM_ENVS):
        for q in (1, -1):
            rows = np.flatnonzero(players[:, e] == q)
            if rows.size == 0:
                continue
            last = rows[-1]
            later_cut = cuts[last:, e].any()
            other_boot |= (lp[e] != q) and not later_cut
        for c in np.flatnonzero(dones[:, e]):
            prev = np.flatnonzero(players[:c, e] == -players[c, e])
            other_done |= prev.size > 0 and not cuts[prev[-1]:c, e].any()
    assert other_boot and other_done


def test_no_side_ever_reads_the_other_sides_vp_estimate() -> None:
    """Perturb every VP estimate of one side -- its bootstrap past the buffer (its own last moment, or the side to
    move's v_vp) and its stored per-row v_vp -- and the other side's returns_vp stay bit for bit. The interleaved
    trace, for contrast, moves (the leak this removes); and a side's own estimate does move its own returns."""
    buf, kw = _real_rollout()
    players = buf.players.numpy()
    lp = kw["last_players"].numpy()
    base = _rerun(buf, kw)
    for q, qi in ((1, 0), (-1, 1)):
        o, oi = -q, 1 - qi
        side_vp = np.array(kw["last_side_vp"], dtype=np.float32, copy=True)
        side_vp[oi] += 0.37
        last_v_vp = kw["last_v_vp"].clone()
        last_v_vp[torch.from_numpy(lp == o)] += 0.37
        saved = buf.values_vp.clone()
        buf.values_vp[torch.from_numpy(players == o)] += 0.37
        moved = _rerun(buf, kw, last_side_vp=side_vp, last_v_vp=last_v_vp)
        mine = players == q
        assert np.array_equal(moved[mine], base[mine])
        assert not np.allclose(moved[~mine], base[~mine])
        # The interleaved trace reads the side to move's v_vp for both sides.
        il_base = _rerun(buf, kw, per_player_gae=False)
        il_moved = _rerun(buf, kw, per_player_gae=False, last_v_vp=last_v_vp)
        assert not np.array_equal(il_moved[mine], il_base[mine])
        buf.values_vp.copy_(saved)
        # Its own estimate past the buffer does reach it.
        side_vp = np.array(kw["last_side_vp"], dtype=np.float32, copy=True)
        side_vp[qi] += 0.37
        last_v_vp = kw["last_v_vp"].clone()
        last_v_vp[torch.from_numpy(lp == q)] += 0.37
        own = _rerun(buf, kw, last_side_vp=side_vp, last_v_vp=last_v_vp)
        assert not np.array_equal(own[mine], base[mine])
        assert np.array_equal(own[~mine], base[~mine])


def test_the_trainer_bootstraps_each_side_from_its_own_moment() -> None:
    """The other side's bootstrap is its own v_vp at its last decision's last captured moment, through the network,
    and finite wherever its chain reaches the buffer's end."""
    buf, kw = _real_rollout()
    own = np.asarray(kw["last_side_vp"])
    assert own.shape == (2, NUM_ENVS)
    assert np.isfinite(_boot(kw)).sum() > NUM_ENVS          # more than the side to move's own value
    assert np.isfinite(buf.returns_vp.numpy()).all()


def test_without_the_flag_returns_vp_is_the_interleaved_trace() -> None:
    """Runs without --per-player-gae are untouched: last_side_vp is not read (NaN changes nothing) and returns_vp
    is _gae_backward's. (Before/after the change, a CPU run of three iterations had identical losses and weights.)"""
    buf, kw = _real_rollout()
    a = _rerun(buf, kw, per_player_gae=False, last_side_vp=None)
    b = _rerun(buf, kw, per_player_gae=False, last_side_vp=np.full((2, NUM_ENVS), np.nan, dtype=np.float32))
    assert np.array_equal(a, b)
    buf._gae_backward((float(kw["gamma"]), float(kw["gae_lambda"]), False, 4, False, "off", False), kw["last_v_win"],
                      kw["last_v_vp"], kw["last_players"])
    assert np.array_equal(buf.returns_vp.numpy(), a)
    assert not np.array_equal(_rerun(buf, kw), a)


def test_a_missing_own_estimate_is_refused() -> None:
    buf, kw = _real_rollout()
    with pytest.raises(ValueError, match="last_side_vp"):
        _rerun(buf, kw, last_side_vp=None)
    lp = kw["last_players"].numpy()
    nan_other = np.array(kw["last_side_vp"], dtype=np.float32, copy=True)
    nan_other[0, lp != 1] = np.nan
    nan_other[1, lp != -1] = np.nan
    with pytest.raises(ValueError, match="negated bootstrap"):
        _rerun(buf, kw, last_side_vp=nan_other)
    # The side to move's own estimate is last_v_vp: a NaN in its last_side_vp slot is never read.
    nan_mover = np.array(kw["last_side_vp"], dtype=np.float32, copy=True)
    nan_mover[0, lp == 1] = np.nan
    nan_mover[1, lp == -1] = np.nan
    assert np.array_equal(_rerun(buf, kw, last_side_vp=nan_mover), _rerun(buf, kw))


def test_held_rows_stay_refused() -> None:
    """The interleaved trace's held-row branch has no per-player counterpart: compute_gae refuses the pair."""
    buf, kw = _real_rollout()
    buf.uses_hold = True
    try:
        with pytest.raises(ValueError, match="held rows"):
            _rerun(buf, kw)
    finally:
        buf.uses_hold = False


def test_last_moment_vp_values_reads_the_last_moment_and_skips_a_die() -> None:
    """Per env and side, the v_vp of the last row of that side's last decision's walk (the obs read by vp_fn); NaN
    where that row waits on a die (outcomes listed) or the side has no decision."""
    players = np.array([[1, 1, -1], [-1, 1, -1]], dtype=np.int8)         # (T=2, N=3)
    obs0 = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
    r0 = AfterstateRows(0, np.array([0, 0, 1, 2]), np.array([1, 1, 1, -1], dtype=np.int8),
                        np.array([AFTER_DECISION, AFTER_ROLL, AFTER_DECISION, BEFORE_ROLL], dtype=np.int8), obs0)
    r0.face_row = np.array([3, 3])
    obs1 = torch.tensor([[0.5], [0.6], [0.7]])
    r1 = AfterstateRows(1, np.array([0, 1, 2]), np.array([-1, 1, -1], dtype=np.int8),
                        np.array([AFTER_DECISION, AFTER_DECISION, BEFORE_ROLL], dtype=np.int8), obs1)
    r1.face_row = np.array([2])
    out = last_moment_vp_values([r0, r1], players, 3, lambda o: o[:, 0] * 10.0)
    nan = float("nan")
    want = np.array([[2.0, 6.0, nan], [5.0, nan, nan]], dtype=np.float32)
    np.testing.assert_allclose(out, want, rtol=1e-6, equal_nan=True)
    assert math.isnan(float(out[1, 2]))      # env 2's USSR walk ends at a die: no blind read
