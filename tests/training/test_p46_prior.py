"""P45's L6 and P46b (owner, 2026-11-14): own-pool opponents held for a whole game (`per_game_own`), and the critic's
prior per opponent and seat (`prior_window`): the running scores, beta, the signed shift, its persistence, and the
trainer's log-odds shift of the win value."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from ai.models.ladder_net import create_ladder_net, ladder_config_from_state_dict
from ai.training import NashPGTrainer
from ai.training.generic_trainer import load_resume_state, save_resume_state
from ai.training.nash_pg import prior_apply
from ai.training.opponent_pool import OpponentPool
from bindings.ts_env import TsVectorizedEnv
from tests.training.test_branch_arms import SHALLOW


class _Tiny(nn.Module):
    def __init__(self, k: float) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.full((2,), k))

    def forward(self, x: Any) -> Any:  # pragma: no cover - never called
        return x


def _pool(per_game: bool = True, prior: int = 0, num_envs: int = 32, frac: float = 1.0, seed: int = 7,
          turn_weight: tuple = (1.0, 1.0)) -> OpponentPool:
    p = OpponentPool([_Tiny(0.0)], num_envs=num_envs, frac=frac, seed=seed, capacity=8, per_game_own=per_game,
                     prior_window=prior, prior_turn_weight=turn_weight)
    for n in (5, 10, 15):
        p.add(_Tiny(float(n)), n * 1_000_000)
    for i in range(num_envs):           # every env starts a game with the members present
        p.on_episode_end(i, victory_points=None)
    return p


def test_off_the_own_pool_keeps_one_opponent_per_iteration() -> None:
    p = _pool(per_game=False)
    assert (p.env_member == -1).all()


def test_on_every_mixed_game_has_an_own_member_for_the_whole_game() -> None:
    p = _pool()
    mixed = p.is_mixed
    assert (p.env_member[mixed] >= 0).all()
    assert set(p.env_member[mixed].tolist()) <= set(p.ids)
    before = p.env_member.copy()
    for _ in range(5):                  # iterations change the iteration's opponent, never a game's member
        p.start_iteration()
    assert (p.env_member == before).all()


def test_only_mixed_envs_get_a_member() -> None:
    p = _pool(frac=0.5)
    assert (p.env_member[~p.is_mixed] == -1).all()
    assert (p.env_member[p.is_mixed] >= 0).all()


def test_a_games_result_goes_wholly_to_its_member() -> None:
    p = _pool(num_envs=4)
    for _ in range(3):
        p.start_iteration()
    oid = int(p.env_member[0])
    g0 = p.games[oid]
    p.on_episode_end(0, victory_points=5.0)
    assert p.games[oid] == pytest.approx(g0 + 1.0)
    for other in p.ids:
        if other != oid:
            assert p.games[other] == 0.0


def test_members_are_drawn_across_the_pool() -> None:
    p = _pool(num_envs=400)
    counts = np.bincount([p.ids.index(int(m)) for m in p.env_member], minlength=len(p.ids))
    assert (counts > 400 / len(p.ids) * 0.6).all()


def test_an_evicted_members_games_fall_back_and_are_counted() -> None:
    p = OpponentPool([_Tiny(0.0)], num_envs=64, frac=1.0, seed=2, capacity=3, per_game_own=True)
    for n in (5, 10):
        p.add(_Tiny(float(n)), n * 1_000_000)
    for i in range(64):
        p.on_episode_end(i, victory_points=None)
    p.add(_Tiny(20.0), 20_000_000)       # capacity 3: the interior member closest to its neighbours goes
    assert set(p.env_member.tolist()) <= set(p.ids) | {-1}
    assert p.stats()["opp_own_orphaned_games"] > 0


def test_a_prior_needs_one_opponent_per_game() -> None:
    with pytest.raises(ValueError, match="per_game_own"):
        OpponentPool([_Tiny(0.0)], num_envs=4, frac=1.0, prior_window=100)


def test_running_scores_and_beta() -> None:
    p = _pool(prior=9)                  # alpha = 0.2
    oid = int(p.env_member[0])
    side = int(p.learner_side[0])
    assert p.prior_beta(oid, side) == pytest.approx(0.0)   # no games yet: starts at the self-play score
    vp = 5.0 if side == 1 else -5.0     # a learner win
    p.on_episode_end(0, victory_points=vp)
    assert p.prior_rate[(oid, side)] == pytest.approx(0.5 + 0.2 * 0.5)
    lg = lambda q: math.log(q / (1 - q))
    assert p.prior_beta(oid, side) == pytest.approx(lg(0.6) - lg(0.5))
    p.prior_selfplay(3.0)               # a US win in self-play
    assert p.prior_sp_us == pytest.approx(0.6)
    us_self = 0.6 if side == 1 else 0.4
    assert p.prior_beta(oid, side) == pytest.approx(lg(0.6) - lg(us_self))


def test_a_new_member_starts_at_fifty_percent() -> None:
    """P46 (2026-11-14): a new member is the learner's snapshot from moments ago, so it starts even, whatever the
    other members' scores are."""
    p = _pool(prior=9)
    p.prior_rate[(p.ids[0], 1)] = 0.7
    p.prior_rate[(p.ids[1], 1)] = 0.6
    assert p.prior_rate_of(p.ids[2], 1) == pytest.approx(0.5)


def test_prior_terms_are_the_two_factors_of_the_shift() -> None:
    p = _pool(prior=9, num_envs=8, frac=0.5, turn_weight=(0.98, 0.46))
    for i in range(8):
        if p.is_mixed[i]:
            p.prior_rate[(int(p.env_member[i]), int(p.learner_side[i]))] = 0.7
    turns = np.array([1, 3, 5, 7, 9, 10, 2, 4])
    for dp in (p.learner_side.copy(), -p.learner_side.copy()):
        beta, wline = p.prior_terms(dp, turns)
        assert np.allclose(beta * wline, p.prior_shift(dp, turns), atol=1e-6)
        assert (beta[~p.is_mixed] == 0).all() and (wline[~p.is_mixed] == 0).all()


def test_the_gain_head_starts_at_zero_and_is_read_from_the_weights() -> None:
    """P46c (owner, 2026-11-14): g is a critic head on the trunk, zero at the start, recovered from the weights."""
    torch.manual_seed(0)
    net = create_ladder_net("cpu", **SHALLOW, prior_gain=True)
    obs = torch.rand(5, int(getattr(net, "TOTAL_OBS_SIZE")))
    out = net.forward_prior_gain(obs)
    assert out.shape == (5, 1) and torch.equal(out, torch.zeros(5, 1))
    cfg = ladder_config_from_state_dict(net.state_dict())
    assert cfg is not None and cfg.get("prior_gain") is True and net.ladder_config().get("prior_gain") is True
    plain = create_ladder_net("cpu", **SHALLOW)
    assert "prior_gain" not in (ladder_config_from_state_dict(plain.state_dict()) or {})
    with pytest.raises(RuntimeError, match="prior-gain head"):
        plain.forward_prior_gain(obs)


def _trainer(prior_gain: bool, **kw: Any) -> NashPGTrainer:
    torch.manual_seed(0)
    dev = torch.device("cpu")
    model = create_ladder_net(dev, **SHALLOW, prior_gain=prior_gain)
    env = TsVectorizedEnv(num_envs=8, base_seed=123)
    return NashPGTrainer(active_net=model, env=env, num_envs=8, buffer_size=16, lr=3e-4, eta=0.1,
                         ref_update_freq=500, cuda_graphs=False, device=dev, **kw)


def test_a_branch_adds_the_gain_head_and_plays_as_saved(tmp_path: Any) -> None:
    a = _trainer(False)
    a.train_iteration()
    path = str(tmp_path / "resume.pt")
    save_resume_state(path, a.active_net, a, iteration=1, total_env_steps=128, elapsed_seconds=1.0, seed=0)
    b = _trainer(True)
    load_resume_state(path, b.active_net, b, seed=0)
    sa, sb = a.active_net.state_dict(), b.active_net.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert any(k.startswith("prior_gain_head.") for k in sb)
    obs = torch.rand(4, int(getattr(a.active_net, "TOTAL_OBS_SIZE")))
    mask = torch.ones(4, 220, dtype=torch.bool)
    with torch.no_grad():
        for x, y in zip(a.active_net(obs, mask), b.active_net(obs, mask)):
            assert torch.allclose(x, y)
        assert torch.equal(getattr(b.active_net, "forward_prior_gain")(obs), torch.zeros(4, 1))


def test_a_learned_gain_needs_the_head() -> None:
    with pytest.raises(ValueError, match="prior-gain head"):
        _trainer(False, critic_prior=True, critic_prior_gain="learned")


def test_the_shift_is_signed_by_the_decider_and_weighted_by_the_turn() -> None:
    p = _pool(prior=9, num_envs=8, frac=0.5, turn_weight=(1.0, 0.5))
    for i in range(8):
        if p.is_mixed[i]:
            p.prior_rate[(int(p.env_member[i]), int(p.learner_side[i]))] = 0.8
    turns = np.full(8, 10)
    learner = p.learner_side.copy()
    s_learner = p.prior_shift(learner, turns)
    s_member = p.prior_shift(-learner, turns)
    lg = lambda q: math.log(q / (1 - q))
    for i in range(8):
        if p.is_mixed[i]:
            assert s_learner[i] == pytest.approx(0.5 * (lg(0.8) - lg(0.5)), rel=1e-5)
            assert s_member[i] == pytest.approx(-s_learner[i])
        else:
            assert s_learner[i] == 0.0 and s_member[i] == 0.0
    assert p.prior_turn_w(1) == pytest.approx(1.0)
    assert p.prior_turn_w(0) == pytest.approx(1.0)          # setup counts as turn 1


def test_the_prior_survives_a_resume() -> None:
    p = _pool(prior=9)
    p.prior_rate[(p.ids[1], -1)] = 0.73
    p.prior_sp_us = 0.47
    blob = p.state_dict()
    q = OpponentPool([_Tiny(0.0), _Tiny(5.0), _Tiny(10.0), _Tiny(15.0)], num_envs=32, frac=1.0, seed=7, capacity=8,
                     per_game_own=True, prior_window=9)
    q.load_state_dict(blob, p.steps, p.paths)
    assert q.prior_rate[(p.ids[1], -1)] == pytest.approx(0.73)
    assert q.prior_sp_us == pytest.approx(0.47)
    assert (q.env_member[q.is_mixed] >= 0).all()


def test_prior_apply_shifts_in_log_odds_and_leaves_zero_shift_alone() -> None:
    v = torch.tensor([-0.5, 0.0, 0.3, 0.999999])
    out = prior_apply(v, torch.zeros(4))
    assert torch.equal(out, v)
    s = torch.tensor([1.0, 1.0, -1.0, 0.0])
    out = prior_apply(v, s)
    p = (v[:3] + 1) / 2
    expect = 2 * torch.sigmoid(torch.log(p / (1 - p)) + s[:3]) - 1
    assert torch.allclose(out[:3], expect, atol=1e-6)
    assert out[3] == v[3]
