"""--update-learner-rows-only (owner, 2026-11-26: "why does the US side get any updates on a USSR exploiter? It is waste
of compute"): against a fixed opponent the update reads the learner's own rows only.

What it must hold to: a seat the learner never plays (a seat-locked run on per-side networks, A39) gets no gradient and
no optimizer step at all -- where without the option the opponent's rows train it through the value loss and the KL
to pi_ref --; the minibatches hold the learner's rows only, every one once an epoch, in the number of minibatches the
whole buffer would take; and where every row is the learner's (self-play) it changes nothing, bit for bit."""

from __future__ import annotations

import copy
from typing import Any, Dict

import numpy as np
import pytest
import torch

from ai.models.ladder_net import LadderNet, create_ladder_net
from ai.training.afterstate_critic import build_targets as build_afterstate_targets
from ai.training.afterstate_critic import learner_seat_rows, next_decision_targets
from ai.training.nash_pg import NashPGTrainer
from ai.training.opponent_pool import OpponentPool
from ai.training.rollout_buffer import RolloutBuffer
from bindings.ts_env import TsVectorizedEnv
from tests.training.test_side_split import CFG


def _net() -> LadderNet:
    torch.manual_seed(0)
    return create_ladder_net("cpu", **{**CFG, "side_nets": True})


def _trainer(**kw: Any) -> NashPGTrainer:
    args: Dict[str, Any] = dict(active_net=_net(), env=TsVectorizedEnv(num_envs=8, base_seed=7), num_envs=8,
                                buffer_size=48, batch_size=96, lr=3e-4, eta=0.1, ref_update_freq=500,
                                cuda_graphs=False, device=torch.device("cpu"), per_player_gae=True,
                                afterstate_value_coef=1.0, afterstate_frac=1.0, predeal_value_coef=1.0,
                                adv_norm_learner_only=True, update_learner_rows_only=True, zero_avoid_coef=0.01)
    args.update(kw)
    return NashPGTrainer(**args)


@pytest.mark.parametrize("kw, needle", [
    (dict(per_player_gae=False, afterstate_value_coef=0.0), "--per-player-gae"),
    (dict(adv_norm_learner_only=False), "--adv-norm-learner-only"),
    (dict(value_selfplay_only=True), "--value-selfplay-only"),
])
def test_it_refuses_what_would_read_or_need_the_opponents_rows(kw: Dict[str, Any], needle: str) -> None:
    """Without --per-player-gae every learner row's advantage bootstraps from the critic at the opponent's next
    decision, so the critic must be fitted there."""
    with pytest.raises(ValueError, match=needle):
        _trainer(**kw)


def test_the_minibatches_hold_every_learner_row_once_in_the_whole_buffers_count() -> None:
    b = RolloutBuffer(buffer_size=8, num_envs=4, obs_dim=3, device="cpu")
    b.learner = torch.tensor([[1.0, 0.0, 1.0, 1.0]] * 8)
    rows = b.learner.view(-1) > 0.5
    batches = list(b.get_batches(8, rows=rows, num_batches=4))     # 32 rows / 8: four, as without the option
    assert len(batches) == 4
    idx = torch.cat([bt[12] for bt in batches])
    assert sorted(idx.tolist()) == torch.nonzero(rows).squeeze(1).tolist()
    assert all(bool((bt[8] > 0.5).all()) for bt in batches)
    assert {int(bt[12].numel()) for bt in batches} == {6}           # 24 learner rows, an even share each


def test_a_row_counts_only_in_a_seat_the_learner_plays() -> None:
    seat = np.array([[0, 1, -1]], dtype=np.int8)                     # self-play, learner US, learner USSR
    t, e = np.zeros(6, dtype=np.int64), np.array([0, 0, 1, 1, 2, 2])
    side = np.array([1, -1, 1, -1, 1, -1])
    assert learner_seat_rows(t, e, side, seat).tolist() == [True, True, True, False, False, True]


def _us_params(net: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in net.named_parameters() if not k.startswith("ussr_net.")}


def _ussr_params(net: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in net.named_parameters() if k.startswith("ussr_net.")}


def _locked(update_learner_rows_only: bool, **kw: Any) -> NashPGTrainer:
    t = _trainer(update_learner_rows_only=update_learner_rows_only, **kw)
    t.opponent_pool = OpponentPool([copy.deepcopy(t.active_net).eval()], num_envs=8, frac=1.0, seed=3,
                                   lock_learner_side=-1)
    return t


@pytest.mark.parametrize("on", [True, False])
def test_a_seat_the_learner_never_plays_is_not_trained_only_with_the_option(on: bool) -> None:
    """A USSR-locked run on per-side networks: with the option the US network is untouched (no gradient, so no AdamW
    step and no decay either); without it the opponent's US rows train it -- the R173 exploiter's US network moved
    20-60% and fell to 0.1% as the US."""
    t = _locked(on)
    us0, ussr0 = _us_params(t.active_net), _ussr_params(t.active_net)
    t.collect_rollouts()
    assert 0.3 < float(t.buffer.learner.mean()) < 0.7
    t.train_step()
    us1, ussr1 = _us_params(t.active_net), _ussr_params(t.active_net)
    assert any(not torch.equal(ussr0[k], ussr1[k]) for k in ussr0), "the learner's own seat must train"
    us_moved = [k for k in us0 if not torch.equal(us0[k], us1[k])]
    if on:
        assert not us_moved, us_moved[:5]
    else:
        assert us_moved


def test_with_every_row_the_learners_it_changes_nothing() -> None:
    """Self-play: every row is the learner's and every pre-deal / afterstate row's seat is one it plays, so the update
    is the one without the option -- the same minibatches in the same order and the same losses, bit for bit."""
    out = []
    for on in (False, True):
        torch.manual_seed(11)
        np.random.seed(11)
        t = _trainer(update_learner_rows_only=on)
        init = [p.detach().clone() for p in t.active_net.parameters()]
        t.collect_rollouts()
        torch.manual_seed(12)
        m = t.train_step()
        out.append((m, [p.detach().clone() for p in t.active_net.parameters()]))
        assert any(not torch.equal(a, b) for a, b in zip(init, out[-1][1])), "the update must train"
    (m0, p0), (m1, p1) = out
    assert all(torch.equal(a, b) for a, b in zip(p0, p1))
    for k in ("policy_loss", "value_loss", "kl_div", "entropy"):
        if k in m0:
            assert m0[k] == m1[k], k


def test_every_position_the_learners_critic_is_read_at_stays_trained() -> None:
    """Owner, 2026-11-26: dropping the opponent's rows must not leave a position the learner's critic is READ at
    untrained. In a USSR-locked per-player-GAE run with the afterstate advantage and the R169 critics:

    (a) the learner's decision rows: read in its chain, and exactly the rows the minibatches hold;
    (b) the afterstate moments of its own decisions' walks (the positions after its move, the opponent possibly next,
        seen from its own side; the after-roll states averaged into a roll's value): every one is the learner's own
        side, and the afterstate rows trained are the same with and without the option; the pre-deal rows dropped
        are the opponent seat's only, and no advantage or return reads a pre-deal value;
    (c) the opponent's decision rows: seen from the opponent's side (its hand), and no learner advantage or return
        changes when the critic's values there -- and at the rollout's end where the opponent is to move -- change.
    """
    t = _locked(True, buffer_size=160)       # long enough to reach a turn end, where the pre-deal rows are filed
    calls: list = []
    orig = t.buffer.compute_gae

    def record(**kw: Any) -> None:
        calls.append(kw)
        orig(**kw)

    t.buffer.compute_gae = record          # type: ignore[method-assign]
    t.collect_rollouts()
    buf, net = t.buffer, t.active_net
    learner = buf.learner > 0.5
    T, N = learner.shape
    # (c) the opponent's rows are seen from the opponent's seat (US here), the learner's from its own (USSR).
    is_us = getattr(net, "is_us")(buf.obs.view(T * N, -1)).view(T, N)
    assert bool(is_us[~learner].all()) and not bool(is_us[learner].any())
    assert bool((buf.players[learner] == -1).all()) and bool((buf.players[~learner] == 1).all())
    # (c) perturb the critic everywhere the opponent's view was read: no learner target moves.
    adv0, ret0, rvp0 = buf.advantages.clone(), buf.returns_win.clone(), buf.returns_vp.clone()
    kw = dict(calls[-1])
    opp_end = kw["last_players"] != -1
    buf.values_win[~learner] += 0.37
    buf.values_vp[~learner] -= 0.21
    kw["last_v_win"] = torch.where(opp_end, kw["last_v_win"] + 0.37, kw["last_v_win"])
    kw["last_v_vp"] = torch.where(opp_end, kw["last_v_vp"] - 0.21, kw["last_v_vp"])
    last_side = kw["last_side_values"].clone()
    last_side[0] += 0.37                   # the US (opponent) seat's own last moments
    kw["last_side_values"] = last_side
    last_vp = np.array(kw["last_side_vp"], dtype=np.float32, copy=True)
    last_vp[0] -= 0.21
    kw["last_side_vp"] = last_vp
    orig(**kw)
    assert torch.allclose(buf.advantages[learner], adv0[learner], atol=1e-6)
    assert torch.allclose(buf.returns_win[learner], ret0[learner], atol=1e-6)
    assert torch.allclose(buf.returns_vp[learner], rvp0[learner], atol=1e-6)
    # (b) every afterstate row of a learner decision's walk is the learner's own side, so the option drops none.
    players = buf.players.cpu().numpy()
    for r in t._as_rows:
        own = learner.cpu().numpy()[r.step, np.asarray(r.envs)]
        assert (np.asarray(r.sides)[own] == -1).all()
    args = (t._as_rows, players, buf.dones.cpu().numpy() > 0.5, buf.rewards.float().cpu().numpy(),
            buf.returns_win.float().cpu().numpy(), learner.cpu().numpy(), torch.device("cpu"))
    o_all, _, _ = build_afterstate_targets(*args)
    o_own, _, _ = build_afterstate_targets(*args, t._learner_seat_rows)
    assert o_all is not None and o_own is not None and torch.equal(o_all, o_own)
    # (b) the pre-deal rows the option drops are the opponent seat's (or filed as a pool game ended) only.
    assert t._predeal_rows, "the run must file pre-deal rows"
    pt = np.concatenate([np.full(2 * len(pd.envs), pd.step, dtype=np.int64) for pd in t._predeal_rows])
    pe = np.concatenate([np.repeat(np.asarray(pd.envs, dtype=np.int64), 2) for pd in t._predeal_rows])
    ps = np.tile(np.array([1, -1], dtype=np.int64), len(pe) // 2)
    dones = buf.dones.cpu().numpy() > 0.5
    _, keep_all = next_decision_targets(pt, pe, ps, players, dones, buf.rewards.float().cpu().numpy(),
                                        buf.returns_win.float().cpu().numpy(), learner.cpu().numpy())
    assert t._learner_seat_rows is not None
    keep_own = keep_all & learner_seat_rows(pt, pe, ps, t._learner_seat_rows) & ~dones[pt, pe]
    dropped = keep_all & ~keep_own
    assert ((ps[dropped] == 1) | dones[pt[dropped], pe[dropped]]).all()
    assert keep_own.any() and (ps[keep_own] == -1).all()
    # (a) the minibatches hold exactly the learner's rows, every one once an epoch.
    seen: list = []
    orig_batches = buf.get_batches

    def batches(*a: Any, **k: Any) -> Any:
        for bt in orig_batches(*a, **k):
            seen.append(bt[12])
            yield bt

    buf.get_batches = batches              # type: ignore[method-assign]
    t.train_step()
    idx = torch.cat(seen)
    want = torch.nonzero(learner.view(-1)).squeeze(1)
    assert sorted(idx.tolist()) == sorted(want.tolist() * t.num_epochs)
