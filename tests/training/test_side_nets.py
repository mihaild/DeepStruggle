"""One side's network of an A39 (--ladder-side-nets) state taken out alone and put back (tools/lib/side_nets.py; owner,
2026-11-26: a seat-locked run trains that side's network alone, the other side not built, run or kept in the
optimizer).

Held to: the round trip is the identity, bit for bit, and the side left out is the base's exactly; the network taken
out computes what the A39 network computes on its side's rows; a resume state taken out restores that side's weights,
pi_ref and Adam moments into a plain network; a launch resuming it must be seat-locked to it; and a locked run on it,
put back, leaves the other side bit-identical."""

from __future__ import annotations

import copy
import os
from typing import Any, Dict

import pytest
import torch

from ai.models import ladder_net as ln
from ai.models.ladder_net import LadderNet, create_ladder_net
from ai.training.generic_trainer import check_side_only, load_resume_state, save_resume_state
from ai.training.nash_pg import NashPGTrainer
from ai.training.opponent_pool import OpponentPool
from bindings.ts_env import TsVectorizedEnv
from tests.training.test_side_split import CFG, _batch
from tools.lib.side_nets import USSR_PREFIX, combine, extract, side_state


def _a39(seed: int = 0) -> LadderNet:
    torch.manual_seed(seed)
    return create_ladder_net("cpu", **{**CFG, "side_nets": True})


def _plain(seed: int = 1) -> LadderNet:
    torch.manual_seed(seed)
    return create_ladder_net("cpu", **CFG)


def _trainer(net: torch.nn.Module, **kw: Any) -> NashPGTrainer:
    args: Dict[str, Any] = dict(active_net=net, env=TsVectorizedEnv(num_envs=8, base_seed=5), num_envs=8,
                                buffer_size=48, batch_size=96, lr=3e-4, eta=0.1, ref_update_freq=500,
                                cuda_graphs=False, device=torch.device("cpu"), per_player_gae=True,
                                afterstate_value_coef=1.0, afterstate_frac=1.0, predeal_value_coef=1.0,
                                adv_norm_learner_only=True, update_learner_rows_only=True)
    args.update(kw)
    return NashPGTrainer(**args)


@pytest.mark.parametrize("side", ["us", "ussr"])
def test_the_round_trip_is_the_identity_and_keeps_the_other_side_bit_for_bit(side: str) -> None:
    base = _a39().state_dict()
    assert all(torch.equal(a, b) for a, b in zip(combine(extract(base, side), base, side).values(), base.values()))
    other = _plain(7).state_dict()                     # another network in the side's place
    if "obs_features" in base:
        other["obs_features"] = base["obs_features"].clone()
    out = combine(other, base, side)
    assert list(out) == list(base)
    for k, v in out.items():
        mine = k.startswith(USSR_PREFIX) == (side == "ussr")
        name = k[len(USSR_PREFIX):] if k.startswith(USSR_PREFIX) else k
        assert torch.equal(v, other[name] if mine else base[k]), k


@pytest.mark.parametrize("side", ["us", "ussr"])
def test_the_network_taken_out_computes_what_its_side_did(side: str) -> None:
    a39 = _a39().eval()
    one = _plain().eval()
    one.load_state_dict(side_state(a39.state_dict(), side))
    obs, masks = _batch(64)
    us = obs[:, ln.ColdWarNetV2.GLOBAL_OFFSET + ln.I_AM_US_GLOBAL] > 0.5
    rows = us if side == "us" else ~us
    with torch.no_grad():
        a, b = a39(obs[rows], masks[rows]), one(obs[rows], masks[rows])
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_combine_refuses_what_does_not_fit() -> None:
    base = _a39().state_dict()
    with pytest.raises(ValueError, match="itself a --ladder-side-nets"):
        combine(base, base, "ussr")
    small = create_ladder_net("cpu", **{**CFG, "hidden_dim": 32}).state_dict()
    with pytest.raises(ValueError, match="shape"):
        combine(small, base, "ussr")


def test_a_resume_state_taken_out_restores_that_sides_weights_reference_and_moments(tmp_path: Any) -> None:
    t = _trainer(_a39())
    t.collect_rollouts()
    t.train_step()
    path = os.path.join(tmp_path, "resume_1000steps.pt")
    save_resume_state(path, t.active_net, t, 3, 1000, 1.0, seed=45)
    blob = torch.load(path, weights_only=False)
    side = extract(blob, "ussr", source=path)
    assert side["side_only"] == "ussr" and side["total_env_steps"] == 1000
    out = os.path.join(tmp_path, "side-ussr_resume_1000steps.pt")
    torch.save(side, out)
    t2 = _trainer(_plain())
    state = load_resume_state(out, t2.active_net, t2, seed=45)
    assert state["side_only"] == "ussr" and state["total_env_steps"] == 1000
    a39 = dict(t.active_net.named_parameters())
    ref = t.reference_net.state_dict()
    moved = 0
    for n, p in t2.active_net.named_parameters():
        assert torch.equal(p, a39[USSR_PREFIX + n]), n
        old = t.optimizer.state.get(a39[USSR_PREFIX + n], {})
        new = t2.optimizer.state.get(p, {})
        assert ("exp_avg" in old) == ("exp_avg" in new), n          # a parameter no loss reached has no moments
        if "exp_avg" in old:
            assert torch.equal(new["exp_avg"], old["exp_avg"]) and torch.equal(new["exp_avg_sq"], old["exp_avg_sq"]), n
            moved += 1
    assert moved > 10
    for k, v in t2.reference_net.state_dict().items():
        assert torch.equal(v, ref[USSR_PREFIX + k]), k
    with pytest.raises(ValueError, match="already the ussr network"):
        extract(side, "ussr")


@pytest.mark.parametrize("kw, needle", [
    (dict(opponent_lock_side="us"), "--opponent-lock-side ussr"),
    (dict(opponent_frac=0.5), "--opponent-frac 1.0"),
    (dict(update_learner_rows_only=False), "--update-learner-rows-only"),
])
def test_a_side_only_state_must_be_played_seat_locked(kw: Dict[str, Any], needle: str) -> None:
    good: Dict[str, Any] = dict(opponent_lock_side="ussr", opponent_frac=1.0, update_learner_rows_only=True)
    check_side_only("ussr", _plain(), **good)
    with pytest.raises(ValueError, match=needle):
        check_side_only("ussr", _plain(), **{**good, **kw})


def test_a_side_only_state_is_not_loaded_into_a_side_split_network() -> None:
    with pytest.raises(ValueError, match="without --ladder-side-nets"):
        check_side_only("ussr", _a39(), "ussr", 1.0, True)
    check_side_only(None, _a39(), None, 0.0, False)


def test_a_locked_run_on_the_side_alone_leaves_the_other_side_bit_identical() -> None:
    """The exploiter route: the USSR network alone trains against the frozen A39 base in the US seat, and put back it
    is the base on the US side exactly."""
    base = _a39()
    base_sd = {k: v.clone() for k, v in base.state_dict().items()}
    one = _plain()
    one.load_state_dict(side_state(base_sd, "ussr"))
    t = _trainer(one)
    t.opponent_pool = OpponentPool([copy.deepcopy(base).eval()], num_envs=8, frac=1.0, seed=3, lock_learner_side=-1)
    t.collect_rollouts()
    t.train_step()
    out = combine(one.state_dict(), base_sd, "ussr")
    assert all(torch.equal(out[k], base_sd[k]) for k in base_sd if not k.startswith(USSR_PREFIX))
    assert any(not torch.equal(out[k], base_sd[k]) for k in base_sd if k.startswith(USSR_PREFIX))
