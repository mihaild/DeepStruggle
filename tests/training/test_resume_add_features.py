"""Branching a trained state into a zero-width observation bit (--resume-add-features, owner 2026-11-26).

uncapped_deficit writes board slots 24 and 25 without their cap and has no width, so a state trained without it
loads unchanged into a network built with it: the bit is added to the network's, pi_ref's and (for an A39 network)
the USSR copy's recorded feature set, the weights are untouched, and a resume without the flag is refused. One side's
network taken out of such a state (tools/lib/side_nets.py) carries the bit, and is put back only beside a base that
reads it too."""

from __future__ import annotations

import os
from typing import Any, Dict

import pytest
import torch

import ts_engine as ts

from ai.models.ladder_net import LadderNet, create_ladder_net
from ai.training.generic_trainer import (OBS_FEATURE_BITS, add_rule_bits_to, load_resume_state,
                                         save_resume_state)
from ai.training.nash_pg import NashPGTrainer
from bindings.ts_env import TsVectorizedEnv, model_obs_features
from tests.training.test_side_split import CFG
from tools.lib.side_nets import combine, extract

E9 = int(ts.OBS_FEATURE_HEADLINE_REVEAL) | int(ts.OBS_FEATURE_OPP_EVENT_GATED)
U = int(ts.OBS_FEATURE_UNCAPPED_DEFICIT)


def _a39(features: int, seed: int = 0) -> LadderNet:
    torch.manual_seed(seed)
    return create_ladder_net("cpu", **{**CFG, "side_nets": True, "obs_features": features})


def _trainer(net: torch.nn.Module) -> NashPGTrainer:
    args: Dict[str, Any] = dict(active_net=net, env=TsVectorizedEnv(num_envs=8, base_seed=5), num_envs=8,
                                buffer_size=48, batch_size=96, lr=3e-4, eta=0.1, ref_update_freq=500,
                                cuda_graphs=False, device=torch.device("cpu"), per_player_gae=True,
                                afterstate_value_coef=1.0, afterstate_frac=1.0, predeal_value_coef=1.0)
    return NashPGTrainer(**args)


def test_the_name_is_the_engines_bit() -> None:
    assert OBS_FEATURE_BITS["uncapped_deficit"] == U == 1 << 11
    assert ts.obs_size_for(E9 | U) == ts.OBS_SIZE


def test_a_state_branched_into_uncapped_deficit_keeps_its_weights(tmp_path: Any) -> None:
    t = _trainer(_a39(E9))
    t.collect_rollouts()
    t.train_step()
    path = os.path.join(tmp_path, "resume_1000steps.pt")
    save_resume_state(path, t.active_net, t, 3, 1000, 1.0, seed=45)

    # Without the flag, a network built for the new default refuses the state: it was trained on another view.
    with pytest.raises(ValueError, match="uncapped_deficit"):
        t_plain = _trainer(_a39(E9 | U, seed=9))
        load_resume_state(path, t_plain.active_net, t_plain, seed=45)

    t2 = _trainer(_a39(E9 | U, seed=9))
    assert t2.obs_features == E9 | U
    state = load_resume_state(path, t2.active_net, t2, seed=45, add_rules=U)
    assert state["total_env_steps"] == 1000
    assert model_obs_features(t2.active_net) == E9 | U
    sd = t2.active_net.state_dict()
    assert int(sd["obs_features"]) == E9 | U and int(sd["ussr_net.obs_features"]) == E9 | U
    assert int(t2.reference_net.state_dict()["obs_features"]) == E9 | U
    old = t.active_net.state_dict()
    for k, v in sd.items():
        if not k.endswith("obs_features"):
            assert torch.equal(v, old[k]), k
    # The bit is already there: adding it again is refused.
    path2 = os.path.join(tmp_path, "resume_2000steps.pt")
    save_resume_state(path2, t2.active_net, t2, 4, 2000, 1.0, seed=45)
    t3 = _trainer(_a39(E9 | U, seed=9))
    with pytest.raises(ValueError, match="already has"):
        load_resume_state(path2, t3.active_net, t3, seed=45, add_rules=U)

    # One side taken out carries the bit; it goes back only beside a base that reads it.
    blob = torch.load(path2, weights_only=False)
    for side in ("us", "ussr"):
        alone = extract(blob, side, source=path2)
        assert int(alone["model_state_dict"]["obs_features"]) == E9 | U
        assert int(alone["reference_state_dict"]["obs_features"]) == E9 | U
        back = combine(alone["model_state_dict"], blob["model_state_dict"], side)
        assert int(back["obs_features"]) == E9 | U and int(back["ussr_net.obs_features"]) == E9 | U
        with pytest.raises(ValueError, match="feature bits"):
            combine(alone["model_state_dict"], old, side)


def test_only_zero_width_bits_are_added() -> None:
    net = _a39(E9)
    add_rule_bits_to(net, U)
    assert model_obs_features(net) == E9 | U
    with pytest.raises(ValueError, match="can be added"):
        add_rule_bits_to(net, int(ts.OBS_FEATURE_OPS_BUDGET))
