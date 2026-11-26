"""Branching a trained state into a zero-width observation bit (--resume-add-features, owner 2026-11-26).

uncapped_deficit writes board slots 24 and 25 without their cap and has no width, so a state trained without it
loads unchanged into a network built with it: the bit is added to the network's, pi_ref's and (for an A39 network)
the USSR copy's recorded feature set, the weights are untouched, and a resume without the flag is refused. One side's
network taken out of such a state (tools/lib/side_nets.py) carries the bit, and is put back only beside a base that
reads it too."""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

import ts_engine as ts

from ai.models.ladder_net import LadderNet, create_ladder_net
from ai.training.feature_branch import swapped_columns
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


R = int(ts.OBS_FEATURE_RELATIVE_EFFECTS)
H = int(ts.OBS_FEATURE_HONEST_CONTEXT)
#: Effect bits each position gets at random: every paired bit and the four space-attempt bits.
_PAIRED = sorted({b for p in ts.OBS_RELATIVE_EFFECT_PAIRS for b in p} | {43, 44, 45, 46})


def _views(n: int, features: int, seed: int = 3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """`n` decision positions with random paired effect bits: their observation without and with RELATIVE_EFFECTS
    (both under `features`), their masks, and whether the decider's attempt slots read the same in the two views."""
    rng = np.random.default_rng(seed)
    runner = ts.VectorizedBatchRunner(n, 4321 + seed)
    for _ in range(30):
        masks = np.asarray(runner.get_action_masks())
        runner.step_flat_all([int(rng.choice(np.flatnonzero(m))) if m.any() else 0 for m in masks], True)
    old, new, same = [], [], []
    for i in range(n):
        st = runner.get_state(i)
        for b in _PAIRED:
            if rng.random() < 0.4:
                st.persistent_effects = int(st.persistent_effects) | (1 << b)
        p = st.ctx().decision_player
        a = np.asarray(ts.extract_observation_features(st, p, features))
        r = np.asarray(ts.extract_observation_features(st, p, features | R))
        old.append(a)
        new.append(r)
        g = 3724 + int(ts.OBS_GLOBAL_EFFECT_BITS)
        same.append(bool(a[g + 43] == r[g + 43] and a[g + 44] == r[g + 44]))
    masks_t = torch.from_numpy(np.asarray(runner.get_action_masks()))
    return (torch.from_numpy(np.stack(old)).float(), torch.from_numpy(np.stack(new)).float(), masks_t,
            np.asarray(same))


def _outputs(net: torch.nn.Module, obs: torch.Tensor, masks: torch.Tensor) -> List[torch.Tensor]:
    with torch.no_grad():
        return [o for o in net.eval()(obs, masks) if isinstance(o, torch.Tensor)]


def test_relative_effects_branch_computes_what_the_network_did() -> None:
    """An A39 network branched into relative_effects (its USSR copy's global input columns swapped pair by pair)
    gives, on the relative view, exactly its outputs on the absolute view -- wherever the decider's attempt slots read
    the same in both (always for a US decider). Where they differ (a USSR decider whose own attempts differ from the
    US's), the outputs may differ: that slot is new information, not a permutation."""
    net = _a39(E9 | U)
    old, new, masks, same = _views(96, E9 | U)
    branched = copy.deepcopy(net)
    add_rule_bits_to(branched, R)
    assert model_obs_features(branched) == E9 | U | R
    a, b = _outputs(net, old, masks), _outputs(branched, new, masks)
    us = old[:, 3724 + 61] > 0.5
    assert us.any() and (~us).any() and bool(same[~us.numpy()].any())
    rows = torch.from_numpy(same)
    for x, y in zip(a, b):
        assert torch.allclose(x[rows], y[rows], atol=1e-5, rtol=0), (x[rows] - y[rows]).abs().max()
    # The US network is untouched; the USSR network's global projection moved by the pairs only.
    sd_a, sd_b = net.state_dict(), branched.state_dict()
    for k in sd_a:
        if k.endswith("obs_features"):
            continue
        if k == "ussr_net.global_proj.0.weight":
            w, v = sd_a[k], sd_b[k]
            for i, j in swapped_columns():
                assert torch.equal(v[:, i], w[:, j]) and torch.equal(v[:, j], w[:, i])
        else:
            assert torch.equal(sd_a[k], sd_b[k]), k


def test_relative_effects_on_resume_permutes_the_network_reference_and_moments(tmp_path: Any) -> None:
    t = _trainer(_a39(E9))
    t.collect_rollouts()
    t.train_step()
    path = os.path.join(tmp_path, "resume_1000steps.pt")
    save_resume_state(path, t.active_net, t, 3, 1000, 1.0, seed=45)
    t2 = _trainer(_a39(E9 | U | H | R, seed=9))
    load_resume_state(path, t2.active_net, t2, seed=45, add_rules=U | H | R)
    # Only R's effect on the observation is compared here (U and H change values, not a permutation): the branch reads
    # the E9 view with R where the state read it without.
    old, new, masks, same = _views(64, E9)
    rows = torch.from_numpy(same)
    a = _outputs(t.active_net, old, masks)
    b = _outputs(t2.active_net, new, masks)
    for x, y in zip(a, b):
        assert torch.allclose(x[rows], y[rows], atol=1e-5, rtol=0)
    ref_a, ref_b = t.reference_net.state_dict(), t2.reference_net.state_dict()
    k = "ussr_net.global_proj.0.weight"
    for i, j in swapped_columns():
        assert torch.equal(ref_b[k][:, i], ref_a[k][:, j])
    # Adam's moments moved with their columns.
    pa = dict(t.active_net.named_parameters())[k]
    pb = dict(t2.active_net.named_parameters())[k]
    ma, mb = t.optimizer.state[pa], t2.optimizer.state[pb]
    for m in ("exp_avg", "exp_avg_sq"):
        for i, j in swapped_columns():
            assert torch.equal(mb[m][:, i], ma[m][:, j]) and torch.equal(mb[m][:, j], ma[m][:, i])


def test_relative_effects_is_refused_for_a_network_reading_both_sides() -> None:
    torch.manual_seed(0)
    plain = create_ladder_net("cpu", **{**CFG, "obs_features": E9})
    with pytest.raises(ValueError, match="both sides"):
        add_rule_bits_to(plain, R)
