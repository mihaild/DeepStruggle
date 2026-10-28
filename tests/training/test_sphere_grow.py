"""P39 arm 6: a sphere trunk grown into a trained network in place. With --ladder-sphere-carry-norm the trunk's output
keeps the stem's norm, so at alpha 0 it is the identity: a state without the trunk resumes into a network with it
(--resume-grow-sphere), alone or with --resume-widen, and the network and pi_ref play exactly what they did; the
saved Adam moments follow their parameters and the grown ones start fresh; the setting is recovered from the
weights."""

import os
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest
import torch

import ts_engine as ts
from ai.models.hyperspherical import SphereTrunk
from ai.models.ladder_net import LadderNet, create_ladder_net, ladder_config_from_state_dict
from ai.training.generic_trainer import load_resume_state


def _net(proj: int = 32, sphere: int = 0, seed: int = 0) -> LadderNet:
    torch.manual_seed(seed)
    extra: Dict[str, Any] = {"sphere_blocks": sphere, "sphere_carry_norm": True} if sphere else {}
    return create_ladder_net("cpu", input_mode="grouped", aggregation="flatten", entity_dim=16,
                             entity_proj_dim=proj, card_self_attention=False, cross_attention=False,
                             per_entity_heads=16, head_context=True, head_static=True,
                             head_entities="country", head_center=True, identity_dim=0,
                             drop_static=True, hidden_dim=64, num_res_blocks=0, num_attn_heads=4,
                             card_lookup=False, card_lookup_heads=0, card_lookup_dim=0,
                             card_lookup_identity_dim=0, categorical_value=False, **extra).eval()


def _batch(n: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    runner = ts.VectorizedBatchRunner(n, 4321)
    rng = np.random.default_rng(3)
    for _ in range(30):
        masks = np.asarray(runner.get_action_masks())
        runner.step_flat_all([int(rng.choice(np.flatnonzero(m))) if m.any() else 0 for m in masks], True)
    obs = torch.from_numpy(np.asarray(runner.get_observations(), dtype=np.float32))
    return obs, torch.from_numpy(np.asarray(runner.get_action_masks()))


def _logits(net: LadderNet, obs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return net(obs, masks)[0]


def _saved(tmp_path: Any, net: LadderNet, ref: LadderNet) -> tuple[str, torch.optim.AdamW]:
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    obs, masks = _batch(8)
    net.train()
    logits, v_win, v_vp = net(obs, masks)
    (logits.float().logsumexp(-1).sum() + v_win.sum() + v_vp.sum()).backward()
    opt.step()
    net.eval()
    blob: Dict[str, Any] = {"model_state_dict": net.state_dict(), "reference_state_dict": ref.state_dict(),
                            "optimizer_state_dict": opt.state_dict(), "total_env_steps": 1000, "iteration": 3,
                            "elapsed_seconds": 1.0, "seed": 44, "torch_rng_state": torch.get_rng_state(),
                            "numpy_rng_state": np.random.get_state(), "ema_state_dict": None}
    path = os.path.join(str(tmp_path), "resume.pt")
    torch.save(blob, path)
    return path, opt


def test_a_carry_norm_trunk_at_alpha_zero_is_the_identity() -> None:
    trunk = SphereTrunk(64, 2, carry_norm=True)
    trunk.identity_init()
    x = torch.randn(5, 64) * 3.0
    assert torch.allclose(trunk(x), x, atol=1e-5)
    with pytest.raises(ValueError):
        SphereTrunk(64, 2).identity_init()                      # a fixed-RMS trunk is not the identity at alpha 0


def test_the_setting_is_recovered_from_the_weights() -> None:
    cfg = ladder_config_from_state_dict(_net(sphere=2).state_dict())
    assert cfg is not None and cfg.get("sphere_carry_norm") is True
    assert _net(sphere=2).ladder_config()["sphere_carry_norm"] is True


@pytest.mark.parametrize("widen", [False, True])
def test_growing_a_sphere_in_place_plays_what_the_saved_network_did(tmp_path: Any, widen: bool) -> None:
    old, ref = _net(32), _net(32, seed=7)
    path, old_opt = _saved(tmp_path, old, ref)
    new, new_ref = _net(64 if widen else 32, sphere=2, seed=1), _net(64 if widen else 32, sphere=2, seed=2)
    trainer = SimpleNamespace(device="cpu", optimizer=torch.optim.AdamW(new.parameters(), lr=1e-3),
                              reference_net=new_ref, total_env_steps=0)
    load_resume_state(path, new, trainer, seed=44, widen=widen, grow_sphere=True)
    obs, masks = _batch()
    for a, b in ((old, new), (ref, new_ref)):
        assert torch.allclose(_logits(a, obs, masks), _logits(b, obs, masks), atol=1e-3)
    # the saved moments follow their parameters; the grown trunk's parameters start without state
    params, old_params = dict(new.named_parameters()), dict(old.named_parameters())
    for n, p in params.items():
        st = trainer.optimizer.state.get(p, {})
        had = "exp_avg" in old_opt.state.get(old_params[n], {}) if n in old_params else False
        assert (n.startswith("sphere.") and not st) or ("exp_avg" in st) == had, n
    head = "policy_head.3.weight"                                 # a shape widening leaves alone
    assert torch.equal(trainer.optimizer.state[params[head]]["exp_avg"], old_opt.state[old_params[head]]["exp_avg"])


def test_growing_refuses_a_fixed_rms_trunk_or_a_state_that_has_one(tmp_path: Any) -> None:
    old, ref = _net(32), _net(32, seed=7)
    path, _ = _saved(tmp_path, old, ref)
    plain_sphere = create_ladder_net("cpu", **{**old.ladder_config(), "sphere_blocks": 2})
    trainer = SimpleNamespace(device="cpu", optimizer=torch.optim.AdamW(plain_sphere.parameters()),
                              reference_net=create_ladder_net("cpu", **{**old.ladder_config(), "sphere_blocks": 2}),
                              total_env_steps=0)
    with pytest.raises(ValueError):
        load_resume_state(path, plain_sphere, trainer, seed=44, grow_sphere=True)
