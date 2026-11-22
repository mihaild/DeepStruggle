"""SimbaV2's hyperspherical trunk (`--ladder-sphere-blocks`, P38 rung 2): the stream stays on the unit sphere and
the heads read exactly sqrt(d) in norm however the weights move; the weight rows are back at unit norm after every
optimiser step; off by default, recorded in the config only when on and recovered from the weights; it replaces the
residual blocks and the trunk norm rather than stacking on them."""

import math

import pytest
import torch

from ai.models.coldwar_net_v2 import create_like
from ai.models.hyperspherical import SphereBlock, SphereTrunk, register_sphere_projection
from ai.models.ladder_net import LadderNet, create_ladder_net, ladder_config_from_state_dict
from ai.training.behavioral_cloning import BehavioralCloningTrainer
from ai.training.nash_pg import make_optimizer
from ai.training.schedule import WeightEMA

CFG = dict(input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=64,
           card_self_attention=False, cross_attention=False, per_entity_heads=16,
           head_context=True, head_static=True, head_entities="country", head_center=True,
           identity_dim=0, drop_static=True, hidden_dim=64, num_res_blocks=0, num_attn_heads=4,
           card_lookup=False, card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0,
           categorical_value=False, branch_head=True, play_mode_head=True)


def _sphere_net(blocks: int = 2) -> LadderNet:
    torch.manual_seed(0)
    return create_ladder_net("cpu", **{**CFG, "sphere_blocks": blocks, "sphere_expansion": 4})


def test_every_block_keeps_the_stream_on_the_unit_sphere_and_the_heads_read_sqrt_d() -> None:
    torch.manual_seed(1)
    trunk = SphereTrunk(32, 3, 4)
    for block in trunk.blocks:                          # weights far from where they started
        assert isinstance(block, SphereBlock)
        block.alpha.data.uniform_(-2.0, 2.0)
        block.scaler.data.mul_(50.0)
    h = torch.nn.functional.normalize(torch.randn(16, 32), dim=-1)
    for block in trunk.blocks:
        h = block(h)
        assert torch.allclose(h.norm(dim=-1), torch.ones(16), atol=1e-5)
    out = trunk(torch.randn(16, 32) * 1000.0)
    assert torch.allclose(out.norm(dim=-1), torch.full((16,), math.sqrt(32)), atol=1e-3)


def test_the_network_reads_a_bounded_trunk_and_trains_through_it() -> None:
    net = _sphere_net()
    obs = torch.randn(8, 3824)
    extract = getattr(net, "extract_features")
    h = extract(obs)
    assert torch.allclose(h.norm(dim=-1), torch.full((8,), math.sqrt(64)), atol=1e-3)
    logits = net(obs, torch.ones(8, 220, dtype=torch.bool))[0]
    logits.float().pow(2).mean().backward()
    block = net.get_submodule("sphere.blocks.0")
    assert isinstance(block, SphereBlock)
    assert block.w1.weight.grad is not None and float(block.w1.weight.grad.abs().sum()) > 0
    assert block.alpha.grad is not None and float(block.alpha.grad.abs().sum()) > 0


def test_weight_rows_are_back_at_unit_norm_after_every_optimiser_step() -> None:
    net = _sphere_net()
    opt = make_optimizer(net, 1e-1)
    for p in net.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    for name, p in net.named_parameters():
        if name.startswith("sphere.") and name.endswith(("w1.weight", "w2.weight")):
            assert torch.allclose(p.norm(dim=1), torch.ones(p.shape[0]), atol=1e-5), name


def test_off_by_default_recorded_only_when_on_and_recovered_from_the_weights() -> None:
    off = create_ladder_net("cpu", **CFG)
    assert getattr(off, "sphere") is None and "sphere_blocks" not in off.ladder_config()
    assert not any(k.startswith("sphere.") for k in off.state_dict())
    assert "sphere_blocks" not in (ladder_config_from_state_dict(off.state_dict()) or {})
    on = _sphere_net(3)
    cfg = on.ladder_config()
    assert cfg["sphere_blocks"] == 3 and cfg["sphere_expansion"] == 4
    found = ladder_config_from_state_dict(on.state_dict()) or {}
    assert found.get("sphere_blocks") == 3 and found.get("sphere_expansion") == 4
    copy = create_like(on)
    assert isinstance(getattr(copy, "sphere"), SphereTrunk) and len(getattr(copy, "sphere").blocks) == 3


def test_it_replaces_the_residual_blocks_and_the_trunk_norm() -> None:
    with pytest.raises(ValueError, match="sphere_blocks"):
        create_ladder_net("cpu", **{**CFG, "num_res_blocks": 2, "sphere_blocks": 2})
    with pytest.raises(ValueError, match="sphere_blocks"):
        create_ladder_net("cpu", **{**CFG, "num_res_blocks": 2, "trunk_norm": True, "sphere_blocks": 2})


def test_the_output_scale_sets_the_heads_input_and_is_recovered_only_off_one() -> None:
    torch.manual_seed(0)
    net = create_ladder_net("cpu", **{**CFG, "sphere_blocks": 2, "sphere_out_scale": 0.5})
    extract = getattr(net, "extract_features")
    h = extract(torch.randn(8, 3824))
    assert torch.allclose(h.pow(2).mean(-1).sqrt(), torch.full((8,), 0.5), atol=1e-4)
    assert net.ladder_config()["sphere_out_scale"] == 0.5
    assert (ladder_config_from_state_dict(net.state_dict()) or {}).get("sphere_out_scale") == 0.5
    plain = _sphere_net()
    assert "sphere_out_scale" not in plain.ladder_config()
    assert not any("out_scale" in k for k in plain.state_dict())


def _rows_unit(net: LadderNet) -> bool:
    assert net.sphere is not None
    return all(torch.allclose(lin.weight.norm(dim=1), torch.ones(lin.weight.shape[0]), atol=1e-5)
               for block in net.sphere.blocks for lin in (block.w1, block.w2))


def _one_step(net: LadderNet, opt: torch.optim.Optimizer) -> None:
    logits, v, _ = net(torch.rand(4, 3824), torch.ones(4, 220, dtype=torch.bool))
    (logits.float().logsumexp(-1).sum() + v.sum()).backward()
    opt.step()
    opt.zero_grad()


def test_every_trainers_optimiser_re_projects_the_sphere() -> None:
    """Audit M-6: the projection is registered on every optimiser that trains a sphere network, not only the RL
    update's; one that trains none of the trunk is left alone."""
    net = _sphere_net()
    _one_step(net, register_sphere_projection(torch.optim.AdamW(net.parameters(), lr=0.1), net))
    assert _rows_unit(net)
    bc = BehavioralCloningTrainer(net, lr=0.1, device="cpu")
    _one_step(net, bc.optimizer)
    assert _rows_unit(net)
    head_only = register_sphere_projection(torch.optim.AdamW(net.val_win_head.parameters(), lr=0.1), net)
    assert not head_only._optimizer_step_post_hooks
    raw = torch.optim.AdamW(net.parameters(), lr=0.1)
    _one_step(net, raw)
    assert not _rows_unit(net)          # the step does break the invariant without the hook


def test_the_weight_average_is_projected_back_to_the_sphere() -> None:
    """Audit M-6: a Euclidean average of unit rows is shorter; the EMA projects it back."""
    live = _sphere_net()
    ema = WeightEMA(live, tau_steps=1000.0)
    opt = make_optimizer(live, 0.1)
    for _ in range(3):
        _one_step(live, opt)
    ema.update(live, 500.0)
    assert _rows_unit(ema.model)        # type: ignore[arg-type]
