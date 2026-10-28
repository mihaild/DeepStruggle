"""P39 arms 4 and 5: per-parameter learning rates. Without them the optimiser is exactly the single group every earlier
run has; with them the groups are contiguous runs in model order carrying their multiplier, the schedule scales each
group by it, and a single-group state maps onto the new layout. muP's multipliers are 1 at the A8 widths, base
fan-in over fan-in at wider ones, and the readouts start at zero."""

from typing import Any, Dict

import pytest
import torch

from ai.models.ladder_net import create_ladder_net
from ai.models.mup import base_shapes_for, combine_lr_mults, mup_lr_mults, policy_head_lr_mults, zero_output_layers
from ai.training.nash_pg import make_optimizer, regroup_optimizer_state

CFG: Dict[str, Any] = dict(input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=256,
                           card_self_attention=False, cross_attention=False, per_entity_heads=64, head_context=True,
                           head_static=True, head_entities="country", head_center=True, identity_dim=0,
                           drop_static=True, hidden_dim=480, num_res_blocks=0, num_attn_heads=4, card_lookup=False,
                           card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0, categorical_value=False,
                           branch_head=True, play_mode_head=True)


def _net(**over: Any) -> torch.nn.Module:
    torch.manual_seed(0)
    return create_ladder_net("cpu", **{**CFG, **over})


def _linear(net: torch.nn.Module, name: str) -> torch.nn.Linear:
    mod = net.get_submodule(name)
    assert isinstance(mod, torch.nn.Linear)
    return mod


def _mults(net: torch.nn.Module) -> Dict[str, float]:
    cfg = getattr(net, "ladder_config")()
    return mup_lr_mults(net, base_shapes_for(cfg, 256, 480, lambda **kw: create_ladder_net("cpu", **kw)), 480)


def test_without_multipliers_the_optimiser_is_the_old_single_group() -> None:
    opt = make_optimizer(_net(), 3e-4)
    assert len(opt.param_groups) == 1 and opt.param_groups[0]["betas"] == (0.9, 0.999)
    assert "lr_mult" not in opt.param_groups[0]


def test_multipliers_make_contiguous_groups_in_model_order() -> None:
    net = _net()
    mults = policy_head_lr_mults(net, 0.5)
    opt = make_optimizer(net, 3e-4, betas=(0.9, 0.95), eps=1e-6, lr_mults=mults)
    order = [p for g in opt.param_groups for p in g["params"]]
    assert order == list(net.parameters())                      # the global order is the model's
    by_param = {id(p): g for g in opt.param_groups for p in g["params"]}
    for name, p in net.named_parameters():
        g = by_param[id(p)]
        assert g["lr"] == pytest.approx(3e-4 * mults.get(name, 1.0)) and g["lr_mult"] == mults.get(name, 1.0)
    assert opt.param_groups[0]["betas"] == (0.9, 0.95) and opt.param_groups[0]["eps"] == 1e-6


def test_a_single_group_state_maps_onto_the_grouped_layout() -> None:
    net = _net()
    old = make_optimizer(net, 3e-4)
    net(torch.randn(4, 3824), torch.ones(4, 220, dtype=torch.bool))[0].sum().backward()
    old.step()
    new = make_optimizer(net, 3e-4, lr_mults=policy_head_lr_mults(net, 0.5))
    new.load_state_dict(regroup_optimizer_state(old.state_dict(), new))
    p0 = next(net.parameters())
    assert torch.equal(new.state[p0]["exp_avg"], old.state[p0]["exp_avg"])
    assert len(new.param_groups) > 1


def test_mup_is_the_identity_at_the_base_widths_and_scales_wider_layers() -> None:
    assert _mults(_net()) == {}
    wide = _net(entity_proj_dim=512, hidden_dim=768)
    m = _mults(wide)
    base_f, wide_f = _linear(_net(), "fusion_in.0").in_features, _linear(wide, "fusion_in.0").in_features
    assert wide_f > base_f and m["fusion_in.0.weight"] == pytest.approx(base_f / wide_f)   # projections + globals
    assert m["policy_head.0.weight"] == pytest.approx(480 / 768)
    assert "lad_board.0.weight" not in m                          # an input projection: its fan-in did not grow


def test_mup_scales_blocks_the_base_lacks_against_the_base_width() -> None:
    m = _mults(_net(entity_proj_dim=512, hidden_dim=768, sphere_blocks=2))
    assert m["sphere.blocks.0.w1.weight"] == pytest.approx(480 / 768)
    assert m["sphere.blocks.0.w2.weight"] == pytest.approx(480 / (4 * 768))


def test_readouts_are_zeroed_and_multipliers_combine() -> None:
    net = _net()
    done = zero_output_layers(net)
    assert "policy_head.3" in done and float(_linear(net, "policy_head.3").weight.detach().abs().sum()) == 0.0
    assert combine_lr_mults({"a": 0.5}, {"a": 0.5, "b": 2.0}) == {"a": 0.25, "b": 2.0}
    assert combine_lr_mults(None, {}) is None
