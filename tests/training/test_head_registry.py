"""Audit M-4: one registry of head modules (ai/models/head_registry.py) that muP, the head L2 term, the policy-head
reset, the head norms, the token-growing resume and the policy-head rate all read. Before, each kept its own list,
and P42's macro head, Cell A's region attention, the card head and the categorical value head were missing from
some of them."""

from typing import Any, Dict

import torch

from ai.models import mup
from ai.models.coldwar_net_v2 import POLICY_HEAD_MODULES
from ai.models.grow import ENTITY_HEADS, grow_macro_state_dict
from ai.models.head_registry import HEADS, POLICY_HEADS, READOUTS, VALUE_HEADS
from ai.models.ladder_net import create_ladder_net
from ai.training.generic_trainer import reset_policy_head
from ai.training.head_l2 import HEAD_L2_MODULES
from bindings.action_encoder import ActionEncoder
from tests.training.test_mup import CFG

#: Every head at once: both per-entity heads, the macro view, region attention, the head norms, a categorical value.
FULL: Dict[str, Any] = {**CFG, "head_entities": "both", "macro_influence": True, "region_attn_dim": 32,
                        "head_norm": True, "categorical_value": True}


def _net(**over: Any) -> torch.nn.Module:
    torch.manual_seed(0)
    return create_ladder_net("cpu", **{**FULL, **over})


def test_every_registered_head_exists_in_the_full_network() -> None:
    net = _net()
    for h in HEADS:
        if h.name in ("val_vp_head",):              # replaced by value_dist_head under a categorical value
            continue
        assert isinstance(getattr(net, h.name, None), torch.nn.Module), h.name
        if h.readout is not None:
            assert isinstance(net.get_submodule(f"{h.name}.{h.readout}"), torch.nn.Linear), h
    norms = getattr(net, "head_norms")
    assert list(norms.keys()) == ["pe_country", "pe_card", "branch_head_net", "play_mode_head_net", "pe_macro"]


def test_every_consumer_reads_the_registry() -> None:
    for name in ("pe_macro", "pe_card", "region_attn"):
        assert name in POLICY_HEAD_MODULES and name in HEAD_L2_MODULES
        assert f"{name}." in mup.POLICY_HEAD_PREFIXES
    assert "value_dist_head" in HEAD_L2_MODULES and "value_dist_head.3" in mup.OUTPUT_LAYERS
    assert "pe_macro.2" in mup.OUTPUT_LAYERS and "head_norms" in HEAD_L2_MODULES
    assert set(HEAD_L2_MODULES) == set(POLICY_HEADS) | set(VALUE_HEADS) | {"head_norms"}
    assert tuple(mup.OUTPUT_LAYERS) == READOUTS
    assert "pe_macro.0.weight" in ENTITY_HEADS


def test_the_policy_head_rate_covers_every_policy_head_parameter() -> None:
    net = _net()
    scaled = set(mup.policy_head_lr_mults(net, 0.5))
    for name, _ in net.named_parameters():
        top = name.split(".")[0]
        assert (name in scaled) == (top in POLICY_HEADS or top == "head_norms"), name


def test_muP_zeroes_the_macro_and_categorical_readouts() -> None:
    net = _net()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.1)
    done = mup.zero_output_layers(net)
    for name in ("pe_macro.2", "value_dist_head.3", "pe_card.2"):
        assert name in done
        out = net.get_submodule(name)
        assert isinstance(out, torch.nn.Linear) and float(out.weight.detach().abs().sum()) == 0.0


def test_the_policy_head_reset_resets_the_new_heads_and_the_head_norms() -> None:
    net = _net()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    done = reset_policy_head(net, opt)
    for name in ("pe_macro", "pe_card", "region_attn", "head_norms"):
        assert name in done, done
    norm = getattr(net, "head_norms")["pe_macro"]
    assert isinstance(norm, torch.nn.LayerNorm)
    assert torch.equal(norm.weight, torch.ones_like(norm.weight)), "the head norm's gain is fresh again"
    assert float(getattr(net, "val_win_head")[0].bias.min()) >= 0.5, "the value heads are kept"


def test_the_macro_head_has_its_head_norm_and_growing_into_the_macro_view_keeps_it_identity() -> None:
    base = _net(macro_influence=False, region_attn_dim=0)
    macro = _net(region_attn_dim=0)
    sd = grow_macro_state_dict(base.state_dict(), macro.state_dict(), ActionEncoder.BASE_ACTION_SIZE)
    assert "head_norms.pe_macro.weight" in sd
    macro.load_state_dict(sd)
    obs = torch.randn(3, 3824)
    mask = torch.ones(3, ActionEncoder.FLAT_ACTION_SIZE, dtype=torch.bool)
    with torch.no_grad():
        a = base.eval()(obs, mask[:, :ActionEncoder.BASE_ACTION_SIZE])[0]
        b = macro.eval()(obs, mask)[0][:, :ActionEncoder.BASE_ACTION_SIZE]
    assert torch.allclose(a, b, atol=1e-5)
