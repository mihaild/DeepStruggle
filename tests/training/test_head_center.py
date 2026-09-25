"""--ladder-head-center: the per-entity heads' features are centred across entities before the last
layer. That removes the one direction the E4 policy cannot see -- a shift common to every country
logit -- so the level cannot drift, while every decision whose legal set is countries only keeps
the same distribution; the final bias gets no gradient; and the setting survives a checkpoint."""

from __future__ import annotations

import copy
from typing import Any, Dict

import pytest
import torch
import torch.nn.functional as F

from ai.models.ladder_net import create_ladder_net, ladder_config_from_state_dict
from bindings.action_encoder import ActionEncoder as A

M2D: Dict[str, Any] = dict(
    input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=64,
    card_self_attention=False, cross_attention=False, per_entity_heads=16, head_context=True,
    head_static=True, head_entities="country", identity_dim=0, card_lookup=False,
    card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0, drop_static=True,
    hidden_dim=64, num_res_blocks=1, num_attn_heads=4, categorical_value=False)


def _inputs(batch: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    import ts_engine as ts
    g = torch.Generator().manual_seed(0)
    obs = torch.randn(batch, ts.OBS_SIZE, generator=g)
    mask = torch.zeros(batch, 220, dtype=torch.uint8)
    mask[:, A.NODE_OFFSET:A.BRANCH_OFFSET] = (torch.rand(batch, 84, generator=g) > 0.5).to(torch.uint8)
    mask[:, A.NODE_OFFSET] = 1                  # countries only, as in every E4 placement
    return obs, mask


def _perturb(net: torch.nn.Module) -> None:
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.3 * torch.randn(p.shape, generator=g))


def test_country_only_decisions_keep_the_same_distribution() -> None:
    torch.manual_seed(0)
    plain = create_ladder_net("cpu", **M2D)
    _perturb(plain)
    centred = create_ladder_net("cpu", **M2D, head_center=True)
    centred.load_state_dict({**plain.state_dict(), "pe_center": torch.ones(())})
    plain.eval(); centred.eval()
    obs, mask = _inputs()
    with torch.no_grad():
        a = F.log_softmax(plain(obs, mask)[0].float(), -1)
        b = F.log_softmax(centred(obs, mask)[0].float(), -1)
    legal = mask.bool()
    assert torch.allclose(a[legal], b[legal], atol=1e-4)


def test_the_country_correction_has_no_common_component_but_its_bias() -> None:
    torch.manual_seed(0)
    net = create_ladder_net("cpu", **M2D, head_center=True)
    _perturb(net)
    net.eval()
    cap: Dict[str, torch.Tensor] = {}
    orig = net._entity_out
    def spy(head: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        cap["out"] = orig(head, x)
        return cap["out"]
    net._entity_out = spy  # type: ignore[method-assign]
    obs, mask = _inputs()
    with torch.no_grad():
        net(obs, mask)
    assert net.pe_country is not None
    last = net.pe_country[-1]
    assert isinstance(last, torch.nn.Linear)
    bias = last.bias
    assert torch.allclose(cap["out"].mean(dim=1), bias.expand(obs.shape[0]), atol=1e-5)


def test_the_final_bias_gets_no_gradient_from_a_country_only_policy_loss() -> None:
    torch.manual_seed(0)
    net = create_ladder_net("cpu", **M2D, head_center=True)
    _perturb(net)
    obs, mask = _inputs()
    lp = F.log_softmax(net(obs, mask)[0].float(), -1)
    act = torch.distributions.Categorical(logits=lp).sample()
    (-lp.gather(1, act[:, None]).mean()).backward()
    assert net.pe_country is not None
    last = net.pe_country[-1]
    assert isinstance(last, torch.nn.Linear)
    g = last.bias.grad
    assert g is not None and float(g.abs().max()) < 1e-5


def test_the_setting_is_recovered_from_the_weights() -> None:
    net = create_ladder_net("cpu", **M2D, head_center=True)
    cfg = ladder_config_from_state_dict(net.state_dict())
    assert cfg is not None and cfg["head_center"] is True
    plain = create_ladder_net("cpu", **M2D)
    cfg0 = ladder_config_from_state_dict(plain.state_dict())
    assert cfg0 is not None and cfg0["head_center"] is False
    assert net.ladder_config()["head_center"] is True
    assert copy.deepcopy(net).head_center


def test_without_per_entity_heads_it_is_refused() -> None:
    cfg = {**M2D, "per_entity_heads": 0, "head_entities": "both"}
    with pytest.raises(ValueError):
        create_ladder_net("cpu", **cfg, head_center=True)


def test_the_cli_offers_it_off_by_default() -> None:
    from ai.training.train import build_parser
    assert build_parser().parse_args([]).ladder_head_center is False
