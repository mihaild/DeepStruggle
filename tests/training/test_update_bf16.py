"""P52 (--update-bf16, --bf16-fp32-heads): the heads-in-fp32 switch changes nothing without autocast, and the flag
refuses to run without --update-bf16. (The autocast path itself needs CUDA; its numerics were measured on the GPU.)"""

from __future__ import annotations

import pytest
import torch

from ai.models.ladder_net import LadderNet, create_ladder_net
from ai.training.nash_pg import NashPGTrainer
from bindings.ts_env import TsVectorizedEnv


def _net() -> LadderNet:
    torch.manual_seed(0)
    return create_ladder_net(torch.device("cpu"), input_mode="grouped", aggregation="flatten", entity_dim=16,
                             entity_proj_dim=64, card_self_attention=False, cross_attention=False,
                             per_entity_heads=16, head_context=True, head_static=True, head_entities="country",
                             head_center=True, identity_dim=0, drop_static=True, hidden_dim=64, num_res_blocks=0,
                             num_attn_heads=4, card_lookup=False, card_lookup_heads=0, card_lookup_dim=0,
                             card_lookup_identity_dim=0, categorical_value=False).eval()


def test_fp32_heads_change_nothing_without_autocast() -> None:
    net = _net()
    env = TsVectorizedEnv(num_envs=8, base_seed=3)
    obs, masks, _ = env.reset_all()
    o, m = torch.from_numpy(obs).float(), torch.from_numpy(masks)
    with torch.no_grad():
        plain = net(o, m), net.forward_value(o), net.forward_with_risk(o, m)
        setattr(net, "fp32_heads", True)
        heads = net(o, m), net.forward_value(o), net.forward_with_risk(o, m)
    for a, b in zip(plain[0] + (plain[1],) + plain[2], heads[0] + (heads[1],) + heads[2]):
        assert torch.equal(a, b)


def test_fp32_heads_need_bf16() -> None:
    with pytest.raises(ValueError, match="needs --update-bf16"):
        NashPGTrainer(active_net=_net(), env=TsVectorizedEnv(num_envs=4, base_seed=7), num_envs=4, buffer_size=4,
                      lr=3e-4, eta=0.1, ref_update_freq=500, cuda_graphs=False, device=torch.device("cpu"),
                      bf16_fp32_heads=True)
