"""The policy-head norms logged every iteration (P38): one per head module, a fresh network's denominators built
without touching the caller's random stream, and a fresh network reads ~1x its own fresh norms."""

import torch

from ai.models.coldwar_net_v2 import POLICY_HEAD_MODULES, fresh_policy_head_norms, policy_head_norms
from ai.models.ladder_net import create_ladder_net

CFG = dict(input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=64,
           card_self_attention=False, cross_attention=False, per_entity_heads=16,
           head_context=True, head_static=True, head_entities="country", head_center=True,
           identity_dim=0, drop_static=True, hidden_dim=64, num_res_blocks=0, num_attn_heads=4,
           card_lookup=False, card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0,
           categorical_value=False, branch_head=True, play_mode_head=True)


def test_every_head_module_is_measured_and_a_fresh_network_reads_about_one() -> None:
    torch.manual_seed(5)
    net = create_ladder_net("cpu", **CFG)
    norms = policy_head_norms(net)
    # Every registered policy head the network has (this one has no card, macro or region-attention head).
    assert set(norms) == {m for m in POLICY_HEAD_MODULES if getattr(net, m, None) is not None}
    assert set(norms) == {"policy_head", "pe_trunk", "pe_country", "branch_head_net", "play_mode_head_net"}
    fresh = fresh_policy_head_norms(net)
    for k, v in norms.items():
        if k not in ("branch_head_net", "play_mode_head_net"):      # zero-initialised last layers make these small
            assert 0.8 < v / fresh[k] < 1.25


def test_the_fresh_denominators_leave_the_random_stream_alone() -> None:
    net = create_ladder_net("cpu", **CFG)
    torch.manual_seed(11)
    expected = torch.rand(3)
    torch.manual_seed(11)
    fresh_policy_head_norms(net)
    assert torch.equal(torch.rand(3), expected)
