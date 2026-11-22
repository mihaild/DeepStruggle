"""The policy-head reset at a branch point (`--resume-reset-policy-head`, P37's last-layer reset): the policy head's
modules get a fresh network's weights and lose their Adam moments; the trunk, the value heads and their moments are
untouched; the run's random stream is unchanged."""

import torch

from ai.models.coldwar_net_v2 import POLICY_HEAD_MODULES
from ai.models.ladder_net import create_ladder_net
from ai.training.generic_trainer import reset_policy_head

CFG = dict(input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=64,
           card_self_attention=False, cross_attention=False, per_entity_heads=16,
           head_context=True, head_static=True, head_entities="country", head_center=True,
           identity_dim=0, drop_static=True, hidden_dim=64, num_res_blocks=0, num_attn_heads=4,
           card_lookup=False, card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0,
           categorical_value=False, branch_head=True, play_mode_head=True)


def _trained() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(0)
    net = create_ladder_net("cpu", **CFG)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
    for _ in range(3):                                     # move every weight and give every parameter moments
        opt.zero_grad()
        logits, v = net(torch.randn(16, 3824), torch.ones(16, 220, dtype=torch.bool))[:2]
        (logits.float().pow(2).mean() + v.float().pow(2).mean()).backward()
        opt.step()
    return net, opt


def test_the_head_is_fresh_and_the_rest_is_kept() -> None:
    net, opt = _trained()
    before = {n: p.detach().clone() for n, p in net.named_parameters()}
    had_moments = {n for n, p in net.named_parameters() if p in opt.state}
    torch.manual_seed(123)
    expected_stream = torch.rand(4)
    torch.manual_seed(123)
    done = reset_policy_head(net, opt, seed=7)
    assert torch.equal(torch.rand(4), expected_stream)                     # the random stream is untouched
    assert set(done) == {m for m in POLICY_HEAD_MODULES if getattr(net, m, None) is not None}
    for name, p in net.named_parameters():
        top = name.split(".")[0]
        if top in POLICY_HEAD_MODULES:
            assert not torch.equal(p, before[name]), name                  # fresh weights...
            assert p not in opt.state, name                                # ...and fresh moments
        else:
            assert torch.equal(p, before[name]), name                      # trunk and value heads as trained
            assert (p in opt.state) == (name in had_moments), name         # and their moments kept


def test_the_reset_is_deterministic_in_its_seed() -> None:
    a, opt_a = _trained()
    b, opt_b = _trained()
    reset_policy_head(a, opt_a, seed=7)
    reset_policy_head(b, opt_b, seed=7)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
