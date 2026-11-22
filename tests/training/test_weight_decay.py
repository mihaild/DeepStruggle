"""The RL update's optimiser (`--weight-decay`, `--no-decay-norms-biases`; P38 rung 1): by default one AdamW group
with every parameter decayed at 1e-4, as every earlier run; with norms and biases excluded, the parameters of fewer
than two dimensions sit in an undecayed second group, and a step decays only the matrices."""

import torch
import torch.nn as nn

from ai.training.nash_pg import make_optimizer


def _net() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.LayerNorm(16), nn.GELU(), nn.Linear(16, 4))


def test_the_default_is_one_group_decayed_as_before() -> None:
    opt = make_optimizer(_net(), 3e-4)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["weight_decay"] == 1e-4


def test_norms_and_biases_can_be_left_undecayed() -> None:
    net = _net()
    opt = make_optimizer(net, 1e-2, weight_decay=0.1, decay_norms_biases=False)
    # Contiguous runs in model order (audit T-14), each decayed exactly when its parameters are matrices.
    for g in opt.param_groups:
        assert g["weight_decay"] == (0.1 if g["params"][0].ndim >= 2 else 0.0)
        assert len({p.ndim >= 2 for p in g["params"]}) == 1
    assert [p for g in opt.param_groups for p in g["params"]] == list(net.parameters())
    # A step with zero gradients changes only what decays: the matrices shrink, gains and biases stay.
    before = {n: p.detach().clone() for n, p in net.named_parameters()}
    for p in net.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    for n, p in net.named_parameters():
        if p.ndim >= 2:
            assert torch.allclose(p, before[n] * (1 - 1e-2 * 0.1))
        else:
            assert torch.equal(p, before[n])
