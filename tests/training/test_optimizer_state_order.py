"""Audit T-14: every optimizer layout indexes the parameters in model order, and a resume maps each Adam moment onto
the parameter it was taken for -- by name when the names were saved, and from the pre-2026-11-22
--no-decay-norms-biases layout (every >= 2-D parameter, then the rest) when they were not."""

from __future__ import annotations

from typing import Dict

import pytest
import torch
import torch.nn as nn

from ai.training.generic_trainer import _load_optimizer_state
from ai.training.nash_pg import make_optimizer, optimizer_param_names, optimizer_state_in_model_order


def _net() -> nn.Module:
    torch.manual_seed(0)
    # Equal-shaped parameters of both kinds (4-wide biases and LayerNorm affines) interleaved with matrices.
    return nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4), nn.Linear(4, 4), nn.Linear(4, 2))


def _step(net: nn.Module, opt: torch.optim.Optimizer) -> Dict[str, torch.Tensor]:
    """One step with a distinct gradient per parameter; each parameter's first moment by name."""
    for k, (_, p) in enumerate(net.named_parameters()):
        p.grad = torch.full_like(p, float(k + 1))
    opt.step()
    by_id = {id(p): n for n, p in net.named_parameters()}
    return {by_id[id(p)]: opt.state[p]["exp_avg"].clone() for g in opt.param_groups for p in g["params"]}


def _moments(net: nn.Module, opt: torch.optim.Optimizer) -> Dict[str, torch.Tensor]:
    return {n: opt.state[p]["exp_avg"] for n, p in net.named_parameters()}


def test_every_layout_indexes_the_parameters_in_model_order() -> None:
    net = _net()
    for kw in (dict(), dict(decay_norms_biases=False), dict(decay_norms_biases=False, lr_mults={"3.weight": 0.5})):
        opt = make_optimizer(net, 1e-3, 1e-2, **kw)  # type: ignore[arg-type]
        assert optimizer_param_names(opt, net) == [n for n, _ in net.named_parameters()]
    groups = make_optimizer(net, 1e-3, 1e-2, decay_norms_biases=False).param_groups
    assert all((g["weight_decay"] == 0.0) == (g["params"][0].ndim < 2) for g in groups)


def test_an_old_no_decay_state_resumes_with_each_moment_on_its_parameter() -> None:
    net = _net()
    old = torch.optim.AdamW([{"params": [p for p in net.parameters() if p.ndim >= 2], "weight_decay": 1e-2},
                             {"params": [p for p in net.parameters() if p.ndim < 2], "weight_decay": 0.0}], lr=1e-3)
    want = _step(net, old)
    saved = optimizer_state_in_model_order(old.state_dict(), None, net.state_dict(),
                                           [n for n, _ in net.named_parameters()])
    for target in (make_optimizer(net, 1e-3, 1e-2), make_optimizer(net, 1e-3, 1e-2, decay_norms_biases=False)):
        _load_optimizer_state(target, saved)
        got = _moments(net, target)
        for n in want:
            assert torch.equal(got[n], want[n]), n


def test_named_states_map_by_name_and_the_default_state_is_untouched() -> None:
    net = _net()
    src = make_optimizer(net, 1e-3, 1e-2, decay_norms_biases=False)
    want = _step(net, src)
    sd = src.state_dict()
    names = optimizer_param_names(src, net)
    target = make_optimizer(net, 1e-3, 1e-2, lr_mults={"0.weight": 0.5})
    _load_optimizer_state(target, optimizer_state_in_model_order(sd, names, net.state_dict(), names))
    got = _moments(net, target)
    assert all(torch.equal(got[n], want[n]) for n in want)
    default = make_optimizer(net, 1e-3, 1e-2)
    _step(net, default)
    dsd = default.state_dict()
    assert optimizer_state_in_model_order(dsd, None, net.state_dict(), names) is dsd


def test_a_state_whose_names_or_shapes_do_not_fit_is_refused() -> None:
    net = _net()
    opt = make_optimizer(net, 1e-3, 1e-2)
    _step(net, opt)
    names = optimizer_param_names(opt, net)
    with pytest.raises(RuntimeError, match="names"):
        optimizer_state_in_model_order(opt.state_dict(), names[:-1] + ["nope"], net.state_dict(), names)
    swapped = names.copy()
    swapped[0], swapped[1] = swapped[1], swapped[0]     # 0.weight (4, 3) <-> 0.bias (4,)
    with pytest.raises(RuntimeError, match="shape"):
        optimizer_state_in_model_order(opt.state_dict(), swapped, net.state_dict(), names)
