"""P44's two trainer additions: L2 toward the initial weights on the heads (`--head-l2-init-coef`) and a resume with a
fresh optimiser (`--resume-fresh-optimizer`)."""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from ai.models.ladder_net import create_ladder_net
from ai.training.generic_trainer import RESUME_FILENAME, load_resume_state, save_resume_state
from ai.training.head_l2 import HEAD_L2_MODULES, head_init_params, head_l2_to_init, head_pairs
from ai.training.nash_pg import BaseNashPGTrainer

CFG = dict(input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=64,
           card_self_attention=False, cross_attention=False, per_entity_heads=16,
           head_context=True, head_static=True, head_entities="country", head_center=True,
           identity_dim=0, drop_static=True, hidden_dim=64, num_res_blocks=0, num_attn_heads=4,
           card_lookup=False, card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0,
           categorical_value=False, branch_head=True, play_mode_head=True, head_norm=True)


def _net(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return create_ladder_net("cpu", **CFG)


def _train(net: nn.Module, opt: torch.optim.Optimizer, n: int = 3) -> None:
    for _ in range(n):
        opt.zero_grad()
        logits, v = net(torch.randn(16, 3824), torch.ones(16, 220, dtype=torch.bool))[:2]
        (logits.float().pow(2).mean() + v.float().pow(2).mean()).backward()
        opt.step()


class _Trainer:
    """What load_resume_state and save_resume_state touch, with NashPG's own set_head_init."""

    set_head_init = BaseNashPGTrainer.set_head_init

    def __init__(self, net: nn.Module, coef: float) -> None:
        self.device = torch.device("cpu")
        self.active_net = net
        self.optimizer = torch.optim.AdamW(net.parameters(), lr=1e-2)
        self.reference_net = _net(1)
        self.total_env_steps = 0
        self.total_iterations = 0
        self.head_l2_init_coef = coef
        self.head_init = head_init_params(net) if coef > 0.0 else {}


def test_the_term_covers_the_heads_and_not_the_trunk() -> None:
    net = _net()
    init = head_init_params(net)
    tops = {n.split(".")[0] for n in init}
    assert tops <= set(HEAD_L2_MODULES)
    assert {"policy_head", "pe_country", "branch_head_net", "play_mode_head_net", "head_norms",
            "val_win_head", "val_vp_head"} <= tops
    assert not any(n.split(".")[0] in ("lad_board", "lad_card", "fusion_in", "global_proj") for n in init)


def test_zero_at_the_start_and_its_gradient_is_twice_the_distance() -> None:
    net = _net()
    pairs = head_pairs(net, head_init_params(net))
    assert float(head_l2_to_init(pairs).detach()) == 0.0
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.01 * torch.randn_like(p))
    net.zero_grad()
    head_l2_to_init(pairs).backward()
    by_name = dict(net.named_parameters())
    init = head_init_params(_net())
    for name, p in by_name.items():
        if name in init:
            assert p.grad is not None, name
            assert torch.allclose(p.grad, 2.0 * (p.detach() - init[name]), atol=1e-6), name
        else:
            assert p.grad is None or float(p.grad.abs().max()) == 0.0, name          # the trunk is left free


def test_another_network_is_refused() -> None:
    net = _net()
    init = head_init_params(net)
    name = next(iter(init))
    with pytest.raises(ValueError, match="do not match"):
        head_pairs(net, {name: torch.zeros(3, 3)})


def test_theta_0_travels_in_the_resume_state(tmp_path) -> None:
    net = _net()
    trainer = _Trainer(net, 1e-3)
    theta_0 = {k: v.clone() for k, v in trainer.head_init.items()}
    _train(net, trainer.optimizer)
    path = os.path.join(tmp_path, RESUME_FILENAME)
    save_resume_state(path, net, trainer, iteration=1, total_env_steps=0, elapsed_seconds=0.0)
    other = _net(7)                                          # another process constructs another initialisation
    revived = _Trainer(other, 1e-3)
    assert not all(torch.equal(revived.head_init[k], theta_0[k]) for k in theta_0)
    load_resume_state(path, other, revived)
    assert all(torch.equal(revived.head_init[k], theta_0[k]) for k in theta_0)   # the run's own theta_0


def test_a_state_without_theta_0_is_refused(tmp_path) -> None:
    net = _net()
    plain = _Trainer(net, 0.0)
    path = os.path.join(tmp_path, RESUME_FILENAME)
    save_resume_state(path, net, plain, iteration=1, total_env_steps=0, elapsed_seconds=0.0)
    with pytest.raises(ValueError, match="theta_0 is not known"):
        load_resume_state(path, _net(), _Trainer(_net(), 1e-3))


def test_a_fresh_optimiser_keeps_the_weights_and_drops_the_moments(tmp_path) -> None:
    net = _net()
    trainer = _Trainer(net, 0.0)
    _train(net, trainer.optimizer)
    assert len(trainer.optimizer.state) > 0
    path = os.path.join(tmp_path, RESUME_FILENAME)
    save_resume_state(path, net, trainer, iteration=1, total_env_steps=0, elapsed_seconds=0.0)
    kept, fresh = _net(3), _net(3)
    t_kept, t_fresh = _Trainer(kept, 0.0), _Trainer(fresh, 0.0)
    load_resume_state(path, kept, t_kept)
    load_resume_state(path, fresh, t_fresh, fresh_optimizer=True)
    assert all(torch.equal(a, b) for a, b in zip(fresh.parameters(), net.parameters()))   # the weights as saved
    assert len(t_kept.optimizer.state) > 0                                              # a plain resume keeps them
    assert len(t_fresh.optimizer.state) == 0                                            # the fresh one starts anew
