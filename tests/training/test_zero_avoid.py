"""P41 J1, the zero-avoiding term: KL(uniform || pi) over the legal moves. Zero at the uniform policy and at a forced
move, unbounded as a legal move's probability goes to zero, blind to masked moves, read on the selected rows only, and
its gradient lifts the starved move; a trainer with the coefficient logs it, and without it logs nothing."""

import math

import pytest
import torch

from ai.models.ladder_net import create_ladder_net
from ai.training import NashPGTrainer
from ai.training.nash_pg import zero_avoid_kl
from bindings.ts_env import TsVectorizedEnv

CFG = dict(input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=64,
           card_self_attention=False, cross_attention=False, per_entity_heads=16,
           head_context=True, head_static=True, head_entities="country", head_center=True,
           identity_dim=0, drop_static=True, hidden_dim=64, num_res_blocks=0, num_attn_heads=4,
           card_lookup=False, card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0,
           categorical_value=False)
ALL = torch.ones(1)
ZERO = pytest.approx(0.0, abs=1e-6)


def _row(values: list, legal: list) -> tuple:
    logits = torch.full((1, len(values)), -1e9)
    for i, v in enumerate(values):
        if legal[i]:
            logits[0, i] = v
    return logits, torch.tensor([legal], dtype=torch.uint8)


def test_zero_at_the_uniform_policy_and_at_a_forced_move() -> None:
    logits, mask = _row([2.0, 2.0, 2.0, 0.0], [1, 1, 1, 0])
    assert float(zero_avoid_kl(logits, mask, ALL)) == ZERO
    logits, mask = _row([5.0, 0.0, 0.0], [1, 0, 0])
    assert float(zero_avoid_kl(logits, mask, ALL)) == ZERO


def test_it_is_the_kl_and_grows_without_bound_as_a_legal_move_starves() -> None:
    # Two legal moves, logit gap g: pi = (s, 1 - s), KL = -0.5 (log s + log(1 - s)) - log 2.
    last = 0.0
    for g in (1.0, 5.0, 10.0, 20.0, 30.0):
        logits, mask = _row([g, 0.0], [1, 1])
        got = float(zero_avoid_kl(logits, mask, ALL))
        log_small = -g - math.log1p(math.exp(-g))
        log_big = -math.log1p(math.exp(-g))
        assert abs(got - (-0.5 * (log_small + log_big) - math.log(2.0))) < 1e-4
        assert got > last
        last = got
    assert last > 10.0                                    # a 30-nat gap costs ~14: unbounded, unlike entropy


def test_masked_moves_do_not_count_and_only_the_selected_rows_do() -> None:
    a, ma = _row([3.0, 0.0, 99.0], [1, 1, 0])
    b, mb = _row([3.0, 0.0, -50.0], [1, 1, 0])
    assert float(zero_avoid_kl(a, ma, ALL)) == float(zero_avoid_kl(b, mb, ALL))
    uniform, mu = _row([0.0, 0.0, 0.0], [1, 1, 0])
    logits = torch.cat([a, uniform])
    masks = torch.cat([ma, mu])
    assert float(zero_avoid_kl(logits, masks, torch.tensor([1.0, 0.0]))) == float(zero_avoid_kl(a, ma, ALL))
    assert float(zero_avoid_kl(logits, masks, torch.tensor([0.0, 1.0]))) == ZERO


def test_its_gradient_lifts_the_starved_move() -> None:
    logits, mask = _row([10.0, 0.0], [1, 1])
    logits.requires_grad_(True)
    zero_avoid_kl(logits, mask, ALL).backward()
    assert logits.grad is not None
    assert float(logits.grad[0, 1]) < 0.0 < float(logits.grad[0, 0])     # descent raises the low logit, lowers the top


def _trainer(coef: float) -> NashPGTrainer:
    torch.manual_seed(0)
    dev = torch.device("cpu")
    return NashPGTrainer(active_net=create_ladder_net(dev, **CFG), env=TsVectorizedEnv(num_envs=2, base_seed=11),
                         num_envs=2, buffer_size=32, lr=3e-4, eta=0.1, ref_update_freq=500, cuda_graphs=False,
                         device=dev, zero_avoid_coef=coef)


def test_the_trainer_logs_it_when_on_and_nothing_when_off() -> None:
    on = _trainer(0.01)
    on.collect_rollouts()
    assert on.train_step()["zero_avoid_kl"] > 0.0
    off = _trainer(0.0)
    off.collect_rollouts()
    assert off.train_step()["zero_avoid_kl"] == 0.0
