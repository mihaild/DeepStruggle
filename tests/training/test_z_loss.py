"""z-loss (--z-loss-coef): off leaves the update bit for bit as it was; on, it pulls the policy logits'
log-normaliser toward 0, and the level is logged either way."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from ai.models import create_coldwar_net
from ai.training import NashPGTrainer


def _trainer(**kw: Any) -> NashPGTrainer:
    from bindings.ts_env import TsVectorizedEnv
    torch.manual_seed(0)
    env = TsVectorizedEnv(num_envs=8, base_seed=123)
    return NashPGTrainer(active_net=create_coldwar_net(torch.device("cpu")), env=env, num_envs=8,
                         buffer_size=16, batch_size=64, cuda_graphs=False, device="cpu", **kw)


def test_off_is_bitwise_the_update_and_still_logs_the_level() -> None:
    torch.manual_seed(1)
    a = _trainer()
    ma = a.train_iteration()
    torch.manual_seed(1)
    b = _trainer(z_loss_coef=0.0)
    mb = b.train_iteration()
    for pa, pb in zip(a.active_net.parameters(), b.active_net.parameters()):
        assert torch.equal(pa, pb)
    assert ma["logit_lse_mean"] == mb["logit_lse_mean"] and ma["z_loss"] == 0.0
    assert ma["logit_lse_absmax"] >= abs(ma["logit_lse_mean"])


def test_on_pulls_the_log_normaliser_toward_zero() -> None:
    t = _trainer(z_loss_coef=1.0)
    t.collect_rollouts()
    first = t.train_step()
    t.collect_rollouts()
    second = t.train_step()
    assert first["z_loss"] > 0.0
    assert abs(second["logit_lse_mean"]) < abs(first["logit_lse_mean"])


def test_a_negative_coefficient_is_refused() -> None:
    with pytest.raises(ValueError):
        _trainer(z_loss_coef=-1e-4)


def test_the_cli_offers_it_off_by_default() -> None:
    from ai.training.train import build_parser
    assert build_parser().parse_args([]).z_loss_coef == 0.0
