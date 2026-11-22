"""Audit M-7: under --update-bf16 the PPO ratio's old log-probs are taken once, before the epochs, from the update's
own forward of the not-yet-updated network on the learner's rows -- not from the fp32 rollout. bf16 autocast runs only
on CUDA, so on the CPU the recomputed log-probs are the rollout's own: a stored learner log-prob shifted after the
rollout is ignored with the flag and changes the update without it."""

from __future__ import annotations

import copy
from typing import Any, Dict

import numpy as np
import torch

from ai.models.ladder_net import create_ladder_net
from ai.training import NashPGTrainer
from bindings.ts_env import TsVectorizedEnv
from tests.training.test_token_main import CFG


def _step(update_bf16: bool, shift: float) -> Dict[str, Any]:
    torch.manual_seed(0)
    dev = torch.device("cpu")
    t: Any = NashPGTrainer(active_net=create_ladder_net(dev, **CFG), env=TsVectorizedEnv(num_envs=4, base_seed=11),
                           num_envs=4, buffer_size=16, batch_size=32, lr=3e-4, eta=0.1, ref_update_freq=500,
                           cuda_graphs=False, device=dev, update_bf16=update_bf16)
    t.collect_rollouts()
    t.buffer.log_probs.add_(shift * (t.buffer.learner > 0.5).float())
    torch.manual_seed(1)
    m = t.train_step()
    return {"m": m, "w": copy.deepcopy(t.active_net.state_dict())}


def test_the_learners_old_log_probs_come_from_the_update_forward_under_the_flag() -> None:
    clean, shifted = _step(True, 0.0), _step(True, 0.3)
    for k, v in clean["w"].items():
        assert torch.allclose(v, shifted["w"][k], atol=1e-6), k
    plain_shifted = _step(False, 0.3)
    assert np.isfinite(plain_shifted["m"]["loss"])
    assert any(not torch.allclose(v, plain_shifted["w"][k], atol=1e-6) for k, v in clean["w"].items())
