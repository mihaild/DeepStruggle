"""A pool must reject incompatible observations before inference, even at equal widths."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import ts_engine as ts

from ai.models.ladder_net import create_ladder_net
from ai.training.nash_pg import NashPGTrainer
from ai.training.opponent_pool import OpponentPool, load_pool
from bindings.ts_env import TsVectorizedEnv
from tests.training.test_ladder_tokens import SHALLOW


@pytest.mark.parametrize("features", [int(ts.OBS_FEATURE_HEADLINE_REVEAL), int(ts.OBS_FEATURE_OPS_BUDGET)])
def test_loaded_opponents_must_match_the_learner(tmp_path: Path, features: int) -> None:
    opponent = create_ladder_net("cpu", **SHALLOW, obs_features=0)
    path = tmp_path / "opponent.pt"
    torch.save(opponent.state_dict(), path)
    nets = load_pool([str(path)], torch.device("cpu"))
    with pytest.raises(ValueError, match="observation features"):
        OpponentPool(nets, 1, obs_features=features)


@pytest.mark.parametrize("group", ["self", "league"])
def test_added_members_are_checked_before_being_frozen(group: str) -> None:
    features = int(ts.OBS_FEATURE_HEADLINE_REVEAL)
    seed = create_ladder_net("cpu", **SHALLOW, obs_features=features)
    pool = OpponentPool([seed], 1, league_capacity=2, obs_features=features)
    wrong = create_ladder_net("cpu", **SHALLOW, obs_features=0)
    with pytest.raises(ValueError, match="observation features"):
        pool.add(wrong, 10, group=group)
    assert len(pool.nets) == 1
    assert wrong.training and all(p.requires_grad for p in wrong.parameters())
    pool.add(create_ladder_net("cpu", **SHALLOW, obs_features=features), 10, group=group)
    assert len(pool.nets) == 2


def test_mixed_seed_features_are_rejected() -> None:
    nets = [create_ladder_net("cpu", **SHALLOW, obs_features=f)
            for f in (0, int(ts.OBS_FEATURE_HEADLINE_REVEAL))]
    with pytest.raises(ValueError, match="observation features"):
        OpponentPool(nets, 1)


def test_direct_trainer_pool_assignment_is_checked_before_rollout() -> None:
    net = create_ladder_net("cpu", **SHALLOW, obs_features=int(ts.OBS_FEATURE_HEADLINE_REVEAL))
    env = TsVectorizedEnv(num_envs=1, base_seed=1)
    trainer = NashPGTrainer(net, env=env, num_envs=1, buffer_size=2, batch_size=2, device="cpu")
    trainer.opponent_pool = OpponentPool([create_ladder_net("cpu", **SHALLOW)], 1)
    before = env.runner.get_state(0).raw_bytes()
    with pytest.raises(ValueError, match="observation features"):
        trainer.collect_rollouts()
    assert env.runner.get_state(0).raw_bytes() == before
