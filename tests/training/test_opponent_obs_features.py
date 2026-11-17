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


def test_a_fixed_opponent_may_differ_in_rule_bits_only_and_keeps_its_own_seat() -> None:
    reveal, gated = int(ts.OBS_FEATURE_HEADLINE_REVEAL), int(ts.OBS_FEATURE_OPP_EVENT_GATED)
    learner, opponent = reveal | gated, reveal
    pool = OpponentPool([create_ladder_net("cpu", **SHALLOW, obs_features=opponent)], 4, frac=1.0,
                        lock_learner_side=1, obs_features=learner, opponent_obs_features=opponent)
    us, ussr = pool.seat_features()
    assert us.tolist() == [learner] * 4 and ussr.tolist() == [opponent] * 4
    # A self-grown member would have the learner's bits, which this pool does not hold.
    with pytest.raises(ValueError, match="observation features"):
        pool.add(create_ladder_net("cpu", **SHALLOW, obs_features=learner), 10)


def test_rule_bit_opponents_need_a_locked_side_in_every_environment() -> None:
    reveal, gated = int(ts.OBS_FEATURE_HEADLINE_REVEAL), int(ts.OBS_FEATURE_OPP_EVENT_GATED)
    nets = [create_ladder_net("cpu", **SHALLOW, obs_features=reveal)]
    with pytest.raises(ValueError, match="locked"):
        OpponentPool(nets, 4, frac=1.0, obs_features=reveal | gated, opponent_obs_features=reveal)
    with pytest.raises(ValueError, match="locked"):
        OpponentPool(nets, 4, frac=0.5, lock_learner_side=1, obs_features=reveal | gated, opponent_obs_features=reveal)


def test_an_opponent_differing_in_observation_bits_is_still_refused() -> None:
    reveal = int(ts.OBS_FEATURE_HEADLINE_REVEAL)
    with pytest.raises(ValueError, match="observation bits"):
        OpponentPool([create_ladder_net("cpu", **SHALLOW, obs_features=0)], 4, frac=1.0, lock_learner_side=1,
                     obs_features=reveal, opponent_obs_features=0)
