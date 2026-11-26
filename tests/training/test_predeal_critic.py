"""P33's pre-deal critic (--predeal-value-coef): the turn-end park's rows, both sides, each with its own
next decision's lambda-return as the target, or the result where the game ends first."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch

from ai.models.ladder_net import create_ladder_net
from ai.training.nash_pg import NashPGTrainer, PredealRows
from bindings.ts_env import TsVectorizedEnv


def test_each_side_gets_its_next_decisions_return_or_the_result() -> None:
    T, E, W = 6, 3, 4
    players = torch.tensor([[1, 1, 1], [1, 1, 1], [-1, -1, -1], [-1, -1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.int8)
    dones = torch.zeros(T, E)
    dones[2, 1] = 1.0                                   # env 1: the USSR's move at step 2 ends the game...
    rewards = torch.zeros(T, E)
    rewards[2, 1] = 1.0                                 # ...and wins it (reward in the mover's frame)
    rets = torch.arange(T * E, dtype=torch.float32).reshape(T, E) / 100.0
    learner = torch.ones(T, E)
    learner[3, 2] = 0.0                                 # env 2: the US's next decision is a pool snapshot's
    buf = SimpleNamespace(players=players, dones=dones, rewards=rewards, returns_win=rets, learner=learner,
                          buffer_size=T)
    us_row, ussr_row = torch.ones(3, W), -torch.ones(3, W)
    fake = SimpleNamespace(buffer=buf, device=torch.device("cpu"),
                           _predeal_rows=[PredealRows(1, np.array([0, 1, 2]), us_row, ussr_row)], _pd_obs=None, _pd_tgt=None,
                           _learner_seat_rows=None)
    NashPGTrainer._build_predeal_targets(cast(Any, fake))
    got = sorted(zip(fake._pd_obs[:, 0].tolist(), fake._pd_tgt.tolist()))
    want = sorted([
        (1.0, float(rets[4, 0])), (-1.0, float(rets[2, 0])),     # env 0: US at step 4, USSR at step 2
        (1.0, -1.0), (-1.0, float(rets[2, 1])),                   # env 1: US never moves again (lost); USSR at 2
        (-1.0, float(rets[2, 2])),                                # env 2: USSR at 2; US's next is the pool's: dropped
    ])
    assert len(got) == len(want) and all(math.isclose(a[1], b[1]) and a[0] == b[0] for a, b in zip(got, want))
    assert fake._predeal_rows == []


def test_a_rollout_trains_the_critic_on_the_park_rows() -> None:
    torch.manual_seed(0)
    dev = torch.device("cpu")
    net = create_ladder_net(dev, input_mode="grouped", aggregation="flatten", entity_dim=16,
                            entity_proj_dim=64, card_self_attention=False, cross_attention=False,
                            per_entity_heads=16, head_context=True, head_static=True,
                            head_entities="country", head_center=True, identity_dim=0,
                            drop_static=True, hidden_dim=64, num_res_blocks=0, num_attn_heads=4,
                            card_lookup=False, card_lookup_heads=0, card_lookup_dim=0,
                            card_lookup_identity_dim=0, categorical_value=False)
    t = NashPGTrainer(active_net=net, env=TsVectorizedEnv(num_envs=4, base_seed=7), num_envs=4,
                      buffer_size=96, lr=3e-4, eta=0.1, ref_update_freq=500, cuda_graphs=False,
                      device=dev, predeal_value_coef=1.0)
    t.collect_rollouts()
    assert t._predeal_rows                              # every env crossed turn 1's end in 96 decisions
    m = t.train_step()
    assert m["predeal_rows"] > 0 and math.isfinite(m["predeal_value_loss"])
