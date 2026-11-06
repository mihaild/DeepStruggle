"""A search plays both sides of its tree under its own network's rule bits (ENG-9's OPP_EVENT_GATED).

Its network models both players, so an old (pre-bit) and a new (gated) searcher can play each other:
each tree is consistent with the network evaluating it, whatever the real opponent's seat plays, and
the real position keeps each side's own rules.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
import ts_engine as ts

from ai.models.ladder_net import create_ladder_net
from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSAgent, BatchedMCTSConfig
from ai.search.pimcts import drain_chance_nodes
from ai.search.rollout_root import RolloutConfig, RolloutRootAgent
from bindings.ts_env import model_obs_features

NATO = 21
EVENT = 110              # flat slot of Resolution::EVENT
GATED = ts.OBS_FEATURE_OPP_EVENT_GATED

SHALLOW: Dict[str, Any] = dict(
    input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=64,
    card_self_attention=False, cross_attention=False, per_entity_heads=16, head_context=True,
    head_static=True, head_entities="country", head_center=True, identity_dim=0, drop_static=True,
    hidden_dim=64, num_res_blocks=0, num_attn_heads=4, card_lookup=False, card_lookup_heads=0,
    card_lookup_dim=0, card_lookup_identity_dim=0, categorical_value=False)


def _net(features: int) -> torch.nn.Module:
    torch.manual_seed(features)
    return create_ladder_net("cpu", **SHALLOW, obs_features=features).eval()


def _nato_node(us: int, ussr: int) -> ts.GameState:
    """The USSR choosing how to play NATO before either prerequisite, the seats as given."""
    s = ts.GameState()
    ts.Engine.init_game(s, 42)
    s.set_card_location(NATO, ts.hand_of(ts.Player.USSR))
    s.current_phase = ts.Phase.ACTION_ROUND
    s.action_round = 1
    s.phasing_player = ts.Player.USSR
    s.ctx().decision_player = ts.Player.USSR
    s.ctx().decision_type = ts.DecisionType.SELECT_PLAY_MODE
    s.ctx().pending_op_card = NATO
    s.seat_obs_features(ts.Player.US, us)
    s.seat_obs_features(ts.Player.USSR, ussr)
    return s


@pytest.mark.parametrize("features, seats, offered", [(GATED, 0, 0), (0, GATED, 1)])
def test_the_cpp_tree_plays_its_networks_rules(features: int, seats: int, offered: int) -> None:
    cs = ts.BatchedSearch(1, obs_features=features)
    cs.reset([_nato_node(seats, seats)], 4, list(random.Random(0).getstate()[1]))
    assert cs.select_leaves(0) == 1
    assert int(np.asarray(cs.masks())[0, EVENT]) == offered


def test_the_rollout_root_plays_its_networks_rules() -> None:
    with pytest.raises(ValueError, match="not legal"):
        ts.RolloutBatch(features=GATED).add(_nato_node(0, 0), [EVENT], 1, None)
    assert ts.RolloutBatch(features=0).add(_nato_node(GATED, GATED), [EVENT], 1, None) > 0


def test_the_python_tree_seats_its_copies_not_the_position() -> None:
    real = _nato_node(GATED, 0)
    mcts = BatchedMCTS(_net(GATED), BatchedMCTSConfig(simulations=2, backend="python"))
    (root,) = mcts._root_states([real])
    assert root.seat_plays_rule(ts.Player.US, GATED) and root.seat_plays_rule(ts.Player.USSR, GATED)
    assert not real.seat_plays_rule(ts.Player.USSR, GATED)


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_an_old_and_a_new_searcher_play_each_other(backend: str) -> None:
    """Every searched move is legal for its real seat, and each seat keeps its own rules."""
    cfg = BatchedMCTSConfig(simulations=4, backend=backend)
    agents: List[Any] = [BatchedMCTSAgent(_net(0), "old", config=cfg),
                         BatchedMCTSAgent(_net(GATED), "new", config=cfg)]
    rollout = [RolloutRootAgent(_net(0), "old", RolloutConfig(k=2, worlds=2, horizon=1, seed=1)),
               RolloutRootAgent(_net(GATED), "new", RolloutConfig(k=2, worlds=2, horizon=1, seed=1))]
    for seat_of_old, searchers in ((ts.Player.USSR, agents), (ts.Player.US, agents), (ts.Player.USSR, rollout)):
        old, new = searchers
        state = _nato_node(0 if seat_of_old == ts.Player.US else GATED, 0 if seat_of_old == ts.Player.USSR else GATED)
        for _ in range(30):
            drain_chance_nodes(state)
            if ts.Engine.is_terminal(state):
                break
            p = state.ctx().decision_player
            agent = old if p == seat_of_old else new
            a = agent.select_action(state, p)
            assert ts.Engine.try_step_flat(state, a), f"{agent.name} chose {a}, illegal for its seat"
        old_side = seat_of_old
        new_side = ts.Player.US if seat_of_old == ts.Player.USSR else ts.Player.USSR
        assert model_obs_features(new.model) & GATED
        assert not state.seat_plays_rule(old_side, GATED)
        assert state.seat_plays_rule(new_side, GATED)
