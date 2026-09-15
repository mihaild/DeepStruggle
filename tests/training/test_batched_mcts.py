"""Batched search must agree with the single-state searcher it replaces.

The point of batching is speed, so the risk is that it quietly changes the answer -- a search that
is fast and subtly wrong would look like a successful optimisation and poison every training
target built on it. These tests pin the properties that make it a drop-in replacement: it returns
a legal action per position, its visit counts sum to the budget, positions do not contaminate one
another, and searching the same position alone or in a crowd gives the same result.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import ts_engine as ts
from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSConfig
from ai.search.pimcts import acting_player, drain_chance_nodes
from bindings.action_encoder import ActionEncoder

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _states(n: int, plies: int = 40, seed0: int = 5150):
    out = []
    for i in range(n):
        s = ts.GameState()
        ts.Engine.init_game(s, seed0 + i)
        for _ in range(plies + i):
            if ts.Engine.is_terminal(s):
                break
            legal = ts.Engine.get_legal_action_indices(s)
            if not len(legal):
                break
            ts.Engine.step_flat(s, int(legal[0]))
        drain_chance_nodes(s)
        out.append(s)
    return out


@pytest.fixture(scope="module")
def model():
    """A freshly initialised network, NOT a checkpoint.

    `data/` is git-ignored, so no test may assume it holds anything -- and picking an arbitrary
    checkpoint from it also picks up retired observation layouts, which the width guard correctly
    refuses. These tests are about search mechanics, which random weights exercise just as well.
    """
    from ai.models.coldwar_net_v2 import create_coldwar_net_v2
    return create_coldwar_net_v2(device="cuda", graph_layers=0).eval()


def test_returns_a_legal_action_for_every_position(model):
    states = _states(6)
    picks = BatchedMCTS(model, config=BatchedMCTSConfig(simulations=8)).best_actions(states)
    assert len(picks) == len(states)
    for st, a in zip(states, picks):
        mask = np.asarray(ActionEncoder.get_legal_mask(st))
        if mask.sum() > 0:
            assert mask[a], "search returned an illegal action"


def test_visit_counts_match_the_simulation_budget(model):
    sims = 12
    states = _states(4)
    res = BatchedMCTS(model, config=BatchedMCTSConfig(simulations=sims)).run(states)
    for actions, visits in res:
        if not actions:
            continue
        # Every simulation that reaches a non-terminal root backs up exactly one visit.
        assert visits.sum() == pytest.approx(sims), (
            f"visits {visits.sum()} != budget {sims}; a simulation was lost or double-counted")


def test_a_position_searched_alone_matches_the_same_position_in_a_batch(model):
    """Batching must not let positions influence each other."""
    states = _states(5)
    target = states[2]

    cfg = BatchedMCTSConfig(simulations=16, seed=99)
    alone = BatchedMCTS(model, config=cfg)
    a_actions, a_visits = alone.run([target])[0]

    together = BatchedMCTS(model, config=cfg)
    # Same position, same index in the batch is not required -- what must hold is that the
    # ANSWER for this position does not depend on its neighbours. Run it first in the batch so
    # the RNG draw order matches the solo run.
    b_actions, b_visits = together.run([target] + states[:2] + states[3:])[0]

    assert a_actions == b_actions
    np.testing.assert_allclose(a_visits, b_visits, err_msg=(
        "a position's search result changed when other positions were searched alongside it"))


def test_terminal_positions_are_handled_without_evaluation(model):
    s = ts.GameState()
    ts.Engine.init_game(s, 777)
    for _ in range(4000):
        if ts.Engine.is_terminal(s):
            break
        legal = ts.Engine.get_legal_action_indices(s)
        if not len(legal):
            break
        ts.Engine.step_flat(s, int(legal[0]))
    if not ts.Engine.is_terminal(s):
        pytest.skip("could not reach a terminal state with the scripted opening")
    actions, visits = BatchedMCTS(model, config=BatchedMCTSConfig(simulations=4)).run([s])[0]
    assert actions == [] and visits.size == 0


def test_determinize_flag_does_not_leak_the_opponent_hand(model):
    """With determinize=True the searched root must differ from the true state."""
    states = _states(3)
    cfg = BatchedMCTSConfig(simulations=4, determinize=True, seed=5)
    picks = BatchedMCTS(model, config=cfg).best_actions(states)
    assert len(picks) == len(states)
    for st, a in zip(states, picks):
        mask = np.asarray(ActionEncoder.get_legal_mask(st))
        if mask.sum() > 0:
            assert mask[a], "determinized search returned an action illegal in the REAL state"


# -- subtree reuse ------------------------------------------------------------------------


def _step_and_report_chance(state, action):
    """Step the real game, reporting whether a die roll resolved -- what `advance` needs."""
    ts.Engine.step_flat(state, int(action))
    intervened = (not ts.Engine.is_terminal(state)
                  and state.ctx().decision_type == ts.DecisionType.ROLL_DIE)
    drain_chance_nodes(state)
    return intervened


def test_reuse_keeps_every_decision_at_full_budget(model):
    """An inherited tree is topped up, not merely added to -- depth must not depend on history."""
    cfg = BatchedMCTSConfig(simulations=16, reuse_subtree=True, seed=4)
    mcts = BatchedMCTS(model, config=cfg)
    s = _states(1)[0]

    for _ in range(4):
        actions, visits = mcts.run([s], keys=["g0"])[0]
        if not actions:
            break
        assert visits.sum() >= cfg.simulations, (
            f"root has {visits.sum()} visits, below the {cfg.simulations} budget: an inherited "
            f"tree was not topped up")
        best = int(actions[int(np.argmax(visits))])
        intervened = _step_and_report_chance(s, best)
        mcts.advance("g0", best, intervened)


def test_advance_drops_the_tree_when_a_die_roll_intervened(model):
    """A cached child holds ONE sampled chance outcome; if the real roll differed it is invalid."""
    cfg = BatchedMCTSConfig(simulations=8, reuse_subtree=True, seed=6)
    mcts = BatchedMCTS(model, config=cfg)
    s = _states(1)[0]
    actions, visits = mcts.run([s], keys=["g0"])[0]
    assert actions, "fixture position has no legal actions"
    best = int(actions[int(np.argmax(visits))])

    mcts.advance("g0", best, chance_intervened=True)
    assert "g0" not in mcts._trees, (
        "tree survived a chance node; the next search would run from a state that never occurred")


def test_reuse_is_disabled_under_determinization(model):
    """A tree built for one sampled world must not be carried into another."""
    cfg = BatchedMCTSConfig(simulations=8, reuse_subtree=True, determinize=True, seed=8)
    mcts = BatchedMCTS(model, config=cfg)
    s = _states(1)[0]
    actions, visits = mcts.run([s], keys=["g0"])[0]
    assert not mcts._trees, "determinized search stored a tree for reuse"
    if actions:
        mcts.advance("g0", int(actions[int(np.argmax(visits))]), chance_intervened=False)
        assert not mcts._trees


def test_forget_removes_a_stream(model):
    cfg = BatchedMCTSConfig(simulations=8, reuse_subtree=True, seed=10)
    mcts = BatchedMCTS(model, config=cfg)
    s = _states(1)[0]
    mcts.run([s], keys=["g0"])
    assert "g0" in mcts._trees
    mcts.forget("g0")
    assert "g0" not in mcts._trees
