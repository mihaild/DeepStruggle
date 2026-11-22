"""Every searcher and every wrapper around one reports its network's feature set, so the harness seats
its side with that network's rule bits (audit S-9.2: `entry:` / `sparse:` exposed no model, and an E8.1
network inside played its unsearched moves under E8's rules), and a wrapper forwards the game hooks a
belief inside needs. The rollout root's root mask is built under the rules its batch plays (S-9.1)."""

from __future__ import annotations

import random
from typing import Any, List

import numpy as np
import pytest

import ts_engine as ts
from ai.search.batched_mcts import BatchedMCTSAgent, BatchedMCTSConfig
from ai.search.critic_argmax import CriticArgmaxAgent
from ai.search.dmcts import determinize
from ai.search.rollout_root import (EntryRolloutAgent, RolloutConfig, RolloutRoot, RolloutRootAgent,
                                    SparseRolloutAgent)
from bindings.ts_env import agent_view
from tests.bindings.test_rollout_batch import _box4_headline
from tests.training.test_search_mixed_rules import EVENT, GATED, _nato_node, _net
from tools.lib.batch_tournament import BatchMatchRunner, _agent_features
from tools.lib.player_agent import RandomAgent

E81 = int(ts.OBS_FEATURE_HEADLINE_REVEAL) | GATED


def _root(belief: int = 0) -> RolloutRootAgent:
    return RolloutRootAgent(_net(E81), "root", RolloutConfig(k=2, worlds=2, horizon=0, seed=1, belief=belief))


def _agents() -> List[Any]:
    return [_root(), EntryRolloutAgent(_root()), SparseRolloutAgent(_root(), 0.5),
            BatchedMCTSAgent(_net(E81), "mcts", config=BatchedMCTSConfig(simulations=2)),
            CriticArgmaxAgent(_net(E81), "critic")]


@pytest.mark.parametrize("i", range(5))
def test_every_searcher_and_wrapper_reports_its_networks_features(i: int) -> None:
    agent = _agents()[i]
    assert agent_view(agent).features == E81
    assert _agent_features(agent) == E81


@pytest.mark.parametrize("wrap", ["entry", "sparse"])
def test_a_wrapped_e81_root_plays_every_move_under_its_rules_and_feeds_its_belief(wrap: str) -> None:
    """A match seats the wrapper's side with the inner network's features: every position it is asked
    about -- searched or not -- has its decider playing the gate. A belief inside sees the games."""
    inner = _root(belief=16)
    agent: Any = EntryRolloutAgent(inner) if wrap == "entry" else SparseRolloutAgent(inner, 0.5)
    seen: List[bool] = []
    forward = inner.root._forward

    def spy(states: Any) -> Any:
        for st in states:
            seen.append(bool(st.seat_plays_rule(st.ctx().decision_player, GATED)))
        return forward(states)

    inner.root._forward = spy   # type: ignore[method-assign]
    res = BatchMatchRunner.play_parallel_matchup(agent, RandomAgent(), games_per_side=1, device="cpu",
                                                 temperature=0.0)
    assert res["total_games"] == 2
    assert len(seen) > 20 and all(seen)          # before the fix: the runner seated feature set 0
    assert agent.searched > 0 and agent.decisions > agent.searched
    assert inner.belief is not None and inner.belief.stats["positions"] > 0


def test_the_root_mask_is_the_moves_legal_under_the_batchs_rules_and_the_callers() -> None:
    """A position saved with no rule bits, searched by a gated network: the root's candidates come from
    the moves legal under both seats, so the batch takes them (before: 'a candidate is not legal')."""
    st = _nato_node(0, 0)
    assert np.asarray(ts.ActionMask.generate_flat_mask(st))[EVENT]
    root = RolloutRoot(_net(GATED), RolloutConfig(k=4, worlds=2, horizon=0, seed=1))
    _lg, masks, _v = root._forward([st])
    assert not masks[0, EVENT] and masks[0].any()
    (pick,) = root.choose([st])
    assert pick != EVENT and np.asarray(ts.ActionMask.generate_flat_mask(st))[pick]
    assert not st.seat_plays_rule(ts.Player.USSR, GATED)     # the caller's position is not reseated


def test_the_python_tree_plays_its_networks_rules() -> None:
    """The Python backend seats both sides of its root copies, as the C++ one does (audit S-9.6)."""
    from_cpp = BatchedMCTSAgent(_net(GATED), "cpp", config=BatchedMCTSConfig(simulations=2)).mcts
    py = BatchedMCTSAgent(_net(GATED), "py", config=BatchedMCTSConfig(simulations=2, backend="python")).mcts
    st = _nato_node(0, 0)
    (node,) = py._search([st])
    assert node is not None and EVENT not in node.actions and node.actions
    (root,) = from_cpp._root_states([st])
    assert not np.asarray(ts.ActionMask.generate_flat_mask(root))[EVENT]


def test_select_action_takes_the_world_mismatch_fallback() -> None:
    """A determinized world's pick illegal at the real position (The Cambridge Five) is replaced by the
    greedy policy in a single match too, not raised (audit S-9.6)."""
    agent = BatchedMCTSAgent(_net(0), "mcts", config=BatchedMCTSConfig(simulations=2))
    st = _nato_node(0, 0)
    legal = np.asarray(ts.ActionMask.generate_flat_mask(st))
    illegal = int(np.flatnonzero(legal == 0)[0])
    agent.mcts.best_actions = lambda states: [illegal] * len(states)   # type: ignore[method-assign]
    a = agent.select_action(st, ts.Player.USSR)
    assert legal[a] and getattr(agent, "world_mismatch_count") == 1


def test_box4_worlds_keep_the_revealed_headline() -> None:
    """dmcts.determinize(keep_revealed=True), the Gumbel / PUCT searchers' box4: the opponent's headline
    revealed by Space Race box 4 stays chosen and in its hand, the rest redealt around it (audit S-9.5)."""
    st = _box4_headline(5)
    card = int(st.headline_ussr_card)
    unknown = ts.CardLocation.HAND_USSR_UNKNOWN
    n_hand = sum(st.get_card_location(c) == unknown for c in range(1, 111))
    rng = random.Random(0)
    hands = set()
    for _ in range(6):
        w = determinize(st, ts.Player.US, rng, keep_revealed=True)
        assert int(w.headline_ussr_card) == card and w.get_card_location(card) == unknown
        assert sum(w.get_card_location(c) == unknown for c in range(1, 111)) == n_hand
        hands.add(tuple(c for c in range(1, 111) if w.get_card_location(c) == unknown))
        assert int(determinize(st, ts.Player.US, rng).headline_ussr_card) == 0
    assert len(hands) > 1
