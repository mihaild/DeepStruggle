"""The rules-derived evaluation and the search that uses it.

The sign convention is the thing most worth pinning. Every value in `heuristic_eval`, `pimcts` and
`heuristic_mcts` is from the US perspective, and this action space invites perspective errors
because the mover can change partway through a turn -- a flipped sign would produce a bot that
plays competently for the wrong side, which looks like weakness rather than like a bug.
"""

from __future__ import annotations

import numpy as np
import pytest

import ts_engine as ts
from ai.search.heuristic_eval import HeuristicWeights, evaluate, explain, region_advantage
from ai.search.heuristic_mcts import HeuristicLeaf, HeuristicMCTSConfig, make_heuristic_mcts_agent
from ai.search.pimcts import PIMCTSAgent
from bot.heuristic_mcts_bot import HeuristicMCTSBot


def _fresh() -> ts.GameState:
    state = ts.GameState()
    ts.Engine.init_game(state, 42)
    return state


class TestEvaluation:
    def test_value_is_bounded_and_us_positive(self) -> None:
        state = _fresh()
        state.victory_points = 15
        v_us = evaluate(state)
        state.victory_points = -15
        v_ussr = evaluate(state)
        assert -1.0 < v_ussr < 0.0 < v_us < 1.0, "US-favourable VP must read positive"

    def test_victory_points_dominate(self) -> None:
        """VP is the win condition; no proxy term may outweigh a decisive VP lead."""
        state = _fresh()
        state.victory_points = 19
        assert evaluate(state) > 0.5

    def test_terminal_returns_exact_utility(self) -> None:
        state = _fresh()
        state.victory_points = 20
        state.current_phase = ts.Phase.GAME_OVER
        assert ts.Engine.is_terminal(state)
        assert evaluate(state) == pytest.approx(float(ts.Engine.get_terminal_utility(state)))

    def test_region_advantage_is_signed_us_positive(self) -> None:
        """`net_delta` is us_score - ussr_score, so the sum inherits that orientation."""
        state = _fresh()
        before = region_advantage(state)
        # Hand the US a battleground it did not have.
        state.set_country(7, 9, 0)      # West Germany, US 9 / USSR 0
        assert region_advantage(state) > before

    def test_defcon_risk_falls_on_the_phasing_player(self) -> None:
        """DEFCON 1 loses the game for whoever causes it, so the risk is not symmetric."""
        state = _fresh()
        state.defcon = 2
        state.phasing_player = ts.Player.US
        us_to_move = explain(state)["defcon_risk"]
        state.phasing_player = ts.Player.USSR
        ussr_to_move = explain(state)["defcon_risk"]
        assert us_to_move < 0.0 < ussr_to_move
        assert us_to_move == pytest.approx(-ussr_to_move)

    def test_no_risk_term_above_defcon_3(self) -> None:
        state = _fresh()
        state.defcon = 4
        state.phasing_player = ts.Player.US
        assert explain(state)["defcon_risk"] == 0.0

    def test_weights_are_not_shared_between_configs(self) -> None:
        """A tuning arm must not mutate every other agent's weights."""
        a, b = HeuristicMCTSConfig(), HeuristicMCTSConfig()
        assert a.weights is not b.weights


class TestLeaf:
    def test_priors_are_a_distribution_over_legal_actions(self) -> None:
        state = _fresh()
        leaf = HeuristicLeaf(HeuristicMCTSConfig())
        legal = np.flatnonzero(np.asarray(ts.Engine.get_flat_action_mask(state)))
        priors, value = leaf(state, legal)
        assert len(priors) == len(legal)
        assert priors.sum() == pytest.approx(1.0)
        assert (priors >= 0).all()
        assert -1.0 <= value <= 1.0

    def test_uniform_prior_when_branching_exceeds_the_cap(self) -> None:
        """The one-ply prior costs an engine step per action, so it is capped rather than
        allowed to dominate the node cost."""
        state = _fresh()
        cfg = HeuristicMCTSConfig(prior_lookahead_max_actions=0)
        legal = np.flatnonzero(np.asarray(ts.Engine.get_flat_action_mask(state)))
        priors, _ = HeuristicLeaf(cfg)(state, legal)
        assert priors == pytest.approx(np.full(len(legal), 1.0 / len(legal)))


class TestAgent:
    def test_search_needs_a_model_or_a_leaf(self) -> None:
        with pytest.raises(ValueError):
            PIMCTSAgent(model=None)

    def test_agent_runs_with_no_network(self) -> None:
        agent = make_heuristic_mcts_agent(config=HeuristicMCTSConfig(simulations=8))
        state = _fresh()
        actions, visits = agent.search(state)
        assert actions and visits.sum() > 0

    def test_selected_action_is_always_legal(self) -> None:
        """The whole game, not one node: an illegal action deadlocks the match loop."""
        bot = HeuristicMCTSBot("US", simulations=4)
        state = _fresh()
        for _ in range(120):
            if ts.Engine.is_terminal(state):
                break
            mask = np.asarray(ts.Engine.get_flat_action_mask(state))
            if not mask.any():
                break
            player = state.ctx().decision_player
            if player == ts.Player.NONE:
                player = state.phasing_player
            flat = bot.select_flat_action(state, player)
            assert mask[flat], f"chose illegal flat {flat}"
            ts.Engine.step_flat(state, flat)

    def test_dict_protocol_refuses_rather_than_guessing(self) -> None:
        """A bot that quietly played an unsearched move would make every row quoting it wrong."""
        bot = HeuristicMCTSBot("US", simulations=4)
        with pytest.raises(NotImplementedError):
            bot.select_action({}, {})
