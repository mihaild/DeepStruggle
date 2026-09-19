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
from ai.search.heuristic_eval import (SCORING_CARD_FOR_REGION, HeuristicWeights,
                                      battleground_counts, evaluate, explain, region_advantage)
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
        state.set_country(7, 9, 0)      # West Germany, US 9 / USSR 0
        assert region_advantage(state) > before

    def test_a_region_is_worth_less_once_its_scoring_card_is_discarded(self) -> None:
        """The point of the liveness weighting: standing in a region nobody can score for is
        worth keeping but not worth paying for."""
        state = _fresh()
        # Enough of Europe to put the US clearly ahead there; one battleground only ties it.
        for cid in (7, 8, 10, 14, 15):
            state.set_country(cid, 9, 0)
        europe = SCORING_CARD_FOR_REGION[int(ts.Region.EUROPE)]
        state.set_card_location(europe, ts.CardLocation.DRAW_DECK)
        live = region_advantage(state)
        state.set_card_location(europe, ts.CardLocation.DISCARD_PILE)
        spent = region_advantage(state)

        # Asserted against Europe's own contribution rather than the board total, which is
        # USSR-positive at setup and would hide the effect.
        europe_delta = float(ts.Scoring.evaluate_region(state, ts.Region.EUROPE).net_delta)
        assert europe_delta > 0, "the fixture should leave the US ahead in Europe"
        assert spent < live, "a discarded scoring card must reduce the value of standing there"
        assert live - spent == pytest.approx(0.5 * europe_delta), (
            "and reduce it by exactly the discarded weighting, not erase it")

    def test_controlled_battlegrounds_are_counted_net(self) -> None:
        state = _fresh()
        ctrl_before, _ = battleground_counts(state)
        state.set_country(7, 9, 0)                       # West Germany: stability 4, US controls
        ctrl_after, _ = battleground_counts(state)
        assert ctrl_after == ctrl_before + 1

    def test_access_counts_a_neighbour_not_just_the_square(self) -> None:
        """Access is where the next Operation can reach, which includes adjacency."""
        state = ts.GameState()
        ts.Engine.init_game(state, 7)
        for cid in range(84):
            state.set_country(cid, 0, 0)
        _, acc_empty = battleground_counts(state)
        # France (8) neighbours West Germany (7), a battleground.
        state.set_country(8, 1, 0)
        _, acc_adj = battleground_counts(state)
        assert acc_adj > acc_empty

    def test_holding_a_scoring_card_is_penalised_and_bites_harder_late(self) -> None:
        """Holding one at turn end loses outright, so the term is a countdown, not a preference."""
        state = _fresh()
        state.turn = 5
        asia = SCORING_CARD_FOR_REGION[int(ts.Region.ASIA)]
        state.set_card_location(asia, ts.hand_of(ts.Player.US))
        state.action_round = 1
        early = explain(state)["held_scoring"]
        state.action_round = 7
        late = explain(state)["held_scoring"]
        assert early < 0 and late < 0, "a held scoring card is bad for its holder"
        assert late < early, "and worse the closer the turn is to ending"

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
