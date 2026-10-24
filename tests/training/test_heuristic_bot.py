"""The terminal bot reads the name-keyed display state supplied by the match runner."""
import ts_engine as ts

from bot.heuristic_bot import HeuristicBot


def test_influence_scoring_uses_the_displayed_influence() -> None:
    state = ts.GameState()
    ts.Engine.init_game(state, 1)
    state.current_phase = ts.Phase.ACTION_ROUND
    state.ctx().decision_player = ts.Player.US
    state.ctx().decision_type = ts.DecisionType.POINT_NODE
    state.set_country(25, 10, 0)  # Iran already overcontrolled.
    state.set_country(24, 0, 0)   # Iraq needs influence.
    action = HeuristicBot("US").select_action(state.to_dict(), {"decision_type": 5, "valid_ids": [25, 24]})
    assert action is not None and action["primary_id"] == 24
