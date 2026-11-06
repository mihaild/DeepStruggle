"""--no-held-scoring-blunder: the blunder-aware reward scores a held-scoring loss as an ordinary loss (winner +1, loser -1)
instead of as a blunder (the holder -1, the winner shielded at 0); unprovoked DEFCON 1 keeps its blunder treatment."""

import numpy as np
import ts_engine as ts

from ai.rewards.reward_calculator import BlunderAwareRewardCalculator


def _held_scoring_end() -> ts.GameState:
    """The US caught holding Europe Scoring at the end of turn 5: the engine's held-scoring game over."""
    st = ts.GameState()
    ts.Engine.init_game(st, 3)
    for name in ("Asia Scoring", "Europe Scoring", "Middle East Scoring", "Central America Scoring",
                 "Southeast Asia Scoring", "Africa Scoring", "South America Scoring"):
        st.set_card_location(int(ts.CardData.get_card_by_name(name)), ts.CardLocation.DISCARD_PILE)
    st.set_card_location(int(ts.CardData.get_card_by_name("Europe Scoring")), ts.hand_of(ts.Player.US))
    st.turn = 5
    st.action_round = 8
    st.victory_points = -20
    st.current_phase = ts.Phase.GAME_OVER
    assert ts.Engine.is_held_scoring_loss(st, ts.Player.US)
    return st


def _rewards(calc: BlunderAwareRewardCalculator, acting: int) -> float:
    st = _held_scoring_end()
    r = calc.compute_step_rewards(acting_players=np.array([acting]), dones=np.array([True]),
                                  terminal_utilities=np.array([-1.0]), prev_victory_points=np.array([-5]),
                                  curr_victory_points=np.array([-20]), states=[st])
    return float(r[0])


def test_by_default_the_holder_is_punished_and_the_winner_shielded() -> None:
    calc = BlunderAwareRewardCalculator()
    assert _rewards(calc, 1) == -1.0          # the US (holder) acted last
    assert _rewards(calc, -1) == 0.0          # the USSR acted last: shielded


def test_without_it_the_loss_is_ordinary() -> None:
    calc = BlunderAwareRewardCalculator(held_scoring_blunder=False)
    assert _rewards(calc, 1) == -1.0          # the loser
    assert _rewards(calc, -1) == 1.0          # the winner gets its +1
