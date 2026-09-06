"""Length-scaled terminal reward: winning sooner must be worth more than winning later.

With gamma = 1 and a terminal-only reward the objective is indifferent to *when* a game ends.
Taking a forced win now returns +1; declining it and winning three turns later also returns
+1, so the policy gradient sees no difference and only risk separates them. That is the most
likely reason the control takes just 80.3% of engine-verified forced wins and stops improving
after ~30M steps, and why perfect-information search does not fix it -- search maximises the
same indifferent objective.
"""

import numpy as np
import pytest
import ts_engine as ts

from ai.rewards.reward_calculator import BlunderAwareRewardCalculator


def _state(turn: int, defcon: int = 5) -> ts.GameState:
    st = ts.GameState()
    ts.Engine.init_game(st, 909)
    st.turn = turn
    st.defcon = defcon
    return st


def _reward(calc, turn: int, acting: int, term_util: float, defcon: int = 5) -> float:
    return float(calc.compute_step_rewards(
        acting_players=np.array([acting]),
        dones=np.array([True]),
        terminal_utilities=np.array([term_util]),
        prev_victory_points=np.array([0]),
        curr_victory_points=np.array([0]),
        states=[_state(turn, defcon)],
    )[0])


def test_disabled_by_default_reproduces_the_old_reward() -> None:
    calc = BlunderAwareRewardCalculator()
    assert calc.decisiveness_turns == 0.0
    for turn in (1, 5, 10):
        assert _reward(calc, turn, acting=1, term_util=1.0) == pytest.approx(1.0)
        assert _reward(calc, turn, acting=-1, term_util=1.0) == pytest.approx(-1.0)


def test_winning_sooner_is_worth_more() -> None:
    """The whole point: the gradient must distinguish a win now from a win later."""
    calc = BlunderAwareRewardCalculator(decisiveness_turns=40.0)
    early = _reward(calc, turn=3, acting=1, term_util=1.0)
    late = _reward(calc, turn=9, acting=1, term_util=1.0)
    assert early > late, f"turn 3 win {early} must beat turn 9 win {late}"
    assert early == pytest.approx(1.0 - 3 / 40)
    assert late == pytest.approx(1.0 - 9 / 40)


def test_losing_later_is_less_bad() -> None:
    """A losing player should prolong the game, not end it -- the turn-3 suicide case."""
    calc = BlunderAwareRewardCalculator(decisiveness_turns=40.0)
    early_loss = _reward(calc, turn=3, acting=-1, term_util=1.0)
    late_loss = _reward(calc, turn=9, acting=-1, term_util=1.0)
    assert late_loss > early_loss, (
        f"losing on turn 9 ({late_loss}) should be preferable to losing on turn 3 "
        f"({early_loss})"
    )


def test_self_inflicted_defcon_loss_is_scaled_too() -> None:
    """An unprovoked DEFCON-1 suicide on turn 3 must cost more than the same on turn 9."""
    calc = BlunderAwareRewardCalculator(decisiveness_turns=40.0)
    # defcon <= 1 with acting == phasing is the unprovoked-blunder branch.
    st_early, st_late = _state(3, defcon=1), _state(9, defcon=1)
    acting = int(st_early.phasing_player)

    def blunder(st):
        return float(calc.compute_step_rewards(
            acting_players=np.array([acting]), dones=np.array([True]),
            terminal_utilities=np.array([-1.0 * acting]),
            prev_victory_points=np.array([0]), curr_victory_points=np.array([0]),
            states=[st])[0])

    assert blunder(st_early) < blunder(st_late) < 0.0, (
        "an early self-inflicted loss must be punished harder than a late one"
    )


def test_zero_sum_is_preserved_for_ordinary_wins() -> None:
    calc = BlunderAwareRewardCalculator(decisiveness_turns=20.0)
    win = _reward(calc, turn=6, acting=1, term_util=1.0)
    loss = _reward(calc, turn=6, acting=-1, term_util=1.0)
    assert win == pytest.approx(-loss), f"not zero-sum: {win} vs {loss}"


def test_scale_never_reaches_zero_or_flips_sign() -> None:
    """A steep slope past the horizon must not make a win worthless or a loss rewarding."""
    calc = BlunderAwareRewardCalculator(decisiveness_turns=4.0)
    r = _reward(calc, turn=10, acting=1, term_util=1.0)
    assert r > 0.0, f"a win must stay positive, got {r}"
    assert _reward(calc, turn=10, acting=-1, term_util=1.0) < 0.0


def test_missing_state_falls_back_to_unscaled() -> None:
    calc = BlunderAwareRewardCalculator(decisiveness_turns=40.0)
    r = float(calc.compute_step_rewards(
        acting_players=np.array([1]), dones=np.array([True]),
        terminal_utilities=np.array([1.0]),
        prev_victory_points=np.array([0]), curr_victory_points=np.array([0]),
        states=None)[0])
    assert r == pytest.approx(1.0), "without a state the turn is unknown; do not guess"
