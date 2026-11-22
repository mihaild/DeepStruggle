"""--vp-potential: potential-based VP shaping, Phi = scale x VP (US side), Phi(terminal) = 0."""

from __future__ import annotations

from typing import List

import numpy as np

from ai.rewards.reward_calculator import VPPotentialShaping, ZeroSumTerminalReward


def _step(calc: VPPotentialShaping, mover: int, prev: int, curr: int, done: bool = False,
          util: float = 0.0) -> float:
    return float(calc.compute_step_rewards(
        acting_players=np.array([mover], dtype=np.int8), dones=np.array([done]),
        terminal_utilities=np.array([util], dtype=np.float32),
        prev_victory_points=np.array([prev], dtype=np.int8),
        curr_victory_points=np.array([curr], dtype=np.int8))[0])


def test_the_shaping_sums_to_zero_over_a_game() -> None:
    calc = VPPotentialShaping(ZeroSumTerminalReward(), 0.01)
    # (mover, VP before, VP after); the last step ends the game, won by the US on VP 9
    steps = [(1, 0, 2), (-1, 2, -1), (-1, -1, -1), (1, -1, 4), (-1, 4, 9)]
    us_frame: List[float] = []
    for k, (m, a, b) in enumerate(steps):
        last = k == len(steps) - 1
        us_frame.append(m * _step(calc, m, a, b, done=last, util=1.0 if last else 0.0))
    # US-frame return: the win alone, every VP payment given back at the end
    assert abs(sum(us_frame) - 1.0) < 1e-6
    # but paid as it happened: the US's first +2 VP is worth +0.02 to the US at once
    assert abs(us_frame[0] - 0.02) < 1e-6
    # and the terminal step takes back the lead the game ended on (VP 4 before the last move)
    assert abs(us_frame[-1] - (1.0 - 0.04)) < 1e-6


def test_the_reward_is_in_the_movers_frame() -> None:
    calc = VPPotentialShaping(ZeroSumTerminalReward(), 0.01)
    assert abs(_step(calc, -1, 0, -3) - 0.03) < 1e-6      # USSR gains 3 VP: +0.03 for the USSR
    assert abs(_step(calc, 1, 0, -3) + 0.03) < 1e-6       # the US moving into the same: -0.03


def test_it_wraps_the_base_calculator() -> None:
    base = ZeroSumTerminalReward()
    base.marker = 7                                       # type: ignore[attr-defined]
    calc = VPPotentialShaping(base, 0.01)
    assert calc.marker == 7                               # attributes reach the base
    assert not hasattr(calc, "on_env_reset")              # absent on the base, absent here
    # a terminal step with no state is the base's ordinary win plus the shaping
    assert abs(_step(calc, 1, 5, 5, done=True, util=1.0) - (1.0 - 0.05)) < 1e-6
