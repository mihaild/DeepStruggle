"""Measures whether a policy takes forced wins and avoids avoidable forced losses.

Win rate barely registers these: a decisive choice arises at roughly 0.6% of decisions, so
a policy can throw away a quarter of them and still look fine. But each one is worth a
whole game, which makes these the most sensitive quality signals available -- and unlike
the behavioural suite's constructed positions, they are measured on states the policy
actually reaches.

Two exclusions matter for the numbers to mean anything, both learned by getting them wrong:

* **Chance nodes.** At ROLL_DIE no choice is being made, so counting the outcome against
  the policy is simply wrong.
* **Already-lost positions.** Where every legal action loses, taking one is not a blunder.
  A held scoring card that must eventually be played is the common case.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import ts_engine as ts

from ai.eval.safety import classify_legal_actions

# (state, player) -> chosen flat action
ActionFn = Callable[[ts.GameState, ts.Player], int]


@dataclass
class DecisiveStats:
    decisions: int = 0
    win_available: int = 0
    win_taken: int = 0
    loss_avoidable: int = 0
    loss_taken: int = 0
    loss_forced: int = 0

    @property
    def win_take_rate(self) -> float:
        return self.win_taken / self.win_available if self.win_available else float("nan")

    @property
    def loss_avoid_rate(self) -> float:
        if not self.loss_avoidable:
            return float("nan")
        return 1.0 - (self.loss_taken / self.loss_avoidable)

    def as_metrics(self) -> dict:
        return {
            "decisive_win_take_rate": self.win_take_rate,
            "decisive_loss_avoid_rate": self.loss_avoid_rate,
            "decisive_win_available": float(self.win_available),
            "decisive_loss_avoidable": float(self.loss_avoidable),
        }


def measure_decisive(
    action_fn: ActionFn,
    num_games: int = 40,
    base_seed: int = 77000,
    max_steps: int = 1500,
) -> DecisiveStats:
    """Plays self-play games and records how the policy handles decisive positions."""
    st_out = DecisiveStats()
    for g in range(num_games):
        state = ts.GameState()
        ts.Engine.init_game(state, base_seed + g)
        steps = 0
        while not ts.Engine.is_terminal(state) and steps < max_steps:
            ctx = state.ctx()
            player = ctx.decision_player if ctx.decision_player != ts.Player.NONE else state.phasing_player
            is_chance = ctx.decision_type == ts.DecisionType.ROLL_DIE
            kinds = {} if is_chance else classify_legal_actions(state, player)
            action = int(action_fn(state, player))

            if kinds:
                st_out.decisions += 1
                wins = [a for a, k in kinds.items() if k == "win"]
                losses = [a for a, k in kinds.items() if k == "loss"]
                if wins:
                    st_out.win_available += 1
                    if kinds.get(action) == "win":
                        st_out.win_taken += 1
                if losses:
                    if len(losses) < len(kinds):
                        st_out.loss_avoidable += 1
                        if kinds.get(action) == "loss":
                            st_out.loss_taken += 1
                    else:
                        st_out.loss_forced += 1

            try:
                ts.Engine.step_flat(state, action)
            except Exception:
                break
            steps += 1
    return st_out
