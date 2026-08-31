"""The batched evaluation path must agree with the sequential one, and only be faster.

The trainer used to evaluate snapshots with TournamentEvaluator.play_matchup, one game and
one decision at a time. That was 30x slower than the vectorized path the tournament CLI
uses, and evaluation cost grows with the opponent list, so a 3-hour run spent 37-61% of its
wall clock evaluating and one A/B arm was starved down to 473 iterations against 1024.

It was also wrong. At a ROLL_DIE node ctx().decision_player is NONE, and play_matchup fell
back to phasing_player -- handing the die roll to a policy network, 134 times per 20 games.
The batched runner resolves chance nodes inside the engine. Before the fix the two paths
disagreed by more than 25 points on the same deterministic matchup.
"""

import numpy as np
import pytest
import ts_engine as ts

from tools.lib.batch_tournament import BatchMatchRunner
from tools.lib.player_agent import HeuristicAgent, NeuralAgent
from tools.lib.tournament_evaluator import TournamentEvaluator, _drain_chance_nodes


def _agent(seed: int) -> NeuralAgent:
    from ai.models.coldwar_net_v2 import create_coldwar_net_v2
    import torch
    torch.manual_seed(seed)
    return NeuralAgent(model=create_coldwar_net_v2("cpu"), name=f"n{seed}", device="cpu")


def test_sequential_path_never_asks_a_policy_to_roll_dice() -> None:
    """A chance node must be resolved by the engine, not chosen by an agent."""
    st = ts.GameState()
    ts.Engine.init_game(st, 4242)
    _drain_chance_nodes(st)

    seen_none = 0
    for _ in range(1500):
        if ts.Engine.is_terminal(st):
            break
        if st.ctx().decision_player == ts.Player.NONE:
            seen_none += 1
        legal = np.flatnonzero(np.asarray(ts.ActionMask.generate_flat_mask(st)))
        ts.Engine.step_flat(st, int(legal[0]) if len(legal) else 0)
        _drain_chance_nodes(st)

    assert seen_none == 0, (
        f"{seen_none} decisions were offered to an agent while decision_player was NONE; "
        f"those are chance nodes and the engine must resolve them"
    )


@pytest.mark.parametrize("opponent_is_neural", [True, False])
def test_batched_and_sequential_agree_exactly_when_deterministic(opponent_is_neural) -> None:
    """Same seeds, same greedy policy: the two paths must produce identical results."""
    a = _agent(1)
    b = _agent(2) if opponent_is_neural else HeuristicAgent()

    seq = TournamentEvaluator.play_matchup(a, b, games_per_side=6, temperature=0.0)
    bat = BatchMatchRunner.play_parallel_matchup(a, b, games_per_side=6, device="cpu",
                                                 temperature=0.0)

    for field in ["a_wins", "b_wins", "draws", "win_rate_a",
                  "a_wins_as_us", "a_losses_as_us",
                  "a_wins_as_ussr", "a_losses_as_ussr"]:
        assert seq[field] == pytest.approx(bat[field]), (
            f"{field}: sequential={seq[field]} batched={bat[field]}; the two evaluation "
            f"paths must be interchangeable"
        )


def test_batched_path_honours_temperature() -> None:
    """deterministic used to be hard-coded True here, which is why the paths diverged."""
    a, b = _agent(3), _agent(4)
    greedy_1 = BatchMatchRunner.play_parallel_matchup(a, b, games_per_side=4, device="cpu",
                                                      temperature=0.0)
    greedy_2 = BatchMatchRunner.play_parallel_matchup(a, b, games_per_side=4, device="cpu",
                                                      temperature=0.0)
    assert greedy_1["a_wins"] == greedy_2["a_wins"], "greedy play must be reproducible"

    # A high temperature must actually change play, or the parameter is being ignored.
    hot = [BatchMatchRunner.play_parallel_matchup(a, b, games_per_side=4, device="cpu",
                                                  temperature=5.0)["a_wins"]
           for _ in range(6)]
    assert len(set(hot)) > 1 or hot[0] != greedy_1["a_wins"], (
        "temperature had no effect on the batched path"
    )
