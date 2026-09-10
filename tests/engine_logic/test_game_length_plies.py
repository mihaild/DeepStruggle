"""The ply numeration, checked against the engine's own action-round schedule.

A ply is one player's single opportunity to act -- one headline, or one action round for one
side -- numbered continuously from the start of the game. The point of pinning it here is that
it is not an independent convention: it has to track `advance_after_action_round`, which
alternates the phasing player USSR -> US and increments `action_round` when the US finishes,
over `max_ar = (turn <= 3) ? 6 : 7`. If that schedule ever changes, these fail.
"""

import ts_engine
import pytest

from ai.game_length import (
    FULL_GAME_PLIES,
    plies_before_turn,
    plies_in_turn,
    ply,
)
from tools.lib.tournament_evaluator import _drain_chance_nodes


def test_numeration_matches_the_stated_scheme() -> None:
    assert ply(1, 0, is_us=False) == 1     # USSR turn 1 headline
    assert ply(1, 0, is_us=True) == 2      # US   turn 1 headline
    assert ply(1, 1, is_us=False) == 3     # USSR turn 1 AR1
    assert ply(1, 6, is_us=True) == 14     # US   turn 1 AR6, the last ply of turn 1
    assert ply(2, 0, is_us=False) == 15    # USSR turn 2 headline
    assert ply(10, 7, is_us=True) == FULL_GAME_PLIES


def test_early_war_turns_are_shorter_than_late_ones() -> None:
    # max_ar is 6 through turn 3 and 7 from turn 4, so 14 plies then 16.
    assert [plies_in_turn(t) for t in (1, 2, 3)] == [14, 14, 14]
    assert [plies_in_turn(t) for t in (4, 7, 10)] == [16, 16, 16]
    assert plies_before_turn(4) == 42
    assert plies_before_turn(11) == FULL_GAME_PLIES == 154


def test_the_terminal_turn_the_engine_reports_for_a_full_game_maps_to_a_full_game() -> None:
    # finish_end_turn increments past 10 before testing, so a game that goes the distance
    # terminates holding turn 11. It has no ply of its own; it is the whole game.
    assert ply(11, 0, is_us=False) == FULL_GAME_PLIES
    assert ply(11, 0, is_us=True) == FULL_GAME_PLIES


@pytest.mark.parametrize("turn", [0, 12])
def test_turns_outside_the_game_are_rejected(turn: int) -> None:
    with pytest.raises(ValueError):
        ply(turn, 1, is_us=False)


def test_ply_never_runs_backwards_across_a_real_game() -> None:
    """Drive a game and check the index is monotonic and inside the schedule.

    This is the property that makes a ply usable as a length: whatever the engine does with
    action rounds, walking a game must never produce a ply that goes down.
    """
    state = ts_engine.GameState()
    ts_engine.Engine.init_game(state, 20260921)
    _drain_chance_nodes(state)

    previous = 0
    steps = 0
    while not ts_engine.Engine.is_terminal(state) and steps < 4000:
        current = ply(int(state.turn), int(state.action_round),
                      is_us=state.phasing_player == ts_engine.Player.US)
        assert 1 <= current <= FULL_GAME_PLIES + 4   # + an AR8's worth of slack
        assert current >= previous, (
            f"ply went backwards at turn {state.turn} AR {state.action_round}: "
            f"{previous} -> {current}")
        previous = current

        legal = ts_engine.Engine.get_legal_action_indices(state)
        if not len(legal):
            break
        ts_engine.Engine.step_flat(state, int(legal[0]))
        _drain_chance_nodes(state)
        steps += 1
