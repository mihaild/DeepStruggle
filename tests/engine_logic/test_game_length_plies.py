"""The ply numeration, checked against the engine's own action-round schedule.

A ply is one player's single opportunity to act -- one headline, or one action round for one
side -- numbered continuously from the start of the game. The point of pinning it here is that
it is not an independent convention: it has to track `advance_after_action_round`, which
alternates the phasing player USSR -> US and increments `action_round` when the US finishes,
over `max_ar = (turn <= 3) ? 6 : 7`. If that schedule ever changes, these fail.
"""

import random

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
    assert ply(1, 0, is_us=False, headline_stage=1) == 1   # turn 1, first headline
    assert ply(1, 0, is_us=False, headline_stage=2) == 2   # turn 1, second headline
    assert ply(1, 1, is_us=False) == 3     # USSR turn 1 AR1
    assert ply(1, 6, is_us=True) == 14     # US   turn 1 AR6, the last ply of turn 1
    assert ply(2, 0, is_us=True, headline_stage=1) == 15   # turn 2, first headline
    assert ply(10, 7, is_us=True) == FULL_GAME_PLIES


@pytest.mark.parametrize("is_us", [False, True])
def test_a_headline_ply_is_ordered_by_resolution_not_by_side(is_us: bool) -> None:
    """The bug this parameter exists for: both sides reveal at once and the higher Ops card goes
    first, so the US headline is the turn's FIRST ply whenever the US played the bigger card.
    Numbering by side made the index run 30 -> 29 inside turn 3."""
    assert ply(3, 0, is_us=is_us, headline_stage=1) == 29
    assert ply(3, 0, is_us=is_us, headline_stage=2) == 30


def test_stage_zero_is_the_first_headline_ply() -> None:
    """While both players are still choosing, no card has resolved: that is the turn's first
    headline ply, and it must not run ahead of the card that resolves next."""
    assert ply(3, 0, is_us=False, headline_stage=0) == 29


def test_a_headline_without_a_stage_is_refused_rather_than_guessed() -> None:
    """A wrong ply returns a plausible number, so this has to fail loudly."""
    with pytest.raises(ValueError, match="headline_stage"):
        ply(3, 0, is_us=True)


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


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 20260921])
def test_ply_never_runs_backwards_across_a_real_game(seed: int) -> None:
    """Drive a game and check the index is monotonic and inside the schedule.

    This is the property that makes a ply usable as a length: whatever the engine does with
    action rounds, walking a game must never produce a ply that goes down.

    What this does NOT cover: an unguided policy drives the VP track to +/-20 in the early war,
    so every one of these walks terminates by autowin somewhere in turns 1-3 and never reaches
    the turn-4 boundary where max_ar goes 6 -> 7. That boundary is pinned by
    `test_early_war_turns_are_shorter_than_late_ones` against the pure function instead. Saying
    so here because the docstring used to imply this walk covered the whole schedule.

    This test was vacuous until it was fixed: it fed indices from the 128-wide per-decision mask
    to `step_flat`, which reads the flat 212-dim space. Every step was refused, the refusal
    discarded, and 4000 iterations ran against a state that never moved, so the assertions below
    passed without ever seeing a second ply.
    """
    rng = random.Random(seed)
    state = ts_engine.GameState()
    ts_engine.Engine.init_game(state, seed)
    _drain_chance_nodes(state)

    previous = 0
    previous_slot = (0, 0)
    steps = 0
    while not ts_engine.Engine.is_terminal(state) and steps < 4000:
        current = ply(int(state.turn), int(state.action_round),
                      is_us=state.phasing_player == ts_engine.Player.US,
                      headline_stage=int(state.headline_stage))
        assert 1 <= current <= FULL_GAME_PLIES + 4   # + an AR8's worth of slack

        # The schedule itself: (turn, action_round) is the engine's own ordering and must never
        # run backwards.
        slot = (int(state.turn), int(state.action_round))
        assert slot >= previous_slot, f"schedule went backwards: {previous_slot} -> {slot}"
        previous_slot = slot

        # Monotonic everywhere, headlines included. This held only outside the headline until
        # `ply` took headline_stage: it numbered the two headline plies USSR-then-US while the
        # engine resolves them in descending Ops, which produced 30 -> 29 inside turn 3.
        assert current >= previous, (
            f"ply went backwards at turn {state.turn} AR {state.action_round} "
            f"stage {state.headline_stage}: {previous} -> {current}")
        previous = current

        mask = ts_engine.Engine.get_flat_action_mask(state)
        legal = [i for i, v in enumerate(mask) if v]
        if not legal:
            break
        ts_engine.Engine.step_flat(state, rng.choice(legal))
        _drain_chance_nodes(state)
        steps += 1

    # Pin that the walk actually moved, so this cannot quietly go vacuous again.
    assert steps > 50, f"only {steps} steps taken; this is not walking a real game"
    assert previous > 1, f"the game never advanced past its first ply (reached {previous})"
