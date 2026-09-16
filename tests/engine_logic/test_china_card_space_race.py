"""The China Card can be played for the Space Race.

It carries no Event of its own, so Operations or the Space Race -- racing with the best Ops
card in the game is a poor play and not an illegal one. Whatever it is played for it passes
to the opponent face down, so it is never discarded.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import List

import pytest
import ts_engine as ts

from tools.lib.corpus_paths import corpus_dir
from tools.lib.ts_replayer_convert import convert_game

_CORPUS = str(corpus_dir())
THE_CHINA_CARD = 6
DUCK_AND_COVER = 4
FIVE_YEAR_PLAN = 5
SPACE = 112     # flat index of the space play mode
PLAY_MODES = (110, 111, 112)


def _holding_the_china_card(box: int) -> ts.GameState:
    """The US at `box`, holding the China Card, with a card in each hand so the turn stands."""
    state = ts.GameState()
    state.rng_state = 42
    state.turn = 10
    state.action_round = 4
    state.current_phase = ts.Phase.ACTION_ROUND
    state.phasing_player = ts.Player.US
    state.china_card_holder = ts.Player.US
    state.china_card_playable = 1
    state.us_space_track = box
    state.set_card_location(DUCK_AND_COVER, ts.hand_of(ts.Player.US))
    state.set_card_location(FIVE_YEAR_PLAN, ts.hand_of(ts.Player.USSR))
    ctx = state.ctx()
    ctx.decision_player = ts.Player.US
    ctx.decision_type = ts.DecisionType.SELECT_CARD
    return state


def _modes(state: ts.GameState) -> List[int]:
    mask = ts.ActionMask.generate_flat_mask(state)
    return [m for m in PLAY_MODES if mask[m]]


def test_the_space_race_is_on_offer_for_the_china_card() -> None:
    state = _holding_the_china_card(box=4)
    ts.Engine.step_flat(state, THE_CHINA_CARD - 1)
    assert state.ctx().decision_type == ts.DecisionType.SELECT_PLAY_MODE
    assert _modes(state) == [111, 112], "Operations and the Space Race, and no Event"


def test_it_is_not_on_offer_when_the_next_box_is_out_of_reach() -> None:
    """Box 8 wants 4 Ops and the China Card has 4, so the ceiling is the attempt, not the Ops:
    with the turn's attempt already spent there is nothing to race for."""
    state = _holding_the_china_card(box=4)
    state.record_space_attempt(ts.Player.US)
    state.record_space_attempt(ts.Player.US)
    ts.Engine.step_flat(state, THE_CHINA_CARD - 1)
    assert _modes(state) == [111]


def _race_with_the_china_card(die: int) -> ts.GameState:
    """Play the China Card for the Space Race and force `die`.

    Forcing the die is a designed affordance, not a test hack: a ROLL_DIE action carries the
    acting player's value in `primary_id` (and the opponent's in `secondary_id`, used only by
    realignment). `decision_type` matches the context, which is what distinguishes it from the
    flat-203 route an earlier version of this test used -- that decoded to CHOOSE_BRANCH and
    worked only because the engine did not check the type.
    """
    state = _holding_the_china_card(box=4)
    ts.Engine.step_flat(state, THE_CHINA_CARD - 1)
    ts.Engine.step_flat(state, SPACE)
    assert state.ctx().decision_type == ts.DecisionType.ROLL_DIE
    assert ts.Engine.try_step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, die, 0, 0)), (
        f"engine refused a forced die of {die}")
    return state


def test_racing_with_it_passes_it_to_the_opponent_rather_than_discarding_it() -> None:
    """Box 4 needs 3 or less, and the card changes hands either way."""
    made = _race_with_the_china_card(3)
    assert int(made.us_space_track) == 5
    assert made.china_card_holder == ts.Player.USSR
    assert made.get_card_location(THE_CHINA_CARD) != ts.CardLocation.DISCARD_PILE

    missed = _race_with_the_china_card(4)
    assert int(missed.us_space_track) == 4, "box 4 advanced on a roll of 4"
    assert missed.china_card_holder == ts.Player.USSR, (
        "the card must pass to the opponent whether or not the attempt succeeds")
    assert missed.get_card_location(THE_CHINA_CARD) != ts.CardLocation.DISCARD_PILE


def test_the_forced_die_threshold_is_three_at_box_four() -> None:
    """Every value, so the boundary is pinned rather than sampled."""
    outcomes = {die: int(_race_with_the_china_card(die).us_space_track) for die in range(1, 7)}
    assert outcomes == {1: 5, 2: 5, 3: 5, 4: 4, 5: 4, 6: 4}, outcomes


def test_a_mismatched_decision_type_cannot_force_a_die() -> None:
    """Pins the door that was closed, so it is not reopened by accident.

    Flat 203-208 decode to CHOOSE_BRANCH. Submitted at a ROLL_DIE node the engine once read
    `primary_id` as the die and consumed the roll -- which is how a coup in a recorded game reached
    resolution without ever rolling.
    """
    state = _holding_the_china_card(box=4)
    ts.Engine.step_flat(state, THE_CHINA_CARD - 1)
    ts.Engine.step_flat(state, SPACE)
    assert state.ctx().decision_type == ts.DecisionType.ROLL_DIE

    before = int(state.us_space_track)
    assert ts.Engine.try_step_flat(state, 203 + 2) is False, (
        "engine accepted a CHOOSE_BRANCH action at a ROLL_DIE node")
    assert int(state.us_space_track) == before
    assert state.ctx().decision_type == ts.DecisionType.ROLL_DIE, (
        "a rejected action consumed the roll")


def test_replay_247_converts_end_to_end() -> None:
    """Turn 10 AR4: the US races to box 5 with the China Card, for 3 VP."""
    path = os.path.join(_CORPUS, "247.json.gz")
    if not os.path.exists(path):
        pytest.skip("corpus not downloaded")
    with gzip.open(path, "rt") as f:
        conv = convert_game(json.load(f))
    assert conv.failure is None, f"replay 247 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 144
