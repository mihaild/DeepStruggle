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
    state.set_card_location(DUCK_AND_COVER, ts.CardLocation.HAND_US)
    state.set_card_location(FIVE_YEAR_PLAN, ts.CardLocation.HAND_USSR)
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


def test_racing_with_it_passes_it_to_the_opponent_rather_than_discarding_it() -> None:
    state = _holding_the_china_card(box=4)
    ts.Engine.step_flat(state, THE_CHINA_CARD - 1)
    ts.Engine.step_flat(state, SPACE)
    assert state.ctx().decision_type == ts.DecisionType.ROLL_DIE
    ts.Engine.step_flat(state, 203 + 2)     # a roll of 3 makes box 5 (needs 3 or less)
    assert int(state.us_space_track) == 5
    assert state.china_card_holder == ts.Player.USSR
    assert state.get_card_location(THE_CHINA_CARD) != ts.CardLocation.DISCARD_PILE


def test_replay_247_converts_end_to_end() -> None:
    """Turn 10 AR4: the US races to box 5 with the China Card, for 3 VP."""
    path = os.path.join(_CORPUS, "247.json.gz")
    if not os.path.exists(path):
        pytest.skip("corpus not downloaded")
    with gzip.open(path, "rt") as f:
        conv = convert_game(json.load(f))
    assert conv.failure is None, f"replay 247 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 144
