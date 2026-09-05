"""Missile Envy's exchange is decided by the hand, so the hand has to be right first.

It takes the opponent's highest Ops card, with no decision to steer unless two tie. A hand
holding one card too many therefore hands over the wrong card, and the wrong event fires.

The turn's hand list in the log is everything a player held at some point during the turn,
including what they picked up part way through it -- SALT Negotiations retrieves a card from
the discard pile mid-turn, and that card appears in the list as though it had been dealt.
"""
import glob
import gzip
import json
import os
from typing import Optional

import pytest
import ts_engine as ts

from tools.lib.ts_replayer_convert import _seed_missile_envy_hand, convert_game

CORPUS = "/workspace/data/datasets/ts_replayer"
SUEZ_CRISIS = 28
RED_SCARE_PURGE = 31
CAMP_DAVID = 65

pytestmark = pytest.mark.skipif(
    not glob.glob(os.path.join(CORPUS, "*.json.gz")),
    reason="ts-replayer corpus not downloaded")


def _convert(replay_id: int):
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return convert_game(json.load(f))


def _state_with_us_hand(cards) -> ts.GameState:
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    for c in range(1, 111):
        if state.get_card_location(c) == ts.CardLocation.HAND_US:
            state.set_card_location(c, ts.CardLocation.DISCARD_PILE)
    for c in cards:
        state.set_card_location(c, ts.CardLocation.HAND_US)
    return state


def _us_hand(state: ts.GameState):
    return {c for c in range(1, 111)
            if state.get_card_location(c) == ts.CardLocation.HAND_US}


def test_a_card_the_giver_could_not_have_held_is_set_aside() -> None:
    """Replay 64 turn 5 AR3: the log hands over Suez Crisis at 3 Ops.

    Red Scare/Purge, at 4, is in the turn's hand list but was retrieved from the discard pile
    four action rounds later, so it cannot have been there to give.
    """
    state = _state_with_us_hand([SUEZ_CRISIS, RED_SCARE_PURGE, CAMP_DAVID])
    _seed_missile_envy_hand(state, SUEZ_CRISIS, ts.Player.US)
    hand = _us_hand(state)
    assert SUEZ_CRISIS in hand
    assert RED_SCARE_PURGE not in hand, "4 Ops would have been taken instead of the logged 3"
    assert CAMP_DAVID in hand, "cards at or below the logged card are left alone"


def test_the_named_card_is_put_in_hand_if_it_is_missing() -> None:
    state = _state_with_us_hand([CAMP_DAVID])
    _seed_missile_envy_hand(state, SUEZ_CRISIS, ts.Player.US)
    assert SUEZ_CRISIS in _us_hand(state)


def test_a_scoring_card_is_never_set_aside() -> None:
    """Missile Envy cannot take one, so a scoring card never competes to be the highest."""
    scoring = next(c for c in range(1, 111) if ts.CardData.get_card_info(c)["is_scoring"])
    state = _state_with_us_hand([SUEZ_CRISIS, scoring])
    _seed_missile_envy_hand(state, SUEZ_CRISIS, ts.Player.US)
    assert scoring in _us_hand(state)


def test_replay_64_converts_end_to_end() -> None:
    conv = _convert(64)
    assert conv.failure is None, f"replay 64 stopped at {conv.failure}"
    # Everything but the last turn, which the recording stops inside -- see
    # tests/test_replay_unfinished_final_turn.py.
    assert conv.entries_converted == 106
