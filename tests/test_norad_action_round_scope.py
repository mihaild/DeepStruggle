"""NORAD claims the end of an action round in which DEFCON reached 2 -- and only that.

While the US controls Canada it adds 1 Influence at the end of each *action round* in which
DEFCON moves to 2. Two things can go wrong and both are silent: firing when the US does not
control Canada, and firing for a drop that happened somewhere other than an action round.

defcon_dropped_to_2 records the drop. The engine clears it when an action round ends; it now
clears it when the headline ends too, so a headline's drop cannot claim the round that follows.
"""
import glob
import gzip
import json
import os

import numpy as np
import pytest
import ts_engine as ts

from tools.lib.ts_replayer_convert import convert_game

CORPUS = "/workspace/data/datasets/ts_replayer"
DUCK_AND_COVER = 4          # degrades DEFCON
SOCIALIST_GOVERNMENTS = 15
NORAD = 106
CANADA = 0


def _base(us_in_canada: int = 4) -> ts.GameState:
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    while (not ts.Engine.is_terminal(state)
           and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
        mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
        ts.Engine.step_flat(state, int(np.flatnonzero(mask)[0]))
    state.set_flag(ts.EffectBits.NORAD_ACTIVE)
    state.set_country(CANADA, us_in_canada, 0)   # stability 2, so 4 US Influence is control
    state.defcon = 3
    for c in range(1, 111):
        if state.get_card_location(c) in (ts.CardLocation.HAND_US,
                                          ts.CardLocation.HAND_USSR):
            state.set_card_location(c, ts.CardLocation.DISCARD_PILE)
    return state


def _play_one_action_round(state: ts.GameState) -> bool:
    """The USSR spends a card; returns whether NORAD claimed the end of the round."""
    state.current_phase = ts.Phase.ACTION_ROUND
    state.action_round = 1
    state.phasing_player = ts.Player.USSR
    state.set_card_location(SOCIALIST_GOVERNMENTS, ts.CardLocation.HAND_USSR)
    state.ctx().decision_player = ts.Player.USSR
    state.ctx().decision_type = ts.DecisionType.SELECT_CARD
    for _ in range(40):
        if ts.Engine.is_terminal(state):
            return False
        if int(state.ctx().resolving_card) == NORAD:
            return True
        mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
        legal = np.flatnonzero(mask)
        if not len(legal):
            return False
        if (state.ctx().decision_type == ts.DecisionType.SELECT_CARD
                and int(state.ctx().resolving_card) == 0
                and state.get_card_location(SOCIALIST_GOVERNMENTS)
                != ts.CardLocation.HAND_USSR):
            return False
        ts.Engine.step_flat(state, int(legal[0]))
    return False


def test_no_norad_without_canada() -> None:
    state = _base(us_in_canada=0)
    state.current_phase = ts.Phase.ACTION_ROUND
    ts.CardHandlers.trigger_event(state, DUCK_AND_COVER, ts.Player.US)
    assert int(state.defcon) == 2
    assert not ts.Scoring.is_controlled_by(state, CANADA, ts.Player.US)
    assert not _play_one_action_round(state)


def test_norad_fires_for_a_drop_inside_the_action_round() -> None:
    state = _base()
    state.current_phase = ts.Phase.ACTION_ROUND
    ts.CardHandlers.trigger_event(state, DUCK_AND_COVER, ts.Player.US)
    assert int(state.defcon) == 2
    assert ts.Scoring.is_controlled_by(state, CANADA, ts.Player.US)
    assert _play_one_action_round(state)


def test_a_headline_drop_does_not_claim_the_first_action_round() -> None:
    """Turn 5 of replay 219: DEFCON falls to 2 in the headline, and the action round that
    follows had nothing to do with it -- the USSR only discards a card to escape Bear Trap."""
    state = _base()
    state.set_card_location(DUCK_AND_COVER, ts.CardLocation.HAND_US)
    state.set_card_location(SOCIALIST_GOVERNMENTS, ts.CardLocation.HAND_USSR)
    state.current_phase = ts.Phase.HEADLINE
    state.headline_stage = 0
    state.action_round = 0
    state.ctx().decision_player = ts.Player.US
    state.ctx().decision_type = ts.DecisionType.SELECT_CARD

    ts.Engine.step(state, ts.MicroAction(ts.DecisionType.SELECT_CARD, DUCK_AND_COVER, 0, 0))
    ts.Engine.step(state, ts.MicroAction(
        ts.DecisionType.SELECT_CARD, SOCIALIST_GOVERNMENTS, 0, 0))
    for _ in range(40):
        if state.current_phase != ts.Phase.HEADLINE:
            break
        mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
        legal = np.flatnonzero(mask)
        if not len(legal):
            break
        ts.Engine.step_flat(state, int(legal[0]))

    assert int(state.defcon) == 2, "the headline took DEFCON to 2"
    assert int(state.defcon_dropped_to_2) == 0, (
        "and the headline's drop does not carry into the action rounds")


@pytest.mark.skipif(not glob.glob(os.path.join(CORPUS, "*.json.gz")),
                    reason="ts-replayer corpus not downloaded")
def test_replay_296_converts_end_to_end() -> None:
    with gzip.open(os.path.join(CORPUS, "296.json.gz"), "rt") as f:
        conv = convert_game(json.load(f))
    assert conv.failure is None, f"replay 296 stopped at {conv.failure}"
