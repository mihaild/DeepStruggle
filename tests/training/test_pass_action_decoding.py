"""The flat action space's pass action must decode as a pass for every decision type.

Action 211 is "decline / stop here". For SELECT_CARD it used to decode with primary_id 0 and
no CONFIRM_DONE flag, so is_confirm_done() answered False for the one action that means
exactly that, and anything reasoning about the decoded action read a pass as "discard card 0".
The card dispatchers happened to accept primary_id 0 as well, so play was unaffected -- but
the decoded action lied about itself.
"""
from typing import List

import numpy as np
import ts_engine as ts

PASS_ACTION = 211
OUR_MAN_IN_TEHRAN = 108


def _drain(state: ts.GameState) -> None:
    while (not ts.Engine.is_terminal(state)
           and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
        mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
        ts.Engine.step_flat(state, int(np.flatnonzero(mask)[0]))


def test_pass_action_decodes_as_confirm_done_for_select_card() -> None:
    """A peeked-card discard is optional, and declining must decode as a pass."""
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    _drain(state)

    # Put the US in a peek: it needs a Middle East country to trigger the event at all.
    state.get_country(15).us_influence = 9  # Iran, comfortably US-controlled
    state.ctx().decision_player = ts.Player.US
    state.ctx().decision_type = ts.DecisionType.SELECT_CARD
    state.ctx().resolving_card = OUR_MAN_IN_TEHRAN
    state.ctx().allow_early_stop = 1

    peek: List[int] = [4, 8, 47, 53, 75]
    for c in peek:
        state.set_card_location(c, ts.CardLocation.PEEKED_TEMP)
    state.ctx().temp_cards = peek
    state.ctx().remaining_steps = len(peek)

    mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
    assert mask[PASS_ACTION] == 1, "declining an optional discard must be legal"

    action = ts.ActionMask.decode_flat_action(state, PASS_ACTION)
    assert action.is_confirm_done(), "the pass action must decode as a pass"
    assert int(action.primary_id) == 0, "primary_id 0 is what the dispatchers match on"


def test_pass_action_decodes_as_confirm_done_for_point_node() -> None:
    """The other decision types already did this; pin it so they stay consistent."""
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    _drain(state)
    state.ctx().decision_type = ts.DecisionType.POINT_NODE
    state.ctx().allow_early_stop = 1

    action = ts.ActionMask.decode_flat_action(state, PASS_ACTION)
    assert action.is_confirm_done()
