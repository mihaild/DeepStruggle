"""The eighth action round is offered, not imposed.

It is not part of the turn: North Sea Oil or a Space Station grants it to the one player who
earned it, and it is theirs to take or to leave. The engine required a card of them like any
other round, so a player who declined could not be reconstructed at all.

At turn 9 AR8 of replay 16 the USSR is on space box 8 against the US's 4, is granted the round,
and declines. The log gives it an entry of its own with an empty body -- "Turn 9, USSR AR8: :"
-- which is a different shape from the bare "Turn 5, USSR AR4" header that marks a skipped
round at the foot of another entry.

The engine may have taken the round away before anyone is asked: it passes a player with an
empty hand automatically. Where it has not, the pass is driven, and a pass with cards still in
hand would be reported rather than forced.
"""
import glob
import gzip
import json
import os
from typing import Dict, List

import pytest
import ts_engine as ts

from tools.lib.ts_replayer_convert import _is_skipped_round, convert_game
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = "/workspace/data/datasets/ts_replayer"
DUCK_AND_COVER, FIVE_YEAR_PLAN = 4, 5
PASS = 211

pytestmark = pytest.mark.skipif(
    not glob.glob(os.path.join(CORPUS, "*.json.gz")),
    reason="ts-replayer corpus not downloaded")


def _game(replay_id: int) -> Dict:
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return json.load(f)


def _at_action_round(turn: int, action_round: int, mover, hand: List[int]) -> ts.GameState:
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    guard = 0
    while state.current_phase != ts.Phase.ACTION_ROUND and guard < 600:
        guard += 1
        ctx = state.ctx()
        if (ctx.decision_player == ts.Player.NONE
                and ctx.decision_type == ts.DecisionType.ROLL_DIE):
            ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
            continue
        legal = [i for i, v in enumerate(ts.ActionMask.generate_flat_mask(state)) if v]
        ts.Engine.step_flat(state, legal[0])
    state.turn = turn
    state.action_round = action_round
    state.phasing_player = mover
    ctx = state.ctx()
    ctx.decision_player = mover
    ctx.decision_type = ts.DecisionType.SELECT_CARD
    loc = ts.CardLocation.HAND_US if mover == ts.Player.US else ts.CardLocation.HAND_USSR
    for c in range(1, 111):
        if state.get_card_location(c) in (ts.CardLocation.HAND_US,
                                          ts.CardLocation.HAND_USSR):
            state.set_card_location(c, ts.CardLocation.DISCARD_PILE)
    for c in hand:
        state.set_card_location(c, loc)
    state.china_card_holder = ts.Player.US if mover == ts.Player.USSR else ts.Player.USSR
    return state


def test_the_eighth_round_may_be_declined_with_cards_in_hand() -> None:
    state = _at_action_round(9, 8, ts.Player.USSR, [DUCK_AND_COVER, FIVE_YEAR_PLAN])
    mask = ts.ActionMask.generate_flat_mask(state)
    assert mask[PASS], "the round is granted, not imposed"
    assert any(mask[i] for i in range(110)), "and playing it is still allowed"


def test_an_ordinary_round_may_not_be_declined() -> None:
    """Only the eighth. A player holding cards owes the other seven."""
    state = _at_action_round(9, 7, ts.Player.USSR, [DUCK_AND_COVER, FIVE_YEAR_PLAN])
    assert not ts.ActionMask.generate_flat_mask(state)[PASS]


def test_the_log_gives_the_declined_round_an_entry_of_its_own() -> None:
    raws = _game(16)["all_turns"]
    entry = next(parse_entry(r) for r in raws
                 if (parse_entry(r).turn, parse_entry(r).phase) == (9, "AR8"))
    assert entry.player == "USSR"
    assert not entry.card
    assert _is_skipped_round(entry)


def test_an_entry_that_plays_something_is_not_a_skipped_round() -> None:
    raws = _game(16)["all_turns"]
    played = next(parse_entry(r) for r in raws
                  if (parse_entry(r).turn, parse_entry(r).phase, parse_entry(r).player)
                  == (9, "AR7", "US"))
    assert played.card == "Che"
    assert not _is_skipped_round(played)


def test_replay_16_converts_end_to_end() -> None:
    """It stopped at turn 5 AR1 on the military ops score, then here at turn 9 AR8."""
    conv = convert_game(_game(16))
    assert conv.failure is None, f"replay 16 stopped at {conv.failure}"
    assert conv.truncated_at is None
    assert conv.entries_converted == conv.entries_total == 155
