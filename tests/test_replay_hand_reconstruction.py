"""A turn's card list is not the hand that was dealt, and the difference is recoverable.

The log names the cards that became *visible* during a turn. That set differs from the dealt
hand at both ends: a card carried over to the next turn is never revealed and so is missing,
and a card picked up part way through the turn is present though it was never dealt.

Both directions matter. A hand one card too large gives Missile Envy the wrong card to take and
Blockade a discard the player did not have; a hand one card too small is a position the rules
cannot produce, and 47% of the corpus's lists are exactly that.
"""
import glob
import gzip
import json
import os

import pytest
import ts_engine as ts

from tools.lib.ts_replayer_convert import (_mid_turn_acquisitions, _pad_hand,
                                           _sort_key_for_keeping, convert_game)

CORPUS = "/workspace/data/datasets/ts_replayer"
SALT_NEGOTIATIONS = 43
ASK_NOT = 77
RED_SCARE_PURGE = 31
DE_STALINIZATION = 33
DUCK_AND_COVER = 4

pytestmark = pytest.mark.skipif(
    not glob.glob(os.path.join(CORPUS, "*.json.gz")),
    reason="ts-replayer corpus not downloaded")


def _raw(num, player, phase, card, text):
    return {"num": str(num), "player": player, "phase": phase, "card": card, "text": text}


def test_a_card_reclaimed_by_salt_was_not_dealt() -> None:
    """SALT Negotiations takes a card out of the discard pile, so it cannot have been dealt."""
    raws = [
        _raw(5, "USSR", "AR1", "Duck and Cover", "Turn 5, USSR AR1: Duck and Cover: ...\n"),
        _raw(5, "US", "AR7", "SALT Negotiations*",
             "Turn 5, US AR7: SALT Negotiations*: Event: SALT Negotiations*\n"
             "US reveals Red Scare/Purge\n"),
    ]
    acquired = _mid_turn_acquisitions(raws, 5, "US", [RED_SCARE_PURGE, DUCK_AND_COVER])
    assert acquired == {RED_SCARE_PURGE: 1}, "reclaimed at the entry that reclaims it, not before"


def test_the_reclaim_is_credited_to_the_side_that_made_it() -> None:
    raws = [_raw(5, "US", "AR7", "SALT Negotiations*",
                 "Turn 5, US AR7: SALT Negotiations*: Event: SALT Negotiations*\n"
                 "US reveals Red Scare/Purge\n")]
    assert _mid_turn_acquisitions(raws, 5, "USSR", [RED_SCARE_PURGE]) == {}


def test_ask_not_draws_as_many_replacements_as_it_discards() -> None:
    """What was discarded was dealt; the replacements were not, and there are as many."""
    raws = [
        _raw(6, "USSR", "AR7", '"Ask Not What Your Country..."*',
             'Turn 6, USSR AR7: "Ask Not What Your Country..."*: Place Influence (3 Ops):\n'
             'USSR +2 in Laos/Cambodia [1][2]\n\n'
             'Event: "Ask Not What Your Country..."*\n'
             'US discards De-Stalinization*\n'),
    ]
    held = [DE_STALINIZATION, RED_SCARE_PURGE, DUCK_AND_COVER]
    acquired = _mid_turn_acquisitions(raws, 6, "US", held)
    assert DE_STALINIZATION not in acquired, "a discarded card was in the dealt hand"
    assert len(acquired) == 1, "one discard means exactly one replacement"


def test_the_weakest_card_is_taken_as_the_replacement() -> None:
    """Nothing records which card was drawn, so the hand is ordered best-first and the tail
    of that order is assumed to be the draw."""
    raws = [
        _raw(6, "USSR", "AR7", '"Ask Not What Your Country..."*',
             'Event: "Ask Not What Your Country..."*\nUS discards De-Stalinization*\n'),
    ]
    acquired = _mid_turn_acquisitions(
        raws, 6, "US", [DE_STALINIZATION, RED_SCARE_PURGE, DUCK_AND_COVER])
    # Red Scare/Purge is 4 Ops, Duck and Cover 3, so Duck and Cover is the weaker of the two.
    assert set(acquired) == {DUCK_AND_COVER}


def test_the_keeping_order_is_ops_then_side() -> None:
    by_ops = sorted([DUCK_AND_COVER, RED_SCARE_PURGE], key=_sort_key_for_keeping)
    assert by_ops == [RED_SCARE_PURGE, DUCK_AND_COVER], "higher Ops is kept first"

    same_ops = [c for c in range(1, 111)
                if int(ts.CardData.get_card_info(c)["ops"]) == 3
                and not ts.CardData.get_card_info(c)["is_scoring"]]
    ordered = sorted(same_ops, key=_sort_key_for_keeping)
    sides = [str(ts.CardData.get_card_info(c)["side"]) for c in ordered]
    ussr_seen = False
    for side in sides:
        if side == "USSR":
            ussr_seen = True
        elif ussr_seen:
            pytest.fail("US and neutral cards come before USSR ones at the same Ops")


def test_a_hand_is_topped_up_to_the_size_actually_dealt() -> None:
    """A player who carries a card over never reveals it, so the list runs short."""
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    held = [c for c in range(1, 111)
            if state.get_card_location(c) == ts.CardLocation.HAND_US][:8]
    padded = _pad_hand(state, held, 9, None, set(held))
    assert len(padded) == 9
    assert not any(ts.CardData.get_card_info(c)["is_scoring"] for c in padded), (
        "a scoring card held at the end of a turn loses the game, so it is never invented")


def test_replay_64_still_converts_end_to_end() -> None:
    with gzip.open(os.path.join(CORPUS, "64.json.gz"), "rt") as f:
        conv = convert_game(json.load(f))
    assert conv.failure is None, f"replay 64 stopped at {conv.failure}"
