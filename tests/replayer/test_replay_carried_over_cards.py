"""The card a player carries into a turn, borrowed from the turn they played it in.

A turn's list holds the cards that became visible during it, and a player carries what they do
not play into the next turn -- one card normally, two when the China Card was among their plays,
since it sits outside the hand limit. So the list is short by exactly that, and the hand the
reconstruction dealt was short with it.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Dict, List, cast

import pytest
import ts_engine as ts

from tools.lib.corpus_paths import corpus_dir
from tools.lib.ts_replayer_convert import (
    _cards_carried_over,
    _played_by_anyone_up_to,
    _ran_out_of_cards,
    _skipped_a_round,
    _under_red_scare,
    _was_trapped,
    card_id,
    convert_game,
)

_CORPUS = str(corpus_dir())


def _game(replay_id: int) -> Dict[str, object]:
    path = os.path.join(_CORPUS, f"{replay_id}.json.gz")
    if not os.path.exists(path):
        pytest.skip("corpus not downloaded")
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _carry(replay_id: int, turn: int, side: str) -> List[int]:
    game = _game(replay_id)
    raws = cast(List[Dict[str, object]], game["all_turns"])
    hands = cast(Dict[str, Dict[str, List[str]]], game["hands"])
    held = [c for c in (card_id(nm) for nm in hands[str(turn)][side.lower()]) if c]
    other = "ussr" if side == "US" else "us"
    claimed = {c for c in (card_id(nm) for nm in hands[str(turn)][other]) if c}
    size = 8 if turn <= 3 else 9
    # A fresh state: it is read only where there is no next turn to borrow from, to ask what a
    # card is worth holding -- whether the US controls Iran, who has the China Card.
    state = ts.GameState()
    ts.Engine.init_game(state, 1)
    return _cards_carried_over(state, raws, hands, turn, side, held, size, claimed)


def test_a_card_already_played_cannot_have_been_held() -> None:
    """It is in the discard pile, and one played again later was drawn again after a reshuffle."""
    game = _game(100)
    raws = cast(List[Dict[str, object]], game["all_turns"])
    spent = _played_by_anyone_up_to(raws, 3)
    hands = cast(Dict[str, Dict[str, List[str]]], game["hands"])
    for turn in (1, 2, 3):
        for side in ("us", "ussr"):
            for nm in hands[str(turn)][side]:
                cid = card_id(nm)
                if cid:
                    assert cid in spent or cid == 6, nm
    for cid in _carry(100, 3, "US"):
        assert cid not in spent


def test_nothing_is_borrowed_where_the_log_says_the_hand_was_empty() -> None:
    """Five Year Plan reaches into the USSR hand and finds nothing: "USSR has no cards to
    discard". A card carried over would have been sitting there."""
    game = _game(101)
    raws = cast(List[Dict[str, object]], game["all_turns"])
    assert _ran_out_of_cards(raws, 5, "USSR")
    assert _carry(101, 5, "USSR") == []


def test_a_skip_under_a_trap_leaves_only_the_small_cards() -> None:
    """A trap takes a card of 2 Ops or more, so a player who *skipped* a round while held by
    one had nothing that big. At turn 5 of replay 114 the USSR is in Bear Trap and skips four
    action rounds -- and is under Red Scare/Purge, which takes an Ops off everything they play,
    so a 2 Ops card is no longer eligible either and 3 is the first that is.
    """
    game = _game(114)
    raws = cast(List[Dict[str, object]], game["all_turns"])
    assert _was_trapped(raws, 5, "USSR")
    assert _skipped_a_round(raws, 5, "USSR")
    assert _under_red_scare(raws, 5, "USSR")
    for cid in _carry(114, 5, "USSR"):
        assert int(ts.CardData.get_card_info(cid)["ops"]) <= 2


def test_playing_through_a_trap_says_nothing_about_the_hand() -> None:
    """A scoring card may always be played out of a trap on the last round, so a turn spent
    playing under one, without a skip, puts no ceiling on what was held. At turn 7 of replay
    237 the USSR lays Quagmire and the US plays Africa Scoring on the very next round."""
    raws = cast(List[Dict[str, object]], _game(237)["all_turns"])
    assert _was_trapped(raws, 7, "US")
    assert not _skipped_a_round(raws, 7, "US")
    conv = convert_game(_game(237))
    assert conv.failure is None, f"replay 237 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 116


def test_a_trap_the_opponent_lays_this_turn_counts_too() -> None:
    """Replay 237 turn 7: the USSR plays Quagmire at AR7 and the US plays their scoring card on
    the very next action round, which the engine will not allow with a bigger card in hand."""
    raws = cast(List[Dict[str, object]], _game(237)["all_turns"])
    assert _was_trapped(raws, 7, "US")


def test_no_scoring_card_is_ever_carried() -> None:
    """Holding one at a turn's end loses the game outright."""
    for replay_id, turn, side in ((100, 4, "US"), (105, 6, "USSR"), (111, 5, "US")):
        for cid in _carry(replay_id, turn, side):
            assert not bool(ts.CardData.get_card_info(cid)["is_scoring"])


def test_the_corpus_still_converts_with_fuller_hands() -> None:
    for replay_id in (100, 105, 111, 114, 237, 78, 96, 73):
        conv = convert_game(_game(replay_id))
        assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
        assert conv.cards_carried_over > 0
