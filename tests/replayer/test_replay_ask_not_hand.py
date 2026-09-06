"""Reconstructing the hand a player discarded from.

"Ask Not What Your Country Can Do For You" discards any number of cards and draws that many
back, so a turn's list holds both the hand as dealt and the replacements, with nothing to say
which is which. Everything the log accounts for is settled first -- what was discarded here,
what was played earlier in the turn -- and the remainder is ordered: the player's own cards and
the neutrals first, then the opponent's by descending Ops.
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
    _mid_turn_acquisitions,
    _sort_key_for_keeping,
    card_id,
    convert_game,
)
from tools.lib.ts_replayer_parse import parse_entry

_CORPUS = str(corpus_dir())
_U2_INCIDENT = 60
_WE_WILL_BURY_YOU = 50
_ALLIANCE_FOR_PROGRESS = 65


def _game(replay_id: int) -> Dict[str, object]:
    path = os.path.join(_CORPUS, f"{replay_id}.json.gz")
    if not os.path.exists(path):
        pytest.skip("corpus not downloaded")
    with gzip.open(path, "rt") as f:
        return json.load(f)


def test_own_and_neutral_cards_are_kept_before_the_opponents() -> None:
    """A hand is dealt from a deck both sides draw from, so holding the opponent's events is
    normal -- holding the ones you did not spend or discard is what to assume least of."""
    us_card = _ALLIANCE_FOR_PROGRESS               # US, 3 Ops
    ussr_card = _U2_INCIDENT                       # USSR, 3 Ops
    assert str(ts.CardData.get_card_info(us_card)["side"]) == "US"
    assert str(ts.CardData.get_card_info(ussr_card)["side"]) == "USSR"
    assert _sort_key_for_keeping(us_card, "US") < _sort_key_for_keeping(ussr_card, "US")
    assert _sort_key_for_keeping(ussr_card, "USSR") < _sort_key_for_keeping(us_card, "USSR")


def test_higher_ops_are_kept_first_among_the_opponents_cards() -> None:
    """They are what a player keeps to spend, so they are likelier held than drawn."""
    assert (_sort_key_for_keeping(_WE_WILL_BURY_YOU, "US")      # USSR, 4 Ops
            < _sort_key_for_keeping(_U2_INCIDENT, "US"))        # USSR, 3 Ops


def test_the_discarded_cards_are_all_taken_as_dealt() -> None:
    """Replay 96 turn 7: six cards discarded to Ask Not, and every one of them was in hand."""
    raws = cast(List[Dict[str, object]], _game(96)["all_turns"])
    entry = next(parse_entry(r) for r in raws
                 if parse_entry(r).turn == 7 and parse_entry(r).phase == "Headline")
    discarded = [card_id(nm) for side, nm in entry.discards if side == "US"]
    assert len(discarded) == 6
    hands = cast(Dict[str, Dict[str, List[str]]], _game(96)["hands"])
    held = [c for c in (card_id(nm) for nm in hands["7"]["us"]) if c]
    acquired = _mid_turn_acquisitions(raws, 7, "US", held)
    assert not (set(discarded) & set(acquired)), "a card discarded here was held, not drawn"


def test_replay_96_keeps_the_card_missile_envy_takes() -> None:
    """Ask Not resolves first -- 3 Ops against Missile Envy's 2 -- and discards six. Its
    seventh prompt was answered from a queue holding U2 Incident, the card Missile Envy was
    about to take for the USSR's 1 VP, and the hand it read was one card short besides."""
    conv = convert_game(_game(96))
    assert conv.failure is None, f"replay 96 stopped at {conv.failure}"
    assert conv.entries_converted == 99


@pytest.mark.parametrize("replay_id", [64, 114, 119, 141, 262])
def test_the_other_hand_reading_games_are_unaffected(replay_id: int) -> None:
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
