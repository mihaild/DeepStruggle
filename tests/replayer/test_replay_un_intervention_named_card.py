"""The card UN Intervention names belongs to the hand it was named from.

A turn's hand list in a ts-replayer log holds the cards that became *visible* during the
turn, not the hand as it was dealt. A card spent only by being named through another card
can therefore be missing from the list altogether, and padding the hand out to nine picks
something else -- so the player holds nothing for the naming card to name.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Dict, List

import pytest

from tools.lib.ts_replayer_convert import (
    _cards_named_through_un_intervention,
    card_id,
    convert_game,
)

_CORPUS = "/workspace/data/datasets/ts_replayer"
_OAS_FOUNDED = 70
_UN_INTERVENTION = 32


def _raw(turn: int, player: str, card: str, text: str) -> Dict[str, object]:
    return {"num": str(turn), "player": player, "phase": "AR7",
            "card": card, "text": text}


def test_the_named_card_is_added_to_the_naming_players_hand() -> None:
    raws: List[Dict[str, object]] = [
        _raw(5, "USSR", "UN Intervention",
             "Turn 5, USSR AR7: UN Intervention: \n"
             "Event: UN Intervention\n"
             "USSR plays OAS Founded\n"
             "Place Influence (1 Ops):\n"
             "USSR +1 in Panama [0][1]\n")
    ]
    hands: Dict[str, List[int]] = {"US": [], "USSR": [_UN_INTERVENTION]}
    assert _cards_named_through_un_intervention(raws, 5, hands) == 1
    assert _OAS_FOUNDED in hands["USSR"]


def test_a_card_already_in_hand_is_left_alone() -> None:
    raws: List[Dict[str, object]] = [
        _raw(5, "USSR", "UN Intervention",
             "Turn 5, USSR AR7: UN Intervention: \n"
             "Event: UN Intervention\n"
             "USSR plays OAS Founded\n")
    ]
    hands: Dict[str, List[int]] = {"US": [], "USSR": [_UN_INTERVENTION, _OAS_FOUNDED]}
    assert _cards_named_through_un_intervention(raws, 5, hands) == 0
    assert hands["USSR"].count(_OAS_FOUNDED) == 1


def test_it_is_taken_from_the_other_hand_when_the_list_misfiled_it() -> None:
    """The entry names the player who played it, and an entry outranks a summary."""
    raws: List[Dict[str, object]] = [
        _raw(5, "USSR", "UN Intervention",
             "Turn 5, USSR AR7: UN Intervention: \n"
             "Event: UN Intervention\n"
             "USSR plays OAS Founded\n")
    ]
    hands: Dict[str, List[int]] = {"US": [_OAS_FOUNDED], "USSR": [_UN_INTERVENTION]}
    assert _cards_named_through_un_intervention(raws, 5, hands) == 1
    assert hands["USSR"] == [_UN_INTERVENTION, _OAS_FOUNDED]
    assert hands["US"] == []


def test_only_un_intervention_moves_cards_this_way() -> None:
    """"X plays Y" is printed by other cards too, where Y came from somewhere else."""
    raws: List[Dict[str, object]] = [
        _raw(5, "USSR", "Missile Envy",
             "Turn 5, USSR AR7: Missile Envy: \n"
             "USSR plays OAS Founded\n")
    ]
    hands: Dict[str, List[int]] = {"US": [], "USSR": []}
    assert _cards_named_through_un_intervention(raws, 5, hands) == 0
    assert hands["USSR"] == []


def test_replay_269_converts_end_to_end() -> None:
    """Turn 5 AR7: UN Intervention naming OAS Founded, its one Op spent on Panama."""
    path = os.path.join(_CORPUS, "269.json.gz")
    if not os.path.exists(path):
        pytest.skip("corpus not downloaded")
    with gzip.open(path, "rt") as f:
        game = json.load(f)
    listed = game["hands"]["5"]["ussr"]
    assert _OAS_FOUNDED not in [card_id(n) for n in listed], (
        "the premise: the turn's list does not mention the card at all")
    conv = convert_game(game)
    assert conv.failure is None, f"replay 269 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 84
