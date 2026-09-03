"""One logged entry can hold operations belonging to both players.

Playing an opponent's card for Operations resolves two things: the event, which is the card
owner's, and the Ops, which are the player's. When both are coups the log prints two "Coup (N
Ops):" headers under one entry, and nothing but the section they sit in says whose each is.
Getting that wrong is not a near miss -- the other player's target is usually not even legal,
since couping a country needs opposing Influence in it.

These run the converter over the real corpus, because the failure is in how the driver
sequences decisions against the engine and no hand-built entry would exercise that.
"""
import glob
import gzip
import json
import os
from typing import Dict, Optional

import pytest

from tools.lib.ts_replayer_convert import convert_game

CORPUS = "/workspace/data/datasets/ts_replayer"


def _convert(replay_id: int):
    path = os.path.join(CORPUS, f"{replay_id}.json.gz")
    with gzip.open(path, "rt") as f:
        return convert_game(json.load(f))


def _failed_at(conv) -> Optional[str]:
    """Where conversion stopped, as "T<turn> <phase>", or None if it finished."""
    if conv.failure is None:
        return None
    return f"T{conv.failure.turn} {conv.failure.phase}"


pytestmark = pytest.mark.skipif(
    not glob.glob(os.path.join(CORPUS, "*.json.gz")),
    reason="ts-replayer corpus not downloaded")


def test_ortega_free_coup_and_the_players_own_coup_are_kept_apart() -> None:
    """Replay 105 turn 9 AR2: the US plays Ortega Elected in Nicaragua for Ops.

    The USSR takes the event's free coup against Costa Rica -- adjacent to Nicaragua, and
    holding US Influence -- and the US then coups Saharan States with the card's own 2 Ops.
    Driving both from one flat target queue spent the USSR's coup first and then aimed the
    US's at Costa Rica, which has no USSR Influence for the US to coup.
    """
    conv = _convert(105)
    assert _failed_at(conv) != "T9 AR2", (
        f"the two coups belong to different players: {conv.failure}")


def test_ches_free_coups_use_the_logged_die() -> None:
    """Replay 113 turn 6 AR7: the US plays Che for a realignment, the USSR gets two coups.

    Their removed Influence is the result of a die, not a target anyone chose. Treated as
    event placements it never reached the outcome search, and the engine rolled freely --
    3 and 1 against the human's 2 and 2.
    """
    conv = _convert(113)
    assert _failed_at(conv) != "T6 AR7", (
        f"Che's free coups must reproduce the logged rolls: {conv.failure}")


def test_replay_105_now_converts_end_to_end() -> None:
    """Ortega was the last thing standing between this game and a complete conversion."""
    conv = _convert(105)
    assert conv.failure is None, f"replay 105 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 143


@pytest.mark.parametrize("replay_id,expect_at_least", [(105, 143), (113, 86)])
def test_no_earlier_entry_is_traded_away(replay_id: int, expect_at_least: int) -> None:
    """A guard on the count, so a fix that buys these entries with earlier ones is caught."""
    conv = _convert(replay_id)
    assert conv.entries_converted >= expect_at_least, (
        f"replay {replay_id} converted {conv.entries_converted} entries, "
        f"expected at least {expect_at_least}: {conv.failure}")


def test_star_wars_pick_from_the_discard_pile_is_read_from_the_log() -> None:
    """Replay 100 turn 10 headline: Star Wars takes a card out of the discard pile.

    The card is played as its event, so the log prints it on an "Event:" line of its own --
    the entry does say which of the pile was taken. The driver only ever answered a card
    request from the discard or reveal lines, so this one had no answer at all.
    """
    conv = _convert(100)
    assert conv.failure is None, f"replay 100 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 144


def test_a_game_the_log_ends_at_defcon_1_is_allowed_to_end() -> None:
    """Replay 104 turn 9 AR7: the USSR plays Star Wars and loses the game to it.

    The US takes How I Learned To Stop Worrying out of the discard pile and sets DEFCON to 1.
    Thermonuclear war costs the phasing player the game, and the phasing player is the USSR.
    Branch selection penalised every branch that ends the game -- a sound default, since
    DEFCON 1 is otherwise never the tie-break to take -- which put the real branch out of
    reach. The entry records defcon 1, so the log settles it.
    """
    conv = _convert(104)
    assert conv.failure is None, f"replay 104 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 128


def test_the_defcon_1_penalty_still_applies_where_the_log_played_on() -> None:
    """Replay 101 turn 4's headline is the game that motivated the penalty."""
    conv = _convert(101)
    assert conv.failure is None, f"replay 101 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 90


def test_an_empty_ops_header_is_a_decline_not_a_skipped_section() -> None:
    """Replay 123 turn 8 AR2: the US plays Ortega, the USSR declines the free coup.

    The log prints "Coup (1 Ops):" with nothing under it, and the US then places 2 Influence
    of its own. Reading the empty header as a section to skip answered the USSR's coup with a
    target from the US's placements -- couping Nigeria, which at DEFCON 2 is thermonuclear
    war and lost the USSR the game outright.
    """
    conv = _convert(123)
    assert _failed_at(conv) != "T8 AR2", (
        f"the declined coup must be answered as a decline: {conv.failure}")
