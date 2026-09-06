"""An event the log never mentions is an event that did not happen.

Some cards choose another card without anyone deciding: Star Wars takes one out of the discard
pile, Five Year Plan discards one at random from the USSR hand and plays it if it is a US card.
Which card comes out changes everything after it, and the log records it -- but there is no
decision to answer, so the seed is searched for instead, and a search can come back with the
wrong card or with nothing.

Nothing downstream noticed. The wrong event fires, moves Influence the log never moved, and the
conversion reports whatever countries ended up different -- at turn 8 AR3 of replay 158 the US
plays Star Wars, which should take Grain Sales To Soviets out of the discard pile; the engine
took Blockade, which removes every US Influence from West Germany, and the entry was reported as
three countries disagreeing rather than as an event that never happened.

So the driver now records every card whose event the engine resolves while driving an entry, and
checks it against what the entry says. The comparison is deliberately generous: a card the entry
mentions anywhere -- played, revealed, returned, discarded, headlined, or fired -- is not an
event from nowhere. Only a card the log knows nothing about at all fails.
"""
import glob
import gzip
import json
import os
from typing import Dict, Set

import pytest

from tools.lib.corpus_paths import corpus_dir
from tools.lib.ts_replayer_convert import (_UNNAMED_EVENTS, _events_the_log_names,
                                           card_id, convert_game)
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = str(corpus_dir())
NORAD = 106



def _game(replay_id: int) -> Dict:
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return json.load(f)


def _entry(replay_id: int, turn: int, phase: str, player: str):
    for raw in _game(replay_id)["all_turns"]:
        e = parse_entry(raw)
        if (e.turn, e.phase, e.player) == (turn, phase, player):
            return e
    raise AssertionError(f"replay {replay_id} has no {player} {phase} of turn {turn}")


def test_the_entry_names_star_wars_and_what_it_took() -> None:
    """Replay 158 turn 8 AR3: Star Wars, then Grain Sales, then the card it drew."""
    e = _entry(158, 8, "AR3", "US")
    named = _events_the_log_names(e)
    assert card_id("Star Wars*") in named
    assert card_id("Grain Sales To Soviets") in named
    assert card_id("Cuban Missile Crisis*") in named, "revealed, so the log knows of it"
    assert card_id("Blockade*") not in named, "and it says nothing of Blockade"


def test_an_event_the_entry_never_names_would_be_reported() -> None:
    """The check is a set difference, and this is the difference it was written for.

    Replay 158's Star Wars entry is fixed now -- Grain Sales draws the Cuban Missile Crisis the
    log records -- so no game in the corpus trips the check any more. What is asserted here is
    the predicate itself: had Blockade been resolved on this entry, it would not have been
    covered by anything the entry names, and the check would have said so.
    """
    e = _entry(158, 8, "AR3", "US")
    named = _events_the_log_names(e)
    blockade = card_id("Blockade*")
    assert blockade is not None
    assert {blockade} - named - _UNNAMED_EVENTS == {blockade}, (
        "an event the entry never mentions survives the difference and fails the check")
    grain_sales = card_id("Grain Sales To Soviets")
    assert grain_sales is not None
    assert {grain_sales} - named - _UNNAMED_EVENTS == set(), (
        "and one the entry does mention does not")


def test_no_game_in_the_corpus_fires_an_event_its_log_never_names() -> None:
    """The check found six games. All six are fixed; it stays here to keep them fixed."""
    for replay_id in (158, 51, 52, 145, 300, 14, 100):
        conv = convert_game(_game(replay_id))
        if conv.failure is not None:
            assert conv.failure.kind != "event the log does not mention", (
                f"replay {replay_id}: {conv.failure}")


def test_norad_is_allowed_to_go_unnamed() -> None:
    """It prints the Influence it places and never a header of its own."""
    assert NORAD in _UNNAMED_EVENTS


def test_a_clean_game_names_everything_it_fires() -> None:
    for replay_id in (14, 100, 127, 144, 150):
        conv = convert_game(_game(replay_id))
        assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"


def test_the_naming_set_covers_every_way_an_entry_can_mention_a_card() -> None:
    """A headline names two cards, and neither is the entry's own card field."""
    e = _entry(150, 8, "Headline", "both")
    named: Set[int] = _events_the_log_names(e)
    assert card_id("The Reformer*") in named
    assert card_id("East European Unrest") in named
