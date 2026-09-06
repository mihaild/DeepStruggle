"""Five Year Plan's discard is drawn where no decision is asked, and the log names it.

The card discards at random from the USSR hand and, if what comes out is a US card, plays its
event. Which card comes out therefore decides the whole entry, and the log records it -- but it
is drawn inside the event, so there is no action to answer it with. The seed is searched for
instead.

Two things kept that search from ever running on the step that mattered:

* The step is not always a decision. Played for Ops first and event second, the event fires the
  moment the Ops run out, and the last of three realignments runs out inside its own chance
  node. At turn 3 AR5 of replay 107 the USSR realigns Angola three times with Five Year Plan
  and the discard was taken by `_drain`, which never consulted the queue.
* The search accepted any seed leaving the named card out of a hand, which is already true
  before the card is drawn. The first step offered -- the one that plays Five Year Plan --
  matched and cleared the queue.

The log drew Nasser, a USSR card that does nothing. The reconstruction drew a US one and
played its event, and the VP it paid stayed on the board until the Special Relationship at
turn 3 AR5 asserted against it.
"""
import glob
import gzip
import json
import os
from typing import Dict, List

import pytest
import ts_engine as ts

from tools.lib.corpus_paths import corpus_dir
from tools.lib.ts_replayer_convert import _force_random_discard, convert_game
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = str(corpus_dir())
FIVE_YEAR_PLAN = 5
NASSER = 15



def _game(replay_id: int) -> Dict:
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return json.load(f)


def _raws(replay_id: int) -> List[Dict]:
    return _game(replay_id)["all_turns"]


def test_the_log_names_the_card_five_year_plan_discarded() -> None:
    """Replay 107 turn 3 AR5: "Event: Five Year Plan / USSR discards Nasser*"."""
    entry = parse_entry(_raws(107)[35])
    assert entry.card is not None and entry.card.startswith("Five Year Plan")
    assert ("USSR", "Nasser*") in entry.discards


def test_a_card_the_ussr_does_not_hold_cannot_be_the_one_discarded() -> None:
    """The step that plays Five Year Plan discards nothing, and must not claim the forcing.

    Accepting "the card is not in a hand" made that step a match, so the queue was cleared
    before the draw it was meant to steer ever happened.
    """
    state = ts.GameState()
    ts.Engine.init_game(state, 1234)
    for c in range(1, 111):
        if state.get_card_location(c) == ts.CardLocation.HAND_USSR:
            state.set_card_location(c, ts.CardLocation.DRAW_DECK)
    assert _force_random_discard(state, [NASSER], action=0) is False


def test_five_year_plan_discards_what_the_log_says_when_the_event_follows_the_ops() -> None:
    """The discard has to be steered on a chance-node step, not only on a decision."""
    conv = convert_game(_game(107))
    assert conv.failure is None, f"replay 107 stopped at {conv.failure}"


@pytest.mark.parametrize("replay_id", [107, 119])
def test_the_five_year_plan_games_convert_end_to_end(replay_id: int) -> None:
    conv = convert_game(_game(replay_id))
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"


def test_a_score_past_the_end_of_the_track_is_the_replayers_arithmetic() -> None:
    """Replay 109 turn 5 AR7: the log reads "Score is USSR 21".

    The VP track runs from 20 to -20 and the game ends the moment either end is reached, so
    the USSR had already won at 20. The engine clamps and stops; the expected value has to be
    clamped with it or the last entry of the game can never match.
    """
    raws = _raws(109)
    last = parse_entry(raws[-1])
    assert last.vp_gains and last.vp_gains[-1][3] == 21
    conv = convert_game(_game(109))
    assert conv.failure is None, f"replay 109 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total
