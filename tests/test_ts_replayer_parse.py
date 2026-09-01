"""Parsing human games from the ts-replayer log format.

The logs are redundant on purpose -- every influence line carries the resulting [US][USSR]
counts, and each entry separately carries the full board, score and DEFCON -- so a parse can be
checked against the log's own arithmetic. These tests pin the grammar and the checks; without
them a silently dropped state change would poison a training set with no visible symptom.
"""

import glob
import gzip
import json

import pytest

from tools.lib.ts_replayer_parse import (country_id, initial_board, parse_entry,
                                         verify_game)

CORPUS = sorted(glob.glob("/workspace/data/datasets/ts_replayer/*.json.gz"))


def _entry(text: str, **kw):
    raw = {"num": kw.pop("num", "3"), "player": kw.pop("player", "US"),
           "phase": kw.pop("phase", "AR2"), "card": kw.pop("card", None),
           "text": text}
    raw.update(kw)
    return parse_entry(raw)


def test_country_names_map_including_the_two_that_differ() -> None:
    assert country_id("France") is not None
    assert country_id("west_germany") == country_id("West Germany")
    # The log writes these differently from the engine.
    assert country_id("UK") == country_id("United Kingdom")
    assert country_id("Dominican Republic") == country_id("Dominican Rep")


def test_influence_line_carries_delta_and_resulting_counts() -> None:
    e = _entry("Place Influence (3 Ops):\nUSSR +2 in Thailand [0][2]\n"
               "USSR +1 in Laos/Cambodia [0][1]")
    assert e.mode == "influence" and e.ops == 3
    assert len(e.influence) == 2
    side, delta, cid, us, ussr = e.influence[0]
    assert (side, delta, us, ussr) == ("USSR", 2, 0, 2)
    assert cid == country_id("Thailand")


def test_negative_influence_is_parsed() -> None:
    e = _entry("USSR -1 in Poland [0][4]")
    assert e.influence[0][1] == -1


def test_coup_target_roll_and_outcome() -> None:
    e = _entry("Coup (2 Ops):\nTarget: Panama\nSUCCESS: 4 [ + 3 - 2x2 = 3 ]")
    assert e.mode == "coup" and e.ops == 2
    assert e.coup_target == country_id("Panama")
    assert e.coup_roll == 4 and e.coup_success is True


def test_realignment_and_war_rolls_are_captured() -> None:
    e = _entry("US rolls 5 (+1) = 6\nDie roll: 2 -- Success! (Needed 4 or less)")
    assert e.realign_rolls == [("US", 5, 1, 6)]
    assert e.die_rolls == [(2, True, 4)]


def test_trap_rolls_are_captured() -> None:
    """Quagmire/Bear Trap escape rolls -- directly relevant to the dominance work."""
    e = _entry("Trap Roll: 4 <= 4 -- Trap Escaped")
    assert e.trap_rolls == [(4, 4, True)]
    e2 = _entry("Trap Roll: 6 > 4 -- Trap Remains in Effect")
    assert e2.trap_rolls == [(6, 4, False)]


def test_prefixed_lines_are_not_silently_dropped() -> None:
    """Regression: these carry a "Turn 9, Cleanup: " prefix.

    Anchoring the patterns at ^ made them miss, and the ignore rule then swallowed them --
    a state change that disappears with nothing to show for it. Roughly 1,900 DEFCON lines
    and 600 VP lines across the corpus were being lost this way.
    """
    e = _entry("Turn 9, Cleanup: USSR gains 2 VP. Score is USSR 5.", score=-5)
    assert e.vp_gains == [("USSR", 2, "USSR", 5)]
    assert e.unparsed == []

    e2 = _entry("Turn 7, US AR4: Brush War: USSR gains 2 VP. Score is USSR 4.", score=-4)
    assert e2.vp_gains and e2.unparsed == []


def test_score_even_and_no_award_forms() -> None:
    assert _entry("US gains 3 VP. Score is even.").vp_gains == [("US", 3, "even", 0)]
    assert _entry("No VP awarded. Score is USSR 8.").score_assertions == [-8]
    assert _entry("No VP awarded. Score is even.").score_assertions == [0]


def test_defcon_milops_and_space_are_captured() -> None:
    e = _entry("DEFCON degrades to 4\nUSSR Military Ops to 3\n"
               "US advances to 1 in the Space Race.")
    assert e.defcon_changes == [("degrades", 4)]
    assert e.milops == [("USSR", 3)]
    assert e.space == [("US", 1)]


def test_initial_board_has_the_fixed_setup_influence() -> None:
    """Starting a running board at zero makes every pre-placed country look mismatched."""
    board = initial_board()
    assert sum(us for us, _ in board.values()) > 0
    assert sum(ussr for _, ussr in board.values()) > 0


@pytest.mark.skipif(not CORPUS, reason="ts-replayer corpus not downloaded")
def test_real_games_reconcile_on_influence_board_and_defcon() -> None:
    """The checks that matter for reconstructing placements must be exact, not approximate."""
    infl = board = defcon = 0
    infl_bad = board_bad = defcon_bad = 0
    for path in CORPUS[:25]:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            game = json.load(f)
        chk, _ = verify_game(game)
        infl += chk.influence_lines
        infl_bad += chk.influence_mismatch
        board += chk.board_checked
        board_bad += chk.board_mismatch
        defcon += chk.defcon_checked
        defcon_bad += chk.defcon_mismatch

    assert infl > 1000 and board > 10000, "corpus slice too small to be meaningful"
    assert infl_bad == 0, f"{infl_bad}/{infl} influence lines disagree with the running board"
    assert board_bad == 0, f"{board_bad}/{board} board cells disagree"
    assert defcon_bad == 0, f"{defcon_bad}/{defcon} DEFCON transitions disagree"
