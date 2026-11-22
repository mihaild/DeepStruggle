"""The page's action log says what each event of a click did to VP and DEFCON.

The second headline commit resolves both headlines in one engine step, and the log described a
click by comparing the state before and after it. At turn 8's headline of the workbench game
nuclear_test_ban, Nuclear Test Ban gave the US 1 VP (DEFCON 3) and Pershing II gave the USSR 1 VP:
the score did not move, and the log said nothing of either. The engine now records each event's
own change (GameState::last_event_tracks) and the log words them when the sum cannot.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest
import ts_engine as ts

from tests.web.js_runner import PUBLIC, run_ts

NUCLEAR_TEST_BAN = 34
PERSHING_II = 99
FIDEL = 8
OLYMPIC_GAMES = 20
CAPTURED_NAZI_SCIENTIST = 18
CHOOSE_BRANCH = 6
RED_SCARE = 31
DE_STALINIZATION = 33
EUROPE_SCORING = 2


def _at_headline(us_card: int, ussr_card: int, defcon: int, vp: int, europe_to: int = 0) -> str:
    """A saved position at turn 1's headline with these two cards; `europe_to` +1 / -1 gives that side 20 Influence
    (and the other none) in every European country."""
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    while state.current_phase == ts.Phase.SETUP:
        legal = [i for i, v in enumerate(ts.ActionMask.generate_flat_mask(state)) if v]
        ts.Engine.step_flat(state, legal[0])
    for player, card in ((ts.Player.US, us_card), (ts.Player.USSR, ussr_card)):
        loc = ts.hand_of(player)
        for c in range(1, 111):
            if state.get_card_location(c) == loc:
                state.set_card_location(c, ts.CardLocation.DISCARD_PILE)
        state.set_card_location(card, loc)
    state.defcon = defcon
    state.victory_points = vp
    if europe_to:
        for c in range(84):
            if ts.MapData.get_country_info(c)["region"] == 0:
                state.set_country(c, 20 if europe_to > 0 else 0, 20 if europe_to < 0 else 0)
    return state.to_save_json()


@pytest.fixture(scope="module")
def logged(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    out = tmp_path_factory.mktemp("event_tracks")
    cases = {
        "cancelling": {"save": _at_headline(NUCLEAR_TEST_BAN, PERSHING_II, 3, 18),
                       "us": NUCLEAR_TEST_BAN, "ussr": PERSHING_II},
        "alone": {"save": _at_headline(NUCLEAR_TEST_BAN, FIDEL, 4, 0),
                  "us": NUCLEAR_TEST_BAN, "ussr": FIDEL},
        # data/replays/cns.tslog.json: the USSR's answer to Olympic Games resolves the sponsor's
        # roll and then Captured Nazi Scientist, whose Space Race advance pays the USSR 2 VP.
        "olympics": {"save": _at_headline(OLYMPIC_GAMES, CAPTURED_NAZI_SCIENTIST, 5, 0),
                     "us": OLYMPIC_GAMES, "ussr": CAPTURED_NAZI_SCIENTIST,
                     "then": [[CHOOSE_BRANCH, 0, 4]]},
        # Red Scare/Purge (4 Ops) resolves before De-Stalinization (3), a USSR Event headlined by
        # the US, whose removals and placements the USSR then makes.
        "purge_first": {"save": _at_headline(DE_STALINIZATION, RED_SCARE, 5, 0),
                        "us": DE_STALINIZATION, "ussr": RED_SCARE, "finish": True},
        # Europe Scoring headlined by the side that controls Europe: Fidel (2 Ops) first, then the win by Europe
        # Control, which the engine's Event record books as 100 VP (E9) while the track moves to 20.
        "europe_us": {"save": _at_headline(EUROPE_SCORING, FIDEL, 5, 3, europe_to=1),
                      "us": EUROPE_SCORING, "ussr": FIDEL},
        "europe_ussr": {"save": _at_headline(NUCLEAR_TEST_BAN, EUROPE_SCORING, 5, -2, europe_to=-1),
                        "us": NUCLEAR_TEST_BAN, "ussr": EUROPE_SCORING},
    }
    spec = out / "cases.json"
    spec.write_text(json.dumps(cases))
    return run_ts("event_tracks.ts", str(out), PUBLIC, str(spec))


def test_two_headlines_that_cancel_are_both_logged(logged: Dict[str, Any]) -> None:
    details: List[str] = logged["cancelling"]["details"]
    assert "  • Nuclear Test Ban (#34): US +1 VP, DEFCON +2" in details, details
    assert "  • Pershing II Deployed (#99): USSR +1 VP" in details, details
    assert not any(d.startswith("  • Victory Points:") for d in details), "the score did not move"


def test_the_step_keeps_the_records(logged: Dict[str, Any]) -> None:
    tracks = logged["cancelling"]["event_tracks"]
    assert tracks[0] == []
    assert [(t["card_id"], t["vp_delta"], t["defcon_delta"]) for t in tracks[1]] == [
        (NUCLEAR_TEST_BAN, 1, 2), (PERSHING_II, -1, 0)]


def test_a_lone_event_is_left_to_the_track_lines(logged: Dict[str, Any]) -> None:
    """Fidel moves neither track, so Nuclear Test Ban's change is the whole change."""
    details: List[str] = logged["alone"]["details"]
    assert "  • Victory Points: 0 (Tie) -> +2 (US) (+2 VP)" in details, details
    assert not any("Nuclear Test Ban (#34):" in d for d in details), details


def test_a_click_reads_from_cause_to_effect(logged: Dict[str, Any]) -> None:
    """The dice, then what each Event did, then the board -- not the board before the roll."""
    entry = logged["olympics"]
    assert entry["text"] == "USSR participates in the Olympic Games (Olympic Games)"
    details: List[str] = entry["details"]
    roll = next(i for i, d in enumerate(details) if d.startswith("  🎲 Olympic Games Competition"))
    olympics = details.index("  • Olympic Games (#20): US +2 VP")
    scientist = details.index("  • Captured Nazi Scientist (#18): USSR +2 VP")
    space = details.index("  • Space Race: USSR 0 -> 1")
    moved = next(i for i, d in enumerate(details) if d.startswith("  • Card #18 (Captured Nazi Scientist) moved:"))
    assert roll < olympics < scientist < space < moved, details


def test_the_headline_entry_reads_in_the_order_the_headlines_resolved(logged: Dict[str, Any]) -> None:
    """Red Scare's effect, then De-Stalinization's moves under its name, then where the cards went."""
    entry = next(e for e in logged["purge_first"]["entries"] if e["title"].startswith("Headlines: "))
    lines: List[str] = entry["lines"]
    purge = lines.index("  • In effect: Red Scare (US)")
    label = lines.index("  ▸ De-Stalinization (#33) — USSR")
    moves = [i for i, line in enumerate(lines) if " Influence: USSR: " in line]
    removed = lines.index("  • De-Stalinization (#33) removed from the game")
    assert moves and purge < label < min(moves) and max(moves) < removed, lines


def test_a_merged_line_is_numbered_only_when_one_step_made_it(logged: Dict[str, Any]) -> None:
    """The commits are steps 1 and 2 and the USSR's answer step 3: the roll and the Space Race
    advance are step 3's, Captured Nazi Scientist's way out of the game spans steps 2 and 3."""
    entry = next(e for e in logged["olympics"]["entries"] if e["title"].startswith("Headlines: "))
    step = dict(zip(entry["lines"], entry["steps"]))
    assert entry["step"] == 2
    roll = next(line for line in entry["lines"] if line.startswith("  🎲 Olympic Games"))
    assert step[roll] == 3
    assert step["  • Space Race: USSR 0 -> 1"] == 3
    assert step["  • Captured Nazi Scientist (#18) removed from the game"] is None


def test_a_win_by_europe_control_is_said_as_that_win(logged: Dict[str, Any]) -> None:
    """In-game Europe Control is booked as 100 VP (E9, `effect_bits::EUROPE_CONTROL_VP`, the region reward's reading):
    the log says the win, beside the track's move to 20, never "+100 VP"."""
    def headline_lines(case: str) -> List[str]:
        # The click's own entry is the game's end; the headline entry holds what it did.
        return next(e for e in logged[case]["entries"] if e["title"].startswith("Headlines: "))["lines"]

    us = headline_lines("europe_us")
    assert "  • Europe Scoring (#2): US controls Europe and wins" in us, us
    assert "  • Victory Points: +3 (US) -> +20 (US) (+17 VP)" in us, us
    assert [(t["card_id"], t["vp_delta"]) for t in logged["europe_us"]["event_tracks"][-1]] == [(EUROPE_SCORING, 100)]
    ussr = headline_lines("europe_ussr")
    assert "  • Europe Scoring (#2): USSR controls Europe and wins" in ussr, ussr
    assert not any("100 VP" in d for d in us + ussr), us + ussr
