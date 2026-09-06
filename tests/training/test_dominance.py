"""Dominance rules: pairs where one option is weakly better in every position.

These are the only local-decision checks in the repo with a position-independent right
answer, so their exceptions have to stay exact. Each exception below exists because the
dominance argument genuinely fails for it, and a future edit that quietly drops one would
turn a defensible metric into a misleading one.
"""

import pytest
import ts_engine as ts

from ai.eval.dominance import (CHINA_CARD, FIVE_YEAR_PLAN, discard_dominance_pairs,
                               hand_cards, space_dominance_alternatives,
                               space_dominance_outcome)


def _state() -> ts.GameState:
    st = ts.GameState()
    ts.Engine.init_game(st, 4242)
    return st


def _card(cid: int):
    return ts.CardData.get_card_info(cid)


def _find(side: str, ops: int, one_time: bool = False, exclude=()):
    for c in range(1, 111):
        i = _card(c)
        if c in exclude or c in (FIVE_YEAR_PLAN, CHINA_CARD):
            continue
        if (str(i["side"]) == side and int(i["ops"]) == ops
                and bool(i["one_time"]) == one_time and not i["is_scoring"]):
            return c
    return None


# -- discard pairs -----------------------------------------------------------------------

def test_opponent_recurring_event_dominates_own_at_equal_ops() -> None:
    st = _state()
    ussr = _find("USSR", 2)
    us = _find("US", 2)
    assert ussr and us, "fixture needs a 2-ops card on each side"
    pairs = discard_dominance_pairs(st, ts.Player.US, {10: ussr, 20: us})
    assert (10, 20, 2) in pairs, "opponent recurring event must dominate an own card"
    assert not any(p[0] == 20 for p in pairs), "own card must never be the dominant side"


def test_unequal_ops_is_never_a_pair() -> None:
    """The whole argument rests on equal Ops; without that there is no dominance."""
    st = _state()
    ussr2, us3 = _find("USSR", 2), _find("US", 3)
    assert ussr2 and us3
    assert discard_dominance_pairs(st, ts.Player.US, {10: ussr2, 20: us3}) == []


def test_five_year_plan_is_never_the_dominant_card() -> None:
    """#5 is the one recurring event whose firing can help its non-owner."""
    assert str(_card(FIVE_YEAR_PLAN)["side"]) == "US"
    assert not _card(FIVE_YEAR_PLAN)["one_time"]
    st = _state()
    ops = int(_card(FIVE_YEAR_PLAN)["ops"])
    ussr_same = _find("USSR", ops)
    assert ussr_same, "fixture needs a USSR card at Five Year Plan's ops"
    # USSR to move: Five Year Plan is the opponent's recurring event, but must not dominate.
    pairs = discard_dominance_pairs(st, ts.Player.USSR,
                                    {10: FIVE_YEAR_PLAN, 20: ussr_same})
    assert not any(p[0] == 10 for p in pairs), "Five Year Plan must be excluded as dominant"


def test_one_time_events_are_excluded_from_the_dominant_side() -> None:
    """Discarding a starred card removes it permanently -- a different argument, left out."""
    st = _state()
    starred = _find("USSR", 2, one_time=True)
    own = _find("US", 2)
    if starred is None or own is None:
        pytest.skip("no 2-ops starred USSR card in this deck")
    pairs = discard_dominance_pairs(st, ts.Player.US, {10: starred, 20: own})
    assert not any(p[0] == 10 for p in pairs)


def test_china_card_is_never_in_a_pair() -> None:
    st = _state()
    other = _find("USSR", int(_card(CHINA_CARD)["ops"]))
    if other is None:
        pytest.skip("no USSR card at the China Card's ops value")
    pairs = discard_dominance_pairs(st, ts.Player.US, {10: other, 20: CHINA_CARD})
    assert all(CHINA_CARD not in (p[0], p[1]) for p in pairs)


# -- space race --------------------------------------------------------------------------

def _give(st: ts.GameState, player: ts.Player, cards) -> None:
    loc = ts.CardLocation.HAND_US if player == ts.Player.US else ts.CardLocation.HAND_USSR
    for c in cards:
        st.set_card_location(c, loc)


def test_spacing_own_card_while_holding_opponent_equal_ops_is_a_violation() -> None:
    st = _state()
    own, opp = _find("US", 2), _find("USSR", 2)
    assert own and opp
    _give(st, ts.Player.US, [own, opp])
    assert opp in hand_cards(st, ts.Player.US)
    assert space_dominance_alternatives(st, ts.Player.US, own) == [opp]
    assert space_dominance_outcome(st, ts.Player.US, own) is False


def test_spacing_the_opponent_card_is_correct() -> None:
    st = _state()
    own, opp = _find("US", 2), _find("USSR", 2)
    assert own and opp
    _give(st, ts.Player.US, [own, opp])
    assert space_dominance_outcome(st, ts.Player.US, opp) is True


def test_no_opportunity_when_no_equal_ops_alternative_is_held() -> None:
    """Reporting violations over all space plays would understate the error; these are the
    plays that must be excluded from the denominator."""
    st = _state()
    own = _find("US", 2)
    opp3 = _find("USSR", 3)
    assert own and opp3
    _give(st, ts.Player.US, [own, opp3])
    assert space_dominance_outcome(st, ts.Player.US, own) is None
