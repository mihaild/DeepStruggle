"""A trap discard says something about the rest of the hand, and the solver now uses it.

A player trapped by Quagmire or Bear Trap must discard, and discarding the opponent's recurring
event is never worse than discarding their own or a neutral card of the same printed Ops. Over the
corpus humans respect this without exception -- 0 of 87 trap discards where the log records both
cards took the dominated side (research/experiments.md §9.5). So a discard of an own or neutral
card is evidence that no opponent recurring event of that Ops was in the hand, and the hand solver
takes it as a preference.

A preference, not a rule: it is a statement about how people play, not about what the rules allow,
so the hard constraints -- everything the log actually establishes -- still outrank it.
"""

from __future__ import annotations

import pytest

import ts_engine as ts

from tools.lib.ts_replayer_hands import (_dominance_excluded,
                                         _dominated_discard_ops)

DUCK_AND_COVER = 32     # US, 3 ops, recurring
FIVE_YEAR_PLAN = 5
CHINA_CARD = 6


class _Facts:
    """Just enough of GameFacts for the helper under test."""

    def __init__(self, discard: dict) -> None:
        self.trap_discard = discard


def _ops(card: int) -> int:
    return int(ts.CardData.get_card_info(card)["ops"])


def _side(card: int) -> str:
    return str(ts.CardData.get_card_info(card)["side"])


def _own_or_neutral_for(side: str) -> int:
    opponent = "USSR" if side == "US" else "US"
    for cid in range(1, 111):
        info = ts.CardData.get_card_info(cid)
        if (str(info["side"]) != opponent and not info["is_scoring"]
                and not info["one_time"] and cid not in (FIVE_YEAR_PLAN, CHINA_CARD)):
            return cid
    raise AssertionError("no own-or-neutral recurring card found")


def _opponent_card_for(side: str) -> int:
    opponent = "USSR" if side == "US" else "US"
    for cid in range(1, 111):
        info = ts.CardData.get_card_info(cid)
        if (str(info["side"]) == opponent and not info["is_scoring"]
                and not info["one_time"] and cid not in (FIVE_YEAR_PLAN, CHINA_CARD)):
            return cid
    raise AssertionError("no opponent recurring card found")


def test_discarding_own_or_neutral_constrains_that_ops_level() -> None:
    """The informative case: they kept whatever opponent event they had, if any -- so they had none."""
    card = _own_or_neutral_for("US")
    facts = _Facts({(4, "US"): card})
    assert _dominated_discard_ops(facts, 4, "US") == _ops(card)


def test_discarding_the_opponents_card_says_nothing() -> None:
    """That is the dominant choice already; it constrains nothing about the rest of the hand."""
    card = _opponent_card_for("US")
    facts = _Facts({(4, "US"): card})
    assert _dominated_discard_ops(facts, 4, "US") is None


def test_a_turn_with_no_trap_discard_says_nothing() -> None:
    assert _dominated_discard_ops(_Facts({}), 4, "US") is None


@pytest.mark.parametrize("card", [FIVE_YEAR_PLAN, CHINA_CARD])
def test_the_cards_the_argument_never_applies_to(card: int) -> None:
    """Five Year Plan can help its non-owner; the China Card is never an ordinary discard."""
    assert _dominance_excluded(card)
    assert _dominated_discard_ops(_Facts({(4, "US"): card}), 4, "US") is None


def test_scoring_and_one_time_cards_are_excluded() -> None:
    scoring = next(c for c in range(1, 111)
                   if ts.CardData.get_card_info(c)["is_scoring"])
    one_time = next(c for c in range(1, 111)
                    if ts.CardData.get_card_info(c)["one_time"]
                    and not ts.CardData.get_card_info(c)["is_scoring"])
    assert _dominance_excluded(scoring)
    assert _dominance_excluded(one_time)


def test_a_plain_recurring_card_is_not_excluded() -> None:
    """Guards the guard: if everything were excluded the helper would be silently inert."""
    info = ts.CardData.get_card_info(DUCK_AND_COVER)
    assert not info["is_scoring"] and not info["one_time"]
    assert not _dominance_excluded(DUCK_AND_COVER)
