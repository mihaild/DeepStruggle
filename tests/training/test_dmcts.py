"""Determinized MCTS must not see what it is not entitled to see.

The whole value of this searcher over `pimcts` is that it is deployable, and that rests entirely
on the determinization being honest: it may reshuffle what the player cannot see and nothing else.
These tests pin that property, because a leak would be invisible in play -- the agent would simply
look stronger than it is, which is the same failure mode as a teacher with privileged information.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

import ts_engine as ts

from ai.search.dmcts import CARD_IDS, determinize, hidden_pool


def _midgame_state(seed: int = 4242, micro_actions: int = 100) -> ts.GameState:
    """A position partway into the game, for tests that need real hidden information.

    `micro_actions` counts engine decisions, not game plies. 100 of them reaches turn 2 AR 2,
    which is the useful window: the deck is down to 7-9 cards so a real discard pile has formed,
    and the opponent still holds 3-5 unknown cards. Both bounds matter -- at 60 the deck is still
    the untouched 22-card early war, and by 140 the opponent's hand is empty, which leaves
    determinization nothing to sample and fails the two tests below outright. A first-legal
    policy also drives the VP track to an autowin around 168, so there is no room to go deeper.

    Until 2026-09-16 this returned the OPENING position regardless of the count: it fed indices
    from the 128-wide per-decision mask to step_flat, which reads the flat 212-dim space, so
    every step was refused and the refusal discarded. Every determinization test below was
    therefore checking a freshly dealt hand.
    """
    s = ts.GameState()
    ts.Engine.init_game(s, seed)
    for _ in range(micro_actions):
        if ts.Engine.is_terminal(s):
            break
        # get_legal_action_indices is the 128-wide PER-DECISION space; step_flat reads the
        # flat 212-dim space. Mixing them refused every step, and the discarded refusal meant
        # this loop returned the OPENING position while claiming to have played `plies`.
        mask = ts.Engine.get_flat_action_mask(s)
        legal = [j for j, v in enumerate(mask) if v]
        if not legal:
            break
        ts.Engine.step_flat(s, int(legal[0]))
    # Pin that the fixture carries hidden information, so it cannot go quietly vacuous again.
    assert not ts.Engine.is_terminal(s), "the walk terminated; there is no hidden state left"
    assert int(s.turn) >= 2, f"expected a position past turn 1, got turn {s.turn}"
    unknown = sum(1 for c in CARD_IDS
                  if s.get_card_location(c) == ts.CardLocation.HAND_USSR_UNKNOWN)
    assert unknown >= 2, f"fixture has only {unknown} unknown USSR cards; nothing to determinize"
    return s


def _locations(state: ts.GameState) -> dict:
    return {cid: state.get_card_location(cid) for cid in CARD_IDS}


@pytest.mark.parametrize("me", [ts.Player.US, ts.Player.USSR])
def test_determinize_preserves_every_count(me):
    """A sampled world must be indistinguishable from the real one by counting alone."""
    s = _midgame_state()
    before = _locations(s)
    out = determinize(s, me, random.Random(7))
    after = _locations(out)

    from collections import Counter
    assert Counter(before.values()) == Counter(after.values()), (
        "determinization changed how many cards are in some location, so the sampled world "
        "is inconsistent with what the player has observed")


@pytest.mark.parametrize("me", [ts.Player.US, ts.Player.USSR])
def test_determinize_never_moves_a_card_the_player_can_see(me):
    """Only the opponent's UNKNOWN cards and the draw deck may move."""
    s = _midgame_state()
    before = _locations(s)
    out = determinize(s, me, random.Random(11))
    after = _locations(out)

    opp_unknown = (ts.CardLocation.HAND_USSR_UNKNOWN if int(me) == int(ts.Player.US)
                   else ts.CardLocation.HAND_US_UNKNOWN)
    movable = {opp_unknown, ts.CardLocation.DRAW_DECK}

    for cid in CARD_IDS:
        if before[cid] != after[cid]:
            assert before[cid] in movable and after[cid] in movable, (
                f"card {cid} moved from {before[cid]} to {after[cid]}; determinization may only "
                f"reshuffle cards the player cannot see")


def test_own_hand_is_never_reshuffled():
    """The searching player's own hand is fully known to it and must survive untouched."""
    s = _midgame_state()
    me = ts.Player.US
    own = {cid for cid in CARD_IDS
           if s.get_card_location(cid) in (ts.CardLocation.HAND_US_UNKNOWN,
                                           ts.CardLocation.HAND_US_KNOWN)}
    assert own, "test state has no US hand; the fixture is not exercising anything"
    out = determinize(s, me, random.Random(3))
    for cid in own:
        assert out.get_card_location(cid) == s.get_card_location(cid), (
            f"card {cid} left the searching player's own hand")


def test_determinization_actually_varies():
    """Different seeds must produce different worlds, or the search gains nothing from sampling."""
    s = _midgame_state()
    a = _locations(determinize(s, ts.Player.US, random.Random(1)))
    b = _locations(determinize(s, ts.Player.US, random.Random(2)))
    assert a != b, "determinize produced identical worlds from different seeds"


def test_hidden_pool_excludes_known_opponent_cards():
    """A card the opponent holds that we have SEEN is not hidden and must not be reshuffled."""
    s = _midgame_state()
    # Force one USSR card to be known to US.
    ussr = [cid for cid in CARD_IDS
            if s.get_card_location(cid) == ts.CardLocation.HAND_USSR_UNKNOWN]
    assert ussr, "fixture has no unknown USSR cards"
    revealed = ussr[0]
    s.set_card_location(revealed, ts.CardLocation.HAND_USSR_KNOWN)

    opp_hand, deck, n = hidden_pool(s, ts.Player.US)
    assert revealed not in opp_hand and revealed not in deck, (
        "a revealed opponent card was treated as hidden")

    out = determinize(s, ts.Player.US, random.Random(5))
    assert out.get_card_location(revealed) == ts.CardLocation.HAND_USSR_KNOWN, (
        "a revealed opponent card was moved by determinization")
