"""The hands behind a log, solved from the rules rather than guessed a turn at a time.

The log says what became visible. The rules say the rest: how many cards a hand holds, what a
player carries, where a spent card goes and when it can come back, what a scoring card may not
do, and what an empty hand or a skipped round proves. Together they pin nearly every hand in
the corpus exactly.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Dict, List, cast

import pytest
import ts_engine as ts

from tools.lib.corpus_paths import corpus_dir
from tools.lib.ts_replayer_convert import card_id, convert_game
from tools.lib.ts_replayer_hands import HAVE_Z3, GameFacts, hand_size, solve_hands

_CORPUS = str(corpus_dir())

pytestmark = pytest.mark.skipif(not HAVE_Z3, reason="z3 is not installed")


def _game(replay_id: int) -> Dict[str, object]:
    path = os.path.join(_CORPUS, f"{replay_id}.json.gz")
    if not os.path.exists(path):
        pytest.skip("corpus not downloaded")
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _solve(replay_id: int):
    game = _game(replay_id)
    return game, solve_hands(cast(List[Dict[str, object]], game["all_turns"]),
                             cast(Dict[str, object], game["hands"]), card_id)


def test_every_hand_is_the_size_the_rules_deal() -> None:
    """Eight in the early war, nine from turn 4, the China Card outside the count."""
    _game_data, hands = _solve(100)
    assert hands is not None
    for turn, sides in hands.items():
        for side, cards in sides.items():
            assert len(cards) == hand_size(turn), f"turn {turn} {side}"
            assert 6 not in cards, "the China Card is not part of the hand limit"


def test_what_the_log_shows_them_spending_is_in_their_hand() -> None:
    game, hands = _solve(100)
    assert hands is not None
    facts = GameFacts(cast(List[Dict[str, object]], game["all_turns"]),
                      cast(Dict[str, object], game["hands"]), card_id)
    for (turn, side), spent in facts.spent.items():
        if facts.drew_mid_turn.get((turn, side)):
            continue          # some of it arrived after the deal; the count is checked below
        for cid in spent:
            if cid in facts.arrived[(turn, side)]:
                continue
            assert cid in hands[turn][side], (
                f"{ts.CardData.get_card_info(cid)['name']} at turn {turn} for {side}")


def test_no_hand_holds_a_scoring_card_it_does_not_play() -> None:
    """Holding one at the end of a turn loses the game, so nobody carries one."""
    game, hands = _solve(100)
    assert hands is not None
    facts = GameFacts(cast(List[Dict[str, object]], game["all_turns"]),
                      cast(Dict[str, object], game["hands"]), card_id)
    for turn, sides in hands.items():
        for side, cards in sides.items():
            for cid in cards:
                if bool(ts.CardData.get_card_info(cid)["is_scoring"]):
                    assert cid in facts.spent[(turn, side)]


def test_a_spent_card_does_not_come_back_before_a_reshuffle() -> None:
    """It is in the discard pile until one, and the log prints where they happen."""
    game, hands = _solve(100)
    assert hands is not None
    facts = GameFacts(cast(List[Dict[str, object]], game["all_turns"]),
                      cast(Dict[str, object], game["hands"]), card_id)
    assert facts.reshuffled_before, "this game reshuffles twice"
    for turn, sides in hands.items():
        for side, cards in sides.items():
            for cid in cards:
                if cid in facts.spent[(turn, side)]:
                    continue
                last = max((u for u in range(1, turn)
                            if cid in facts.spent[(u, "US")] or cid in facts.spent[(u, "USSR")]),
                           default=0)
                if last:
                    assert any(last < r <= turn for r in facts.reshuffled_before), (
                        f"{ts.CardData.get_card_info(cid)['name']} was spent at turn {last}")


def test_the_two_hands_never_hold_the_same_card() -> None:
    _game_data, hands = _solve(105)
    assert hands is not None
    for turn, sides in hands.items():
        assert not (set(sides["US"]) & set(sides["USSR"])), f"turn {turn}"


def test_what_they_do_not_spend_they_carry() -> None:
    game, hands = _solve(105)
    assert hands is not None
    facts = GameFacts(cast(List[Dict[str, object]], game["all_turns"]),
                      cast(Dict[str, object], game["hands"]), card_id)
    for turn in sorted(hands):
        if turn + 1 not in hands:
            continue
        for side in ("US", "USSR"):
            for cid in hands[turn][side]:
                if cid in facts.spent[(turn, side)] or cid in facts.taken[(turn, side)]:
                    continue
                assert cid in hands[turn + 1][side], (
                    f"{ts.CardData.get_card_info(cid)['name']} vanished after turn {turn}")


def test_every_turn_deals_a_full_hand() -> None:
    """The deck does not run out. It is reshuffled when it empties -- turn 10 included -- so
    there is no turn on which a player is dealt short, and a round with nothing in it at the
    foot of the file is the recording stopping rather than a player with nothing to play."""
    for replay_id in (72, 27, 48, 53, 70, 180):
        game = _game(replay_id)
        hands = solve_hands(cast(List[Dict[str, object]], game["all_turns"]),
                            cast(Dict[str, object], game["hands"]), card_id)
        assert hands is not None, f"replay {replay_id} has no model"
        for turn, sides in hands.items():
            for side, cards in sides.items():
                assert len(cards) == hand_size(turn), f"replay {replay_id} turn {turn} {side}"


def test_a_card_discarded_before_missile_envy_reads_the_hand_is_not_capped() -> None:
    """At turn 7's headline of replay 96 "Ask Not" discards six cards and draws six, and only
    then does Missile Envy read the hand and take U-2 Incident at 3 Ops. "We Will Bury You" at
    4 Ops was one of the six, so it was in the hand the turn dealt and gone before Envy looked.
    """
    game, hands = _solve(96)
    assert hands is not None
    facts = GameFacts(cast(List[Dict[str, object]], game["all_turns"]),
                      cast(Dict[str, object], game["hands"]), card_id)
    assert (7, "US", 50) in facts.gone_before_envy
    assert 50 in hands[7]["US"]
    assert facts.envy_took[(7, "US")] == 60


def test_the_games_that_taught_the_model_its_rules_convert() -> None:
    """Replay 16's declined eighth round, 96's Ask Not draws feeding Missile Envy, 100's Our
    Man in Tehran peek, 103's SALT reclaim, 72's announced-and-unwritten last round."""
    for replay_id in (16, 96, 100, 103, 72):
        conv = convert_game(_game(replay_id))
        assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
        assert conv.hands_solved, f"replay {replay_id} fell back to the heuristics"
