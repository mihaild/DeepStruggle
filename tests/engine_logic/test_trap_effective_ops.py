"""A trap judges a card on what it is worth now, not on what is printed.

Quagmire and Bear Trap require a card of 2 Ops or more to be discarded. Containment adds one to
every US card and Brezhnev Doctrine one to every USSR card; Red Scare/Purge takes one away from
the side it is played against, and the two cancel exactly. So which cards are eligible moves
with what is in play -- the same reading Latin American Debt Crisis already takes of its own 3.

At turn 6 AR1 of ts-replayer game 229 the USSR is caught by Bear Trap and discards Panama Canal
Returned, 1 Op printed and 2 under Brezhnev Doctrine. On printed Ops the card was not offered
at all; and once offered, the handler judged it the same way and let it fall through to an
ordinary play, so the engine asked how the USSR wanted to spend Ops it was never given.

A discard has no target region, so the Asia bonuses Vietnam Revolts and the China Card carry
are not counted: those are for operations conducted there.
"""
from typing import List

import pytest
import ts_engine as ts
from bindings.action_encoder import ActionEncoder

PANAMA_CANAL = 64        # 1 Op
DUCK_AND_COVER = 4       # 3 Ops
NUCLEAR_TEST_BAN = 34    # 4 Ops
EUROPE_SCORING = 2       # scoring, 0 Ops
PASS = ActionEncoder.CONFIRM_DONE_INDEX
def _bit(name: str) -> int:
    import re
    hdr = open("engine/include/ts/constants.hpp").read()
    m = re.search(rf"{name}\s*=\s*1(?:ull|ULL)?\s*<<\s*(\d+)", hdr)
    assert m, f"no effect bit named {name}"
    return 1 << int(m.group(1))


BEAR_TRAP = _bit("BEAR_TRAP_ACTIVE")
QUAGMIRE = _bit("QUAGMIRE_ACTIVE")
BREZHNEV = _bit("BREZHNEV_DOCTRINE_ACTIVE")
CONTAINMENT = _bit("CONTAINMENT_ACTIVE")
PURGE_US = _bit("PURGE_US_ACTIVE")
PURGE_USSR = _bit("PURGE_USSR_ACTIVE")


def _trapped(mover, trap: int, effects: List[int], hand: List[int]) -> ts.GameState:
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    guard = 0
    while state.current_phase != ts.Phase.ACTION_ROUND and guard < 600:
        guard += 1
        ctx = state.ctx()
        if (ctx.decision_player == ts.Player.NONE
                and ctx.decision_type == ts.DecisionType.ROLL_DIE):
            ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
            continue
        legal = [i for i, v in enumerate(ts.ActionMask.generate_flat_mask(state)) if v]
        ts.Engine.step_flat(state, legal[0])
    state.turn, state.action_round = 6, 1
    state.phasing_player = mover
    ctx = state.ctx()
    ctx.decision_player, ctx.decision_type = mover, ts.DecisionType.SELECT_CARD
    loc = ts.hand_of(ts.Player.US) if mover == ts.Player.US else ts.hand_of(ts.Player.USSR)
    for c in range(1, 111):
        if state.get_card_location(c) in (ts.hand_of(ts.Player.US),
                                          ts.hand_of(ts.Player.USSR)):
            state.set_card_location(c, ts.CardLocation.DISCARD_PILE)
    for c in hand:
        state.set_card_location(c, loc)
    state.china_card_holder = ts.Player.US if mover == ts.Player.USSR else ts.Player.USSR
    state.china_card_playable = 0
    state.persistent_effects |= trap
    for f in effects:
        state.persistent_effects |= f
    return state


def _offered(state: ts.GameState) -> List[int]:
    mask = ts.ActionMask.generate_flat_mask(state)
    return [i + 1 for i in range(110) if mask[i]]


HAND = [PANAMA_CANAL, DUCK_AND_COVER, NUCLEAR_TEST_BAN, EUROPE_SCORING]


def test_a_one_op_card_is_not_eligible_on_its_own() -> None:
    state = _trapped(ts.Player.USSR, BEAR_TRAP, [], HAND)
    assert PANAMA_CANAL not in _offered(state)
    assert DUCK_AND_COVER in _offered(state)


def test_brezhnev_doctrine_makes_it_eligible() -> None:
    """Replay 229's case: 1 Op printed, 2 in play."""
    state = _trapped(ts.Player.USSR, BEAR_TRAP, [BREZHNEV], HAND)
    assert PANAMA_CANAL in _offered(state)


def test_containment_does_the_same_for_the_us_in_a_quagmire() -> None:
    state = _trapped(ts.Player.US, QUAGMIRE, [CONTAINMENT], HAND)
    assert PANAMA_CANAL in _offered(state)


def test_red_scare_takes_the_bonus_back() -> None:
    for mover, trap, plus, minus in ((ts.Player.USSR, BEAR_TRAP, BREZHNEV, PURGE_USSR),
                                     (ts.Player.US, QUAGMIRE, CONTAINMENT, PURGE_US)):
        assert PANAMA_CANAL not in _offered(_trapped(mover, trap, [minus], HAND))
        assert PANAMA_CANAL not in _offered(_trapped(mover, trap, [plus, minus], HAND)), (
            "the two cancel exactly")


def test_a_scoring_card_is_never_a_discard() -> None:
    state = _trapped(ts.Player.USSR, BEAR_TRAP, [BREZHNEV], HAND)
    assert EUROPE_SCORING not in _offered(state)


def test_the_card_the_mask_offers_is_discarded_and_rolled_for() -> None:
    """The handler must judge it the same way, or it falls through to an ordinary play."""
    state = _trapped(ts.Player.USSR, BEAR_TRAP, [BREZHNEV], HAND)
    assert PANAMA_CANAL in _offered(state)
    ts.Engine.step_flat(state, PANAMA_CANAL - 1)
    ctx = state.ctx()
    assert ctx.decision_type == ts.DecisionType.ROLL_DIE, (
        f"expected the trap's escape roll, found {ctx.decision_type}")
    assert ctx.decision_player == ts.Player.NONE
    assert state.get_card_location(PANAMA_CANAL) == ts.CardLocation.DISCARD_PILE


def test_only_scoring_cards_when_nothing_reaches_two_ops() -> None:
    state = _trapped(ts.Player.USSR, BEAR_TRAP, [], [PANAMA_CANAL, EUROPE_SCORING])
    assert _offered(state) == [EUROPE_SCORING]
