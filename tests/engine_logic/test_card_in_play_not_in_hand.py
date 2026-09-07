"""An event that reads a hand does not find the card being played in it.

The engine leaves an Ops card in its owner's hand until the play finishes, and an opponent's
card played for Operations fires its own event -- so an event that reaches into that hand can
find the very card in front of it.

At turn 9 AR6 of ts-replayer game 229 the USSR plays Grain Sales To Soviets for Operations. Its
hand is empty but for that card, and the log reads "USSR has no cards in hand to reveal", so
the US should simply take Grain Sales' 2 Ops. Instead the US's event drew Grain Sales itself
and offered it back: the engine asked the US how it wished to play it.

Three other events read a hand the same way and would have made the same mistake -- Five Year
Plan and Terrorism would discard the card in play, Missile Envy would take it -- so all four
skip whatever `resolving_card` names. Fired any other way that card is not in a hand at all,
and skipping it changes nothing.
"""
from typing import List

import pytest
import ts_engine as ts

GRAIN_SALES = 67          # US, 2 Ops
FIVE_YEAR_PLAN = 5        # US, 3 Ops
TERRORISM = 92            # USSR
MISSILE_ENVY = 49         # neutral, 2 Ops
NUCLEAR_TEST_BAN = 34     # 4 Ops
DUCK_AND_COVER = 4        # 3 Ops
PLAY_OPS, EVENT_FIRST = 111, 115


def _action_round(mover) -> ts.GameState:
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
    state.turn, state.action_round = 9, 6
    state.phasing_player = mover
    ctx = state.ctx()
    ctx.decision_player, ctx.decision_type = mover, ts.DecisionType.SELECT_CARD
    for c in range(1, 111):
        if state.get_card_location(c) in (ts.hand_of(ts.Player.US),
                                          ts.hand_of(ts.Player.USSR)):
            state.set_card_location(c, ts.CardLocation.DISCARD_PILE)
    state.china_card_holder = ts.Player.US if mover == ts.Player.USSR else ts.Player.USSR
    state.china_card_playable = 0
    return state


def _play_for_ops(mover, card: int, hand: List[int]) -> ts.GameState:
    """Play `card` for Operations, event first, holding nothing else but `hand`."""
    state = _action_round(mover)
    loc = ts.hand_of(ts.Player.US) if mover == ts.Player.US else ts.hand_of(ts.Player.USSR)
    state.set_card_location(card, loc)
    for c in hand:
        state.set_card_location(c, loc)
    ts.Engine.step_flat(state, card - 1)
    ts.Engine.step_flat(state, PLAY_OPS)
    if state.ctx().decision_type == ts.DecisionType.CHOOSE_TIMING_BRANCH:
        ts.Engine.step_flat(state, EVENT_FIRST)
    return state


def test_grain_sales_does_not_offer_back_the_card_being_played() -> None:
    """Replay 229's case: the USSR's hand is empty but for Grain Sales itself."""
    state = _play_for_ops(ts.Player.USSR, GRAIN_SALES, [])
    ctx = state.ctx()
    assert ctx.decision_type == ts.DecisionType.SELECT_OP_MODE, (
        f"expected the US to be given 2 Ops, found {ctx.decision_type}")
    assert ctx.decision_player == ts.Player.US
    assert int(ctx.pending_op_card) == GRAIN_SALES
    assert int(ctx.pending_ops_value) == 2


def test_grain_sales_still_draws_a_card_the_opponent_really_holds() -> None:
    state = _play_for_ops(ts.Player.USSR, GRAIN_SALES, [NUCLEAR_TEST_BAN])
    assert state.ctx().decision_type == ts.DecisionType.CHOOSE_BRANCH, (
        "there is a card to offer, so the US chooses whether to play it")


def test_five_year_plan_discards_the_other_card_and_not_itself() -> None:
    """The USSR plays it for Ops holding one other card, which is the only candidate.

    The played card reaches the discard pile either way, through the ordinary cleanup once
    its event has fired, so what settles it is which *other* card moves: with the card in
    play still in the pool the draw is a coin toss, and without it there is only one answer.
    """
    state = _play_for_ops(ts.Player.USSR, FIVE_YEAR_PLAN, [NUCLEAR_TEST_BAN])
    assert not ts.in_hand_of(state.get_card_location(NUCLEAR_TEST_BAN), ts.Player.USSR), (
        "the only card the USSR actually holds is the one discarded")


def test_terrorism_still_discards_from_the_opponents_hand() -> None:
    """Terrorism is neutral, so this cannot arise through it -- the guard must not bite.

    Only a card associated with the *other* side fires an event when played for Operations,
    so only such a card can be in play while its own event reads the hand it came from. Grain
    Sales and Five Year Plan are both US cards and both reachable that way; Terrorism and
    Missile Envy are neutral and carry the same guard for consistency alone. What matters
    here is that guarding them took nothing away.
    """
    assert ts.CardData.get_card_info(TERRORISM)["side"] == "NONE"
    state = _action_round(ts.Player.US)
    state.set_card_location(TERRORISM, ts.hand_of(ts.Player.US))
    state.set_card_location(NUCLEAR_TEST_BAN, ts.hand_of(ts.Player.USSR))
    ts.Engine.step_flat(state, TERRORISM - 1)
    ts.Engine.step_flat(state, 110)                       # as an Event
    assert state.get_card_location(NUCLEAR_TEST_BAN) == ts.CardLocation.DISCARD_PILE, (
        "the opponent still loses a card")


def test_an_empty_hand_stays_empty() -> None:
    """Holding only the card in play, Five Year Plan finds nothing to discard."""
    state = _play_for_ops(ts.Player.USSR, FIVE_YEAR_PLAN, [])
    held = [c for c in range(1, 111)
            if ts.in_hand_of(state.get_card_location(c), ts.Player.USSR)]
    assert held == [], "nothing was in hand to begin with"
