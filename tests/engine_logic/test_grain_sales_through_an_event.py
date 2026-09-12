"""Grain Sales To Soviets grants its Ops however it was set off.

The card draws one card at random from the USSR hand. The US may play it or return it, and if
it returns it -- or the USSR had no cards to begin with -- the US uses Grain Sales' own 2 Ops.

Any decision taken while `resolving_card` is set is handed to that card's own handler whatever
its type. Both of the branch paths clear the field on their way out; the no-cards path did not.
Played directly that field is already 0 and nothing showed, but fired through another card's
event it is not: at turn 8 AR7 of ts-replayer game 219 the US plays Five Year Plan, whose
event has the USSR discard Grain Sales, and the op mode the US then chose reached Grain Sales'
CHOOSE_BRANCH reader instead. INFLUENCE is 0, which that reader takes for "play the drawn
card", and there was no drawn card -- so the 2 Influence became a play mode for card 0.
"""
from typing import List, Optional, Tuple

import pytest
import ts_engine as ts

FIVE_YEAR_PLAN = 5
GRAIN_SALES = 67
NUCLEAR_TEST_BAN = 34
PLAY_EVENT, BRANCH_PLAY_DRAWN, BRANCH_RETURN, OP_INFLUENCE = 110, 203, 204, 116
GOLDEN = 0x9E3779B97F4A7C15


def _at_action_round() -> ts.GameState:
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
    state.turn, state.action_round = 8, 7
    state.phasing_player = ts.Player.US
    ctx = state.ctx()
    ctx.decision_player, ctx.decision_type = ts.Player.US, ts.DecisionType.SELECT_CARD
    for c in range(1, 111):
        if state.get_card_location(c) in (ts.hand_of(ts.Player.US),
                                          ts.hand_of(ts.Player.USSR)):
            state.set_card_location(c, ts.CardLocation.DISCARD_PILE)
    state.china_card_holder = ts.Player.USSR
    state.china_card_playable = 0
    return state


def _grain_sales_directly(ussr_hand: List[int]) -> ts.GameState:
    state = _at_action_round()
    state.set_card_location(GRAIN_SALES, ts.hand_of(ts.Player.US))
    for c in ussr_hand:
        state.set_card_location(c, ts.hand_of(ts.Player.USSR))
    ts.Engine.step_flat(state, GRAIN_SALES - 1)
    ts.Engine.step_flat(state, PLAY_EVENT)
    return state


def _grain_sales_through_five_year_plan(ussr_extra: List[int]) -> ts.GameState:
    """Five Year Plan discards at random, so the seed is searched until it discards this."""
    state = _at_action_round()
    state.set_card_location(FIVE_YEAR_PLAN, ts.hand_of(ts.Player.US))
    state.set_card_location(GRAIN_SALES, ts.hand_of(ts.Player.USSR))
    for c in ussr_extra:
        state.set_card_location(c, ts.hand_of(ts.Player.USSR))
    ts.Engine.step_flat(state, FIVE_YEAR_PLAN - 1)
    base = int(state.rng_state)
    for k in range(4000):
        probe = state.clone()
        probe.rng_state = (base + k * GOLDEN) % (1 << 64)
        try:
            ts.Engine.step_flat(probe, PLAY_EVENT)
        except Exception:
            continue
        if not ts.in_hand_of(probe.get_card_location(GRAIN_SALES), ts.Player.USSR):
            state.rng_state = probe.rng_state
            break
    else:
        pytest.skip("no seed made Five Year Plan discard Grain Sales")
    ts.Engine.step_flat(state, PLAY_EVENT)
    return state


def _spend_two_ops(state: ts.GameState) -> Tuple[int, int]:
    """Choose influence and report (points to place, targets offered)."""
    assert state.ctx().decision_type == ts.DecisionType.SELECT_OP_MODE, (
        f"expected an op mode, found {state.ctx().decision_type}")
    assert int(state.ctx().pending_op_card) == GRAIN_SALES
    assert int(state.ctx().pending_ops_value) == 2
    ts.Engine.step_flat(state, OP_INFLUENCE)
    assert state.ctx().decision_type == ts.DecisionType.POINT_NODE
    mask = ts.ActionMask.generate_flat_mask(state)
    return int(state.ctx().remaining_steps), sum(1 for i in range(212) if mask[i])


def test_played_directly_with_an_empty_opposing_hand() -> None:
    steps, targets = _spend_two_ops(_grain_sales_directly([]))
    assert steps == 2 and targets > 0


def test_played_directly_and_the_drawn_card_returned() -> None:
    state = _grain_sales_directly([NUCLEAR_TEST_BAN])
    assert state.ctx().decision_type == ts.DecisionType.CHOOSE_BRANCH
    ts.Engine.step_flat(state, BRANCH_RETURN)
    steps, targets = _spend_two_ops(state)
    assert steps == 2 and targets > 0


def test_fired_through_five_year_plan_with_an_empty_opposing_hand() -> None:
    """The case that was broken: no branch is offered, so nothing cleared resolving_card."""
    steps, targets = _spend_two_ops(_grain_sales_through_five_year_plan([]))
    assert steps == 2 and targets > 0


def test_fired_through_five_year_plan_and_the_drawn_card_returned() -> None:
    state = _grain_sales_through_five_year_plan([NUCLEAR_TEST_BAN])
    assert state.ctx().decision_type == ts.DecisionType.CHOOSE_BRANCH
    assert int(state.ctx().resolving_card) == GRAIN_SALES
    ts.Engine.step_flat(state, BRANCH_RETURN)
    steps, targets = _spend_two_ops(state)
    assert steps == 2 and targets > 0


def test_fired_through_five_year_plan_and_the_drawn_card_played() -> None:
    """Playing it asks for that card's play mode, and it is that card, not card 0."""
    state = _grain_sales_through_five_year_plan([NUCLEAR_TEST_BAN])
    ts.Engine.step_flat(state, BRANCH_PLAY_DRAWN)
    ctx = state.ctx()
    assert ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE
    assert int(ctx.pending_op_card) == NUCLEAR_TEST_BAN
    assert int(ctx.resolving_card) == 0
