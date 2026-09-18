"""A headline ends with the stack where it found it.

An event that grants Ops does not finish when it is triggered: it opens an op mode and the
frame it was fired in stays open until those Ops are spent. Missile Envy fires the card it
takes inside a pushed frame and pops it only where the event finishes at once, so a card like
ABM Treaty -- which improves DEFCON and then hands its player 4 Ops -- leaves the frame behind.

Spending those Ops during a headline returned straight to the headline machinery without
unwinding. At turn 6's headline of ts-replayer game 240 the USSR headlines Missile Envy, the US
hands over ABM Treaty, its event improves DEFCON to 4 and grants the USSR 3 Ops (4 less one for
the Red Scare/Purge the US headlined alongside), and the USSR coups Argentina. The headline then
ended a frame deep, and turn 6 AR1 was refused because the engine was inside a headline that
was over.

A frame holding an unanswered op mode is Ops still owed inside the headline, and the stack is
there precisely to resume them -- so the unwind stops at one, exactly as the action round does.
"""
import pytest
import ts_engine as ts

MISSILE_ENVY = 49
ABM_TREATY = 57
RED_SCARE = 31
DUCK_AND_COVER = 4


def _at_headline() -> ts.GameState:
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    guard = 0
    while state.current_phase == ts.Phase.SETUP and guard < 600:
        guard += 1
        legal = [i for i, v in enumerate(ts.ActionMask.generate_flat_mask(state)) if v]
        ts.Engine.step_flat(state, legal[0])
    assert state.current_phase == ts.Phase.HEADLINE
    state.turn = 6
    for c in range(1, 111):
        if state.get_card_location(c) in (ts.hand_of(ts.Player.US),
                                          ts.hand_of(ts.Player.USSR)):
            state.set_card_location(c, ts.CardLocation.DISCARD_PILE)
    state.china_card_holder = ts.Player.US
    state.china_card_playable = 0
    return state


def _headline_missile_envy_against_abm() -> ts.GameState:
    """The USSR headlines Missile Envy; the US holds only ABM Treaty to give."""
    state = _at_headline()
    state.set_card_location(MISSILE_ENVY, ts.hand_of(ts.Player.USSR))
    state.set_card_location(ABM_TREATY, ts.hand_of(ts.Player.US))
    state.set_card_location(RED_SCARE, ts.hand_of(ts.Player.US))
    for _ in range(2):
        p = state.ctx().decision_player
        card = MISSILE_ENVY if p == ts.Player.USSR else RED_SCARE
        ts.Engine.step_flat(state, card - 1)
    return state


def _play_out(state: ts.GameState, limit: int = 200) -> ts.GameState:
    guard = 0
    while state.current_phase == ts.Phase.HEADLINE and guard < limit:
        guard += 1
        ctx = state.ctx()
        if (ctx.decision_player == ts.Player.NONE
                and ctx.decision_type == ts.DecisionType.ROLL_DIE):
            ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
            continue
        legal = [i for i, v in enumerate(ts.ActionMask.generate_flat_mask(state)) if v]
        if not legal:
            break
        ts.Engine.step_flat(state, legal[0])
    return state


def test_the_taken_card_grants_ops_inside_a_pushed_frame() -> None:
    """The setup this rests on: ABM Treaty does not finish when it is triggered."""
    state = _headline_missile_envy_against_abm()
    assert state.ctx().decision_type == ts.DecisionType.SELECT_OP_MODE, (
        "ABM Treaty improved DEFCON and now owes its player Ops")
    assert int(state.ctx_stack_depth) == 1, "fired inside a frame of its own"


def test_the_headline_ends_with_the_stack_empty() -> None:
    state = _play_out(_headline_missile_envy_against_abm())
    assert state.current_phase == ts.Phase.ACTION_ROUND
    assert int(state.ctx_stack_depth) == 0, (
        "the frame ABM Treaty was fired in is gone by the time the action rounds begin")
    assert state.ctx().decision_type == ts.DecisionType.SELECT_CARD


def test_the_ops_are_actually_spent_before_the_unwind() -> None:
    """Unwinding must not cut the Ops short -- the stack exists to resume them."""
    state = _headline_missile_envy_against_abm()
    assert int(state.ctx().pending_ops_value) > 0
    ops = int(state.ctx().pending_ops_value)
    ts.Engine.step_flat(state, 112)                  # influence (P17 slot)
    assert int(state.ctx().remaining_steps) == ops, (
        "every Op granted is still there to spend")
