"""Blockade and Latin American Debt Crisis judge the escape discard by effective Ops.

Both let the US escape by discarding a card of "3 or more Ops", and that is measured as the US
would actually get them: Containment's +1 makes a printed 2 enough, Red Scare/Purge's -1 makes
a printed 3 insufficient, and the two cancel. The engine used the printed value, so at turn 1
AR5 of ts-replayer game 107 the US discarded 2 Ops Fidel under Containment and kept West
Germany, while the engine refused the discard and stripped West Germany instead. All nine
sub-3 escapes across the 287 downloaded human games have Containment in play.
"""
from typing import List

import numpy as np
import pytest
import ts_engine as ts

BLOCKADE = 10
LATIN_AMERICAN_DEBT_CRISIS = 95


def _card_with_ops(ops: int) -> int:
    for c in range(1, 111):
        info = ts.CardData.get_card_info(c)
        if int(info["ops"]) == ops and not info["is_scoring"] and c not in (6, BLOCKADE):
            return c
    raise AssertionError(f"no {ops} Ops card exists")


def _resolving(card: int, flags: List[int], probe: int) -> ts.GameState:
    """The US facing `card`'s demand holding only `probe`, with `flags` in play.

    The probe has to be in hand before the event fires: with nothing that could qualify, the
    demand resolves itself on the spot and never asks for a discard at all.
    """
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    while (not ts.Engine.is_terminal(state)
           and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
        mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
        ts.Engine.step_flat(state, int(np.flatnonzero(mask)[0]))
    state.current_phase = ts.Phase.ACTION_ROUND
    for flag in flags:
        state.set_flag(flag)
    # Empty the US hand so only the probe card is offered.
    for c in range(1, 111):
        if ts.in_hand_of(state.get_card_location(c), ts.Player.US):
            state.set_card_location(c, ts.CardLocation.DISCARD_PILE)
    state.set_card_location(probe, ts.hand_of(ts.Player.US))
    ts.CardHandlers.trigger_event(state, card, ts.Player.USSR)
    return state


def _offered(state: ts.GameState, card: int) -> bool:
    mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
    return bool(mask[card - 1])


@pytest.mark.parametrize("demand", [BLOCKADE, LATIN_AMERICAN_DEBT_CRISIS])
def test_containment_makes_a_two_ops_card_enough(demand: int) -> None:
    probe = _card_with_ops(2)
    state = _resolving(demand, [ts.EffectBits.CONTAINMENT_ACTIVE], probe)
    assert _offered(state, probe), "2 Ops plus Containment reaches the 3 Ops demand"


@pytest.mark.parametrize("demand", [BLOCKADE, LATIN_AMERICAN_DEBT_CRISIS])
def test_two_ops_alone_is_not_enough(demand: int) -> None:
    probe = _card_with_ops(2)
    state = _resolving(demand, [], probe)
    assert not _offered(state, probe), "2 Ops alone falls short of the 3 Ops demand"


@pytest.mark.parametrize("demand", [BLOCKADE, LATIN_AMERICAN_DEBT_CRISIS])
def test_red_scare_pushes_a_three_ops_card_below(demand: int) -> None:
    probe = _card_with_ops(3)
    state = _resolving(demand, [ts.EffectBits.PURGE_US_ACTIVE], probe)
    assert not _offered(state, probe), "Red Scare takes a printed 3 below the demand"


@pytest.mark.parametrize("demand", [BLOCKADE, LATIN_AMERICAN_DEBT_CRISIS])
def test_containment_and_red_scare_cancel(demand: int) -> None:
    three = _card_with_ops(3)
    state = _resolving(demand, [ts.EffectBits.CONTAINMENT_ACTIVE,
                                ts.EffectBits.PURGE_US_ACTIVE], three)
    assert _offered(state, three), "the two modifiers cancel, leaving the printed value"


@pytest.mark.parametrize("demand", [BLOCKADE, LATIN_AMERICAN_DEBT_CRISIS])
def test_plain_three_ops_card_is_enough(demand: int) -> None:
    probe = _card_with_ops(3)
    state = _resolving(demand, [], probe)
    assert _offered(state, probe), "a printed 3 meets the demand with no modifiers"
