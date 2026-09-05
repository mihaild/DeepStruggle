"""Space Race eligibility uses effective Ops, not the value printed on the card.

The Ops modifiers that decide what a card can buy on the board decide what it can buy on the
space track too. At turn 4 AR3 of ts-replayer game 113 the USSR raced with OAS Founded -- a
1 Ops card, one short of the 2 that box 3 requires -- while Brezhnev Doctrine was active,
making it effectively 2. The engine read the printed value, so it offered no Space Race at
all and the entry would not convert. Red Scare/Purge cuts the same way in the other
direction, and the two cancel.
"""
from typing import List

import numpy as np
import pytest
import ts_engine as ts

from ai.eval.positions import PLAY_MODE_ACTION

SPACE_MODE = PLAY_MODE_ACTION["space"]
THE_CHINA_CARD = 6
OAS_FOUNDED = 70            # 1 Ops
DUCK_AND_COVER = 4          # 3 Ops
NUCLEAR_TEST_BAN = 34       # 4 Ops, the only rung that reaches box 8


def _at_box(player: ts.Player, box: int, hand: List[int]) -> ts.GameState:
    """An action round for `player`, standing on `box`, holding exactly `hand`."""
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    while (not ts.Engine.is_terminal(state)
           and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
        mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
        ts.Engine.step_flat(state, int(np.flatnonzero(mask)[0]))
    state.current_phase = ts.Phase.ACTION_ROUND
    state.action_round = 1
    state.phasing_player = player
    if player == ts.Player.US:
        state.us_space_track = box
    else:
        state.ussr_space_track = box
    state.set_space_turns_used(player, 0)
    loc = ts.CardLocation.HAND_US if player == ts.Player.US else ts.CardLocation.HAND_USSR
    for c in range(1, 111):
        if state.get_card_location(c) == loc:
            state.set_card_location(c, ts.CardLocation.DISCARD_PILE)
    for c in hand:
        state.set_card_location(c, loc)
    state.ctx().decision_player = player
    state.ctx().decision_type = ts.DecisionType.SELECT_CARD
    return state


def _may_race(state: ts.GameState, card: int) -> bool:
    """Whether selecting `card` offers Space Race as a play mode. Consumes the state."""
    ts.Engine.step(state, ts.MicroAction(ts.DecisionType.SELECT_CARD, card, 0, 0))
    if state.ctx().decision_type != ts.DecisionType.SELECT_PLAY_MODE:
        return False
    return bool(np.asarray(ts.ActionMask.generate_flat_mask(state))[SPACE_MODE])


def test_a_one_ops_card_cannot_race_unaided() -> None:
    assert not _may_race(_at_box(ts.Player.USSR, 2, [OAS_FOUNDED]), OAS_FOUNDED), (
        "1 printed Op is one short of what box 3 requires")


def test_brezhnev_doctrine_lifts_a_one_ops_card_into_range() -> None:
    """Replay 113 turn 4 AR3, the game that found this."""
    state = _at_box(ts.Player.USSR, 2, [OAS_FOUNDED])
    state.set_flag(ts.EffectBits.BREZHNEV_DOCTRINE_ACTIVE)
    assert _may_race(state, OAS_FOUNDED), (
        "Brezhnev Doctrine makes OAS Founded effectively 2 Ops, enough for box 3")


def test_containment_lifts_a_us_card() -> None:
    state = _at_box(ts.Player.US, 2, [OAS_FOUNDED])
    state.set_flag(ts.EffectBits.CONTAINMENT_ACTIVE)
    assert _may_race(state, OAS_FOUNDED)


@pytest.mark.parametrize("player,flag", [
    (ts.Player.US, ts.EffectBits.PURGE_US_ACTIVE),
    (ts.Player.USSR, ts.EffectBits.PURGE_USSR_ACTIVE),
])
def test_red_scare_purge_pushes_a_card_out_of_range(player: ts.Player, flag: int) -> None:
    """Box 5 needs 3 Ops; Red Scare leaves Duck and Cover worth 2."""
    assert _may_race(_at_box(player, 4, [DUCK_AND_COVER]), DUCK_AND_COVER)
    state = _at_box(player, 4, [DUCK_AND_COVER])
    state.set_flag(flag)
    assert not _may_race(state, DUCK_AND_COVER), (
        "Red Scare/Purge costs an Op on the space track as it does on the board")


def test_the_two_modifiers_cancel() -> None:
    state = _at_box(ts.Player.USSR, 4, [DUCK_AND_COVER])
    state.set_flag(ts.EffectBits.BREZHNEV_DOCTRINE_ACTIVE)
    state.set_flag(ts.EffectBits.PURGE_USSR_ACTIVE)
    assert _may_race(state, DUCK_AND_COVER), (
        "+1 and -1 leave the printed 3 Ops, still enough for box 5")


def test_vietnam_revolts_does_not_pay_for_a_space_attempt() -> None:
    """Its bonus buys Operations in Southeast Asia; the space track is in no region."""
    state = _at_box(ts.Player.USSR, 2, [OAS_FOUNDED])
    state.set_flag(ts.EffectBits.VIETNAM_REVOLTS_ACTIVE)
    assert not _may_race(state, OAS_FOUNDED)


def test_the_china_card_may_race() -> None:
    """A poor play -- it is the best Ops card in the game and passes to the opponent either
    way -- and a legal one. At turn 10 AR4 of ts-replayer game 247 the US races to box 5 with
    it, on 3 of its 4 Ops."""
    assert _may_race(_at_box(ts.Player.US, 4, [THE_CHINA_CARD]), THE_CHINA_CARD)



def test_box_8_still_demands_four_effective_ops() -> None:
    """A modifier can carry a 3 Ops card to the last box, but the requirement stands."""
    assert not _may_race(_at_box(ts.Player.USSR, 7, [DUCK_AND_COVER]), DUCK_AND_COVER)
    state = _at_box(ts.Player.USSR, 7, [DUCK_AND_COVER])
    state.set_flag(ts.EffectBits.BREZHNEV_DOCTRINE_ACTIVE)
    assert _may_race(state, DUCK_AND_COVER)
    assert _may_race(_at_box(ts.Player.USSR, 7, [NUCLEAR_TEST_BAN]), NUCLEAR_TEST_BAN)
