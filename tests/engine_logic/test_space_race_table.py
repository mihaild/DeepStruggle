"""The space race track, pinned box by box.

Boxes 3 to 7 had their roll thresholds the wrong way round -- alternating 4/3/4/3/4 where the
game alternates 3/4/3/4/3 -- because the boxes themselves were named in the wrong order: "Man
in Orbit" before "Man in Space", and so on up the track. Every one of the 1,190 space attempts
in the 287 downloaded human games agrees with the table below.

Each row is (box, min Ops to attempt it, highest roll that succeeds, VP first, VP second). The
Ops requirement is a property of where you stand rather than where you are going: below box 4
a 2 Ops card will do, at boxes 4 to 6 you need 3, and at box 7 only a 4 Ops card reaches box 8.
"""
from typing import List, Tuple

import numpy as np
import pytest
import ts_engine as ts

from ai.eval.positions import PLAY_MODE_ACTION

SPACE_MODE = PLAY_MODE_ACTION["space"]
THE_CHINA_CARD = 6

# box, min_ops, max_roll, vp_first, vp_second
TRACK: List[Tuple[int, int, int, int, int]] = [
    (1, 2, 3, 2, 1),   # Earth Satellite
    (2, 2, 4, 0, 0),   # Animal in Space -- two attempts per turn
    (3, 2, 3, 2, 0),   # Man in Space
    (4, 2, 4, 0, 0),   # Man in Earth Orbit -- opponent headlines first
    (5, 3, 3, 3, 1),   # Lunar Orbit
    (6, 3, 4, 0, 0),   # Eagle/Bear Has Landed -- discard a held card
    (7, 3, 3, 4, 2),   # Space Shuttle
    (8, 4, 2, 2, 0),   # Space Station -- eight action rounds
]


def _card_with_ops(ops: int) -> int:
    for c in range(1, 111):
        info = ts.CardData.get_card_info(c)
        if int(info["ops"]) == ops and not info["is_scoring"] and c != THE_CHINA_CARD:
            return c
    raise AssertionError(f"no {ops} Ops card exists")


def _at_box(us_box: int, ussr_box: int, card: int) -> ts.GameState:
    """A US action round with the tracks set and `card` in hand, ready to select."""
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    while (not ts.Engine.is_terminal(state)
           and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
        mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
        ts.Engine.step_flat(state, int(np.flatnonzero(mask)[0]))
    state.current_phase = ts.Phase.ACTION_ROUND
    state.action_round = 1
    state.us_space_track = us_box
    state.ussr_space_track = ussr_box
    state.victory_points = 0
    state.set_card_location(card, ts.hand_of(ts.Player.US))
    state.phasing_player = ts.Player.US
    state.ctx().decision_player = ts.Player.US
    state.ctx().decision_type = ts.DecisionType.SELECT_CARD
    return state


def _race(state: ts.GameState, card: int, roll: int) -> None:
    ts.Engine.step(state, ts.MicroAction(ts.DecisionType.SELECT_CARD, card, 0, 0))
    mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
    assert mask[SPACE_MODE] == 1, "Space Race must be an available play mode"
    ts.Engine.step(state, ts.MicroAction(
        ts.DecisionType.SELECT_PLAY_MODE, int(ts.PlayMode.SPACE), roll, 0))
    ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, roll, 0, 0))


@pytest.mark.parametrize("box,min_ops,max_roll,vp_first,vp_second", TRACK)
def test_highest_roll_that_advances(box: int, min_ops: int, max_roll: int,
                                    vp_first: int, vp_second: int) -> None:
    state = _at_box(box - 1, 0, _card_with_ops(min_ops))
    _race(state, _card_with_ops(min_ops), max_roll)
    assert int(state.us_space_track) == box, (
        f"a roll of {max_roll} must enter box {box}")
    assert int(state.victory_points) == vp_first, (
        f"box {box} awards {vp_first} VP to whoever reaches it first")


@pytest.mark.parametrize("box,min_ops,max_roll,vp_first,vp_second", TRACK)
def test_lowest_roll_that_fails(box: int, min_ops: int, max_roll: int,
                                vp_first: int, vp_second: int) -> None:
    if max_roll + 1 > 6:
        pytest.skip("box 8 succeeds on 1-2, and 3 through 6 all fail the same way")
    state = _at_box(box - 1, 0, _card_with_ops(min_ops))
    _race(state, _card_with_ops(min_ops), max_roll + 1)
    assert int(state.us_space_track) == box - 1, (
        f"a roll of {max_roll + 1} must fail to enter box {box}")
    assert int(state.victory_points) == 0, "a failed attempt awards nothing"


@pytest.mark.parametrize("box,min_ops,max_roll,vp_first,vp_second", TRACK)
def test_second_player_award(box: int, min_ops: int, max_roll: int,
                             vp_first: int, vp_second: int) -> None:
    """Arriving after the opponent pays the second-place award."""
    state = _at_box(box - 1, box, _card_with_ops(min_ops))
    _race(state, _card_with_ops(min_ops), max_roll)
    assert int(state.us_space_track) == box
    assert int(state.victory_points) == vp_second, (
        f"box {box} awards {vp_second} VP to whoever arrives second")


@pytest.mark.parametrize("box,min_ops,max_roll,vp_first,vp_second", TRACK)
def test_ops_requirement(box: int, min_ops: int, max_roll: int,
                         vp_first: int, vp_second: int) -> None:
    """One Ops short of the requirement, the Space Race is not on offer at all."""
    if min_ops <= 2:
        pytest.skip("no legal card has fewer than 2 Ops")
    short = _card_with_ops(min_ops - 1)
    state = _at_box(box - 1, 0, short)
    ts.Engine.step(state, ts.MicroAction(ts.DecisionType.SELECT_CARD, short, 0, 0))
    mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
    assert mask[SPACE_MODE] == 0, (
        f"box {box} needs {min_ops} Ops, so a {min_ops - 1} Ops card must not reach it")
