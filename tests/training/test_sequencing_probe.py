"""P0 probe 4, against the position it was written from.

`experiments.md` §25: the USSR held UN Intervention, Tear Down this Wall and Grain Sales to
Soviets at a space box requiring 3 Ops, and spent UN Intervention on Tear Down this Wall — the
card that could have spaced itself — leaving Grain Sales with one exit fewer. A probe that does
not catch the case it was written from catches nothing, so that position is the test.

It is constructed rather than replayed: `data/replays/` is git-ignored, so a test that read the
original self-play log would pass on one machine and fail everywhere else.
"""

import numpy as np

import ts_engine as ts

from ai.eval.positions import (PLAY_MODE_ACTION, PositionBuilder, card_action, legal_mask,
                               legal_play_modes, step_to_play_mode)
from ai.eval.sequencing import classify, is_companion_node, legal_companions, new_counts

UN_INTERVENTION, TEAR_DOWN_THIS_WALL, GRAIN_SALES = 32, 96, 67
MOROCCO = 46      # Africa, so coupable while DEFCON is 2
# Next box requires 3 Ops here. Read off the engine, not off the box table: at 3 both cards are
# spaceable and at 7 neither is, so this is the one position that reproduces §25's asymmetry.
SPACE_BOX = 4


def section_25_position() -> ts.GameState:
    return PositionBuilder(
        hand=(UN_INTERVENTION, TEAR_DOWN_THIS_WALL, GRAIN_SALES),
        side=ts.Player.USSR,
        defcon=2,
        turn=10,
        action_round=1,
        ussr_space=SPACE_BOX,
        # Grain Sales is a card the USSR must not play only while it has influence somewhere
        # the US can coup at DEFCON 2 -- see blunders.defcon_suicide_cards. Morocco is in
        # Africa; the opening setup puts USSR influence only in Europe and the Middle East,
        # which are closed at DEFCON 2, so this has to be placed explicitly.
        influence=((MOROCCO, ts.Player.USSR, 2),),
    ).build()


def companion_node(state: ts.GameState) -> ts.GameState:
    """Play UN Intervention as an event, landing on the companion choice."""
    nxt = step_to_play_mode(state, UN_INTERVENTION).clone()
    ts.Engine.step_flat(nxt, PLAY_MODE_ACTION["event"])
    return nxt


def test_the_position_splits_the_two_cards_on_spaceability() -> None:
    """The asymmetry the whole probe turns on, asserted rather than assumed."""
    st = section_25_position()
    assert "space" in legal_play_modes(step_to_play_mode(st, TEAR_DOWN_THIS_WALL))
    assert "space" not in legal_play_modes(step_to_play_mode(st, GRAIN_SALES))


def test_un_intervention_reaches_a_companion_node_here() -> None:
    node = companion_node(section_25_position())
    assert is_companion_node(node), "playing UN Intervention must ask for a companion"
    companions = legal_companions(legal_mask(node))
    assert TEAR_DOWN_THIS_WALL in companions and GRAIN_SALES in companions


def test_spending_it_on_the_card_that_could_space_itself_is_flagged() -> None:
    """The §25 error exactly."""
    st = section_25_position()
    node = companion_node(st)
    counts = new_counts()
    classify(node, st, legal_mask(node), card_action(TEAR_DOWN_THIS_WALL), counts)

    assert counts.committed.get("un_intervention_off_target", 0) == 1, (
        "Tear Down this Wall is not a card the USSR must not play, so spending the scarce "
        "exit on it left the card that needed it in hand"
    )
    assert counts.opportunities.get("un_intervention_off_target", 0) == 1


def test_spending_it_on_the_card_that_needs_it_is_not_flagged() -> None:
    st = section_25_position()
    node = companion_node(st)
    counts = new_counts()
    classify(node, st, legal_mask(node), card_action(GRAIN_SALES), counts)

    assert counts.opportunities.get("un_intervention_off_target", 0) == 1, \
        "the chance was still there -- it was taken correctly"
    assert counts.committed.get("un_intervention_off_target", 0) == 0


def test_a_hand_with_nothing_it_must_not_play_is_not_an_opportunity() -> None:
    """No problem in hand means no mistake available, so it must not enter the denominator."""
    st = PositionBuilder(
        hand=(UN_INTERVENTION, TEAR_DOWN_THIS_WALL, 23),   # Marshall Plan, not a banned card
        side=ts.Player.USSR,
        defcon=2, turn=10, action_round=1, ussr_space=SPACE_BOX,
        # No coupable influence, so Grain Sales is not a card it must not play -- and the hand
        # holds none of the ones that are banned unconditionally.
    ).build()
    node = companion_node(st)
    counts = new_counts()
    classify(node, st, legal_mask(node), card_action(TEAR_DOWN_THIS_WALL), counts)
    assert counts.opportunities.get("un_intervention_off_target", 0) == 0


def test_an_ordinary_card_selection_is_not_a_companion_node() -> None:
    """The two share DecisionType.SELECT_CARD; only pending_op_card separates them."""
    st = section_25_position()
    assert st.ctx().decision_type == ts.DecisionType.SELECT_CARD
    assert not is_companion_node(st)


def test_legal_companions_uses_the_flat_action_convention() -> None:
    """A card is selected by flat action `card_id - 1`, not by its id."""
    mask = np.zeros(212, dtype=np.uint8)
    mask[card_action(GRAIN_SALES)] = 1
    mask[card_action(TEAR_DOWN_THIS_WALL)] = 1
    assert legal_companions(mask) == sorted((GRAIN_SALES, TEAR_DOWN_THIS_WALL))
