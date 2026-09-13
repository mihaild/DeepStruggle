"""The sequencing probe, against the position it was written from.

The USSR held UN Intervention, Tear Down this Wall and Grain Sales to Soviets at a space box
requiring 3 Ops, and spent UN Intervention on Tear Down this Wall — the card that could have
spaced itself — leaving Grain Sales with one exit fewer. A probe that does not catch the case it
was written from catches nothing, so that position is the test.

It is constructed rather than replayed: saved game logs are git-ignored, so a test that read the
original self-play log would pass on one machine and fail everywhere else.

Both US cards are DEFCON-suicide here: Tear Down This Wall grants its coup *in Europe*,
overriding the DEFCON 2 rule that closes the region, so USSR influence in a European battleground
exposes it. Under a narrower taxonomy only Grain Sales was banned, and this position could fire
only the blunter of the two rules.
"""

import numpy as np

import ts_engine as ts

from ai.eval.blunders import defcon_suicide_cards
from ai.eval.positions import (PLAY_MODE_ACTION, PositionBuilder, card_action, legal_mask,
                               legal_play_modes, step_to_play_mode)
from ai.eval.sequencing import classify, is_companion_node, legal_companions, new_counts

UN_INTERVENTION, TEAR_DOWN_THIS_WALL, GRAIN_SALES = 32, 96, 67
MARSHALL_PLAN, US_JAPAN_PACT = 23, 27
ALGERIA = 47          # an African battleground -- only a battleground coup degrades DEFCON
EUROPE_REGION = 0
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
        # Grain Sales hands the US the Operations to coup with, and only a *battleground* coup
        # degrades DEFCON -- so it takes influence in an African battleground, not merely
        # somewhere in Africa. Tear Down This Wall needs no help here: the opening setup leaves
        # the USSR with 3 in East Germany, a European battleground.
        influence=((ALGERIA, ts.Player.USSR, 2),),
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


def test_both_cards_are_ones_the_ussr_must_not_play() -> None:
    """What makes this §25's position rather than a generic one."""
    banned = defcon_suicide_cards(section_25_position(), ts.Player.USSR)
    assert TEAR_DOWN_THIS_WALL in banned, "exposed by USSR influence in East Germany"
    assert GRAIN_SALES in banned, "exposed by USSR influence in Algeria"


def test_un_intervention_reaches_a_companion_node_here() -> None:
    node = companion_node(section_25_position())
    assert is_companion_node(node), "playing UN Intervention must ask for a companion"
    companions = legal_companions(legal_mask(node))
    assert TEAR_DOWN_THIS_WALL in companions and GRAIN_SALES in companions


def test_spending_it_on_the_card_that_could_space_itself_is_flagged() -> None:
    """The §25 error exactly: the scarce exit goes to the card that had its own."""
    st = section_25_position()
    node = companion_node(st)
    counts = new_counts()
    classify(node, st, legal_mask(node), card_action(TEAR_DOWN_THIS_WALL), counts)

    assert counts.committed.get("un_intervention_on_spaceable", 0) == 1
    assert counts.opportunities.get("un_intervention_on_spaceable", 0) == 1
    # Both cards are ones it must not play, so the blunter rule sees a correct choice here.
    assert counts.committed.get("un_intervention_off_target", 0) == 0


def test_spending_it_on_the_card_that_needs_it_is_not_flagged() -> None:
    st = section_25_position()
    node = companion_node(st)
    counts = new_counts()
    classify(node, st, legal_mask(node), card_action(GRAIN_SALES), counts)

    assert counts.opportunities.get("un_intervention_on_spaceable", 0) == 1, \
        "the chance was still there -- it was taken correctly"
    assert counts.committed.get("un_intervention_on_spaceable", 0) == 0
    assert counts.committed.get("un_intervention_off_target", 0) == 0


def test_a_hand_with_nothing_it_must_not_play_is_not_an_opportunity() -> None:
    """No problem in hand means no mistake available, so it must not enter the denominator."""
    st = PositionBuilder(
        hand=(UN_INTERVENTION, MARSHALL_PLAN, US_JAPAN_PACT),
        side=ts.Player.USSR,
        defcon=2, turn=10, action_round=1, ussr_space=SPACE_BOX,
        # Clearing Europe removes the East Germany influence that would otherwise expose Tear
        # Down This Wall; the opening leaves nothing in Africa or the Americas to begin with.
        clear_influence=tuple(
            (cid, ts.Player.USSR) for cid in range(84)
            if int(ts.MapData.get_country_info(cid)["region"]) == EUROPE_REGION),
    ).build()
    node = companion_node(st)
    counts = new_counts()
    classify(node, st, legal_mask(node), card_action(MARSHALL_PLAN), counts)
    assert counts.opportunities.get("un_intervention_off_target", 0) == 0
    assert counts.opportunities.get("un_intervention_on_spaceable", 0) == 0


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
