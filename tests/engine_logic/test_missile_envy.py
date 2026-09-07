"""Missile Envy (#49) resolves by the card's own side, and hands itself over afterwards.

The ruling: the opponent gives up their highest-Ops non-scoring card. If that card is
yours or neutral you resolve its event; otherwise you get Operations equal to its value,
subject to Containment / Brezhnev Doctrine / Red Scare. Missile Envy then passes to the
opponent, who must play it on their next action round.

The side-dispatch was correct. The hand-over was not: the generic post-play cleanup
discarded Missile Envy out of the hand the handler had just placed it in, leaving
forced_card_id pointing at a card nobody held. The action mask only forces a card that is
actually in hand, so the forced play vanished silently and the card was simply gone.
"""

from typing import Any, List, Optional

import numpy as np
import pytest
import ts_engine as ts

from ai.eval.positions import PLAY_MODE_ACTION, PositionBuilder
from bindings.action_encoder import ActionEncoder

MISSILE_ENVY = 49
MARSHALL_PLAN, NUCLEAR_TEST_BAN, WE_WILL_BURY_YOU = 23, 34, 50
DUCK_AND_COVER = 4  # a 3-Ops US card, so a +1 modifier is visible below the cap of 4


def _ussr_plays_missile_envy(us_hand: List[int], **kwargs: Any) -> ts.GameState:
    """USSR plays Missile Envy for its event; the US hand's best card is us_hand[0]."""
    st = PositionBuilder(
        hand=[MISSILE_ENVY, 14, 16], side=ts.Player.USSR, turn=6, defcon=4,
        opponent_hand=us_hand, **kwargs,
    ).build()
    ts.Engine.step_flat(st, MISSILE_ENVY - 1)
    ts.Engine.step_flat(st, PLAY_MODE_ACTION["event"])
    return st


def test_opponent_card_gives_plain_ops_and_not_its_event() -> None:
    """Marshall Plan is a US card: the USSR gets 4 Ops, and the event must not fire."""
    st = _ussr_plays_missile_envy([MARSHALL_PLAN, 1, 2])
    ctx = st.ctx()

    assert ctx.decision_type == ts.DecisionType.SELECT_OP_MODE
    assert ctx.decision_player == ts.Player.USSR
    assert int(ctx.pending_op_card) == MARSHALL_PLAN
    assert int(ctx.pending_ops_value) == 4
    # Marshall Plan would place US influence across Europe; nothing may have happened.
    assert int(st.victory_points) == 0
    assert int(st.defcon) == 4


def test_neutral_card_resolves_its_event() -> None:
    """Nuclear Test Ban is neutral, so the USSR resolves it: DEFCON up, VP to the USSR."""
    st = _ussr_plays_missile_envy([NUCLEAR_TEST_BAN, 1, 2])

    assert int(st.defcon) == 5, "Nuclear Test Ban raises DEFCON by two"
    assert int(st.victory_points) < 0, "the USSR played it, so the USSR scores it"


def test_own_card_resolves_its_event() -> None:
    """We Will Bury You is a USSR card, so the USSR resolving player fires it."""
    st = _ussr_plays_missile_envy([WE_WILL_BURY_YOU, 1, 2])

    assert int(st.defcon) == 3, "We Will Bury You degrades DEFCON by one"


@pytest.mark.parametrize("flag,expected", [
    (None, 3),
    (ts.EffectBits.BREZHNEV_DOCTRINE_ACTIVE, 4),
    (ts.EffectBits.PURGE_USSR_ACTIVE, 2),
])
def test_ops_value_respects_operations_modifiers(flag: Optional[int], expected: int) -> None:
    """The Ops branch is subject to the usual modifiers, not the card's printed value."""
    kwargs = {"flags": (flag,)} if flag is not None else {}
    st = _ussr_plays_missile_envy([DUCK_AND_COVER, 1, 2], **kwargs)

    assert st.ctx().decision_type == ts.DecisionType.SELECT_OP_MODE
    assert int(st.ctx().pending_ops_value) == expected


def test_the_card_passes_to_the_opponent_who_must_play_it() -> None:
    """Missile Envy must survive cleanup in the opponent's hand and be forced next AR."""
    st = _ussr_plays_missile_envy([MARSHALL_PLAN, 1, 2])

    assert ts.in_hand_of(st.get_card_location(MISSILE_ENVY), ts.Player.US), \
        "cleanup must not discard the card the event just handed to the US"
    assert int(st.forced_card_id) == MISSILE_ENVY
    assert st.forced_card_player == ts.Player.US

    # Play out the USSR's Ops, then the US must be offered Missile Envy and nothing else.
    for _ in range(20):
        if ts.Engine.is_terminal(st):
            break
        ctx = st.ctx()
        if ctx.decision_player == ts.Player.NONE and ctx.decision_type == ts.DecisionType.ROLL_DIE:
            ts.Engine.step(st, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
            continue
        legal = np.flatnonzero(ActionEncoder.get_legal_mask(st))
        if not len(legal):
            break
        if ctx.decision_type == ts.DecisionType.SELECT_CARD and ctx.decision_player == ts.Player.US:
            assert [int(a) + 1 for a in legal if a < 110] == [MISSILE_ENVY], \
                "the US is forced to play Missile Envy back"
            return
        ts.Engine.step_flat(st, int(legal[0]))

    pytest.fail("the US never reached a card-selection node")

def test_duck_and_cover_event_does_not_trigger_when_ussr_completes_ops() -> None:
    """Bug 1 reproduction: When USSR takes Duck and Cover via Missile Envy, USSR gets Ops and NO event occurs."""
    st = PositionBuilder(
        hand=[MISSILE_ENVY, 14, 16], side=ts.Player.USSR, turn=6, defcon=2,
        opponent_hand=[DUCK_AND_COVER, 1, 2],
        influence=[(83, ts.Player.USSR, 1)],
    ).build()

    # USSR plays Missile Envy for Event
    ts.Engine.step_flat(st, MISSILE_ENVY - 1)
    ts.Engine.step_flat(st, PLAY_MODE_ACTION["event"])

    assert st.ctx().decision_type == ts.DecisionType.SELECT_OP_MODE
    assert st.ctx().decision_player == ts.Player.USSR
    assert int(st.ctx().pending_op_card) == DUCK_AND_COVER
    assert int(st.ctx().pending_ops_value) == 3

    # USSR uses Ops for Influence
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_OP_MODE, 0, 0, 0))
    # Place 3 influence in Uruguay (#83)
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.POINT_NODE, 83, 0, 0))
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.POINT_NODE, 83, 0, 0))
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.POINT_NODE, 83, 0, 0))

    # Duck and Cover event MUST NOT have triggered:
    # 1. Game must NOT be over (DEFCON did not drop to 1)
    assert not ts.Engine.is_terminal(st), "DEFCON 1 suicide must not occur because opponent event should not trigger"
    assert int(st.defcon) == 2, "DEFCON must remain at 2 (no Duck and Cover event)"
    assert int(st.victory_points) == 0, "US must not gain VP from Duck and Cover"
    assert st.get_card_location(DUCK_AND_COVER) == ts.CardLocation.DISCARD_PILE


def test_forced_missile_envy_can_only_be_played_for_ops() -> None:
    """Bug 2 reproduction: The recipient of Missile Envy MUST use it for Operations on their next AR (no Event, no Space)."""
    st = PositionBuilder(
        hand=[MISSILE_ENVY, 14, 16], side=ts.Player.USSR, turn=6, defcon=4,
        opponent_hand=[DUCK_AND_COVER, 1, 2],
        influence=[(83, ts.Player.USSR, 1)],
    ).build()

    # USSR plays Missile Envy for Event
    ts.Engine.step_flat(st, MISSILE_ENVY - 1)
    ts.Engine.step_flat(st, PLAY_MODE_ACTION["event"])

    # USSR spends Ops (Influence)
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_OP_MODE, 0, 0, 0))
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.POINT_NODE, 83, 0, 0))
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.POINT_NODE, 83, 0, 0))
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.POINT_NODE, 83, 0, 0))

    # Advance to US turn
    assert st.phasing_player == ts.Player.US
    assert int(st.forced_card_id) == MISSILE_ENVY
    assert st.forced_card_player == ts.Player.US

    # At SELECT_CARD, US must ONLY be allowed to select Missile Envy, even holding other cards (1, 2)
    card_mask = ActionEncoder.get_legal_mask(st)
    legal_cards = [int(a) + 1 for a in np.flatnonzero(card_mask) if a < 110]
    assert legal_cards == [MISSILE_ENVY], f"US must be forced to select only Missile Envy, got {legal_cards}"

    # US selects Missile Envy
    ts.Engine.step_flat(st, MISSILE_ENVY - 1)
    assert st.ctx().decision_type == ts.DecisionType.SELECT_PLAY_MODE

    # Legal play modes must ONLY include OPS (flat action 111)
    mask = ActionEncoder.get_legal_mask(st)
    legal_actions = np.flatnonzero(mask)

    assert 111 in legal_actions, "PlayMode::OPS (111) must be legal"
    assert 110 not in legal_actions, "PlayMode::EVENT (110) must be ILLEGAL when forced to play for Ops"
    assert 112 not in legal_actions, "PlayMode::SPACE (112) must be ILLEGAL when forced to play for Ops"

    # Step OPS (111) and place influence
    ts.Engine.step_flat(st, PLAY_MODE_ACTION["ops"])
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_OP_MODE, 0, 0, 0))
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.POINT_NODE, 83, 0, 0))
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.POINT_NODE, 83, 0, 0))

    # After forced play completes:
    assert int(st.forced_card_id) == 0, "forced_card_id must be cleared after playing"
    assert st.forced_card_player == ts.Player.NONE, "forced_card_player must be cleared after playing"
    assert st.get_card_location(MISSILE_ENVY) == ts.CardLocation.DISCARD_PILE, "Missile Envy must be discarded after ops"

