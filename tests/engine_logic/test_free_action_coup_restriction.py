"""KAL-007 and Glasnost restrict the free action their Event grants, not the card.

The rule these pin down:
  * Played for Operations, either card behaves like any other -- coup included.
  * Operations that arrive *from the Event* may not be spent on a coup, whoever receives
    them: the owner playing their own card for the Event, or the owner being handed the
    Event because the opponent played the card for Ops.

So with The Reformer in play and the US playing Glasnost for Ops, the US may coup with
Glasnost's four Ops while the USSR, who receives the Event, may not.

The engine used to bar coups whenever either card was the source of the Ops, which broke the
first half: at turn 8 AR1 of ts-replayer game 105 the US played Glasnost for its 4 Ops and
couped Mexico, and the engine could only place influence.
"""
from typing import List

import numpy as np
import pytest
import ts_engine as ts

GLASNOST = 90
KAL_007 = 89
SOUTH_KOREA = 44
BULGARIA = 20

MODE_ACTION = {"influence": 116, "coup": 117, "realign": 118}


def _fresh() -> ts.GameState:
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    while (not ts.Engine.is_terminal(state)
           and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
        mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
        ts.Engine.step_flat(state, int(np.flatnonzero(mask)[0]))
    state.defcon = 5
    state.current_phase = ts.Phase.ACTION_ROUND
    state.action_round = 1
    state.set_flag(ts.EffectBits.THE_REFORMER_PLAYED)   # Glasnost's free action needs this
    for cid in range(84):
        state.set_country(cid, 2, 2)                    # coup targets everywhere
    state.set_country(SOUTH_KOREA, 9, 0)                # KAL-007's free action needs this
    return state


def _modes(state: ts.GameState) -> List[str]:
    mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
    return [name for name, idx in MODE_ACTION.items() if mask[idx]]


def _play_for_ops(card: int, player: ts.Player, event_first: bool) -> ts.GameState:
    state = _fresh()
    state.phasing_player = player
    hand = ts.CardLocation.HAND_US if player == ts.Player.US else ts.CardLocation.HAND_USSR
    state.set_card_location(card, hand)
    state.ctx().decision_player = player
    state.ctx().decision_type = ts.DecisionType.SELECT_CARD

    ts.Engine.step(state, ts.MicroAction(ts.DecisionType.SELECT_CARD, card, 0, 0))
    ts.Engine.step(state, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, 1, 0, 0))
    if state.ctx().decision_type == ts.DecisionType.CHOOSE_TIMING_BRANCH:
        ts.Engine.step(state, ts.MicroAction(
            ts.DecisionType.CHOOSE_TIMING_BRANCH, 1 if event_first else 0, 0, 0))
    return state


@pytest.mark.parametrize("card,owner", [(GLASNOST, ts.Player.USSR), (KAL_007, ts.Player.US)])
def test_own_card_played_for_ops_allows_coup(card: int, owner: ts.Player) -> None:
    """Playing your own card for Ops does not fire its Event, so nothing is restricted."""
    state = _play_for_ops(card, owner, event_first=False)
    assert state.ctx().decision_player == owner
    assert "coup" in _modes(state)


@pytest.mark.parametrize("card,opponent", [(GLASNOST, ts.Player.US), (KAL_007, ts.Player.USSR)])
def test_opponent_card_played_for_ops_allows_coup(card: int, opponent: ts.Player) -> None:
    """The Ops belong to the player who played the card, and they may coup with them."""
    state = _play_for_ops(card, opponent, event_first=False)
    assert state.ctx().decision_player == opponent
    assert "coup" in _modes(state)


@pytest.mark.parametrize("card,owner,opponent", [
    (GLASNOST, ts.Player.USSR, ts.Player.US),
    (KAL_007, ts.Player.US, ts.Player.USSR),
])
def test_event_granted_ops_forbid_coup(card: int, owner: ts.Player,
                                       opponent: ts.Player) -> None:
    """Ops from the Event go to the card's owner and may not be couped with."""
    state = _play_for_ops(card, opponent, event_first=True)
    assert state.ctx().decision_player == owner, "the Event's Ops belong to the card's owner"
    assert "coup" not in _modes(state)
    assert "realign" in _modes(state), "realignment stays available"


@pytest.mark.parametrize("card,owner", [(GLASNOST, ts.Player.USSR), (KAL_007, ts.Player.US)])
def test_event_played_directly_forbids_coup(card: int, owner: ts.Player) -> None:
    state = _fresh()
    state.phasing_player = owner
    ts.CardHandlers.trigger_event(state, card, owner)
    assert state.ctx().decision_player == owner
    assert "coup" not in _modes(state)


def test_restriction_does_not_leak_into_the_players_own_ops() -> None:
    """The US plays Glasnost for Ops: the USSR's Event Ops are restricted, the US's are not."""
    state = _play_for_ops(GLASNOST, ts.Player.US, event_first=True)
    assert state.ctx().decision_player == ts.Player.USSR
    assert "coup" not in _modes(state)

    # The USSR spends its four restricted Ops on influence.
    ts.Engine.step(state, ts.MicroAction(ts.DecisionType.SELECT_OP_MODE, 0, 0, 0))
    for _ in range(12):
        if state.ctx().decision_type != ts.DecisionType.POINT_NODE:
            break
        ts.Engine.step(state, ts.MicroAction(ts.DecisionType.POINT_NODE, BULGARIA, 0, 0))

    assert state.ctx().decision_player == ts.Player.US, "the US still owes its own Ops"
    assert state.ctx().decision_type == ts.DecisionType.SELECT_OP_MODE
    assert "coup" in _modes(state), "the Event's restriction must not carry over"
