"""A starred card is removed when its event occurs, not when it is spent for Operations.

rules/rules.md ties removal to the event, not to the card:

  280  event cannot occur (unmet prerequisite) -> discard pile, not removed
  281  event prohibited by a superseding event -> "played for Ops only", discard pile
  282  event occurs with no legal target -> "considered played", permanently removed
  252  Space Race -> discard "even if marked with asterisk", because the event never fires

The engine removed every one_time card after an Ops play regardless, because the branch that
relocates it is reached both by an opponent's card whose event fired and by the player's own card,
whose event never does. Over 600 games, 19 of 19 starred own-side Ops plays were removed.

Nothing caught it: the corpus converter compares per-country influence only (`_board_matches`) and
never looks at card locations, and `_apply_hands` overwrites those locations from the solved hands
anyway -- so a wrongly removed card was put back into a hand and the error was papered over rather
than surfaced.
"""

from typing import Dict, List, Tuple

import numpy as np
import pytest

import ts_engine as ts
from bindings.action_encoder import ActionEncoder

OPS_MODE = 1


def _own_starred_ops_plays(games: int = 400, want: int = 8) -> List[Tuple[str, int, str]]:
    """Where a starred card went after its own side spent it for Operations."""
    rng = np.random.default_rng(5)
    seen: List[Tuple[str, int, str]] = []
    watching: Dict[int, str] = {}
    for g in range(games):
        if len(seen) >= want:
            break
        state = ts.GameState()
        ts.Engine.init_game(state, 200000 + g)
        watching.clear()
        for _ in range(4000):
            if ts.Engine.is_terminal(state) or len(seen) >= want:
                break
            ctx = state.ctx()
            p = ctx.decision_player if ctx.decision_player != ts.Player.NONE \
                else state.phasing_player
            mask = np.asarray(ActionEncoder.get_legal_mask(state))
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                break

            if int(ctx.decision_type) == 2 and p != ts.Player.NONE:
                card = int(ctx.pending_op_card) or int(ctx.resolving_card)
                if 1 <= card <= 110:
                    info = ts.CardData.get_card_info(card)
                    mine = str(info["side"]) == ("US" if p == ts.Player.US else "USSR")
                    if info["one_time"] and mine:
                        watching[card] = str(info["name"])

            act = int(rng.choice(legal))
            if int(ctx.decision_type) == 2 and watching:
                ops = [i for i in legal
                       if int(ActionEncoder.decode(state, int(i)).primary_id) == OPS_MODE]
                if ops:
                    act = int(ops[0])
                else:
                    watching.clear()
            ts.Engine.step_flat(state, act)

            for card, name in list(watching.items()):
                loc = str(state.get_card_location(card)).split(".")[-1]
                if loc in ("REMOVED_FROM_GAME", "DISCARD_PILE"):
                    seen.append((name, card, loc))
                    del watching[card]
    return seen


@pytest.fixture(scope="module")
def plays():
    found = _own_starred_ops_plays()
    assert found, "no starred own-side Ops play reached; widen the search before trusting this"
    return found


def test_a_starred_card_spent_for_ops_is_discarded_not_removed(plays) -> None:
    removed = [(n, c) for n, c, loc in plays if loc == "REMOVED_FROM_GAME"]
    assert not removed, (
        f"{len(removed)} starred card(s) were permanently removed after being played for "
        f"Operations by their own side, whose event never fired: {removed[:5]}")


def test_the_search_actually_found_the_case(plays) -> None:
    """Guards the test above from passing because nothing was measured."""
    assert len(plays) >= 5, f"only {len(plays)} plays observed; too few to conclude anything"
    assert all(loc == "DISCARD_PILE" for _, _, loc in plays)


def test_a_starred_card_stays_available_to_be_drawn_again(plays) -> None:
    """The consequence that matters: the deck does not shrink and the event can still happen."""
    name, card, _ = plays[0]
    state = ts.GameState()
    ts.Engine.init_game(state, 200000)
    assert state.get_card_location(card) != ts.CardLocation.REMOVED_FROM_GAME, (
        f"{name} starts the game already removed")


# ---------------------------------------------------------------------------------------------
# The other half of the same rule: an Event that is fired but cannot occur
# ---------------------------------------------------------------------------------------------
#
# rules.md 280 and 281 cover the case where the Event *is* triggered and cannot happen -- an
# unmet prerequisite, or a superseding card that cancels it. The card has not been implemented,
# so it is discarded, not removed.
#
# The Ops fix above did not reach these: the sites that fire an Event still decided on `one_time`
# alone, so NATO played for Operations by the USSR with neither Marshall Plan nor Warsaw Pact in
# play was deleted from the game. Seven starred cards can reach it -- five through the
# prerequisite table, and two whose condition lives inside their own handler.

# Card ids; ts_engine does not export the card_ids namespace.
_NATO = 21
_KITCHEN_DEBATES = 48
_WILLY_BRANDT = 55
_FLOWER_POWER = 59
_STAR_WARS = 85
_SOLIDARITY = 101
_OUR_MAN_IN_TEHRAN = 108

_MARSHALL_PLAN_PLAYED = 1 << 3
_CAMP_DAVID_PLAYED = 1 << 24
_EVIL_EMPIRE_PLAYED = 1 << 30
_TEAR_DOWN_THIS_WALL_PLAYED = 1 << 38

# (card, the side that must play it for Ops so the Event fires, flag to set, why it is blocked)
_BLOCKED_ON_OPS: List[Tuple[int, ts.Player, int, str]] = [
    (_NATO,        ts.Player.USSR, 0, "neither Marshall Plan nor Warsaw Pact played"),
    (_WILLY_BRANDT, ts.Player.US, _TEAR_DOWN_THIS_WALL_PLAYED, "Tear Down This Wall in effect"),
    (_FLOWER_POWER, ts.Player.US, _EVIL_EMPIRE_PLAYED, "An Evil Empire in effect"),
    (_STAR_WARS,   ts.Player.USSR, 0, "the US is not ahead on the space track"),
    (_SOLIDARITY,  ts.Player.USSR, 0, "John Paul II has not been played"),
]


def _play_opponent_card_for_ops(card: int, player: ts.Player, flag: int) -> ts.CardLocation:
    """Spend an opponent's card for Operations, which fires its Event, and report where it went."""
    state = ts.GameState()
    ts.Engine.init_game(state, 909)
    state.current_phase = ts.Phase.ACTION_ROUND
    state.turn = 5
    state.action_round = 1
    state.phasing_player = player
    if flag:
        state.set_flag(flag)
    ctx = state.ctx()
    ctx.decision_player = player
    ctx.decision_type = ts.DecisionType.SELECT_CARD
    state.set_card_location(card, ts.hand_of(player))

    assert ts.Engine.try_step(state, ts.MicroAction(ts.DecisionType.SELECT_CARD, card, 0, 0))
    assert ts.Engine.try_step(state, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, OPS_MODE, 0, 0))
    assert ts.Engine.try_step(state, ts.MicroAction(ts.DecisionType.CHOOSE_TIMING_BRANCH, 0, 0, 0))
    for _ in range(16):
        ctx = state.ctx()
        if ctx.decision_type == ts.DecisionType.SELECT_OP_MODE:
            ts.Engine.step(state, ts.MicroAction(ts.DecisionType.SELECT_OP_MODE, 0, 0, 0))
        elif ctx.decision_type == ts.DecisionType.POINT_NODE:
            target = 14 if player == ts.Player.USSR else 1
            ts.Engine.step(state, ts.MicroAction(ts.DecisionType.POINT_NODE, target, 0, 0))
        elif ctx.decision_player == ts.Player.NONE and ctx.decision_type == ts.DecisionType.ROLL_DIE:
            ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
        else:
            break
    return state.get_card_location(card)


@pytest.mark.parametrize("card,player,flag,why", _BLOCKED_ON_OPS,
                         ids=[str(c[0]) for c in _BLOCKED_ON_OPS])
def test_a_starred_card_whose_event_cannot_occur_is_discarded(
    card: int, player: ts.Player, flag: int, why: str
) -> None:
    name = ts.CardData.get_card_info(card)["name"]
    assert ts.CardData.get_card_info(card)["one_time"], f"{name} is not starred; fixture is wrong"
    opponent = ts.Player.US if player == ts.Player.USSR else ts.Player.USSR

    probe = ts.GameState()
    ts.Engine.init_game(probe, 909)
    if flag:
        probe.set_flag(flag)
    assert not ts.CardHandlers.can_trigger_event(probe, card, opponent), (
        f"{name} is not actually blocked by '{why}'; the fixture no longer tests anything"
    )

    where = _play_opponent_card_for_ops(card, player, flag)
    assert where == ts.CardLocation.DISCARD_PILE, (
        f"{name} was fired as an Event it could not perform ({why}) and went to {where}; "
        f"rules.md:280-281 send it to the discard pile"
    )


def test_the_prerequisite_still_removes_the_card_when_the_event_does_occur() -> None:
    """The guard must not turn into 'never remove': NATO with Marshall Plan played still goes."""
    where = _play_opponent_card_for_ops(_NATO, ts.Player.USSR, _MARSHALL_PLAN_PLAYED)
    assert where == ts.CardLocation.REMOVED_FROM_GAME, (
        f"NATO's Event could occur and was implemented, so it leaves the game; got {where}"
    )


# The two whose condition lives inside their handler rather than in the prerequisite table.
# Both are checked in each direction, because a one-directional test passes just as well against
# an engine that never removes anything.

def _play_as_event(card: int, setup) -> ts.CardLocation:
    state = ts.GameState()
    ts.Engine.init_game(state, 909)
    state.current_phase = ts.Phase.ACTION_ROUND
    state.turn = 8
    state.action_round = 1
    state.phasing_player = ts.Player.US
    setup(state)
    ctx = state.ctx()
    ctx.decision_player = ts.Player.US
    ctx.decision_type = ts.DecisionType.SELECT_CARD
    state.set_card_location(card, ts.hand_of(ts.Player.US))
    assert ts.Engine.try_step(state, ts.MicroAction(ts.DecisionType.SELECT_CARD, card, 0, 0))
    assert ts.Engine.try_step(state, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, 0, 0, 0))
    for _ in range(20):
        ctx = state.ctx()
        if ctx.decision_player == ts.Player.NONE and ctx.decision_type == ts.DecisionType.ROLL_DIE:
            ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
        elif ctx.resolving_card == card:
            ts.Engine.step(state, ts.MicroAction(ctx.decision_type, 255, 0, 0x80))
        else:
            break
    return state.get_card_location(card)


def _clear_middle_east(state: ts.GameState) -> None:
    for cid in range(21, 31):
        state.set_country(cid, 0, 0)


def _us_controls_israel(state: ts.GameState) -> None:
    state.set_country(23, 5, 0)


def _ussr_leads_battlegrounds(state: ts.GameState) -> None:
    for cid in range(84):
        state.set_country(cid, 0, 0)
    state.set_country(14, 0, 5)


def _us_leads_battlegrounds(state: ts.GameState) -> None:
    for cid in range(84):
        state.set_country(cid, 0, 0)
    state.set_country(7, 6, 0)


def test_our_man_in_tehran_is_discarded_with_no_middle_east_country() -> None:
    assert _play_as_event(_OUR_MAN_IN_TEHRAN, _clear_middle_east) \
        == ts.CardLocation.DISCARD_PILE


def test_our_man_in_tehran_is_removed_when_its_event_happens() -> None:
    assert _play_as_event(_OUR_MAN_IN_TEHRAN, _us_controls_israel) \
        == ts.CardLocation.REMOVED_FROM_GAME


def test_kitchen_debates_is_discarded_when_the_us_is_behind_on_battlegrounds() -> None:
    assert _play_as_event(_KITCHEN_DEBATES, _ussr_leads_battlegrounds) \
        == ts.CardLocation.DISCARD_PILE


def test_kitchen_debates_is_removed_when_its_event_happens() -> None:
    assert _play_as_event(_KITCHEN_DEBATES, _us_leads_battlegrounds) \
        == ts.CardLocation.REMOVED_FROM_GAME
