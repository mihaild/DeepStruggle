"""The card Grain Sales draws for the US, in every way Grain Sales can be played.

The US plays the drawn card for Operations while the USSR is the phasing player -- Grain Sales
played by the USSR, or headlined by it -- and the engine judged the card against the phasing
player instead of the player whose hand holds it. Three things followed:

* a USSR card the US coups, realigns or places Influence with owed its Event to nobody, so it was
  dropped when the USSR played Grain Sales for Operations (Ops first) or headlined it;
* a US or neutral starred card the US played for Operations was removed from the game as if its
  Event had occurred, when the USSR headlined Grain Sales or played it Ops first;
* nothing about Flower Power was wrong, and is pinned here: the US pays 2 VP for a war card it
  plays for anything but the Space Race, whichever way Grain Sales handed it over.

Played by the US itself, or by the USSR for its Event, the card was already right; those cases
stay in the matrix as the control. (The Coup/Realign case of the USSR's Event is
test_grain_sales_drawn_card_event_after_roll.py.)
"""
from typing import Callable, Dict, List

import pytest

import ts_engine as ts
from bindings.action_encoder import ActionEncoder
from tests.engine_logic.test_grain_sales_through_an_event import GRAIN_SALES, PLAY_EVENT, _at_action_round
from tests.engine_logic.test_headline_ops_card_discarded import CAMBRIDGE_FIVE, _at_headline, _set_hand

EB = ts.EffectBits
DUCK_AND_COVER = 4
NUCLEAR_TEST_BAN = 34

QUAGMIRE = 42          # a USSR card, starred: the US playing it for Operations owes the USSR its Event
MARSHALL_PLAN = 23     # a US card, starred
SALT_NEGOTIATIONS = 43  # a neutral card, starred
WARS = (11, 13, 24, 36, 102)   # Korean, Arab-Israeli, Indo-Pakistani, Brush, Iran-Iraq: USSR, USSR, neutral x3

PLAY_MODES: Dict[str, int] = {
    "event": ActionEncoder.PLAY_MODE_OFFSET + int(ts.Resolution.EVENT),
    "space": ActionEncoder.PLAY_MODE_OFFSET + int(ts.Resolution.SPACE),
    "influence": ActionEncoder.PLAY_MODE_OFFSET + int(ts.Resolution.OPS_INFLUENCE),
    "coup": ActionEncoder.PLAY_MODE_OFFSET + int(ts.Resolution.OPS_COUP),
    "realign": ActionEncoder.PLAY_MODE_OFFSET + int(ts.Resolution.OPS_REALIGN),
}
OPS_MODES = ("influence", "coup", "realign")
OPS_FIRST = PLAY_MODES["influence"]


def _legal(state: ts.GameState) -> List[int]:
    return [i for i, v in enumerate(ts.ActionMask.generate_flat_mask(state)) if v]


def _roll(state: ts.GameState) -> None:
    ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))


def _is_chance(state: ts.GameState) -> bool:
    ctx = state.ctx()
    return ctx.decision_player == ts.Player.NONE and ctx.decision_type == ts.DecisionType.ROLL_DIE


def _at_offer(state: ts.GameState, drawn: int) -> bool:
    ctx = state.ctx()
    return (int(ctx.resolving_card) == GRAIN_SALES and ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE
            and ctx.decision_player == ts.Player.US and int(ctx.pending_op_card) == drawn)


def _to_offer(state: ts.GameState, drawn: int) -> ts.GameState:
    for _ in range(200):
        if _at_offer(state, drawn):
            return state
        if _is_chance(state):
            _roll(state)
        else:
            ts.Engine.step_flat(state, _legal(state)[0])
    raise AssertionError("the US was never offered the drawn card")


def _action_round(drawn: int) -> ts.GameState:
    state = _at_action_round()
    state.set_card_location(drawn, ts.hand_of(ts.Player.USSR))
    return state


def _us_plays_grain_sales(drawn: int) -> ts.GameState:
    state = _action_round(drawn)
    for card in (GRAIN_SALES, DUCK_AND_COVER):
        state.set_card_location(card, ts.hand_of(ts.Player.US))
    ts.Engine.step_flat(state, GRAIN_SALES - 1)
    ts.Engine.step_flat(state, PLAY_EVENT)
    return _to_offer(state, drawn)


def _ussr_plays_grain_sales(play_mode: int) -> Callable[[int], ts.GameState]:
    def build(drawn: int) -> ts.GameState:
        state = _action_round(drawn)
        state.phasing_player = ts.Player.USSR
        state.ctx().decision_player = ts.Player.USSR
        state.set_card_location(GRAIN_SALES, ts.hand_of(ts.Player.USSR))
        ts.Engine.step_flat(state, GRAIN_SALES - 1)
        ts.Engine.step_flat(state, play_mode)
        return _to_offer(state, drawn)
    return build


def _headlines_grain_sales(owner: ts.Player) -> Callable[[int], ts.GameState]:
    def build(drawn: int) -> ts.GameState:
        state = _at_headline()
        us_card = GRAIN_SALES if owner == ts.Player.US else DUCK_AND_COVER
        ussr_card = GRAIN_SALES if owner == ts.Player.USSR else CAMBRIDGE_FIVE
        _set_hand(state, ts.Player.US, [us_card, DUCK_AND_COVER, NUCLEAR_TEST_BAN])
        _set_hand(state, ts.Player.USSR, [ussr_card, drawn])
        for _ in range(2):
            ts.Engine.step_flat(state, (us_card if state.ctx().decision_player == ts.Player.US else ussr_card) - 1)
        return _to_offer(state, drawn)
    return build


#: name -> (builder of the position where the US is offered `drawn`, whether it is a headline)
PATHS: Dict[str, Callable[[int], ts.GameState]] = {
    "ar_us_plays": _us_plays_grain_sales,
    "ar_ussr_event_first": _ussr_plays_grain_sales(PLAY_MODES["event"]),
    "ar_ussr_ops_first": _ussr_plays_grain_sales(OPS_FIRST),
    "hl_us_headlines": _headlines_grain_sales(ts.Player.US),
    "hl_ussr_headlines": _headlines_grain_sales(ts.Player.USSR),
}
HEADLINE_PATHS = {"hl_us_headlines", "hl_ussr_headlines"}


def _us_plays(offer: ts.GameState, mode: str, headline: bool) -> ts.GameState:
    """The US answers the offer with `mode`; the play is driven to its end."""
    state = offer.clone()
    ts.Engine.step_flat(state, PLAY_MODES[mode])
    for _ in range(200):
        ctx = state.ctx()
        if ts.Engine.is_terminal(state):
            break
        if _is_chance(state):
            _roll(state)
        elif headline and state.current_phase != ts.Phase.HEADLINE:
            break
        elif (not headline and ctx.decision_type == ts.DecisionType.SELECT_CARD and int(ctx.resolving_card) == 0
              and int(state.ctx_stack_depth) == 0):
            break
        else:
            legal = _legal(state)
            if not legal:
                break
            ts.Engine.step_flat(state, legal[0])
    return state


@pytest.mark.parametrize("path", list(PATHS))
@pytest.mark.parametrize("mode", OPS_MODES)
def test_a_ussr_card_the_us_plays_for_ops_owes_the_ussr_its_event(path: str, mode: str) -> None:
    after = _us_plays(PATHS[path](QUAGMIRE), mode, path in HEADLINE_PATHS)
    assert after.has_flag(EB.QUAGMIRE_ACTIVE)
    assert after.get_card_location(QUAGMIRE) == ts.CardLocation.REMOVED_FROM_GAME


@pytest.mark.parametrize("path", list(PATHS))
@pytest.mark.parametrize("drawn", [MARSHALL_PLAN, SALT_NEGOTIATIONS], ids=["us_card", "neutral_card"])
@pytest.mark.parametrize("mode", OPS_MODES + ("space",))
def test_a_us_or_neutral_card_played_for_ops_triggers_no_event_and_is_not_removed(
        path: str, drawn: int, mode: str) -> None:
    """Spent for Operations (or the Space Race) it is discarded: a starred card leaves the game only
    when its Event occurs, and a card the US plays for Ops fires nobody's Event."""
    after = _us_plays(PATHS[path](drawn), mode, path in HEADLINE_PATHS)
    assert after.get_card_location(drawn) == ts.CardLocation.DISCARD_PILE
    if drawn == MARSHALL_PLAN:
        assert not after.has_flag(EB.MARSHALL_PLAN_PLAYED)


@pytest.mark.parametrize("path", list(PATHS))
@pytest.mark.parametrize("war", WARS)
def test_flower_power_charges_the_us_two_vp_for_a_drawn_war_card_except_for_the_space_race(
        path: str, war: int) -> None:
    offer = PATHS[path](war)
    for mode in PLAY_MODES:
        plain = _us_plays(offer, mode, path in HEADLINE_PATHS)
        flowered_offer = offer.clone()
        flowered_offer.set_flag(EB.FLOWER_POWER_ACTIVE)
        flowered = _us_plays(flowered_offer, mode, path in HEADLINE_PATHS)
        # Same position and same dice, so the war's own result cancels: what is left is the charge.
        charge = int(flowered.victory_points) - int(plain.victory_points)
        assert charge == (0 if mode == "space" else -2), f"{path} / war {war} / {mode}: charge {charge} VP"
