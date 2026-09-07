"""Cancelling Cuban Missile Crisis is the US player's choice, not a fixed country.

Couping while the crisis stands against you cancels it, and the US pays 2 Influence from West
Germany *or* Turkey -- whichever it likes, when both can pay. Taken silently and always from
West Germany, that was right in eight of the corpus's ten US cancellations and wrong in two: at
turn 5 AR7 of ts-replayer game 234 the US holds 5 in West Germany and 2 in Turkey and pays from
Turkey.

The choice is asked before the coup's die, so the coup is already staged in temp_cards and the
chance node opens on the far side of the answer with nothing to rebuild. Where only one country
can pay there is nothing to choose and execute_coup settles it as before; the USSR side is
Cuba alone and has no choice at all.
"""
from typing import List

import pytest

from tools.lib.corpus_paths import corpus_dir
import ts_engine as ts

WEST_GERMANY, TURKEY, CUBA = 7, 12, 71
ANGOLA = 59
CUBAN_MISSILE_CRISIS = 40
DUCK_AND_COVER = 4
OP_COUP = 117
PASS = 211


def _bit(name: str) -> int:
    import re
    hdr = open("engine/include/ts/constants.hpp").read()
    m = re.search(rf"{name}\s*=\s*1(?:ull|ULL)?\s*<<\s*(\d+)", hdr)
    assert m, f"no effect bit named {name}"
    return 1 << int(m.group(1))


CMC_ACTIVE_USSR = _bit("CMC_ACTIVE_USSR")


CMC_ACTIVE_US = _bit("CMC_ACTIVE_US")


def _us_about_to_coup(west_germany: int, turkey: int) -> ts.GameState:
    """The US, holding a card, one action round into a crisis the USSR played.

    The engine offers the payoff at the head of the round, so a test that wants the coup
    declines it first -- see _coup_angola.
    """
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    guard = 0
    while state.current_phase != ts.Phase.ACTION_ROUND and guard < 600:
        guard += 1
        ctx = state.ctx()
        if (ctx.decision_player == ts.Player.NONE
                and ctx.decision_type == ts.DecisionType.ROLL_DIE):
            ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
            continue
        legal = [i for i, v in enumerate(ts.ActionMask.generate_flat_mask(state)) if v]
        ts.Engine.step_flat(state, legal[0])
    state.turn, state.action_round = 5, 7
    state.phasing_player = ts.Player.US
    ctx = state.ctx()
    ctx.decision_player, ctx.decision_type = ts.Player.US, ts.DecisionType.SELECT_CARD
    for c in range(1, 111):
        if state.get_card_location(c) in (ts.hand_of(ts.Player.US),
                                          ts.hand_of(ts.Player.USSR)):
            state.set_card_location(c, ts.CardLocation.DISCARD_PILE)
    state.set_card_location(DUCK_AND_COVER, ts.hand_of(ts.Player.US))
    state.china_card_holder = ts.Player.USSR
    state.china_card_playable = 0
    state.set_country(WEST_GERMANY, west_germany, 0)
    state.set_country(TURKEY, turkey, 0)
    state.set_country(ANGOLA, 0, 3)            # somewhere with USSR Influence to coup
    state.defcon = 5                            # so an Africa coup is legal
    state.persistent_effects |= CMC_ACTIVE_USSR
    return state


def _coup_angola(state: ts.GameState) -> ts.GameState:
    if int(state.ctx().resolving_card) == CUBAN_MISSILE_CRISIS:
        ts.Engine.step_flat(state, PASS)          # decline at the head of the round
    ts.Engine.step_flat(state, DUCK_AND_COVER - 1)
    ts.Engine.step_flat(state, 111)             # for Operations
    ts.Engine.step_flat(state, OP_COUP)
    ts.Engine.step_flat(state, 119 + ANGOLA)    # target
    return state


def _offered(state: ts.GameState) -> List[int]:
    """The countries on offer. The decline, where there is one, decodes to 255, not a country."""
    mask = ts.ActionMask.generate_flat_mask(state)
    ids = [int(ts.ActionMask.decode_flat_action(state, i).primary_id)
           for i in range(212) if mask[i]]
    return [c for c in ids if 0 <= c < 84]


def test_the_us_is_asked_when_either_country_could_pay() -> None:
    state = _coup_angola(_us_about_to_coup(west_germany=5, turkey=2))
    ctx = state.ctx()
    assert ctx.decision_type == ts.DecisionType.POINT_NODE
    assert ctx.decision_player == ts.Player.US
    assert int(ctx.resolving_card) == CUBAN_MISSILE_CRISIS
    assert sorted(_offered(state)) == sorted([WEST_GERMANY, TURKEY])


def test_paying_from_turkey_leaves_west_germany_alone() -> None:
    """Replay 234's choice."""
    state = _coup_angola(_us_about_to_coup(west_germany=5, turkey=2))
    ts.Engine.step_flat(state, 119 + TURKEY)
    assert int(state.get_country(TURKEY).us_influence) == 0
    assert int(state.get_country(WEST_GERMANY).us_influence) == 5
    assert state.ctx().decision_type == ts.DecisionType.ROLL_DIE, (
        "and the coup that provoked it goes on to roll")


def test_paying_from_west_germany_leaves_turkey_alone() -> None:
    state = _coup_angola(_us_about_to_coup(west_germany=5, turkey=2))
    ts.Engine.step_flat(state, 119 + WEST_GERMANY)
    assert int(state.get_country(WEST_GERMANY).us_influence) == 3
    assert int(state.get_country(TURKEY).us_influence) == 2
    assert state.ctx().decision_type == ts.DecisionType.ROLL_DIE


def test_no_choice_is_offered_when_only_one_country_can_pay() -> None:
    """Nothing to decide, so the coup goes straight to its die as it always did.

    The payment then happens inside the coup, which is where it always did -- so it shows
    once the die has been taken, not before.
    """
    state = _coup_angola(_us_about_to_coup(west_germany=5, turkey=1))
    assert state.ctx().decision_type == ts.DecisionType.ROLL_DIE
    assert int(state.ctx().resolving_card) != CUBAN_MISSILE_CRISIS
    ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 3, 0, 0))
    assert int(state.get_country(WEST_GERMANY).us_influence) == 3
    assert int(state.get_country(TURKEY).us_influence) == 1


def _offer_at_head_of_round(payer, **influence) -> ts.GameState:
    """The payoff as the engine poses it at the head of a round: a decision that may be declined.

    Built directly rather than by playing a turn out, because what is under test here is the
    decision itself -- that it offers the right countries, that declining leaves the crisis
    standing, and that paying does not open a coup's die. That the engine reaches this at the
    head of a round is covered end to end by replay 264.
    """
    state = _us_about_to_coup(west_germany=influence.get("west_germany", 5),
                              turkey=influence.get("turkey", 2))
    if payer == ts.Player.USSR:
        state.persistent_effects &= ~CMC_ACTIVE_USSR
        state.persistent_effects |= CMC_ACTIVE_US
        state.set_country(CUBA, 0, influence.get("cuba", 3))
    state.phasing_player = payer
    ctx = state.ctx()
    ctx.decision_player = payer
    ctx.decision_type = ts.DecisionType.POINT_NODE
    ctx.resolving_card = CUBAN_MISSILE_CRISIS
    ctx.remaining_steps = 1
    ctx.allow_early_stop = 1          # as the head of a round poses it
    return state                       # nothing staged behind it: no coup provoked this


def test_the_payoff_offers_a_decline_at_the_head_of_a_round() -> None:
    state = _offer_at_head_of_round(ts.Player.US)
    mask = ts.ActionMask.generate_flat_mask(state)
    assert mask[PASS], "paying is a choice, and usually declined"
    assert sorted(_offered(state)) == sorted([WEST_GERMANY, TURKEY])


def test_declining_leaves_the_crisis_standing() -> None:
    state = _offer_at_head_of_round(ts.Player.US)
    ts.Engine.step_flat(state, PASS)
    assert state.persistent_effects & CMC_ACTIVE_USSR, "nothing was paid"
    assert state.ctx().decision_type == ts.DecisionType.SELECT_CARD, (
        "and the player still has their action round to take")


def test_paying_at_the_head_of_a_round_opens_no_die() -> None:
    """No coup provoked it, so nothing follows but the player's own card."""
    state = _offer_at_head_of_round(ts.Player.US)
    ts.Engine.step_flat(state, 119 + TURKEY)
    assert int(state.get_country(TURKEY).us_influence) == 0
    assert not (state.persistent_effects & CMC_ACTIVE_USSR)
    assert state.ctx().decision_type == ts.DecisionType.SELECT_CARD


def test_the_ussr_pays_from_cuba_and_nowhere_else() -> None:
    state = _offer_at_head_of_round(ts.Player.USSR)
    assert _offered(state) == [CUBA]
    ts.Engine.step_flat(state, 119 + CUBA)
    assert int(state.get_country(CUBA).ussr_influence) == 1
    assert not (state.persistent_effects & CMC_ACTIVE_US)


def test_replay_264_converts_end_to_end() -> None:
    """The USSR pays out of Cuba, then plays "We Will Bury You" for its 4 Operations."""
    import gzip
    import json
    import os
    path = os.path.join(str(corpus_dir()), "264.json.gz")
    if not os.path.exists(path):
        pytest.skip("corpus not downloaded")
    from tools.lib.ts_replayer_convert import convert_game
    with gzip.open(path, "rt") as f:
        conv = convert_game(json.load(f))
    assert conv.failure is None, f"replay 264 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 144


def test_the_crisis_is_gone_once_it_is_paid_for() -> None:
    state = _coup_angola(_us_about_to_coup(west_germany=5, turkey=2))
    assert state.persistent_effects & CMC_ACTIVE_USSR
    ts.Engine.step_flat(state, 119 + TURKEY)
    assert not (state.persistent_effects & CMC_ACTIVE_USSR)
