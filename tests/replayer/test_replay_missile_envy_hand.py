"""Missile Envy's exchange is decided by the hand, so the hand has to be right first.

It takes the opponent's highest Ops card, with no decision to steer unless two tie. A hand
holding one card too many therefore hands over the wrong card, and the wrong event fires.

The turn's hand list in the log is everything a player held at some point during the turn,
including what they picked up part way through it -- SALT Negotiations retrieves a card from
the discard pile mid-turn, and that card appears in the list as though it had been dealt.

"Highest" is the card's value to the giver (owner's ruling, 2026-11-18): Containment, Brezhnev
Doctrine and Red Scare/Purge count, Vietnam Revolts does not, so a modifier can turn a card the
cap would have set aside into a tie the giver chose from.
"""
import ts_engine as ts

from tools.lib.ts_replayer_convert import _cap_draw_deck, _envy_modifier_flags, _seed_missile_envy_hand
from tests.replay_corpus import ReplayCorpus

DUCK_AND_COVER = 4
CONTAINMENT = 25
CIA_CREATED = 26
SUEZ_CRISIS = 28
RED_SCARE_PURGE = 31
MISSILE_ENVY = 49
BREZHNEV_DOCTRINE = 51
CAMP_DAVID = 65


def _state_with_us_hand(cards) -> ts.GameState:
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    for c in range(1, 111):
        if ts.in_hand_of(state.get_card_location(c), ts.Player.US):
            state.set_card_location(c, ts.CardLocation.DISCARD_PILE)
    for c in cards:
        state.set_card_location(c, ts.hand_of(ts.Player.US))
    return state


def _us_hand(state: ts.GameState):
    return {c for c in range(1, 111)
            if ts.in_hand_of(state.get_card_location(c), ts.Player.US)}


def test_a_card_the_giver_could_not_have_held_is_set_aside() -> None:
    """Replay 64 turn 5 AR3: the log hands over Suez Crisis at 3 Ops.

    Red Scare/Purge, at 4, is in the turn's hand list but was retrieved from the discard pile
    four action rounds later, so it cannot have been there to give.
    """
    state = _state_with_us_hand([SUEZ_CRISIS, RED_SCARE_PURGE, CAMP_DAVID])
    _seed_missile_envy_hand(state, SUEZ_CRISIS, ts.Player.US)
    hand = _us_hand(state)
    assert SUEZ_CRISIS in hand
    assert RED_SCARE_PURGE not in hand, "4 Ops would have been taken instead of the logged 3"
    assert CAMP_DAVID in hand, "cards at or below the logged card are left alone"


def test_the_named_card_is_put_in_hand_if_it_is_missing() -> None:
    state = _state_with_us_hand([CAMP_DAVID])
    _seed_missile_envy_hand(state, SUEZ_CRISIS, ts.Player.US)
    assert SUEZ_CRISIS in _us_hand(state)


def test_a_scoring_card_is_never_set_aside() -> None:
    """Missile Envy cannot take one, so a scoring card never competes to be the highest."""
    scoring = next(c for c in range(1, 111) if ts.CardData.get_card_info(c)["is_scoring"])
    state = _state_with_us_hand([SUEZ_CRISIS, scoring])
    _seed_missile_envy_hand(state, SUEZ_CRISIS, ts.Player.US)
    assert scoring in _us_hand(state)


def test_under_purge_a_card_that_ties_the_logged_one_stays() -> None:
    """Purged, the US values Suez Crisis's 3 and Red Scare/Purge's 4 at 2 and 3: still above. But
    Duck and Cover's 3 ties Suez Crisis at 2, so it may have been in the hand the US chose from."""
    state = _state_with_us_hand([SUEZ_CRISIS, DUCK_AND_COVER, RED_SCARE_PURGE, CIA_CREATED])
    state.set_flag(int(ts.EffectBits.PURGE_US_ACTIVE))
    _seed_missile_envy_hand(state, SUEZ_CRISIS, ts.Player.US)
    hand = _us_hand(state)
    assert {SUEZ_CRISIS, DUCK_AND_COVER, CIA_CREATED} <= hand
    assert RED_SCARE_PURGE not in hand


def test_under_containment_a_four_ties_a_logged_three() -> None:
    state = _state_with_us_hand([SUEZ_CRISIS, RED_SCARE_PURGE])
    state.set_flag(int(ts.EffectBits.CONTAINMENT_ACTIVE))
    _seed_missile_envy_hand(state, SUEZ_CRISIS, ts.Player.US)
    assert RED_SCARE_PURGE in _us_hand(state), "3 and 4 are both 4 under Containment"


def test_a_modifier_on_the_other_side_is_not_the_givers() -> None:
    state = _state_with_us_hand([SUEZ_CRISIS, RED_SCARE_PURGE])
    state.set_flag(int(ts.EffectBits.BREZHNEV_DOCTRINE_ACTIVE))
    state.set_flag(int(ts.EffectBits.PURGE_USSR_ACTIVE))
    _seed_missile_envy_hand(state, SUEZ_CRISIS, ts.Player.US)
    assert RED_SCARE_PURGE not in _us_hand(state)


def test_a_headline_resolving_first_sets_the_modifier_the_cap_reads() -> None:
    """Headlined against Missile Envy, Containment / Brezhnev Doctrine / Red Scare/Purge resolve
    first, inside the same engine step, so the cap must value the hand as they will leave it."""
    assert _envy_modifier_flags({"US": CONTAINMENT, "USSR": MISSILE_ENVY}) == [
        int(ts.EffectBits.CONTAINMENT_ACTIVE)]
    assert _envy_modifier_flags({"US": MISSILE_ENVY, "USSR": BREZHNEV_DOCTRINE}) == [
        int(ts.EffectBits.BREZHNEV_DOCTRINE_ACTIVE)]
    assert _envy_modifier_flags({"US": MISSILE_ENVY, "USSR": RED_SCARE_PURGE}) == [
        int(ts.EffectBits.PURGE_US_ACTIVE)]
    assert _envy_modifier_flags({"US": CONTAINMENT}) == [], "no Missile Envy headlined"

    state = _state_with_us_hand([SUEZ_CRISIS, RED_SCARE_PURGE])
    _seed_missile_envy_hand(state, SUEZ_CRISIS, ts.Player.US,
                            extra_flags=_envy_modifier_flags({"US": CONTAINMENT, "USSR": MISSILE_ENVY}))
    assert RED_SCARE_PURGE in _us_hand(state)
    assert not state.has_flag(int(ts.EffectBits.CONTAINMENT_ACTIVE)), "the state itself is not changed"


def test_the_draw_deck_cap_values_cards_as_the_giver_would() -> None:
    state = _state_with_us_hand([SUEZ_CRISIS])
    for c in (DUCK_AND_COVER, RED_SCARE_PURGE):
        state.set_card_location(c, ts.CardLocation.DRAW_DECK)
    # Under Containment every card the US could draw is worth at most 4, as Suez Crisis is.
    assert _cap_draw_deck(state, SUEZ_CRISIS, ts.Player.US, [int(ts.EffectBits.CONTAINMENT_ACTIVE)]) == []
    parked = _cap_draw_deck(state, SUEZ_CRISIS, ts.Player.US)
    assert RED_SCARE_PURGE in parked and DUCK_AND_COVER not in parked


def test_replay_64_converts_end_to_end(corpus: ReplayCorpus) -> None:
    conv = corpus.converted(64)
    assert conv.failure is None, f"replay 64 stopped at {conv.failure}"
    # Everything but the last turn, which the recording stops inside -- see
    # tests/replayer/test_replay_unfinished_final_turn.py.
    assert conv.entries_converted == 106
