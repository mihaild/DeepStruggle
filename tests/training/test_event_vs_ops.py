"""The rule that makes the Event/Ops comparison meaningful, and the pairing that makes it fair.

If placing influence did not cost double in a country the opponent controls, there would be no
reason for a free-placement event to prefer different targets from ordinary Operations, and the
probe would be measuring nothing.
"""
from __future__ import annotations

import pytest

import ts_engine as ts
from ai.eval.event_vs_ops import DEFAULT_CARDS, _country_flags
from ai.eval.positions import PositionBuilder

POLAND = next(c for c in range(84)
              if str(ts.MapData.get_country_info(c)["name"]) == "Poland")


def test_control_is_what_the_probe_says_it_is() -> None:
    """The probe splits countries by `Scoring::is_controlled_by`, so that boundary must be the
    engine's own and must actually move with influence.

    The 2-Ops-per-point premium that motivates the whole comparison lives in
    `Operations::get_influence_cost` (`engine/src/ops.cpp`), which the bindings do not export --
    so it is cited rather than asserted here. What is asserted is the predicate the probe
    partitions on.
    """
    contested = PositionBuilder(hand=(1,), side=ts.Player.US, turn=5,
                                influence=((POLAND, ts.Player.USSR, 1),)).build()
    held = PositionBuilder(hand=(1,), side=ts.Player.US, turn=5,
                           influence=((POLAND, ts.Player.USSR, 4),)).build()
    assert not ts.Scoring.is_controlled_by(contested, POLAND, ts.Player.USSR)
    assert ts.Scoring.is_controlled_by(held, POLAND, ts.Player.USSR)


def test_every_probe_card_is_held_by_its_own_side() -> None:
    """The pairing depends on it: playing an *opponent's* card for Ops fires their event, so the
    two branches would no longer differ in one decision only."""
    for card, holder in DEFAULT_CARDS:
        side = str(ts.CardData.get_card_info(card)["side"])
        assert side == ("US" if holder == ts.Player.US else "USSR"), (
            f"{ts.CardData.get_card_name(card)} is a {side} card but the probe has "
            f"{holder} holding it; playing it for Ops would fire the event")


def test_reachable_is_a_subset_of_controlled() -> None:
    """'Controlled but still contestable' must never count a country nobody controls."""
    st = PositionBuilder(hand=(1,), side=ts.Player.US, turn=5,
                         influence=((POLAND, ts.Player.USSR, 4),)).build()
    legal = list(range(84))
    controlled, reachable = _country_flags(st, ts.Player.US, legal)
    assert ((reachable == 1) <= (controlled == 1)).all()
    assert controlled[POLAND] == 1.0


def test_a_country_held_far_beyond_the_threshold_is_not_reachable() -> None:
    """A country the opponent holds overwhelmingly is nobody's target, and must not inflate the
    headline number."""
    deep = PositionBuilder(hand=(1,), side=ts.Player.US, turn=5,
                           influence=((POLAND, ts.Player.USSR, 12),)).build()
    _controlled, reachable = _country_flags(deep, ts.Player.US, [POLAND])
    assert reachable[0] == 0.0


@pytest.mark.parametrize("card,holder", DEFAULT_CARDS)
def test_probe_cards_exist_and_carry_operations(card: int, holder: ts.Player) -> None:
    info = ts.CardData.get_card_info(card)
    assert int(info["ops"]) >= 2, "a card with no Ops has no Ops branch to compare against"
    assert not bool(info["is_scoring"])
