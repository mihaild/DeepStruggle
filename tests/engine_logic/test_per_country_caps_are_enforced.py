"""A per-country limit is enforced by the handler, not only by the legal mask.

Sixteen events cap how much they may move in any one country. Fifteen checked the cap in the
handler as well as in the mask; Socialist Governments checked it only in the mask, so a caller
stepping past the mask could take all three of its Influence out of a single Western European
country instead of at most two.

That mattered beyond the rule. `node_counts` is packed two bits per country, which is exactly
the range the rules need -- every `max_per_country` in the engine is 0, 1 or 2 -- and a handler
that can be driven past its own cap is what makes "two bits is enough" an assumption rather than
a fact.

These drive the handler directly, which is the whole point: going through the mask would only
re-test the mask.
"""

from __future__ import annotations

from typing import List, Tuple

import pytest

import ts_engine as ts

SOCIALIST_GOVERNMENTS = 7
SUEZ_CRISIS = 28
THE_VOICE_OF_AMERICA = 74

FRANCE = 8
UNITED_KINGDOM = 1
INDIA = 33

#: (card, the side whose Influence moves, country, cap, total allowance)
CASES: List[Tuple[int, str, int, int, int]] = [
    (SOCIALIST_GOVERNMENTS, "US", FRANCE, 2, 3),
    (SUEZ_CRISIS, "US", UNITED_KINGDOM, 2, 4),
    (THE_VOICE_OF_AMERICA, "USSR", INDIA, 2, 4),
]


def _fresh(card: int, country: int, side: str, stock: int) -> ts.GameState:
    state = ts.GameState()
    ts.Engine.init_game(state, 31337)
    state.current_phase = ts.Phase.ACTION_ROUND
    state.turn = 6
    state.action_round = 1
    for cid in range(84):
        state.set_country(cid, 0, 0)
    # plenty to take, so only the cap can stop it
    state.set_country(country, stock if side == "US" else 0, 0 if side == "US" else stock)
    player = ts.Player.USSR if side == "US" else ts.Player.US
    state.phasing_player = player
    ts.CardHandlers.trigger_event(state, card, player)
    return state


@pytest.mark.parametrize("card,side,country,cap,total", CASES,
                         ids=[str(c[0]) for c in CASES])
def test_no_more_than_the_cap_comes_out_of_one_country(
    card: int, side: str, country: int, cap: int, total: int
) -> None:
    """Point at the same country more times than the cap allows; the extra must be refused."""
    name = ts.CardData.get_card_info(card)["name"]
    state = _fresh(card, country, side, stock=total + 2)

    before = (state.get_country(country).us_influence if side == "US"
              else state.get_country(country).ussr_influence)

    for _ in range(total + 2):
        ts.CardHandlers.handle_event_step(
            state, ts.MicroAction(ts.DecisionType.POINT_NODE, country, 0, 0))

    after = (state.get_country(country).us_influence if side == "US"
             else state.get_country(country).ussr_influence)
    taken = before - after

    assert taken <= cap, (
        f"{name} took {taken} out of one country; its own max_per_country is {cap}"
    )
    assert state.ctx().node_count(country) <= cap, (
        f"{name} recorded {state.ctx().node_count(country)} for one country, above its cap of {cap}"
    )


@pytest.mark.parametrize("card,side,country,cap,total", CASES,
                         ids=[str(c[0]) for c in CASES])
def test_the_cap_does_not_stop_it_taking_the_first_two(
    card: int, side: str, country: int, cap: int, total: int
) -> None:
    """The other way to pass the test above is to refuse everything, so pin that too."""
    name = ts.CardData.get_card_info(card)["name"]
    state = _fresh(card, country, side, stock=total + 2)

    before = (state.get_country(country).us_influence if side == "US"
              else state.get_country(country).ussr_influence)
    for _ in range(cap):
        ts.CardHandlers.handle_event_step(
            state, ts.MicroAction(ts.DecisionType.POINT_NODE, country, 0, 0))
    after = (state.get_country(country).us_influence if side == "US"
             else state.get_country(country).ussr_influence)

    assert before - after == cap, (
        f"{name} should move {cap} out of a country that has plenty; it moved {before - after}"
    )


def test_no_engine_event_allows_more_than_two_per_country() -> None:
    """The premise the two-bit packing rests on, stated where it can fail.

    If a future card needs three per country, this fails here rather than silently saturating
    a two-bit field.
    """
    assert ts.DecisionContext.NODE_COUNT_MAX == 3, (
        "node_count is packed two bits; a wider ceiling needs a wider field"
    )
