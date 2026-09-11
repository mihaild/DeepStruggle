"""Which cards a side must not play at DEFCON 2, clause by clause.

"Must not play" means the *opponent's* event fires whether the holder likes it or not and takes
DEFCON to 1 by the holder's own action. A card whose own event the holder chooses is not here:
declining is free, so playing it for Operations is safe and only the Event is the mistake.
Olympic Games is the standing example and has its own rule.

Only a coup in a **battleground** degrades DEFCON, so every "hands the opponent Operations" card
is conditioned on battleground influence rather than influence anywhere in the region. Checking
any country made these cards look dangerous in positions where they were not.
"""

import ts_engine as ts

from ai.eval.blunders import (CIA_CREATED, DUCK_AND_COVER, FIVE_YEAR_PLAN, GRAIN_SALES,
                              HOW_I_LEARNED, KAL_007, LONE_GUNMAN, NUCLEAR_SUBS_ACTIVE,
                              OLYMPIC_GAMES, ORTEGA_ELECTED, STAR_WARS, TEAR_DOWN_THIS_WALL,
                              WE_WILL_BURY_YOU, defcon_suicide_cards)
from ai.eval.positions import PositionBuilder

ALGERIA, CUBA, NICARAGUA, MOROCCO = 47, 71, 69, 46
EUROPE_REGION = 0
COUPABLE_REGIONS = (3, 4, 5)      # Africa, Central America, South America
NEUTRAL_HAND = (23, 27)          # Marshall Plan, US/Japan Pact -- dangerous to neither side


# PositionBuilder applies `clear_influence` *after* `influence` (positions.py:93-96), so a
# country named in both ends at zero. Anything the test places explicitly is kept out of the
# clear list rather than placed and then silently wiped.
def _cleared(player: ts.Player, regions, keep=()) -> tuple[tuple[int, ts.Player], ...]:
    kept = {int(c) for c, *_ in keep}
    return tuple((cid, player) for cid in range(84)
                 if int(ts.MapData.get_country_info(cid)["region"]) in regions
                 and cid not in kept)


def ussr(hand=NEUTRAL_HAND, influence=(), clear_europe=True, flags=(), discard=()):
    return PositionBuilder(
        hand=hand, side=ts.Player.USSR, defcon=2, turn=8, action_round=1,
        influence=influence, flags=flags, discard=discard,
        clear_influence=(_cleared(ts.Player.USSR, (EUROPE_REGION,), influence)
                         if clear_europe else ()),
    ).build()


def us(hand=NEUTRAL_HAND, influence=(), flags=(), discard=(), us_space=0, ussr_space=0,
       clear_coupable=True):
    return PositionBuilder(
        hand=hand, side=ts.Player.US, defcon=2, turn=8, action_round=1,
        influence=influence, flags=flags, discard=discard,
        us_space=us_space, ussr_space=ussr_space,
        # The opening leaves the US 1 in South Africa, an African battleground, so a
        # position with nothing exposed has to say so explicitly.
        clear_influence=(_cleared(ts.Player.US, COUPABLE_REGIONS, influence)
                         if clear_coupable else ()),
    ).build()


# --- USSR ---------------------------------------------------------------------------------

def test_duck_and_cover_and_kal_007_are_unconditional() -> None:
    """Both degrade DEFCON from their own text; there is nothing for the USSR to avoid."""
    banned = defcon_suicide_cards(ussr(), ts.Player.USSR)
    assert DUCK_AND_COVER in banned and KAL_007 in banned


def test_cia_and_grain_sales_need_a_battleground_to_coup() -> None:
    """They hand the US Operations; without a battleground exposed there is nothing to lose."""
    empty = defcon_suicide_cards(ussr(), ts.Player.USSR)
    assert CIA_CREATED not in empty and GRAIN_SALES not in empty

    exposed = defcon_suicide_cards(ussr(influence=((ALGERIA, ts.Player.USSR, 2),)),
                                   ts.Player.USSR)
    assert CIA_CREATED in exposed and GRAIN_SALES in exposed


def test_influence_in_a_non_battleground_is_not_enough() -> None:
    """Morocco is African and coupable, but losing it does not end the game."""
    banned = defcon_suicide_cards(ussr(influence=((MOROCCO, ts.Player.USSR, 2),)),
                                  ts.Player.USSR)
    assert GRAIN_SALES not in banned and CIA_CREATED not in banned


def test_tear_down_this_wall_is_exposed_by_european_battlegrounds_alone() -> None:
    """It grants its coup *in Europe*, overriding the DEFCON 2 rule that closes the region."""
    assert TEAR_DOWN_THIS_WALL not in defcon_suicide_cards(ussr(), ts.Player.USSR)
    # An African battleground does not expose it -- the card's coup is region-locked to Europe.
    africa_only = defcon_suicide_cards(ussr(influence=((ALGERIA, ts.Player.USSR, 2),)),
                                       ts.Player.USSR)
    assert TEAR_DOWN_THIS_WALL not in africa_only
    # The opening setup's East Germany influence is a European battleground.
    assert TEAR_DOWN_THIS_WALL in defcon_suicide_cards(ussr(clear_europe=False), ts.Player.USSR)


def test_nuclear_subs_disarms_the_cards_that_hand_the_us_operations() -> None:
    """Nuclear Subs exempts *US* battleground coups, so these stop being dangerous."""
    pos = ussr(influence=((ALGERIA, ts.Player.USSR, 2),), flags=(NUCLEAR_SUBS_ACTIVE,),
               clear_europe=False)
    banned = defcon_suicide_cards(pos, ts.Player.USSR)
    assert GRAIN_SALES not in banned and CIA_CREATED not in banned
    assert TEAR_DOWN_THIS_WALL not in banned
    # The cards that degrade DEFCON from their own text are untouched by it.
    assert DUCK_AND_COVER in banned and KAL_007 in banned


def test_five_year_plan_is_dangerous_only_with_something_for_it_to_hit() -> None:
    """It discards a random USSR card and fires it if it is a US event."""
    assert FIVE_YEAR_PLAN not in defcon_suicide_cards(ussr(), ts.Player.USSR)
    with_target = ussr(hand=(DUCK_AND_COVER, 23))
    assert FIVE_YEAR_PLAN in defcon_suicide_cards(with_target, ts.Player.USSR)


def test_the_ussr_may_play_its_own_defcon_cards_for_operations() -> None:
    """We Will Bury You is the USSR's own event, so declining it is free."""
    assert WE_WILL_BURY_YOU not in defcon_suicide_cards(ussr(clear_europe=False),
                                                        ts.Player.USSR)


# --- US -----------------------------------------------------------------------------------

def test_we_will_bury_you_is_unconditional_for_the_us() -> None:
    """A USSR event that degrades DEFCON outright; the US cannot decline it."""
    assert WE_WILL_BURY_YOU in defcon_suicide_cards(us(), ts.Player.US)


def test_lone_gunman_needs_an_exposed_us_battleground() -> None:
    """The opening already exposes it -- the US starts with 1 in South Africa."""
    assert LONE_GUNMAN in defcon_suicide_cards(us(clear_coupable=False), ts.Player.US)
    assert LONE_GUNMAN not in defcon_suicide_cards(us(), ts.Player.US)
    exposed = defcon_suicide_cards(us(influence=((ALGERIA, ts.Player.US, 2),)), ts.Player.US)
    assert LONE_GUNMAN in exposed


def test_ortega_is_about_cuba_specifically() -> None:
    """Its free coup reaches a neighbour of Nicaragua; Cuba is the battleground among them."""
    assert ORTEGA_ELECTED not in defcon_suicide_cards(us(), ts.Player.US)
    # Influence in Nicaragua itself is not a battleground and does not expose it.
    assert ORTEGA_ELECTED not in defcon_suicide_cards(
        us(influence=((NICARAGUA, ts.Player.US, 2),)), ts.Player.US)
    assert ORTEGA_ELECTED in defcon_suicide_cards(
        us(influence=((CUBA, ts.Player.US, 1),)), ts.Player.US)


def test_star_wars_needs_the_space_lead_and_a_dangerous_discard() -> None:
    """The pick is mandatory and fizzles only on an empty pile, so the discard decides it."""
    behind = us(us_space=1, ussr_space=3, discard=(DUCK_AND_COVER,))
    assert STAR_WARS not in defcon_suicide_cards(behind, ts.Player.US)

    ahead_safe = us(us_space=3, ussr_space=1, discard=(23,))
    assert STAR_WARS not in defcon_suicide_cards(ahead_safe, ts.Player.US)

    ahead_dangerous = us(us_space=3, ussr_space=1, discard=(DUCK_AND_COVER,))
    assert STAR_WARS in defcon_suicide_cards(ahead_dangerous, ts.Player.US)


def test_star_wars_ignores_the_cards_whose_operations_go_to_the_opponent() -> None:
    """Lone Gunman, Ortega and Five Year Plan act through the USSR, not the US's own hand."""
    for safe_card in (LONE_GUNMAN, ORTEGA_ELECTED, FIVE_YEAR_PLAN):
        pos = us(us_space=3, ussr_space=1, discard=(safe_card,))
        assert STAR_WARS not in defcon_suicide_cards(pos, ts.Player.US), safe_card


def test_star_wars_counts_how_i_learned_and_olympic_games() -> None:
    """How I Learned sets DEFCON outright; Olympic Games ends the game through a boycott."""
    for danger in (HOW_I_LEARNED, OLYMPIC_GAMES):
        pos = us(us_space=3, ussr_space=1, discard=(danger,))
        assert STAR_WARS in defcon_suicide_cards(pos, ts.Player.US), danger
