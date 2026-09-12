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
                              HOW_I_LEARNED, JUNTA, KAL_007, LONE_GUNMAN, NUCLEAR_SUBS_ACTIVE,
                              OLYMPIC_GAMES, ORTEGA_ELECTED, STAR_WARS, TEAR_DOWN_THIS_WALL,
                              WE_WILL_BURY_YOU, defcon_suicide_cards, is_us_event,
                              us_played_degraders)
from ai.eval.positions import PositionBuilder

ALGERIA, CUBA, NICARAGUA, MOROCCO = 47, 71, 69, 46
AMERICAS_REGIONS = (4, 5)
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


def ussr(hand=NEUTRAL_HAND, influence=(), clear_europe=True, flags=(), discard=(),
         us_space=0, ussr_space=0):
    return PositionBuilder(
        hand=hand, side=ts.Player.USSR, defcon=2, turn=8, action_round=1,
        influence=influence, flags=flags, discard=discard,
        us_space=us_space, ussr_space=ussr_space,
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


def test_the_unconditional_degraders_reach_the_ussr_anywhere() -> None:
    """Taken from a self-play game that ended this way.

    The US took Duck and Cover out of a pile that also held Junta, and the USSR lost from a hand
    holding two safe alternatives. These three degrade from their own text, so where the USSR
    stands is irrelevant -- the position below has no influence anywhere couppable.
    """
    for degrader in (DUCK_AND_COVER, KAL_007, HOW_I_LEARNED):
        pos = ussr(discard=(degrader,), us_space=3, ussr_space=1)
        assert STAR_WARS in defcon_suicide_cards(pos, ts.Player.USSR), degrader


def test_the_coup_degraders_each_need_a_battleground_in_their_own_region() -> None:
    """Junta reaches the Americas, Tear Down This Wall Europe, the Ops cards either."""
    # Junta: exposed by Cuba (Americas battleground), not by Algeria (African one).
    assert STAR_WARS in defcon_suicide_cards(
        ussr(discard=(JUNTA,), influence=((CUBA, ts.Player.USSR, 2),),
             us_space=3, ussr_space=1), ts.Player.USSR)
    assert STAR_WARS not in defcon_suicide_cards(
        ussr(discard=(JUNTA,), influence=((ALGERIA, ts.Player.USSR, 2),),
             us_space=3, ussr_space=1), ts.Player.USSR)
    # Grain Sales / CIA Created hand the US Operations, so either region exposes them.
    for region_country in (CUBA, ALGERIA):
        assert STAR_WARS in defcon_suicide_cards(
            ussr(discard=(GRAIN_SALES,), influence=((region_country, ts.Player.USSR, 2),),
                 us_space=3, ussr_space=1), ts.Player.USSR), region_country
    # Tear Down This Wall is Europe-locked: the opening's East Germany influence exposes it.
    assert STAR_WARS in defcon_suicide_cards(
        ussr(discard=(TEAR_DOWN_THIS_WALL,), clear_europe=False,
             us_space=3, ussr_space=1), ts.Player.USSR)
    assert STAR_WARS not in defcon_suicide_cards(
        ussr(discard=(TEAR_DOWN_THIS_WALL,), influence=((CUBA, ts.Player.USSR, 2),),
             us_space=3, ussr_space=1), ts.Player.USSR)


def test_a_coup_degrader_with_nothing_to_coup_is_not_a_danger() -> None:
    """No battleground exposed anywhere means the pick cannot reach DEFCON."""
    for degrader in (JUNTA, GRAIN_SALES, CIA_CREATED, TEAR_DOWN_THIS_WALL):
        pos = ussr(discard=(degrader,), us_space=3, ussr_space=1)
        assert STAR_WARS not in defcon_suicide_cards(pos, ts.Player.USSR), degrader


def test_five_year_plan_only_fires_us_events() -> None:
    """It discards a random USSR card and fires it *only if it is a US event*.

    Junta and How I Learned are neutral, so a hand holding only those is safe from it even
    though both are degraders in the Star Wars sense.
    """
    assert not is_us_event(JUNTA) and not is_us_event(HOW_I_LEARNED)
    assert is_us_event(DUCK_AND_COVER) and is_us_event(GRAIN_SALES)
    for neutral in (JUNTA, HOW_I_LEARNED):
        pos = ussr(hand=(neutral, 23), influence=((CUBA, ts.Player.USSR, 2),))
        assert FIVE_YEAR_PLAN not in us_played_degraders(pos, subs=False), neutral
    pos = ussr(hand=(DUCK_AND_COVER, 23))
    assert FIVE_YEAR_PLAN in us_played_degraders(pos, subs=False)


def test_five_year_plan_in_the_discard_needs_a_degrader_in_hand() -> None:
    bare = ussr(hand=(23, 27), discard=(FIVE_YEAR_PLAN,), us_space=3, ussr_space=1)
    assert STAR_WARS not in defcon_suicide_cards(bare, ts.Player.USSR)
    loaded = ussr(hand=(DUCK_AND_COVER, 23), discard=(FIVE_YEAR_PLAN,),
                  us_space=3, ussr_space=1)
    assert STAR_WARS in defcon_suicide_cards(loaded, ts.Player.USSR)


def test_nuclear_subs_leaves_the_direct_degraders_alone() -> None:
    """It exempts US *coups*, so the coup-based degraders go and the direct ones stay."""
    pos = ussr(discard=(JUNTA,), influence=((CUBA, ts.Player.USSR, 2),),
               us_space=3, ussr_space=1, flags=(NUCLEAR_SUBS_ACTIVE,))
    assert JUNTA not in us_played_degraders(pos, subs=True)
    assert DUCK_AND_COVER in us_played_degraders(pos, subs=True)


def test_the_ussr_star_wars_clause_still_needs_the_space_lead_and_a_degrader() -> None:
    def case(**kw):
        base = dict(discard=(DUCK_AND_COVER,), us_space=3, ussr_space=1)
        base.update(kw)
        return defcon_suicide_cards(ussr(**base), ts.Player.USSR)

    # Without the lead the event does nothing at all.
    assert STAR_WARS not in case(us_space=1, ussr_space=3)
    # An empty-of-degraders pile leaves nothing worth taking.
    assert STAR_WARS not in case(discard=(23,))


def test_cards_the_ussr_can_decline_are_not_star_wars_dangers() -> None:
    """Lone Gunman and Ortega hand the USSR the Operations; it simply does not coup.

    We Will Bury You is excluded for a different reason: the US playing it degrades DEFCON by
    the US's own action, so the US would be choosing its own loss.
    """
    for safe in (LONE_GUNMAN, ORTEGA_ELECTED, OLYMPIC_GAMES, WE_WILL_BURY_YOU):
        pos = ussr(discard=(safe,), us_space=3, ussr_space=1)
        assert STAR_WARS not in defcon_suicide_cards(pos, ts.Player.USSR), safe


def test_nuclear_subs_disarms_a_coup_only_discard() -> None:
    """With only Junta in the pile and its coup exempted, there is nothing left to fear."""
    pos = ussr(discard=(JUNTA,), influence=((CUBA, ts.Player.USSR, 2),),
               us_space=3, ussr_space=1, flags=(NUCLEAR_SUBS_ACTIVE,))
    assert STAR_WARS not in defcon_suicide_cards(pos, ts.Player.USSR)


def test_five_year_plan_sees_star_wars_once_it_is_banned() -> None:
    """Whatever is in the banned set is what a random discard can hit."""
    pos = ussr(hand=(STAR_WARS, 23), discard=(DUCK_AND_COVER,), us_space=3, ussr_space=1)
    banned = defcon_suicide_cards(pos, ts.Player.USSR)
    assert STAR_WARS in banned and FIVE_YEAR_PLAN in banned


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
