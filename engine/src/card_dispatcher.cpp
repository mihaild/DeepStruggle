#include "ts/card_handlers.hpp"
#include "ts/invariant.hpp"
#include "ts/war_events.hpp"
#include "ts/card_data.hpp"
#include "ts/map_data.hpp"
#include "ts/scoring.hpp"
#include "ts/space_race.hpp"
#include "ts/ops.hpp"
#include "ts/prng.hpp"
#include "ts/defcon.hpp"
#include <algorithm>

namespace ts {

namespace early_war {
    bool trigger_duck_and_cover(GameState& state, Player p) noexcept;
    bool trigger_five_year_plan(GameState& state, Player p) noexcept;
    bool trigger_socialist_governments(GameState& state, Player p) noexcept;
    bool trigger_fidel(GameState& state, Player p) noexcept;
    bool trigger_vietnam_revolts(GameState& state, Player p) noexcept;
    bool trigger_blockade(GameState& state, Player p) noexcept;
    bool trigger_korean_war(GameState& state, Player p, uint8_t forced_roll = 0) noexcept;
    bool trigger_romanian_abdication(GameState& state, Player p) noexcept;
    bool trigger_arab_israeli_war(GameState& state, Player p, uint8_t forced_roll = 0) noexcept;
    bool trigger_comecon(GameState& state, Player p) noexcept;
    bool trigger_nasser(GameState& state, Player p) noexcept;
    bool trigger_warsaw_pact(GameState& state, Player p) noexcept;
    bool trigger_de_gaulle(GameState& state, Player p) noexcept;
    bool trigger_captured_nazi_scientist(GameState& state, Player p) noexcept;
    bool trigger_truman_doctrine(GameState& state, Player p) noexcept;
    bool trigger_olympic_games(GameState& state, Player p) noexcept;
    bool trigger_nato(GameState& state, Player p) noexcept;
    bool trigger_independent_reds(GameState& state, Player p) noexcept;
    bool trigger_marshall_plan(GameState& state, Player p) noexcept;
    bool trigger_indo_pakistani_war(GameState& state, Player p) noexcept;
    bool trigger_containment(GameState& state, Player p) noexcept;
    bool trigger_cia_created(GameState& state, Player p) noexcept;
    bool trigger_us_japan_pact(GameState& state, Player p) noexcept;
    bool trigger_suez_crisis(GameState& state, Player p) noexcept;
    bool trigger_east_european_unrest(GameState& state, Player p) noexcept;
    bool trigger_decolonization(GameState& state, Player p) noexcept;
    bool trigger_red_scare_purge(GameState& state, Player p) noexcept;
    bool trigger_un_intervention(GameState& state, Player p) noexcept;
    bool trigger_de_stalinization(GameState& state, Player p) noexcept;
    bool trigger_nuclear_test_ban(GameState& state, Player p) noexcept;
    bool trigger_formosan_resolution(GameState& state, Player p) noexcept;
    bool trigger_defectors(GameState& state, Player p) noexcept;
    bool trigger_cambridge_five(GameState& state, Player p) noexcept;
    bool trigger_special_relationship(GameState& state, Player p) noexcept;
    bool trigger_norad(GameState& state, Player p) noexcept;
}

namespace mid_war {
    bool trigger_arms_race(GameState& state, Player p) noexcept;
    bool trigger_brush_war(GameState& state, Player p) noexcept;
    bool trigger_cuban_missile_crisis(GameState& state, Player p) noexcept;
    bool trigger_nuclear_subs(GameState& state, Player p) noexcept;
    bool trigger_quagmire(GameState& state, Player p) noexcept;
    bool trigger_salt_negotiations(GameState& state, Player p) noexcept;
    bool trigger_bear_trap(GameState& state, Player p) noexcept;
    bool trigger_summit(GameState& state, Player p) noexcept;
    bool trigger_how_i_learned_to_stop_worrying(GameState& state, Player p) noexcept;
    bool trigger_junta(GameState& state, Player p) noexcept;
    bool trigger_kitchen_debates(GameState& state, Player p) noexcept;
    bool trigger_missile_envy(GameState& state, Player p) noexcept;
    bool trigger_we_will_bury_you(GameState& state, Player p) noexcept;
    bool trigger_brezhnev_doctrine(GameState& state, Player p) noexcept;
    bool trigger_portuguese_empire(GameState& state, Player p) noexcept;
    bool trigger_south_african_unrest(GameState& state, Player p) noexcept;
    bool trigger_allende(GameState& state, Player p) noexcept;
    bool trigger_willy_brandt(GameState& state, Player p) noexcept;
    bool trigger_muslim_revolution(GameState& state, Player p) noexcept;
    bool trigger_abm_treaty(GameState& state, Player p) noexcept;
    bool trigger_cultural_revolution(GameState& state, Player p) noexcept;
    bool trigger_flower_power(GameState& state, Player p) noexcept;
    bool trigger_u2_incident(GameState& state, Player p) noexcept;
    bool trigger_opec(GameState& state, Player p) noexcept;
    bool trigger_lone_gunman(GameState& state, Player p) noexcept;
    bool trigger_colonial_rear_guards(GameState& state, Player p) noexcept;
    bool trigger_panama_canal(GameState& state, Player p) noexcept;
    bool trigger_camp_david(GameState& state, Player p) noexcept;
    bool trigger_puppet_governments(GameState& state, Player p) noexcept;
    bool trigger_grain_sales(GameState& state, Player p) noexcept;
    bool trigger_john_paul_ii(GameState& state, Player p) noexcept;
    bool trigger_latin_death_squads(GameState& state, Player p) noexcept;
    bool trigger_oas_founded(GameState& state, Player p) noexcept;
    bool trigger_nixon_china_card(GameState& state, Player p) noexcept;
    bool trigger_sadat_expels_soviets(GameState& state, Player p) noexcept;
    bool trigger_shuttle_diplomacy(GameState& state, Player p) noexcept;
    bool trigger_voice_of_america(GameState& state, Player p) noexcept;
    bool trigger_liberation_theology(GameState& state, Player p) noexcept;
    bool trigger_ussuri_river(GameState& state, Player p) noexcept;
    bool trigger_ask_not(GameState& state, Player p) noexcept;
    bool trigger_alliance_for_progress(GameState& state, Player p) noexcept;
    bool trigger_one_small_step(GameState& state, Player p) noexcept;
    bool trigger_che(GameState& state, Player p) noexcept;
    bool trigger_our_man_in_tehran(GameState& state, Player p) noexcept;
}

namespace late_war {
    bool trigger_iranian_hostage_crisis(GameState& state, Player p) noexcept;
    bool trigger_iron_lady(GameState& state, Player p) noexcept;
    bool trigger_reagan_bombs_libya(GameState& state, Player p) noexcept;
    bool trigger_star_wars(GameState& state, Player p) noexcept;
    bool trigger_north_sea_oil(GameState& state, Player p) noexcept;
    bool trigger_the_reformer(GameState& state, Player p) noexcept;
    bool trigger_marine_barracks_bombing(GameState& state, Player p) noexcept;
    bool trigger_soviets_shoot_down_kal(GameState& state, Player p) noexcept;
    bool trigger_glasnost(GameState& state, Player p) noexcept;
    bool trigger_ortega_elected(GameState& state, Player p) noexcept;
    bool trigger_terrorism(GameState& state, Player p) noexcept;
    bool trigger_iran_contra(GameState& state, Player p) noexcept;
    bool trigger_chernobyl(GameState& state, Player p) noexcept;
    bool trigger_latin_debt_crisis(GameState& state, Player p) noexcept;
    bool trigger_tear_down_this_wall(GameState& state, Player p) noexcept;
    bool trigger_an_evil_empire(GameState& state, Player p) noexcept;
    bool trigger_aldrich_ames(GameState& state, Player p) noexcept;
    bool trigger_pershing_ii(GameState& state, Player p) noexcept;
    bool trigger_wargames(GameState& state, Player p) noexcept;
    bool trigger_solidarity(GameState& state, Player p) noexcept;
    bool trigger_iran_iraq_war(GameState& state, Player p) noexcept;
    bool trigger_yuri_and_samantha(GameState& state, Player p) noexcept;
    bool trigger_awacs_sale(GameState& state, Player p) noexcept;
}

bool CardHandlers::can_trigger_event(const GameState& state, uint8_t card_id, Player player) noexcept {
    // The China Card has no event and can only be played for Operations
    if (card_id == card_ids::THE_CHINA_CARD) {
        return false;
    }

    switch (card_id) {
        case card_ids::DEFECTORS: {
            if (player == Player::US && state.current_phase == Phase::ACTION_ROUND && state.phasing_player != Player::USSR) {
                return false;
            }
            return true;
        }
        case card_ids::NATO:
            return state.has_flag(effect_bits::MARSHALL_PLAN_PLAYED) || state.has_flag(effect_bits::WARSAW_PACT_PLAYED);
        case card_ids::SOCIALIST_GOVERNMENTS:
            return !state.has_flag(effect_bits::IRON_LADY_PLAYED);
        case card_ids::ARAB_ISRAELI_WAR:
            return !state.has_flag(effect_bits::CAMP_DAVID_PLAYED);
        case card_ids::MUSLIM_REVOLUTION:
            return !state.has_flag(effect_bits::AWACS_PLAYED);
        case card_ids::OPEC:
            return !state.has_flag(effect_bits::NORTH_SEA_OIL_PLAYED);
        case card_ids::WILLY_BRANDT:
            return !state.has_flag(effect_bits::TEAR_DOWN_THIS_WALL_PLAYED);
        case card_ids::FLOWER_POWER:
            return !state.has_flag(effect_bits::EVIL_EMPIRE_PLAYED);
        case card_ids::SOLIDARITY:
            return state.has_flag(effect_bits::JOHN_PAUL_II_PLAYED);
        case card_ids::THE_CAMBRIDGE_FIVE:
            return state.turn < 8; // Not in Late War
        case card_ids::STAR_WARS:
            return state.us_space_track > state.ussr_space_track;
        default:
            return true;
    }
}

// Cards whose "can this Event do anything" test lives inside the handler rather than in the
// prerequisite table above, hoisted here so the handler and the removal decision share one
// definition. Adding a card here without routing its handler through the same predicate is the
// drift this exists to prevent.
namespace {

bool us_controls_a_middle_east_country(const GameState& state) noexcept {
    for (uint8_t i = 0; i < 84; ++i) {
        if (MapData::get_country(i).region == Region::MIDDLE_EAST &&
            Scoring::is_controlled_by(state, i, Player::US)) {
            return true;
        }
    }
    return false;
}

bool us_leads_on_battlegrounds(const GameState& state) noexcept {
    uint8_t us_bgs = 0;
    uint8_t ussr_bgs = 0;
    for (uint8_t i = 0; i < 84; ++i) {
        if (!MapData::get_country(i).battleground) continue;
        Player ctrl = Scoring::get_country_control(state, i);
        if (ctrl == Player::US) us_bgs++;
        else if (ctrl == Player::USSR) ussr_bgs++;
    }
    return us_bgs > ussr_bgs;
}

// Does a scoring card in the US hand name the region `country` is in?
//
// The Cambridge Five reveals the US's scoring cards and lets the USSR place in a region one of
// them names. Both the legal mask and the validation ask this, and they used to carry separate
// copies of the same seven-way table over a list of card ids copied out of the hand. The hand
// is the authority; the reveal is recorded by marking those cards known, not by listing them.
bool cambridge_five_allows(const GameState& state, uint8_t country) noexcept {
    if (country >= 84) return false;
    const auto& c = MapData::get_country(country);
    for (uint8_t sc = 1; sc <= 110; ++sc) {
        if (!in_hand_of(state.card_locations[sc], Player::US)) continue;
        if (!CardData::is_scoring_card(sc)) continue;
        switch (sc) {
            case card_ids::ASIA_SCORING:            if (c.region == Region::ASIA) return true; break;
            case card_ids::EUROPE_SCORING:          if (c.region == Region::EUROPE) return true; break;
            case card_ids::MIDDLE_EAST_SCORING:     if (c.region == Region::MIDDLE_EAST) return true; break;
            case card_ids::CENTRAL_AMERICA_SCORING: if (c.region == Region::CENTRAL_AMERICA) return true; break;
            case card_ids::SOUTH_AMERICA_SCORING:   if (c.region == Region::SOUTH_AMERICA) return true; break;
            case card_ids::AFRICA_SCORING:          if (c.region == Region::AFRICA) return true; break;
            case card_ids::SE_ASIA_SCORING:         if (c.in_southeast_asia) return true; break;
            default: break;
        }
    }
    return false;
}

// Is any card sitting in `where`? Used where a set of cards is represented by their location
// rather than by a list kept alongside it.
bool any_card_at(const GameState& state, CardLocation where) noexcept {
    for (uint8_t c = 1; c <= 110; ++c) {
        if (state.card_locations[c] == where) return true;
    }
    return false;
}

} // namespace

// Missile Envy takes the opponent's highest-Ops card, scoring cards excepted. The set it may
// take is therefore a property of that hand and is recomputed wherever it is needed, so there
// is no stored list to fall out of step with the hand it describes.
//
// `in_play` is the card currently resolving: the engine leaves an Ops card in its owner's hand
// until the play finishes, so the event can otherwise find the very card in front of it.
uint8_t CardHandlers::highest_takeable_ops(const GameState& state, Player giver) noexcept {
    const uint8_t in_play = state.ctx().resolving_card;
    uint8_t best = 0;
    for (uint8_t c = 1; c <= 110; ++c) {
        if (c == in_play || !in_hand_of(state.card_locations[c], giver)) continue;
        if (CardData::is_scoring_card(c)) continue;
        const uint8_t ops = CardData::get_card(c).ops;
        if (ops > best) best = ops;
    }
    return best;
}

bool CardHandlers::missile_envy_may_take(const GameState& state, uint8_t card, Player giver,
                                         uint8_t best_ops) noexcept {
    if (best_ops == 0 || card < 1 || card > 110) return false;
    if (card == state.ctx().resolving_card) return false;
    if (!in_hand_of(state.card_locations[card], giver)) return false;
    if (CardData::is_scoring_card(card)) return false;
    return CardData::get_card(card).ops == best_ops;
}

bool CardHandlers::event_has_effect(const GameState& state, uint8_t card_id, Player player) noexcept {
    if (!can_trigger_event(state, card_id, player)) return false;
    switch (card_id) {
        case card_ids::OUR_MAN_IN_TEHRAN: return us_controls_a_middle_east_country(state);
        case card_ids::KITCHEN_DEBATES:   return us_leads_on_battlegrounds(state);
        default: return true;
    }
}

void CardHandlers::relocate_played_card(GameState& state, uint8_t card, bool event_occurred,
                                       Handover handover) noexcept {
    if (card == 0 || card > 110) return;
    if (card == card_ids::KITCHEN_DEBATES) return;
    if (handover == Handover::Respect && keeps_own_card_location(state, card)) return;

    if (card == card_ids::SHUTTLE_DIPLOMACY &&
        state.has_flag(effect_bits::SHUTTLE_DIPLOMACY_ACTIVE)) {
        state.card_locations[card] = CardLocation::ONGOING_EVENT;
        return;
    }
    state.card_locations[card] = (CardData::get_card(card).one_time && event_occurred)
        ? CardLocation::REMOVED_FROM_GAME : CardLocation::DISCARD_PILE;
}

bool CardHandlers::trigger_event(GameState& state, uint8_t card_id, Player player, uint8_t forced_roll) noexcept {
    if (!can_trigger_event(state, card_id, player)) {
        return true; // Unmet prerequisite: event does not occur
    }

    switch (card_id) {
        // Scoring Cards
        case card_ids::ASIA_SCORING:
            Scoring::score_region(state, Region::ASIA); return true;
        case card_ids::EUROPE_SCORING:
            Scoring::score_region(state, Region::EUROPE); return true;
        case card_ids::MIDDLE_EAST_SCORING:
            Scoring::score_region(state, Region::MIDDLE_EAST); return true;
        case card_ids::CENTRAL_AMERICA_SCORING:
            Scoring::score_region(state, Region::CENTRAL_AMERICA); return true;
        case card_ids::SE_ASIA_SCORING:
            Scoring::score_southeast_asia(state); return true;
        case card_ids::SOUTH_AMERICA_SCORING:
            Scoring::score_region(state, Region::SOUTH_AMERICA); return true;
        case card_ids::AFRICA_SCORING:
            Scoring::score_region(state, Region::AFRICA); return true;

        // Early War
        case card_ids::DUCK_AND_COVER: return early_war::trigger_duck_and_cover(state, player);
        case card_ids::FIVE_YEAR_PLAN: return early_war::trigger_five_year_plan(state, player);
        case card_ids::SOCIALIST_GOVERNMENTS: return early_war::trigger_socialist_governments(state, player);
        case card_ids::FIDEL: return early_war::trigger_fidel(state, player);
        case card_ids::VIETNAM_REVOLTS: return early_war::trigger_vietnam_revolts(state, player);
        case card_ids::BLOCKADE: return early_war::trigger_blockade(state, player);
        case card_ids::KOREAN_WAR: return early_war::trigger_korean_war(state, player, forced_roll);
        case card_ids::ROMANIAN_ABDICATION: return early_war::trigger_romanian_abdication(state, player);
        case card_ids::ARAB_ISRAELI_WAR: return early_war::trigger_arab_israeli_war(state, player, forced_roll);
        case card_ids::COMECON: return early_war::trigger_comecon(state, player);
        case card_ids::NASSER: return early_war::trigger_nasser(state, player);
        case card_ids::WARSAW_PACT: return early_war::trigger_warsaw_pact(state, player);
        case card_ids::DE_GAULLE: return early_war::trigger_de_gaulle(state, player);
        case card_ids::CAPTURED_NAZI_SCIENTIST: return early_war::trigger_captured_nazi_scientist(state, player);
        case card_ids::TRUMAN_DOCTRINE: return early_war::trigger_truman_doctrine(state, player);
        case card_ids::OLYMPIC_GAMES: return early_war::trigger_olympic_games(state, player);
        case card_ids::NATO: return early_war::trigger_nato(state, player);
        case card_ids::INDEPENDENT_REDS: return early_war::trigger_independent_reds(state, player);
        case card_ids::MARSHALL_PLAN: return early_war::trigger_marshall_plan(state, player);
        case card_ids::INDO_PAKISTANI_WAR: return early_war::trigger_indo_pakistani_war(state, player);
        case card_ids::CONTAINMENT: return early_war::trigger_containment(state, player);
        case card_ids::CIA_CREATED: return early_war::trigger_cia_created(state, player);
        case card_ids::US_JAPAN_PACT: return early_war::trigger_us_japan_pact(state, player);
        case card_ids::SUEZ_CRISIS: return early_war::trigger_suez_crisis(state, player);
        case card_ids::EAST_EUROPEAN_UNREST: return early_war::trigger_east_european_unrest(state, player);
        case card_ids::DECOLONIZATION: return early_war::trigger_decolonization(state, player);
        case card_ids::RED_SCARE_PURGE: return early_war::trigger_red_scare_purge(state, player);
        case card_ids::UN_INTERVENTION: return early_war::trigger_un_intervention(state, player);
        case card_ids::DE_STALINIZATION: return early_war::trigger_de_stalinization(state, player);
        case card_ids::NUCLEAR_TEST_BAN: return early_war::trigger_nuclear_test_ban(state, player);
        case card_ids::FORMOSAN_RESOLUTION: return early_war::trigger_formosan_resolution(state, player);
        case card_ids::DEFECTORS: return early_war::trigger_defectors(state, player);
        case card_ids::THE_CAMBRIDGE_FIVE: return early_war::trigger_cambridge_five(state, player);
        case card_ids::SPECIAL_RELATIONSHIP: return early_war::trigger_special_relationship(state, player);
        case card_ids::NORAD: return early_war::trigger_norad(state, player);

        // Mid War
        case card_ids::ARMS_RACE: return mid_war::trigger_arms_race(state, player);
        case card_ids::BRUSH_WAR: return mid_war::trigger_brush_war(state, player);
        case card_ids::CUBAN_MISSILE_CRISIS: return mid_war::trigger_cuban_missile_crisis(state, player);
        case card_ids::NUCLEAR_SUBS: return mid_war::trigger_nuclear_subs(state, player);
        case card_ids::QUAGMIRE: return mid_war::trigger_quagmire(state, player);
        case card_ids::SALT_NEGOTIATIONS: return mid_war::trigger_salt_negotiations(state, player);
        case card_ids::BEAR_TRAP: return mid_war::trigger_bear_trap(state, player);
        case card_ids::SUMMIT: return mid_war::trigger_summit(state, player);
        case card_ids::HOW_I_LEARNED_TO_STOP_WORRYING: return mid_war::trigger_how_i_learned_to_stop_worrying(state, player);
        case card_ids::JUNTA: return mid_war::trigger_junta(state, player);
        case card_ids::KITCHEN_DEBATES: return mid_war::trigger_kitchen_debates(state, player);
        case card_ids::MISSILE_ENVY: return mid_war::trigger_missile_envy(state, player);
        case card_ids::WE_WILL_BURY_YOU: return mid_war::trigger_we_will_bury_you(state, player);
        case card_ids::BREZHNEV_DOCTRINE: return mid_war::trigger_brezhnev_doctrine(state, player);
        case card_ids::PORTUGUESE_EMPIRE_CRUMBLES: return mid_war::trigger_portuguese_empire(state, player);
        case card_ids::SOUTH_AFRICAN_UNREST: return mid_war::trigger_south_african_unrest(state, player);
        case card_ids::ALLENDE: return mid_war::trigger_allende(state, player);
        case card_ids::WILLY_BRANDT: return mid_war::trigger_willy_brandt(state, player);
        case card_ids::MUSLIM_REVOLUTION: return mid_war::trigger_muslim_revolution(state, player);
        case card_ids::ABM_TREATY: return mid_war::trigger_abm_treaty(state, player);
        case card_ids::CULTURAL_REVOLUTION: return mid_war::trigger_cultural_revolution(state, player);
        case card_ids::FLOWER_POWER: return mid_war::trigger_flower_power(state, player);
        case card_ids::U2_INCIDENT: return mid_war::trigger_u2_incident(state, player);
        case card_ids::OPEC: return mid_war::trigger_opec(state, player);
        case card_ids::LONE_GUNMAN: return mid_war::trigger_lone_gunman(state, player);
        case card_ids::COLONIAL_REAR_GUARDS: return mid_war::trigger_colonial_rear_guards(state, player);
        case card_ids::PANAMA_CANAL_RETURNED: return mid_war::trigger_panama_canal(state, player);
        case card_ids::CAMP_DAVID_ACCORDS: return mid_war::trigger_camp_david(state, player);
        case card_ids::PUPPET_GOVERNMENTS: return mid_war::trigger_puppet_governments(state, player);
        case card_ids::GRAIN_SALES: return mid_war::trigger_grain_sales(state, player);
        case card_ids::JOHN_PAUL_II: return mid_war::trigger_john_paul_ii(state, player);
        case card_ids::LATIN_DEATH_SQUADS: return mid_war::trigger_latin_death_squads(state, player);
        case card_ids::OAS_FOUNDED: return mid_war::trigger_oas_founded(state, player);
        case card_ids::NIXON_PLAYS_THE_CHINA_CARD: return mid_war::trigger_nixon_china_card(state, player);
        case card_ids::SADAT_EXPELS_SOVIETS: return mid_war::trigger_sadat_expels_soviets(state, player);
        case card_ids::SHUTTLE_DIPLOMACY: return mid_war::trigger_shuttle_diplomacy(state, player);
        case card_ids::THE_VOICE_OF_AMERICA: return mid_war::trigger_voice_of_america(state, player);
        case card_ids::LIBERATION_THEOLOGY: return mid_war::trigger_liberation_theology(state, player);
        case card_ids::USSURI_RIVER_SKIRMISH: return mid_war::trigger_ussuri_river(state, player);
        case card_ids::ASK_NOT_WHAT_YOUR_COUNTRY_CAN_DO_FOR_YOU: return mid_war::trigger_ask_not(state, player);
        case card_ids::ALLIANCE_FOR_PROGRESS: return mid_war::trigger_alliance_for_progress(state, player);
        case card_ids::ONE_SMALL_STEP: return mid_war::trigger_one_small_step(state, player);
        case card_ids::CHE: return mid_war::trigger_che(state, player);
        case card_ids::OUR_MAN_IN_TEHRAN: return mid_war::trigger_our_man_in_tehran(state, player);

        // Late War
        case card_ids::IRANIAN_HOSTAGE_CRISIS: return late_war::trigger_iranian_hostage_crisis(state, player);
        case card_ids::THE_IRON_LADY: return late_war::trigger_iron_lady(state, player);
        case card_ids::REAGAN_BOMBS_LIBYA: return late_war::trigger_reagan_bombs_libya(state, player);
        case card_ids::STAR_WARS: return late_war::trigger_star_wars(state, player);
        case card_ids::NORTH_SEA_OIL: return late_war::trigger_north_sea_oil(state, player);
        case card_ids::THE_REFORMER: return late_war::trigger_the_reformer(state, player);
        case card_ids::MARINE_BARRACKS_BOMBING: return late_war::trigger_marine_barracks_bombing(state, player);
        case card_ids::SOVIETS_SHOOT_DOWN_KAL_007: return late_war::trigger_soviets_shoot_down_kal(state, player);
        case card_ids::GLASNOST: return late_war::trigger_glasnost(state, player);
        case card_ids::ORTEGA_ELECTED_IN_NICARAGUA: return late_war::trigger_ortega_elected(state, player);
        case card_ids::TERRORISM: return late_war::trigger_terrorism(state, player);
        case card_ids::IRAN_CONTRA: return late_war::trigger_iran_contra(state, player);
        case card_ids::CHERNOBYL: return late_war::trigger_chernobyl(state, player);
        case card_ids::LATIN_AMERICAN_DEBT_CRISIS: return late_war::trigger_latin_debt_crisis(state, player);
        case card_ids::TEAR_DOWN_THIS_WALL: return late_war::trigger_tear_down_this_wall(state, player);
        case card_ids::AN_EVIL_EMPIRE: return late_war::trigger_an_evil_empire(state, player);
        case card_ids::ALDRICH_AMES: return late_war::trigger_aldrich_ames(state, player);
        case card_ids::PERSHING_II_DEPLOYED: return late_war::trigger_pershing_ii(state, player);
        case card_ids::WARGAMES: return late_war::trigger_wargames(state, player);
        case card_ids::SOLIDARITY: return late_war::trigger_solidarity(state, player);
        case card_ids::IRAN_IRAQ_WAR: return late_war::trigger_iran_iraq_war(state, player);
        case card_ids::YURI_AND_SAMANTHA: return late_war::trigger_yuri_and_samantha(state, player);
        case card_ids::AWACS_SALE: return late_war::trigger_awacs_sale(state, player);

        default:
            return true;
    }
}

bool CardHandlers::handle_event_step(GameState& state, const MicroAction& action) noexcept {
    uint8_t card = state.ctx().resolving_card;
    // At a chance node there is no decision_player, so the roller is whoever staged the die.
    // This used to read a card slot that held a player code for a coup or a war and a *country
    // id* for CHE and Ortega -- harmless only because the id that aliases onto "US" is the
    // United Kingdom, which neither card can target.
    Player p = (state.ctx().decision_player != Player::NONE)
        ? state.ctx().decision_player
        : ((state.ctx().decision_type == DecisionType::ROLL_DIE
            && state.ctx().roll_actor != Player::NONE)
               ? state.ctx().roll_actor : state.phasing_player);

    switch (card) {
        case card_ids::BLOCKADE: {
            if (action.is_confirm_done() || action.primary_id == 0) {
                state.countries[countries::WEST_GERMANY].us_influence = 0;
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t card_id = action.primary_id;
            // The escape needs 3 or more Ops as the US would actually get them: Containment's +1
            // makes a printed 2 enough, Red Scare's -1 makes a printed 3 insufficient. All nine
            // sub-3 escapes across the 287 downloaded human games have Containment in play.
            if (card_id >= 1 && card_id <= 110 && in_hand_of(state.card_locations[card_id], Player::US) &&
                Operations::get_effective_ops(state, card_id, Player::US) >= 3) {
                state.card_locations[card_id] = CardLocation::DISCARD_PILE;
                state.ctx().resolving_card = 0;
                return true;
            }
            return false;
        }

        case card_ids::SOCIALIST_GOVERNMENTS: {
            if (action.is_confirm_done()) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t cid = action.primary_id;
            // At most two from any one country -- three Influence, no more than two anywhere.
            // The cap was in the mask and in the card's own max_per_country, but not here, so a
            // caller stepping past the mask could take all three out of one country. It is the
            // only one of the sixteen per-country limits that the handler did not enforce.
            if (cid < 84 && MapData::get_country(cid).in_western_europe
                && state.countries[cid].us_influence > 0
                && state.ctx().node_count(cid) < 2) {
                state.countries[cid].remove_influence(Player::US, 1);
                state.ctx().bump_node_count(cid);
                if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                if (state.ctx().remaining_steps == 0) {
                    state.ctx().resolving_card = 0;
                    return true;
                }
            }
            return false;
        }

        case card_ids::COMECON: {
            if (action.is_confirm_done()) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t cid = action.primary_id;
            if (cid < 84 && MapData::get_country(cid).in_eastern_europe && !Scoring::is_controlled_by(state, cid, Player::US) && !state.ctx().is_visited(cid)) {
                state.countries[cid].add_influence(Player::USSR, 1);
                state.ctx().bump_node_count(cid);
                state.ctx().mark_visited(cid);
                if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                if (state.ctx().remaining_steps == 0) {
                    state.ctx().resolving_card = 0;
                    return true;
                }
            }
            return false;
        }

        case card_ids::KOREAN_WAR:
        case card_ids::ARAB_ISRAELI_WAR:
        case card_ids::INDO_PAKISTANI_WAR:
        case card_ids::BRUSH_WAR:
        case card_ids::IRAN_IRAQ_WAR:
            return war_helpers::handle_war_step(state, card, p, action);

        case card_ids::WARSAW_PACT: {
            if (state.ctx().decision_type == DecisionType::CHOOSE_BRANCH) {
                if (action.primary_id == 0) {
                    // Remove all US inf from 4 countries in Eastern Europe
                    uint8_t count = 0;
                    for (uint8_t i = 0; i < 84; ++i) {
                        if (MapData::get_country(i).in_eastern_europe && state.countries[i].us_influence > 0) {
                            count++;
                        }
                    }
                    if (count == 0) {
                        state.ctx().resolving_card = 0;
                        return true;
                    }
                    state.ctx().decision_type = DecisionType::POINT_NODE;
                    state.ctx().remaining_steps = std::min<uint8_t>(4, count);
                    state.ctx().max_per_country = 1;
                    // P17 6a: mandatory -- remove from 4 / add 5 -- exact counts, and already clamped
                    state.ctx().allow_early_stop = 0;
                    state.ctx().visited_nodes = {};
                    return false;
                } else {
                    // Add 5 USSR inf in Eastern Europe (max 2 per country)
                    state.ctx().decision_type = DecisionType::POINT_NODE;
                    state.ctx().remaining_steps = 5;
                    state.ctx().max_per_country = 2;
                    // P17 6a: mandatory -- remove from 4 / add 5 -- exact counts, and already clamped
                    state.ctx().allow_early_stop = 0;
                    state.ctx().clear_node_counts();
                    return false;
                }
            } else if (state.ctx().decision_type == DecisionType::POINT_NODE) {
                if (action.is_confirm_done() || action.primary_id >= 84) {
                    state.ctx().resolving_card = 0;
                    return true;
                }
                uint8_t cid = action.primary_id;
                if (cid < 84 && MapData::get_country(cid).in_eastern_europe) {
                    if (state.ctx().max_per_country == 1) {
                        if (state.countries[cid].us_influence > 0 && !state.ctx().is_visited(cid)) {
                            state.countries[cid].us_influence = 0;
                            state.ctx().mark_visited(cid);
                        } else {
                            return false;
                        }
                    } else {
                        if (state.ctx().node_count(cid) < 2) {
                            state.countries[cid].add_influence(Player::USSR, 1);
                            state.ctx().bump_node_count(cid);
                        } else {
                            return false;
                        }
                    }
                    if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                    if (state.ctx().remaining_steps == 0) {
                        state.ctx().resolving_card = 0;
                        return true;
                    }
                    if (state.ctx().max_per_country == 1) {
                        uint8_t rem = 0;
                        for (uint8_t i = 0; i < 84; ++i) {
                            if (MapData::get_country(i).in_eastern_europe && state.countries[i].us_influence > 0 && !state.ctx().is_visited(i)) {
                                rem++;
                            }
                        }
                        if (rem == 0) {
                            state.ctx().resolving_card = 0;
                            return true;
                        }
                    }
                    return false;
                }
            }
            return false;
        }

        case card_ids::TRUMAN_DOCTRINE: {
            if (action.is_confirm_done()) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t cid = action.primary_id;
            if (cid < 84 && MapData::get_country(cid).region == Region::EUROPE) {
                if (Scoring::get_country_control(state, cid) == Player::NONE) {
                    state.countries[cid].ussr_influence = 0;
                    state.ctx().resolving_card = 0;
                    return true;
                }
            }
            return false;
        }

        case card_ids::OLYMPIC_GAMES: {
            if (state.ctx().decision_type == DecisionType::CHOOSE_BRANCH) {
                if (action.primary_id == 0) {
                    // Participate: transition to ROLL_DIE
                    state.ctx().decision_player = Player::NONE;
                    state.ctx().decision_type = DecisionType::ROLL_DIE;
                    state.ctx().pending_roll = RollType::OLYMPIC_GAMES;
                    return false;
                } else {
                    // Boycott: DEFCON degrades by 1, and sponsor conducts Operations (as if played 4 Ops)
                    Player sponsor = get_opponent(state.ctx().decision_player);
                    if (state.defcon > 1) {
                        state.defcon--;
                        if (state.defcon == 2) state.defcon_dropped_to_2 = 1;
                    }
                    if (state.defcon == 1) {
                        resolve_defcon_one_loss(state, state.ctx().decision_player);
                        state.ctx().resolving_card = 0;
                        return true;
                    }
                    state.ctx().pending_op_card = card_ids::OLYMPIC_GAMES;
                    state.ctx().pending_ops_value = Operations::grant_ops(state, 4, sponsor);
                    state.ctx().decision_player = sponsor;
                    state.ctx().decision_type = DecisionType::SELECT_OP_MODE;
                    state.ctx().resolving_card = 0;
                    return false;
                }
            } else if (state.ctx().decision_type == DecisionType::ROLL_DIE) {
                Player sponsor = state.phasing_player;
                uint8_t forced_sp = (action.primary_id >= 1 && action.primary_id <= 6) ? action.primary_id : 0;
                uint8_t forced_opp = (action.secondary_id >= 1 && action.secondary_id <= 6) ? action.secondary_id : 0;
                uint8_t sp_roll = (forced_sp > 0) ? (forced_sp + 2) : (Prng::roll_d6(state.rng_state) + 2);
                uint8_t opp_roll = (forced_opp > 0) ? forced_opp : Prng::roll_d6(state.rng_state);
                while (sp_roll == opp_roll && forced_sp == 0 && forced_opp == 0) {
                    sp_roll = Prng::roll_d6(state.rng_state) + 2;
                    opp_roll = Prng::roll_d6(state.rng_state);
                }
                state.last_die_roll = sp_roll;
                state.last_opp_die_roll = opp_roll;
                Player winner = (sp_roll > opp_roll) ? sponsor : get_opponent(sponsor);
                state.last_roll = DieRollRecord{
                    .type = RollType::OLYMPIC_GAMES,
                    .roller = sponsor,
                    .card_id = card_ids::OLYMPIC_GAMES,
                    .country_id = 255,
                    .roll1 = sp_roll,
                    .mod1 = 2,
                    .roll2 = opp_roll,
                    .mod2 = 0,
                    .success = (winner == sponsor),
                    .net_delta = static_cast<int8_t>((winner == sponsor) ? 2 : -2)
                };
                int32_t vp_delta = (winner == Player::US) ? 2 : -2;
                state.victory_points = static_cast<int8_t>(std::clamp(static_cast<int32_t>(state.victory_points) + vp_delta, -20, 20));
                if (state.victory_points >= 20 || state.victory_points <= -20) state.current_phase = Phase::GAME_OVER;
                state.ctx().resolving_card = 0;
                return true;
            }
            return true;
        }
        case card_ids::INDEPENDENT_REDS: {
            if (action.is_confirm_done()) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t cid = action.primary_id;
            if (cid == countries::YUGOSLAVIA || cid == countries::ROMANIA || cid == countries::BULGARIA ||
                cid == countries::HUNGARY || cid == countries::CZECHOSLOVAKIA) {
                uint8_t ussr_inf = state.countries[cid].ussr_influence;
                uint8_t us_inf = state.countries[cid].us_influence;
                if (ussr_inf > us_inf) {
                    state.countries[cid].add_influence(Player::US, ussr_inf - us_inf);
                }
                state.ctx().resolving_card = 0;
                return true;
            }
            return false;
        }

        case card_ids::MARSHALL_PLAN: {
            if (action.is_confirm_done()) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t cid = action.primary_id;
            if (cid < 84 && MapData::get_country(cid).in_western_europe && !Scoring::is_controlled_by(state, cid, Player::USSR) && !state.ctx().is_visited(cid)) {
                state.countries[cid].add_influence(Player::US, 1);
                state.ctx().mark_visited(cid);
                if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                if (state.ctx().remaining_steps == 0) {
                    state.ctx().resolving_card = 0;
                    return true;
                }
            }
            return false;
        }

        case card_ids::SUEZ_CRISIS: {
            if (action.is_confirm_done()) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t cid = action.primary_id;
            constexpr std::array<uint8_t, 3> SUEZ_COUNTRIES = {
                countries::UNITED_KINGDOM, countries::FRANCE, countries::ISRAEL
            };
            bool valid = false;
            for (uint8_t v : SUEZ_COUNTRIES) if (v == cid) valid = true;
            if (valid && state.countries[cid].us_influence > 0 && state.ctx().node_count(cid) < 2) {
                state.countries[cid].remove_influence(Player::US, 1);
                state.ctx().bump_node_count(cid);
                if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                if (state.ctx().remaining_steps == 0) {
                    state.ctx().resolving_card = 0;
                    return true;
                }
            }
            return false;
        }

        case card_ids::EAST_EUROPEAN_UNREST: {
            if (action.is_confirm_done()) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t cid = action.primary_id;
            if (cid < 84 && MapData::get_country(cid).in_eastern_europe && state.countries[cid].ussr_influence > 0) {
                uint8_t remove_amt = (state.turn >= 8) ? 2 : 1; // Late War: remove 2
                state.countries[cid].remove_influence(Player::USSR, remove_amt);
                state.ctx().mark_visited(cid);
                if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                if (state.ctx().remaining_steps == 0) {
                    state.ctx().resolving_card = 0;
                    return true;
                }
            }
            return false;
        }

        case card_ids::DECOLONIZATION: {
            if (action.is_confirm_done()) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t cid = action.primary_id;
            if (cid < 84 && (MapData::get_country(cid).region == Region::AFRICA || MapData::get_country(cid).in_southeast_asia) && !state.ctx().is_visited(cid)) {
                state.countries[cid].add_influence(Player::USSR, 1);
                state.ctx().mark_visited(cid);
                state.ctx().bump_node_count(cid);
                if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                if (state.ctx().remaining_steps == 0) {
                    state.ctx().resolving_card = 0;
                    return true;
                }
            }
            return false;
        }

        case card_ids::UN_INTERVENTION: {
            if (state.has_flag(effect_bits::U2_INCIDENT_ACTIVE)) {
                state.victory_points = static_cast<int8_t>(std::max(-20, state.victory_points - 1));
                if (state.victory_points <= -20) {
                    state.current_phase = Phase::GAME_OVER;
                    state.ctx().resolving_card = 0;
                    return true;
                }
            }
            uint8_t chosen_card = action.primary_id;
            Player opp = get_opponent(p);
            if (chosen_card >= 1 && chosen_card <= 110 && in_hand_of(state.card_locations[chosen_card], p)) {
                if (CardData::get_card(chosen_card).side == opp && !CardData::is_scoring_card(chosen_card)) {
                    state.card_locations[chosen_card] = CardLocation::DISCARD_PILE;
                    state.ctx().pending_op_card = chosen_card;
                    // Region::ASIA grants the conditional bonuses up front exactly as a normal
                    // Ops play does, leaving the budget ladder to withdraw them if the player
                    // steps out of the region. Without it the named card was worth only its
                    // printed Ops: at turn 2 AR4 of ts-replayer game 108 the USSR named CIA
                    // Created under Vietnam Revolts and placed influence in two Southeast
                    // Asian countries, where the engine could afford only one.
                    state.ctx().pending_ops_value = Operations::grant_ops_for_card(state, chosen_card, p);
                    // UN Intervention uses the named card's Ops "without triggering the Event".
                    // advance_after_ops otherwise sees an opponent card sitting in
                    // pending_op_card on the default OPS_FIRST branch and fires it -- turn 1
                    // AR3 of ts-replayer game 100 gave the USSR Nasser's 2 Influence in Egypt
                    // on a US action round.
                    state.ctx().suppress_op_card_event = 1;
                    state.ctx().decision_type = DecisionType::SELECT_OP_MODE;
                    state.ctx().resolving_card = 0;
                    return false;
                }
            }
            state.ctx().resolving_card = 0;
            return true;
        }

        case card_ids::DE_STALINIZATION: {
            // Stage 1 (removal) and Stage 2 (placement)
            uint8_t cid = action.primary_id;
            // How many have been removed so far is what the allowance has been spent down by,
            // so no separate tally is kept. It used to live in the staged-card count, which
            // meant that count said "cards" for every other card and "influence points" here.
            const uint8_t removed = static_cast<uint8_t>(
                de_stalinization::MAX_REMOVALS - state.ctx().remaining_steps);

            // Opening the placement phase: place back exactly what was removed, at most two per
            // country, and no declining. The per-country tracking starts fresh because the
            // countries removed from are not the countries placed into.
            auto begin_placement = [&state](uint8_t to_place) {
                state.ctx().event_stage = de_stalinization::STAGE_PLACE;
                state.ctx().remaining_steps = to_place;
                state.ctx().max_per_country = 2;
                state.ctx().allow_early_stop = 0;
                state.ctx().visited_nodes = {};
                state.ctx().clear_node_counts();
            };

            if (state.ctx().event_stage == de_stalinization::STAGE_REMOVE) {
                if (action.is_confirm_done()) {
                    if (removed == 0) {
                        state.ctx().resolving_card = 0;
                        return true;
                    }
                    begin_placement(removed);
                    return false;
                }
                if (cid < 84 && state.countries[cid].ussr_influence > 0) {
                    state.countries[cid].remove_influence(Player::USSR, 1);
                    if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                    if (state.ctx().remaining_steps == 0) {
                        begin_placement(de_stalinization::MAX_REMOVALS);
                        return false;
                    }
                    return false;
                }
            } else {
                // Stage 2: Placement into non-US controlled countries (max 2 per country)
                if (cid < 84 && !Scoring::is_controlled_by(state, cid, Player::US) && state.ctx().node_count(cid) < 2) {
                    state.countries[cid].add_influence(Player::USSR, 1);
                    state.ctx().bump_node_count(cid);
                    if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                    if (state.ctx().remaining_steps == 0) {
                        state.ctx().resolving_card = 0;
                        return true;
                    }
                    return false;
                }
            }
            return false;
        }

        case card_ids::SALT_NEGOTIATIONS: {
            uint8_t card_retrieved = action.primary_id;
            if (card_retrieved >= 1 && card_retrieved <= 110 && state.card_locations[card_retrieved] == CardLocation::DISCARD_PILE) {
                state.card_locations[card_retrieved] = hand_of(p, /*known=*/true);
            }
            state.ctx().resolving_card = 0;
            return true;
        }

        case card_ids::SUMMIT: {
            if (state.ctx().decision_type == DecisionType::ROLL_DIE) {
                uint8_t us_dom_count = 0;
                uint8_t ussr_dom_count = 0;

                for (uint8_t r = 0; r < 6; ++r) {
                    auto summary = Scoring::evaluate_region(state, static_cast<Region>(r));
                    if (summary.us_status == RegionalStatus::DOMINATION || summary.us_status == RegionalStatus::CONTROL) {
                        us_dom_count++;
                    }
                    if (summary.ussr_status == RegionalStatus::DOMINATION || summary.ussr_status == RegionalStatus::CONTROL) {
                        ussr_dom_count++;
                    }
                }

                uint8_t forced_us = (action.primary_id >= 1 && action.primary_id <= 6) ? action.primary_id : 0;
                uint8_t forced_ussr = (action.secondary_id >= 1 && action.secondary_id <= 6) ? action.secondary_id : 0;
                uint8_t us_roll = (forced_us > 0) ? forced_us : Prng::roll_d6(state.rng_state);
                uint8_t ussr_roll = (forced_ussr > 0) ? forced_ussr : Prng::roll_d6(state.rng_state);
                state.last_die_roll = us_roll;
                state.last_opp_die_roll = ussr_roll;

                int16_t us_total = us_roll + us_dom_count;
                int16_t ussr_total = ussr_roll + ussr_dom_count;

                bool us_wins = (us_total > ussr_total);
                bool ussr_wins = (ussr_total > us_total);
                Player summit_winner = us_wins ? Player::US : (ussr_wins ? Player::USSR : Player::NONE);

                state.last_roll = DieRollRecord{
                    .type = RollType::SUMMIT,
                    .roller = p,
                    .card_id = card_ids::SUMMIT,
                    .country_id = 255,
                    .roll1 = us_roll,
                    .mod1 = static_cast<int8_t>(us_dom_count),
                    .roll2 = ussr_roll,
                    .mod2 = static_cast<int8_t>(ussr_dom_count),
                    .success = (summit_winner != Player::NONE),
                    .net_delta = static_cast<int8_t>(us_wins ? 2 : (ussr_wins ? -2 : 0))
                };

                if (us_total == ussr_total) {
                    state.ctx().resolving_card = 0;
                    return true;
                }

                if (us_total > ussr_total) {
                    state.victory_points = static_cast<int8_t>(std::min(20, state.victory_points + 2));
                    if (state.victory_points >= 20) {
                        state.current_phase = Phase::GAME_OVER;
                        state.ctx().resolving_card = 0;
                        return true;
                    }
                    state.ctx().decision_player = Player::US;
                    state.ctx().decision_type = DecisionType::CHOOSE_BRANCH;
                    // The winner picks one of the three DEFCON options; declining is not one
                    // of them. Set explicitly rather than inherited -- see trigger_war.
                    state.ctx().allow_early_stop = 0;
                    return false;
                } else {
                    state.victory_points = static_cast<int8_t>(std::max(-20, state.victory_points - 2));
                    if (state.victory_points <= -20) {
                        state.current_phase = Phase::GAME_OVER;
                        state.ctx().resolving_card = 0;
                        return true;
                    }
                    state.ctx().decision_player = Player::USSR;
                    state.ctx().decision_type = DecisionType::CHOOSE_BRANCH;
                    state.ctx().allow_early_stop = 0;   // as above
                    return false;
                }
            } else if (state.ctx().decision_type == DecisionType::CHOOSE_BRANCH) {
                // P17 section 4: "set DEFCON to V", not +1/-1. primary_id is the level the mask
                // offered -- {current-1, current, current+1} clipped to 1..5 -- so choosing the
                // current level is how "may ... by 1" expresses leaving it alone, which the old
                // pair of branches could not say at all.
                const int target = std::clamp(static_cast<int>(action.primary_id), 1, 5);
                const int before = static_cast<int>(state.defcon);
                state.defcon = static_cast<uint8_t>(target);
                if (target < before && target == 2) state.defcon_dropped_to_2 = 1;
                if (target == 1) {
                    // As everywhere else the phasing player takes the loss, so lowering DEFCON
                    // to 1 while your opponent is phasing wins the game rather than losing it.
                    resolve_defcon_one_loss(state, state.ctx().decision_player);
                }
                state.ctx().resolving_card = 0;
                return true;
            }
            return true;
        }
        case card_ids::HOW_I_LEARNED_TO_STOP_WORRYING: {
            uint8_t new_defcon = std::clamp(action.primary_id, static_cast<uint8_t>(1), static_cast<uint8_t>(5));
            state.defcon = new_defcon;
            if (state.defcon == 2) state.defcon_dropped_to_2 = 1;
            if (state.defcon == 1) {
                resolve_defcon_one_loss(state, state.ctx().decision_player);
            }
            state.ctx().resolving_card = 0;
            return true;
        }

        case card_ids::CUBAN_MISSILE_CRISIS: {
            // Where the payer pays from, or a decline. Declining is only on offer at the head
            // of an action round: inside a coup it is not a choice, since couping without
            // paying loses the game, and the mask offers no early stop there.
            const Player payer = state.ctx().decision_player;
            if (!action.is_confirm_done()) {
                uint8_t cid = action.primary_id;
                if (payer == Player::US) {
                    if (cid != countries::WEST_GERMANY && cid != countries::TURKEY) return false;
                    if (state.countries[cid].us_influence < 2) return false;
                    state.countries[cid].remove_influence(Player::US, 2);
                    state.clear_flag(effect_bits::CMC_ACTIVE_USSR);
                } else if (payer == Player::USSR) {
                    if (cid != countries::CUBA) return false;
                    if (state.countries[cid].ussr_influence < 2) return false;
                    state.countries[cid].remove_influence(Player::USSR, 2);
                    state.clear_flag(effect_bits::CMC_ACTIVE_US);
                } else {
                    return false;
                }
            }
            state.ctx().resolving_card = 0;
            state.ctx().remaining_steps = 0;
            state.ctx().allow_early_stop = 0;
            if (state.ctx().pending_roll == RollType::COUP) {
                // A coup provoked this and is already staged; open its die.
                state.ctx().decision_player = Player::NONE;
                state.ctx().decision_type = DecisionType::ROLL_DIE;
            } else {
                // Asked at the head of the action round, so the player still has it to play.
                state.ctx().decision_player = payer;
                state.ctx().decision_type = DecisionType::SELECT_CARD;
            }
            return false;
        }

        case card_ids::JUNTA: {
            if (action.is_confirm_done()) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t cid = action.primary_id;
            if (cid < 84 && (MapData::get_country(cid).region == Region::CENTRAL_AMERICA || MapData::get_country(cid).region == Region::SOUTH_AMERICA)) {
                state.countries[cid].add_influence(p, 2);
                // Transition to Ops in CA/SA. The event's own Ops, so Influence is barred
                // from these and not from the player's own -- see the SELECT_OP_MODE handler.
                state.ctx().pending_op_card = card_ids::JUNTA;
                state.ctx().pending_ops_value = Operations::grant_ops(state, 2, state.ctx().decision_player);
                state.ctx().event_granted_ops = 1;
                state.ctx().resolving_card = 0;
                state.ctx().decision_type = DecisionType::SELECT_OP_MODE;
                return false;
            }
            return false;
        }

        case card_ids::MISSILE_ENVY: {
            if (action.is_confirm_done() || action.primary_id == 0 || action.primary_id == 255) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t chosen_card = action.primary_id;
            if (chosen_card < 1 || chosen_card > 110) return false;

            Player opp = state.ctx().decision_player;
            // The card must be one Missile Envy could actually take -- highest Ops among the
            // giver's non-scoring cards. Asked of the hand rather than of a list stored when
            // the tie was found: the hand is the thing the rule talks about, and a list can
            // only ever agree with it or be wrong.
            if (!CardHandlers::missile_envy_may_take(state, chosen_card, opp,
                                             CardHandlers::highest_takeable_ops(state, opp))) {
                return false;
            }
            Player p_player = get_opponent(opp);
            if (!in_hand_of(state.card_locations[chosen_card], opp)) return false;

            state.card_locations[chosen_card] = hand_of(p_player, /*known=*/true);
            state.card_locations[card_ids::MISSILE_ENVY] = hand_of(opp, /*known=*/true);
            state.forced_card_player = opp;
            state.forced_card_id = card_ids::MISSILE_ENVY;

            const auto& c_info = CardData::get_card(chosen_card);
            if (c_info.side == p_player || c_info.side == Player::NONE) {
                state.ctx().decision_player = p_player;
                state.ctx().resolving_card = chosen_card;
                const bool fired = event_has_effect(state, chosen_card, p_player);
                bool done = trigger_event(state, chosen_card, p_player);
                relocate_played_card(state, chosen_card, fired, Handover::Respect);
                if (done) state.ctx().resolving_card = 0;
                return done;
            } else {
                state.ctx().pending_op_card = chosen_card;
                state.ctx().pending_ops_value = Operations::grant_ops_for_card(state, chosen_card, p_player);
                state.ctx().decision_player = p_player;
                state.ctx().decision_type = DecisionType::SELECT_OP_MODE;
                state.ctx().timing_branch = 255;
                // The tied-card route into the same place; see trigger_missile_envy for why
                // timing_branch alone leaves a starred card removed from the game.
                state.ctx().suppress_op_card_event = 1;
                state.ctx().resolving_card = 0;
                return false;
            }
        }

        case card_ids::GRAIN_SALES: {
            // The drawn card is the one being looked at, so it is the card at PEEKED_TEMP.
            uint8_t drawn_card = 0;
            for (uint8_t c = 1; c <= 110; ++c) {
                if (state.card_locations[c] == CardLocation::PEEKED_TEMP) { drawn_card = c; break; }
            }
            if (drawn_card == 0) {
                // Nothing staged: the event drew nothing, and there is no choice to answer.
                state.ctx().resolving_card = 0;
                return true;
            }
            if (action.primary_id == 0) {
                // Play the drawn card. It must leave PEEKED_TEMP before its Event can fire --
                // if it is Our Man in Tehran, an event that peeks five cards of its own, a card
                // still sitting there would be inside its own peek and could discard itself.
                state.card_locations[drawn_card] = hand_of(Player::US, /*known=*/true);
                state.ctx().pending_op_card = drawn_card;
                state.ctx().resolving_card = 0;
                state.ctx().decision_type = DecisionType::SELECT_PLAY_MODE;
                return false;
            } else {
                // Return it and conduct 2 Ops. It goes back to the hand it came from, marked
                // known: the US has seen it, and that is exactly what the location records.
                state.card_locations[drawn_card] = hand_of(Player::USSR, /*known=*/true);
                state.ctx().pending_op_card = card_ids::GRAIN_SALES;
                state.ctx().pending_ops_value = Operations::grant_ops(state, 2, Player::US);
                state.ctx().resolving_card = 0;
                state.ctx().decision_type = DecisionType::SELECT_OP_MODE;
                return false;
            }
        }

        case card_ids::WARGAMES: {
            if (action.primary_id == 0) {
                // Give 6 VP to opponent and end game
                Player opp = get_opponent(p);
                int32_t vp_delta = (opp == Player::US) ? 6 : -6;
                state.victory_points = static_cast<int8_t>(std::clamp(static_cast<int32_t>(state.victory_points) + vp_delta, -20, 20));
                state.current_phase = Phase::GAME_OVER;
                state.ctx().resolving_card = 0;
                return true;
            } else {
                // Pass
                state.ctx().resolving_card = 0;
                return true;
            }
        }

        case card_ids::STAR_WARS: {
            if (action.is_confirm_done() || action.primary_id == 0 || action.primary_id == 255) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t card_id = action.primary_id;
            if (card_id >= 1 && card_id <= 110 && state.card_locations[card_id] == CardLocation::DISCARD_PILE) {
                if (!state.push_context()) {
                    invariant_failed("event chain too deep; card", static_cast<int>(card_id));
                }
                state.ctx().decision_player = Player::US;
                state.ctx().resolving_card = card_id;
                const bool fired = event_has_effect(state, card_id, Player::US);
                bool done = trigger_event(state, card_id, Player::US);
                relocate_played_card(state, card_id, fired, Handover::Respect);
                if (done) state.pop_context();
                return done;
            }
            return false;
        }

        case card_ids::SPECIAL_RELATIONSHIP: {
            if (action.is_confirm_done()) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t cid = action.primary_id;
            if (cid < 84) {
                if (!state.has_flag(effect_bits::NATO_ACTIVE)) {
                    const auto& uk = MapData::get_country(countries::UNITED_KINGDOM);
                    bool is_adj = false;
                    for (uint8_t n = 0; n < uk.num_neighbors; ++n) {
                        if (uk.neighbors[n] == cid) is_adj = true;
                    }
                    if (is_adj) {
                        state.countries[cid].add_influence(Player::US, 1);
                        state.ctx().resolving_card = 0;
                        return true;
                    }
                } else {
                    if (MapData::get_country(cid).in_western_europe) {
                        state.countries[cid].add_influence(Player::US, 2);
                        state.ctx().resolving_card = 0;
                        return true;
                    }
                }
            }
            return false;
        }

        case card_ids::COLONIAL_REAR_GUARDS: {
            if (action.is_confirm_done()) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t cid = action.primary_id;
            if (cid < 84 && (MapData::get_country(cid).region == Region::AFRICA || MapData::get_country(cid).in_southeast_asia) && !state.ctx().is_visited(cid)) {
                state.countries[cid].add_influence(Player::US, 1);
                state.ctx().mark_visited(cid);
                if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                if (state.ctx().remaining_steps == 0) {
                    state.ctx().resolving_card = 0;
                    return true;
                }
            }
            return false;
        }

        case card_ids::PUPPET_GOVERNMENTS: {
            if (action.is_confirm_done()) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t cid = action.primary_id;
            if (cid < 84 && state.countries[cid].us_influence == 0 && state.countries[cid].ussr_influence == 0 && !state.ctx().is_visited(cid)) {
                state.countries[cid].add_influence(Player::US, 1);
                state.ctx().mark_visited(cid);
                if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                if (state.ctx().remaining_steps == 0) {
                    state.ctx().resolving_card = 0;
                    return true;
                }
            }
            return false;
        }

        case card_ids::OAS_FOUNDED: {
            if (action.is_confirm_done()) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t cid = action.primary_id;
            if (cid < 84 && (MapData::get_country(cid).region == Region::CENTRAL_AMERICA || MapData::get_country(cid).region == Region::SOUTH_AMERICA) && state.ctx().node_count(cid) < 2) {
                state.countries[cid].add_influence(Player::US, 1);
                state.ctx().bump_node_count(cid);
                if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                if (state.ctx().remaining_steps == 0) {
                    state.ctx().resolving_card = 0;
                    return true;
                }
            }
            return false;
        }

        case card_ids::THE_VOICE_OF_AMERICA: {
            if (action.is_confirm_done()) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t cid = action.primary_id;
            if (cid < 84 && MapData::get_country(cid).region != Region::EUROPE && state.countries[cid].ussr_influence > 0 && state.ctx().node_count(cid) < 2) {
                state.countries[cid].remove_influence(Player::USSR, 1);
                state.ctx().bump_node_count(cid);
                if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                if (state.ctx().remaining_steps == 0) {
                    state.ctx().resolving_card = 0;
                    return true;
                }
            }
            return false;
        }

        case card_ids::LIBERATION_THEOLOGY: {
            if (action.is_confirm_done()) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t cid = action.primary_id;
            if (cid < 84 && MapData::get_country(cid).region == Region::CENTRAL_AMERICA && state.ctx().node_count(cid) < 2) {
                state.countries[cid].add_influence(Player::USSR, 1);
                state.ctx().bump_node_count(cid);
                if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                if (state.ctx().remaining_steps == 0) {
                    state.ctx().resolving_card = 0;
                    return true;
                }
            }
            return false;
        }

        case card_ids::USSURI_RIVER_SKIRMISH: {
            if (action.is_confirm_done()) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t cid = action.primary_id;
            if (cid < 84 && MapData::get_country(cid).region == Region::ASIA && state.ctx().node_count(cid) < 2) {
                state.countries[cid].add_influence(Player::US, 1);
                state.ctx().bump_node_count(cid);
                if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                if (state.ctx().remaining_steps == 0) {
                    state.ctx().resolving_card = 0;
                    return true;
                }
            }
            return false;
        }

        case card_ids::MUSLIM_REVOLUTION: {
            if (action.is_confirm_done()) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t cid = action.primary_id;
            if (cid < 84 && !state.ctx().is_visited(cid)) {
                state.countries[cid].us_influence = 0;
                state.ctx().mark_visited(cid);
                if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                if (state.ctx().remaining_steps == 0) {
                    state.ctx().resolving_card = 0;
                    return true;
                }
            }
            return false;
        }

        case card_ids::THE_REFORMER: {
            if (action.is_confirm_done()) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t cid = action.primary_id;
            if (cid < 84 && MapData::get_country(cid).region == Region::EUROPE && state.ctx().node_count(cid) < 2) {
                state.countries[cid].add_influence(Player::USSR, 1);
                state.ctx().bump_node_count(cid);
                if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                if (state.ctx().remaining_steps == 0) {
                    state.ctx().resolving_card = 0;
                    return true;
                }
            }
            return false;
        }

        case card_ids::MARINE_BARRACKS_BOMBING: {
            if (action.is_confirm_done()) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t cid = action.primary_id;
            if (cid < 84 && MapData::get_country(cid).region == Region::MIDDLE_EAST && state.countries[cid].us_influence > 0 && state.ctx().node_count(cid) < 2) {
                state.countries[cid].remove_influence(Player::US, 1);
                state.ctx().bump_node_count(cid);
                if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                if (state.ctx().remaining_steps == 0) {
                    state.ctx().resolving_card = 0;
                    return true;
                }
            }
            return false;
        }

        case card_ids::LATIN_AMERICAN_DEBT_CRISIS: {
            if (state.ctx().decision_player == Player::US) {
                if (action.is_confirm_done() || action.primary_id == 0) {
                    state.ctx().decision_player = Player::USSR;
                    state.ctx().decision_type = DecisionType::POINT_NODE;
                    state.ctx().remaining_steps = 2;
                    state.ctx().max_per_country = 1;
                    state.ctx().allow_early_stop = 1;
                    state.ctx().visited_nodes = {};
                    return false;
                }
                uint8_t card_id = action.primary_id;
                // Effective Ops, as for Blockade above.
                if (card_id >= 1 && card_id <= 110 && in_hand_of(state.card_locations[card_id], Player::US) &&
                    Operations::get_effective_ops(state, card_id, Player::US) >= 3) {
                    state.card_locations[card_id] = CardLocation::DISCARD_PILE;
                    state.ctx().resolving_card = 0;
                    return true;
                }
                return false;
            } else {
                if (action.is_confirm_done()) {
                    state.ctx().resolving_card = 0;
                    return true;
                }
                uint8_t cid = action.primary_id;
                if (cid < 84 && MapData::get_country(cid).region == Region::SOUTH_AMERICA && state.countries[cid].ussr_influence > 0 && !state.ctx().is_visited(cid)) {
                    state.countries[cid].add_influence(Player::USSR, state.countries[cid].ussr_influence);
                    state.ctx().mark_visited(cid);
                    if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                    if (state.ctx().remaining_steps == 0) {
                        state.ctx().resolving_card = 0;
                        return true;
                    }
                    return false;
                }
                return false;
            }
        }

        case card_ids::PERSHING_II_DEPLOYED: {
            if (action.is_confirm_done()) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t cid = action.primary_id;
            if (cid < 84 && MapData::get_country(cid).in_western_europe && state.countries[cid].us_influence > 0 && !state.ctx().is_visited(cid)) {
                state.countries[cid].remove_influence(Player::US, 1);
                state.ctx().mark_visited(cid);
                if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                if (state.ctx().remaining_steps == 0) {
                    state.ctx().resolving_card = 0;
                    return true;
                }
            }
            return false;
        }

        case card_ids::THE_CAMBRIDGE_FIVE: {
            if (action.is_confirm_done()) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t cid = action.primary_id;
            if (cid < 84) {
                if (cambridge_five_allows(state, cid)) {
                    state.countries[cid].add_influence(Player::USSR, 1);
                    state.ctx().resolving_card = 0;
                    return true;
                }
            }
            return false;
        }

        case card_ids::SOUTH_AFRICAN_UNREST: {
            {
                uint8_t cid = action.primary_id;
                if (cid < 84) {
                    if (cid == countries::SOUTH_AFRICA) {
                        state.countries[countries::SOUTH_AFRICA].add_influence(Player::USSR, 1);
                        if (state.ctx().event_stage == 0) {
                            // The placement both readings share. The choice comes next.
                            state.ctx().event_stage = 1;
                            state.ctx().decision_type = DecisionType::POINT_NODE;
                            state.ctx().decision_player = Player::USSR;
                            state.ctx().remaining_steps = 1;
                            state.ctx().allow_early_stop = 0;
                            return false;
                        }
                        // Stage 1: the rest goes here too, and the card is done.
                        state.ctx().resolving_card = 0;
                        return true;
                    }
                    const auto& sa = MapData::get_country(countries::SOUTH_AFRICA);
                    bool is_adj = false;
                    for (uint8_t n = 0; n < sa.num_neighbors; ++n) {
                        if (sa.neighbors[n] == cid) { is_adj = true; break; }
                    }
                    if (is_adj) {
                        // The split. Two placements of 1, not one of 2: the card says "a single
                        // country" but at turn 5 AR2 of replay 112 the USSR put 1 in Botswana
                        // and 1 in Angola, so the pair may be split and each is its own node.
                        const bool first = (state.ctx().event_stage == 1);
                        // Placed directly, never through the Ops path: an event that says
                        // "add Influence" pays no doubled cost for a country the opponent
                        // controls.
                        state.countries[cid].add_influence(Player::USSR, 1);
                        if (!first) {
                            state.ctx().resolving_card = 0;
                            return true;
                        }
                        // One more adjacent placement, and South Africa is off the mask now.
                        state.ctx().event_stage = 2;
                        state.ctx().decision_type = DecisionType::POINT_NODE;
                        state.ctx().decision_player = Player::USSR;
                        state.ctx().remaining_steps = 1;
                        state.ctx().allow_early_stop = 0;   // as above
                        return false;
                    }
                }
                return false;
            }
        }

        case card_ids::CHERNOBYL: {
            uint8_t region_chosen = action.primary_id;
            state.set_flag(effect_bits::CHERNOBYL_ACTIVE);
            state.persistent_effects &= ~effect_bits::CHERNOBYL_REGION_MASK;
            state.persistent_effects |= (static_cast<uint64_t>(region_chosen & 0x7) << effect_bits::CHERNOBYL_REGION_SHIFT);
            state.ctx().resolving_card = 0;
            return true;
        }

        case card_ids::ASK_NOT_WHAT_YOUR_COUNTRY_CAN_DO_FOR_YOU: {
            if (action.is_confirm_done() || action.primary_id == 0 || action.primary_id == 255) {
                // Each card was discarded as it was chosen, so how many to draw is what the
                // allowance has been spent down by -- no list of the cards themselves is kept.
                uint8_t discard_count = static_cast<uint8_t>(ask_not::MAX_DISCARDS - state.ctx().remaining_steps);
                for (uint8_t k = 0; k < discard_count; ++k) {
                    uint8_t draw_cards[111];
                    uint8_t draw_cnt = 0;
                    for (uint8_t i = 1; i <= 110; ++i) {
                        if (state.card_locations[i] == CardLocation::DRAW_DECK) {
                            draw_cards[draw_cnt++] = i;
                        }
                    }
                    if (draw_cnt == 0) {
                        for (uint8_t i = 1; i <= 110; ++i) {
                            if (i == card_ids::THE_CHINA_CARD) continue;
                            if (state.card_locations[i] == CardLocation::DISCARD_PILE) {
                                state.card_locations[i] = CardLocation::DRAW_DECK;
                                draw_cards[draw_cnt++] = i;
                            }
                        }
                    }
                    if (draw_cnt > 0) {
                        uint32_t chosen_idx = Prng::random_index(state.rng_state, draw_cnt);
                        uint8_t drawn = draw_cards[chosen_idx];
                        state.card_locations[drawn] = hand_of(Player::US);
                    }
                }
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t card_id = action.primary_id;
            if (card_id >= 1 && card_id <= 110 && in_hand_of(state.card_locations[card_id], Player::US)) {
                // Discarded as it is chosen. "Already chosen" is then simply "no longer in the
                // hand", which the mask reads directly -- where a parallel list of the chosen
                // ids used to say the same thing in a second place, and outlived the decision
                // it belonged to.
                state.card_locations[card_id] = CardLocation::DISCARD_PILE;
                if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                return false;
            }
            return false;
        }

        case card_ids::ALDRICH_AMES: {
            uint8_t card_id = action.primary_id;
            if (card_id >= 1 && card_id <= 110 && in_hand_of(state.card_locations[card_id], Player::US)) {
                state.card_locations[card_id] = CardLocation::DISCARD_PILE;
            }
            state.ctx().resolving_card = 0;
            return true;
        }

        case card_ids::CHE: {
            // The coup resolves at a chance node of its own, like every other die in the
            // engine. Rolling it here, inside the target choice, left it the one coup a caller
            // could not steer without knowing to put the value in secondary_id.
            if (state.ctx().decision_type == DecisionType::ROLL_DIE) {
                uint8_t cid = state.ctx().roll_target;
                uint8_t forced = action.primary_id;
                uint8_t che_ops = Operations::get_modified_ops(state, 3, Player::USSR);
                // execute_coup credits the military operations; crediting them again here
                // spent each coup twice, so a single coup reached the cap of 5 where it should
                // have earned 3, and the second one was free.
                auto coup_res = Operations::execute_coup(state, Player::USSR, cid, che_ops, forced);

                // If US influence was removed and this was coup 1, offer coup 2
                if (coup_res.opp_inf_removed > 0 && state.ctx().event_stage == 0) {
                    state.ctx().mark_visited(cid);
                    state.ctx().event_stage = 1;   // the second coup is now the one on offer
                    state.ctx().decision_player = Player::USSR;
                    state.ctx().decision_type = DecisionType::POINT_NODE;
                    state.ctx().remaining_steps = 1;
                    state.ctx().allow_early_stop = 1;
                    state.ctx().resolving_card = card_ids::CHE;
                    return false;
                }
                state.ctx().resolving_card = 0;
                return true;
            }
            if (action.is_confirm_done()) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t cid = action.primary_id;
            const auto& c_info = MapData::get_country(cid);
            if (cid < 84 && !c_info.battleground &&
                (c_info.region == Region::CENTRAL_AMERICA || c_info.region == Region::SOUTH_AMERICA || c_info.region == Region::AFRICA) &&
                !state.ctx().is_visited(cid) &&
                Operations::can_coup(state, Player::USSR, cid)) {
                state.ctx().pending_roll = RollType::COUP;
                state.ctx().roll_target = cid;
                state.ctx().roll_actor = Player::USSR;
                state.ctx().decision_player = Player::NONE;
                state.ctx().decision_type = DecisionType::ROLL_DIE;
                return false;
            }
            state.ctx().resolving_card = 0;
            return true;
        }

        case card_ids::ORTEGA_ELECTED_IN_NICARAGUA: {
            if (state.ctx().decision_type == DecisionType::ROLL_DIE) {   // as Che, above
                uint8_t target_cid = state.ctx().roll_target;
                uint8_t forced = action.primary_id;
                uint8_t ortega_ops = Operations::get_modified_ops(state, 2, Player::USSR);
                Operations::execute_coup(state, Player::USSR, target_cid, ortega_ops, forced);
                state.ctx().resolving_card = 0;
                return true;
            }
            if (action.is_confirm_done()) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t target_cid = action.primary_id;
            const auto& nic = MapData::get_country(countries::NICARAGUA);
            bool is_adj = false;
            for (uint8_t n = 0; n < nic.num_neighbors; ++n) {
                if (nic.neighbors[n] == target_cid) { is_adj = true; break; }
            }
            if (is_adj && target_cid < 84 && Operations::can_coup(state, Player::USSR, target_cid)) {
                state.ctx().pending_roll = RollType::COUP;
                state.ctx().roll_target = target_cid;
                state.ctx().roll_actor = Player::USSR;
                state.ctx().decision_player = Player::NONE;
                state.ctx().decision_type = DecisionType::ROLL_DIE;
                return false;
            }
            state.ctx().resolving_card = 0;
            return true;
        }

        case card_ids::SOVIETS_SHOOT_DOWN_KAL_007: {
            state.ctx().resolving_card = 0;
            return true;
        }

        case card_ids::NORAD: {
            if (action.is_confirm_done() || action.primary_id == 255) {
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t cid = action.primary_id;
            if (cid < 84 && state.countries[cid].us_influence > 0) {
                state.countries[cid].add_influence(Player::US, 1);
                state.ctx().resolving_card = 0;
                return true;
            }
            return false;
        }

        case card_ids::OUR_MAN_IN_TEHRAN: {
            if (action.is_confirm_done() || action.primary_id == 0 || action.primary_id == 255) {
                // Return remaining cards in PEEKED_TEMP back to DRAW_DECK
                for (uint8_t i = 1; i <= 110; ++i) {
                    if (state.card_locations[i] == CardLocation::PEEKED_TEMP) {
                        state.card_locations[i] = CardLocation::DRAW_DECK;
                    }
                }
                state.ctx().resolving_card = 0;
                return true;
            }
            uint8_t chosen_card = action.primary_id;
            if (chosen_card >= 1 && chosen_card <= 110 &&
                state.card_locations[chosen_card] == CardLocation::PEEKED_TEMP) {
                // The peek is the set of cards at PEEKED_TEMP, so discarding one removes it
                // from the set by moving it. A parallel list of the same ids was kept alongside
                // and shuffled down on every discard; it could only ever agree with the
                // locations, and the check just above already trusted the locations over it.
                state.card_locations[chosen_card] = CardLocation::DISCARD_PILE;
                if (!any_card_at(state, CardLocation::PEEKED_TEMP)) {
                    state.ctx().resolving_card = 0;
                    return true;
                }
                return false;
            }
            return false;
        }

        default:
            state.ctx().resolving_card = 0;
            return true;
    }
}

void CardHandlers::get_event_action_mask(const GameState& state, uint8_t* mask_out, size_t* out_size) noexcept {
    uint8_t card = state.ctx().resolving_card;
    DecisionType dt = state.ctx().decision_type;
    Player p = state.ctx().decision_player;

    if (dt == DecisionType::POINT_NODE) {
        *out_size = 84;
        for (uint8_t i = 0; i < 84; ++i) {
            mask_out[i] = 0;
            const auto& c_info = MapData::get_country(i);

            switch (card) {
                // Cancelling Cuban Missile Crisis: the US pays 2 Influence from West Germany
                // or from Turkey, and is asked only when both can pay.
                case card_ids::CUBAN_MISSILE_CRISIS:
                    if (p == Player::US) {
                        if ((i == countries::WEST_GERMANY || i == countries::TURKEY) &&
                            state.countries[i].us_influence >= 2) {
                            mask_out[i] = 1;
                        }
                    } else if (p == Player::USSR) {
                        if (i == countries::CUBA &&
                            state.countries[i].ussr_influence >= 2) {
                            mask_out[i] = 1;
                        }
                    }
                    break;
                case card_ids::WARSAW_PACT:
                    if (state.ctx().max_per_country == 1) {
                        // Remove all US influence from 4 countries in Eastern Europe
                        if (c_info.in_eastern_europe && state.countries[i].us_influence > 0 && !state.ctx().is_visited(i)) {
                            mask_out[i] = 1;
                        }
                    } else {
                        // Add 5 USSR influence in Eastern Europe (max 2 per country)
                        if (c_info.in_eastern_europe && state.ctx().node_count(i) < 2) {
                            mask_out[i] = 1;
                        }
                    }
                    break;
                case card_ids::DE_STALINIZATION:
                    if (state.ctx().event_stage == de_stalinization::STAGE_REMOVE) {
                        // Stage 1: Removal of USSR influence
                        if (state.countries[i].ussr_influence > 0) {
                            mask_out[i] = 1;
                        }
                    } else {
                        // Stage 2: Placement into non-US controlled countries (max 2 per country)
                        if (!Scoring::is_controlled_by(state, i, Player::US) && state.ctx().node_count(i) < 2) {
                            mask_out[i] = 1;
                        }
                    }
                    break;
                case card_ids::SOCIALIST_GOVERNMENTS:
                    if (c_info.in_western_europe && state.countries[i].us_influence > 0 && state.ctx().node_count(i) < 2) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::COMECON:
                    if (c_info.in_eastern_europe && !Scoring::is_controlled_by(state, i, Player::US) && !state.ctx().is_visited(i)) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::MARSHALL_PLAN:
                    if (c_info.in_western_europe && !Scoring::is_controlled_by(state, i, Player::USSR) && !state.ctx().is_visited(i)) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::INDEPENDENT_REDS:
                    if ((i == countries::YUGOSLAVIA || i == countries::ROMANIA || i == countries::BULGARIA ||
                         i == countries::HUNGARY || i == countries::CZECHOSLOVAKIA) && state.countries[i].ussr_influence > state.countries[i].us_influence) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::THE_CAMBRIDGE_FIVE:
                    if (cambridge_five_allows(state, i)) mask_out[i] = 1;
                    break;
                case card_ids::LATIN_AMERICAN_DEBT_CRISIS:
                    if (c_info.region == Region::SOUTH_AMERICA && state.countries[i].ussr_influence > 0 && !state.ctx().is_visited(i)) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::SPECIAL_RELATIONSHIP:
                    if (!state.has_flag(effect_bits::NATO_ACTIVE)) {
                        const auto& uk = MapData::get_country(countries::UNITED_KINGDOM);
                        for (uint8_t n = 0; n < uk.num_neighbors; ++n) {
                            if (uk.neighbors[n] == i) mask_out[i] = 1;
                        }
                    } else {
                        if (c_info.in_western_europe) mask_out[i] = 1;
                    }
                    break;
                case card_ids::TRUMAN_DOCTRINE:
                    if (c_info.region == Region::EUROPE && Scoring::get_country_control(state, i) == Player::NONE && state.countries[i].ussr_influence > 0) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::INDO_PAKISTANI_WAR:
                case card_ids::BRUSH_WAR:
                case card_ids::IRAN_IRAQ_WAR:
                    war_helpers::get_war_target_mask(state, card, p, mask_out);
                    break;
                case card_ids::SOUTH_AFRICAN_UNREST: {
                    // Stage 0: South Africa alone -- both readings require it.
                    if (state.ctx().event_stage == 0) {
                        if (i == countries::SOUTH_AFRICA) mask_out[i] = 1;
                        break;
                    }
                    // Stage 1 is the choice: South Africa spends the rest there, an adjacent
                    // country starts the split. Stage 2 is the split's second half, where
                    // South Africa is no longer on offer.
                    if (state.ctx().event_stage == 1 && i == countries::SOUTH_AFRICA) {
                        mask_out[i] = 1;
                        break;
                    }
                    const auto& sa = MapData::get_country(countries::SOUTH_AFRICA);
                    for (uint8_t n = 0; n < sa.num_neighbors; ++n) {
                        if (sa.neighbors[n] == i) mask_out[i] = 1;
                    }
                    break;
                }
                case card_ids::NORAD:
                    if (state.countries[i].us_influence > 0) mask_out[i] = 1;
                    break;
                case card_ids::CHE:
                    // can_coup, not just the card's own restrictions: a free coup is still a
                    // coup, so it needs opponent influence to remove, and it is still bound by
                    // the DEFCON regional limits, NATO and The Reformer. Filtering on region and
                    // battleground alone offered coups the rules forbid -- see ORTEGA below.
                    if (!c_info.battleground &&
                        (c_info.region == Region::CENTRAL_AMERICA || c_info.region == Region::SOUTH_AMERICA || c_info.region == Region::AFRICA) &&
                        !state.ctx().is_visited(i) &&
                        Operations::can_coup(state, Player::USSR, i)) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::JUNTA:
                    if (c_info.region == Region::CENTRAL_AMERICA || c_info.region == Region::SOUTH_AMERICA) mask_out[i] = 1;
                    break;
                case card_ids::COLONIAL_REAR_GUARDS:
                    if ((c_info.region == Region::AFRICA || c_info.in_southeast_asia) && !state.ctx().is_visited(i)) mask_out[i] = 1;
                    break;
                case card_ids::PUPPET_GOVERNMENTS:
                    if (state.countries[i].us_influence == 0 && state.countries[i].ussr_influence == 0 && !state.ctx().is_visited(i)) mask_out[i] = 1;
                    break;
                case card_ids::OAS_FOUNDED:
                    if ((c_info.region == Region::CENTRAL_AMERICA || c_info.region == Region::SOUTH_AMERICA) && state.ctx().node_count(i) < 2) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::THE_VOICE_OF_AMERICA:
                    if (c_info.region != Region::EUROPE && state.countries[i].ussr_influence > 0 && state.ctx().node_count(i) < 2) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::LIBERATION_THEOLOGY:
                    if (c_info.region == Region::CENTRAL_AMERICA && state.ctx().node_count(i) < 2) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::USSURI_RIVER_SKIRMISH:
                    if (c_info.region == Region::ASIA && state.ctx().node_count(i) < 2) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::THE_REFORMER:
                    if (c_info.region == Region::EUROPE && state.ctx().node_count(i) < 2) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::MARINE_BARRACKS_BOMBING:
                    if (c_info.region == Region::MIDDLE_EAST && state.countries[i].us_influence > 0 && state.ctx().node_count(i) < 2) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::SUEZ_CRISIS:
                    if ((i == countries::UNITED_KINGDOM || i == countries::FRANCE || i == countries::ISRAEL) &&
                        state.countries[i].us_influence > 0 && state.ctx().node_count(i) < 2) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::EAST_EUROPEAN_UNREST:
                    if (c_info.in_eastern_europe && state.countries[i].ussr_influence > 0 && !state.ctx().is_visited(i)) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::DECOLONIZATION:
                    if ((c_info.region == Region::AFRICA || c_info.in_southeast_asia) && !state.ctx().is_visited(i)) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::MUSLIM_REVOLUTION:
                    if ((i == countries::SUDAN || i == countries::IRAN || i == countries::IRAQ ||
                         i == countries::EGYPT || i == countries::LIBYA || i == countries::SAUDI_ARABIA ||
                         i == countries::SYRIA || i == countries::JORDAN) && state.countries[i].us_influence > 0 && !state.ctx().is_visited(i)) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::PERSHING_II_DEPLOYED:
                    if (c_info.in_western_europe && state.countries[i].us_influence > 0 && !state.ctx().is_visited(i)) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::ORTEGA_ELECTED_IN_NICARAGUA: {
                    // Adjacency alone is not enough. At turn 9 AR 2 of ts-replayer game 139 the
                    // US played Ortega for Ops, the event handed the USSR its free coup, and this
                    // offered Cuba -- which held US 0 / USSR 3, so there was nothing there to
                    // coup. Cuba being a battleground, taking it dropped DEFCON from 2 to 1 and
                    // ended the game against the phasing player, turning an illegal move into a
                    // winning one.
                    const auto& nic = MapData::get_country(countries::NICARAGUA);
                    for (uint8_t n = 0; n < nic.num_neighbors; ++n) {
                        if (nic.neighbors[n] == i && Operations::can_coup(state, Player::USSR, i)) {
                            mask_out[i] = 1;
                        }
                    }
                    break;
                }
                default:
                    mask_out[i] = 1;
                    break;
            }
        }
    } else if (dt == DecisionType::SELECT_CARD) {
        *out_size = 112;
        for (uint8_t i = 0; i < 112; ++i) mask_out[i] = 0;

        switch (card) {
            case card_ids::STAR_WARS:
                for (uint8_t i = 1; i <= 110; ++i) {
                    if (state.card_locations[i] == CardLocation::DISCARD_PILE && !CardData::is_scoring_card(i) && CardHandlers::can_trigger_event(state, i, Player::US)) {
                        mask_out[i] = 1;
                    }
                }
                break;
            case card_ids::SALT_NEGOTIATIONS:
                for (uint8_t i = 1; i <= 110; ++i) {
                    if (state.card_locations[i] == CardLocation::DISCARD_PILE && !CardData::is_scoring_card(i)) {
                        mask_out[i] = 1;
                    }
                }
                break;
            case card_ids::OUR_MAN_IN_TEHRAN:
                for (uint8_t c = 1; c <= 110; ++c) {
                    if (state.card_locations[c] == CardLocation::PEEKED_TEMP) mask_out[c] = 1;
                }
                break;
            case card_ids::MISSILE_ENVY: {
                // Whichever of the giver's cards Missile Envy could take: the non-scoring ones
                // of highest Ops. Recomputed rather than stored, exactly as SALT Negotiations
                // and Star Wars recompute their offer from the discard pile.
                uint8_t best = CardHandlers::highest_takeable_ops(state, p);
                for (uint8_t c = 1; c <= 110; ++c) {
                    if (CardHandlers::missile_envy_may_take(state, c, p, best)) mask_out[c] = 1;
                }
                break;
            }
            case card_ids::ALDRICH_AMES:
                for (uint8_t i = 1; i <= 110; ++i) {
                    if (in_hand_of(state.card_locations[i], Player::US)) {
                        mask_out[i] = 1;
                    }
                }
                break;
            case card_ids::BLOCKADE:
            case card_ids::LATIN_AMERICAN_DEBT_CRISIS:
                for (uint8_t i = 1; i <= 110; ++i) {
                    // Effective Ops, matching what both dispatchers accept.
                    if (in_hand_of(state.card_locations[i], Player::US) &&
                        Operations::get_effective_ops(state, i, Player::US) >= 3) {
                        mask_out[i] = 1;
                    }
                }
                break;
            case card_ids::ASK_NOT_WHAT_YOUR_COUNTRY_CAN_DO_FOR_YOU:
                // A chosen card has already gone to the discard pile, so whatever is still in
                // hand is what may still be chosen.
                for (uint8_t i = 1; i <= 110; ++i) {
                    if (in_hand_of(state.card_locations[i], p)) mask_out[i] = 1;
                }
                break;
            case card_ids::UN_INTERVENTION:
                // ENG-1. "Play this card simultaneously with a card containing your opponent's
                // associated Event", so the companion is an opponent-associated, non-scoring
                // card in hand. This case is the REACHABLE definition: the branch in
                // action_mask.cpp that had the rule right sits below a `resolving_card != 0`
                // test that delegates here and returns, so it never ran and the default offered
                // the player's whole hand -- their own cards and scoring cards included.
                for (uint8_t i = 1; i <= 110; ++i) {
                    if (in_hand_of(state.card_locations[i], p) &&
                        CardData::is_opponent_card(i, p) &&
                        !CardData::is_scoring_card(i)) {
                        mask_out[i] = 1;
                    }
                }
                break;
            default:
                for (uint8_t i = 1; i <= 110; ++i) {
                    if (in_hand_of(state.card_locations[i], p)) mask_out[i] = 1;
                }
                break;
        }
    } else if (dt == DecisionType::CHOOSE_BRANCH) {
        *out_size = 8;
        for (uint8_t i = 0; i < 8; ++i) mask_out[i] = 0;
        mask_out[0] = 1;
        mask_out[1] = 1;
        if (card == card_ids::HOW_I_LEARNED_TO_STOP_WORRYING) {
            mask_out[0] = 0; mask_out[1] = 1; mask_out[2] = 1; mask_out[3] = 1; mask_out[4] = 1; mask_out[5] = 1;
        } else if (card == card_ids::SUMMIT) {
            // P17 section 4: DEFCON LEVELS, not branch indices, so this agrees with the flat
            // mask -- one definition of legality (P14). "may degrade or improve ... by 1" is
            // {current-1, current, current+1} clipped to 1..5, and choosing the current level
            // is how "may" says leave it alone. The old 0/1/2 encoding meant improve/degrade/
            // unchanged, which the flat side has no way to spell.
            mask_out[0] = 0; mask_out[1] = 0; mask_out[2] = 0;
            const int d = static_cast<int>(state.defcon);
            for (int v = d - 1; v <= d + 1; ++v) {
                if (v >= 1 && v <= 5) mask_out[v] = 1;
            }
        } else if (card == card_ids::CHERNOBYL) {
            mask_out[2] = 1; mask_out[3] = 1; mask_out[4] = 1; mask_out[5] = 1;
        }
    }
}

} // namespace ts
