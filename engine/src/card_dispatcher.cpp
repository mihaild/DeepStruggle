#include "ts/card_handlers.hpp"
#include "ts/card_data.hpp"
#include "ts/map_data.hpp"
#include "ts/scoring.hpp"
#include "ts/space_race.hpp"
#include "ts/ops.hpp"
#include "ts/prng.hpp"
#include <algorithm>

namespace ts {

namespace early_war {
    bool trigger_duck_and_cover(GameState& state, Player p) noexcept;
    bool trigger_five_year_plan(GameState& state, Player p) noexcept;
    bool trigger_socialist_governments(GameState& state, Player p) noexcept;
    bool trigger_fidel(GameState& state, Player p) noexcept;
    bool trigger_vietnam_revolts(GameState& state, Player p) noexcept;
    bool trigger_blockade(GameState& state, Player p) noexcept;
    bool trigger_korean_war(GameState& state, Player p) noexcept;
    bool trigger_romanian_abdication(GameState& state, Player p) noexcept;
    bool trigger_arab_israeli_war(GameState& state, Player p) noexcept;
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
    bool trigger_de_stalinization(GameState& state, Player p) noexcept;
    bool trigger_nuclear_test_ban(GameState& state, Player p) noexcept;
    bool trigger_formosan_resolution(GameState& state, Player p) noexcept;
    bool trigger_defectors(GameState& state, Player p) noexcept;
    bool trigger_cambridge_five(GameState& state, Player p) noexcept;
    bool trigger_special_relationship(GameState& state, Player p) noexcept;
    bool trigger_norad(GameState& state, Player p) noexcept;
}

namespace mid_war {
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
    switch (card_id) {
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
        default:
            return true;
    }
}

bool CardHandlers::trigger_event(GameState& state, uint8_t card_id, Player player) noexcept {
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
        case card_ids::KOREAN_WAR: return early_war::trigger_korean_war(state, player);
        case card_ids::ROMANIAN_ABDICATION: return early_war::trigger_romanian_abdication(state, player);
        case card_ids::ARAB_ISRAELI_WAR: return early_war::trigger_arab_israeli_war(state, player);
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
        case card_ids::DE_STALINIZATION: return early_war::trigger_de_stalinization(state, player);
        case card_ids::NUCLEAR_TEST_BAN: return early_war::trigger_nuclear_test_ban(state, player);
        case card_ids::FORMOSAN_RESOLUTION: return early_war::trigger_formosan_resolution(state, player);
        case card_ids::DEFECTORS: return early_war::trigger_defectors(state, player);
        case card_ids::THE_CAMBRIDGE_FIVE: return early_war::trigger_cambridge_five(state, player);
        case card_ids::SPECIAL_RELATIONSHIP: return early_war::trigger_special_relationship(state, player);
        case card_ids::NORAD: return early_war::trigger_norad(state, player);

        // Mid War
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
    Player p = state.ctx().decision_player;

    if (action.is_confirm_done()) {
        // Early stop confirmed
        state.ctx().decision_type = DecisionType::NONE;
        state.ctx().resolving_card = 0;
        return true;
    }

    switch (card) {
        case card_ids::SOCIALIST_GOVERNMENTS: {
            uint8_t cid = action.primary_id;
            if (cid < 84 && MapData::get_country(cid).in_western_europe && state.countries[cid].us_influence > 0) {
                state.countries[cid].remove_influence(Player::US, 1);
                state.ctx().node_counts[cid]++;
                if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                if (state.ctx().remaining_steps == 0) {
                    state.ctx().resolving_card = 0;
                    return true;
                }
            }
            return false;
        }

        case card_ids::COMECON: {
            uint8_t cid = action.primary_id;
            if (cid < 84 && MapData::get_country(cid).in_eastern_europe && !Scoring::is_controlled_by(state, cid, Player::US)) {
                state.countries[cid].add_influence(Player::USSR, 1);
                state.ctx().node_counts[cid]++;
                state.ctx().mark_visited(cid);
                if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                if (state.ctx().remaining_steps == 0) {
                    state.ctx().resolving_card = 0;
                    return true;
                }
            }
            return false;
        }

        case card_ids::WARSAW_PACT: {
            if (state.ctx().decision_type == DecisionType::CHOOSE_BRANCH) {
                if (action.primary_id == 0) {
                    // Remove all US inf from 4 countries in Eastern Europe
                    state.ctx().decision_type = DecisionType::POINT_NODE;
                    state.ctx().remaining_steps = 4;
                    state.ctx().max_per_country = 1;
                    state.ctx().allow_early_stop = 1;
                    return false;
                } else {
                    // Add 5 USSR inf in Eastern Europe (max 2 per country)
                    state.ctx().decision_type = DecisionType::POINT_NODE;
                    state.ctx().remaining_steps = 5;
                    state.ctx().max_per_country = 2;
                    state.ctx().allow_early_stop = 1;
                    return false;
                }
            } else if (state.ctx().decision_type == DecisionType::POINT_NODE) {
                uint8_t cid = action.primary_id;
                if (cid < 84 && MapData::get_country(cid).in_eastern_europe) {
                    if (state.ctx().max_per_country == 1) {
                        state.countries[cid].us_influence = 0;
                    } else {
                        state.countries[cid].add_influence(Player::USSR, 1);
                        state.ctx().node_counts[cid]++;
                    }
                    if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                    if (state.ctx().remaining_steps == 0) {
                        state.ctx().resolving_card = 0;
                        return true;
                    }
                }
            }
            return false;
        }

        case card_ids::TRUMAN_DOCTRINE: {
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
            if (action.primary_id == 0) {
                // Participate: both roll 1d6 (sponsor gets +2)
                Player sponsor = get_opponent(state.ctx().decision_player);
                uint8_t sp_roll = Prng::roll_d6(state.rng_state) + 2;
                uint8_t opp_roll = Prng::roll_d6(state.rng_state);
                while (sp_roll == opp_roll) {
                    sp_roll = Prng::roll_d6(state.rng_state) + 2;
                    opp_roll = Prng::roll_d6(state.rng_state);
                }
                Player winner = (sp_roll > opp_roll) ? sponsor : get_opponent(sponsor);
                int32_t vp_delta = (winner == Player::US) ? 2 : -2;
                state.victory_points = static_cast<int8_t>(std::clamp(static_cast<int32_t>(state.victory_points) + vp_delta, -20, 20));
                if (state.victory_points >= 20 || state.victory_points <= -20) state.current_phase = Phase::GAME_OVER;
                state.ctx().resolving_card = 0;
                return true;
            } else {
                // Boycott: sponsor receives 2 VP and conducts 4 Ops
                Player sponsor = get_opponent(state.ctx().decision_player);
                int32_t vp_delta = (sponsor == Player::US) ? 2 : -2;
                state.victory_points = static_cast<int8_t>(std::clamp(static_cast<int32_t>(state.victory_points) + vp_delta, -20, 20));
                if (state.victory_points >= 20 || state.victory_points <= -20) state.current_phase = Phase::GAME_OVER;

                state.ctx().decision_player = sponsor;
                state.ctx().pending_op_card = card_ids::OLYMPIC_GAMES;
                state.ctx().pending_ops_value = 4;
                state.ctx().decision_type = DecisionType::SELECT_OP_MODE;
                return false;
            }
        }

        case card_ids::INDEPENDENT_REDS: {
            uint8_t cid = action.primary_id;
            constexpr std::array<uint8_t, 5> VALID_REDS = {
                countries::YUGOSLAVIA, countries::ROMANIA, countries::BULGARIA,
                countries::HUNGARY, countries::CZECHOSLOVAKIA
            };
            bool valid = false;
            for (uint8_t v : VALID_REDS) if (v == cid) valid = true;
            if (valid) {
                uint8_t stab = MapData::get_country(cid).stability;
                state.countries[cid].add_influence(Player::US, stab);
                state.ctx().resolving_card = 0;
                return true;
            }
            return false;
        }

        case card_ids::MARSHALL_PLAN: {
            uint8_t cid = action.primary_id;
            if (cid < 84 && MapData::get_country(cid).in_western_europe && !Scoring::is_controlled_by(state, cid, Player::USSR)) {
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

        case card_ids::INDO_PAKISTANI_WAR: {
            uint8_t target = action.primary_id;
            if (target != countries::INDIA && target != countries::PAKISTAN) return false;
            Player opp = get_opponent(p);

            if (p == Player::US) {
                state.us_mil_ops = static_cast<uint8_t>(std::min(5, static_cast<int>(state.us_mil_ops) + 2));
            } else {
                state.ussr_mil_ops = static_cast<uint8_t>(std::min(5, static_cast<int>(state.ussr_mil_ops) + 2));
            }

            int16_t mod = 0;
            const auto& c_info = MapData::get_country(target);
            for (uint8_t i = 0; i < c_info.num_neighbors; ++i) {
                if (Scoring::is_controlled_by(state, c_info.neighbors[i], opp)) mod--;
            }

            uint8_t roll = Prng::roll_d6(state.rng_state);
            if (roll + mod >= 4) {
                int32_t vp_delta = (p == Player::US) ? 2 : -2;
                state.victory_points = static_cast<int8_t>(std::clamp(static_cast<int32_t>(state.victory_points) + vp_delta, -20, 20));
                uint8_t opp_inf = state.countries[target].get_influence(opp);
                state.countries[target].remove_influence(opp, opp_inf);
                state.countries[target].add_influence(p, opp_inf);
                if (state.victory_points >= 20 || state.victory_points <= -20) state.current_phase = Phase::GAME_OVER;
            }

            if (p == Player::US && state.has_flag(effect_bits::FLOWER_POWER_ACTIVE)) {
                state.victory_points = static_cast<int8_t>(std::max(-20, state.victory_points - 2));
                if (state.victory_points <= -20) state.current_phase = Phase::GAME_OVER;
            }

            state.ctx().resolving_card = 0;
            return true;
        }

        case card_ids::SUEZ_CRISIS: {
            uint8_t cid = action.primary_id;
            constexpr std::array<uint8_t, 4> SUEZ_COUNTRIES = {
                countries::UNITED_KINGDOM, countries::FRANCE, countries::ISRAEL, countries::EGYPT
            };
            bool valid = false;
            for (uint8_t v : SUEZ_COUNTRIES) if (v == cid) valid = true;
            if (valid && state.countries[cid].us_influence > 0 && state.ctx().node_counts[cid] < 2) {
                state.countries[cid].remove_influence(Player::US, 1);
                state.ctx().node_counts[cid]++;
                if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                if (state.ctx().remaining_steps == 0) {
                    state.ctx().resolving_card = 0;
                    return true;
                }
            }
            return false;
        }

        case card_ids::EAST_EUROPEAN_UNREST: {
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
            uint8_t cid = action.primary_id;
            if (cid < 84 && (MapData::get_country(cid).region == Region::AFRICA || MapData::get_country(cid).in_southeast_asia)) {
                state.countries[cid].add_influence(Player::USSR, 1);
                state.ctx().mark_visited(cid);
                if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                if (state.ctx().remaining_steps == 0) {
                    state.ctx().resolving_card = 0;
                    return true;
                }
            }
            return false;
        }

        case card_ids::DE_STALINIZATION: {
            // Stage 1 (removal) and Stage 2 (placement)
            uint8_t cid = action.primary_id;
            if (state.ctx().temp_card_cnt < 4 && state.countries[cid].ussr_influence > 0) {
                state.countries[cid].remove_influence(Player::USSR, 1);
                state.ctx().temp_card_cnt++; // Tracks total removed
                if (state.ctx().remaining_steps > 0) state.ctx().remaining_steps--;
                if (state.ctx().remaining_steps == 0) {
                    // Transition to placement
                    state.ctx().remaining_steps = state.ctx().temp_card_cnt;
                    state.ctx().max_per_country = 2;
                    state.ctx().visited_nodes = {};
                    state.ctx().node_counts = {};
                    return false;
                }
            } else if (state.ctx().remaining_steps > 0) {
                // Placement stage
                if (cid < 84 && !Scoring::is_controlled_by(state, cid, Player::US) && state.ctx().node_counts[cid] < 2) {
                    state.countries[cid].add_influence(Player::USSR, 1);
                    state.ctx().node_counts[cid]++;
                    state.ctx().remaining_steps--;
                    if (state.ctx().remaining_steps == 0) {
                        state.ctx().resolving_card = 0;
                        return true;
                    }
                }
            }
            return false;
        }

        case card_ids::BRUSH_WAR: {
            uint8_t cid = action.primary_id;
            if (cid >= 84 || MapData::get_country(cid).stability > 2) return false;

            // NATO check
            if (p == Player::USSR && state.has_flag(effect_bits::NATO_ACTIVE) &&
                MapData::get_country(cid).region == Region::EUROPE && Scoring::is_controlled_by(state, cid, Player::US)) {
                return false;
            }

            Player opp = get_opponent(p);
            if (p == Player::US) {
                state.us_mil_ops = static_cast<uint8_t>(std::min(5, static_cast<int>(state.us_mil_ops) + 3));
            } else {
                state.ussr_mil_ops = static_cast<uint8_t>(std::min(5, static_cast<int>(state.ussr_mil_ops) + 3));
            }

            int16_t mod = 0;
            const auto& c_info = MapData::get_country(cid);
            for (uint8_t i = 0; i < c_info.num_neighbors; ++i) {
                if (Scoring::is_controlled_by(state, c_info.neighbors[i], opp)) mod--;
            }

            uint8_t roll = Prng::roll_d6(state.rng_state);
            if (roll + mod >= 3) {
                int32_t vp_delta = (p == Player::US) ? 1 : -1;
                state.victory_points = static_cast<int8_t>(std::clamp(static_cast<int32_t>(state.victory_points) + vp_delta, -20, 20));
                uint8_t opp_inf = state.countries[cid].get_influence(opp);
                state.countries[cid].remove_influence(opp, opp_inf);
                state.countries[cid].add_influence(p, opp_inf);
                if (state.victory_points >= 20 || state.victory_points <= -20) state.current_phase = Phase::GAME_OVER;
            }

            if (p == Player::US && state.has_flag(effect_bits::FLOWER_POWER_ACTIVE)) {
                state.victory_points = static_cast<int8_t>(std::max(-20, state.victory_points - 2));
                if (state.victory_points <= -20) state.current_phase = Phase::GAME_OVER;
            }

            state.ctx().resolving_card = 0;
            return true;
        }

        case card_ids::SALT_NEGOTIATIONS: {
            uint8_t card_retrieved = action.primary_id;
            if (card_retrieved >= 1 && card_retrieved <= 110 && state.card_locations[card_retrieved] == CardLocation::DISCARD_PILE) {
                state.card_locations[card_retrieved] = (p == Player::US) ? CardLocation::HAND_US : CardLocation::HAND_USSR;
            }
            state.ctx().resolving_card = 0;
            return true;
        }

        case card_ids::SUMMIT: {
            if (action.primary_id == 0) {
                state.defcon = static_cast<uint8_t>(std::min(5, static_cast<int>(state.defcon) + 1));
            } else if (action.primary_id == 1) {
                if (state.defcon > 1) {
                    state.defcon--;
                    if (state.defcon == 2) state.defcon_dropped_to_2_in_ar = 1;
                    if (state.defcon == 1) {
                        Player loser = state.phasing_player;
                        state.victory_points = (loser == Player::US) ? -20 : 20;
                        state.current_phase = Phase::GAME_OVER;
                    }
                }
            }
            state.ctx().resolving_card = 0;
            return true;
        }

        case card_ids::HOW_I_LEARNED_TO_STOP_WORRYING: {
            uint8_t new_defcon = std::clamp(action.primary_id, static_cast<uint8_t>(1), static_cast<uint8_t>(5));
            state.defcon = new_defcon;
            if (state.defcon == 2) state.defcon_dropped_to_2_in_ar = 1;
            if (state.defcon == 1) {
                Player loser = state.phasing_player;
                state.victory_points = (loser == Player::US) ? -20 : 20;
                state.current_phase = Phase::GAME_OVER;
            }
            state.ctx().resolving_card = 0;
            return true;
        }

        case card_ids::JUNTA: {
            uint8_t cid = action.primary_id;
            if (cid < 84 && (MapData::get_country(cid).region == Region::CENTRAL_AMERICA || MapData::get_country(cid).region == Region::SOUTH_AMERICA)) {
                state.countries[cid].add_influence(p, 2);
                // Transition to Ops in CA/SA
                state.ctx().pending_op_card = card_ids::JUNTA;
                state.ctx().pending_ops_value = 2;
                state.ctx().decision_type = DecisionType::SELECT_OP_MODE;
                return false;
            }
            return false;
        }

        case card_ids::GRAIN_SALES: {
            if (action.primary_id == 0) {
                // Play drawn card
                uint8_t drawn_card = state.ctx().temp_cards[0];
                state.card_locations[drawn_card] = CardLocation::HAND_US;
                state.ctx().pending_op_card = drawn_card;
                state.ctx().decision_type = DecisionType::SELECT_PLAY_MODE;
                return false;
            } else {
                // Return card, US conducts 2 Ops
                state.ctx().pending_op_card = card_ids::GRAIN_SALES;
                state.ctx().pending_ops_value = 2;
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
            uint8_t card_id = action.primary_id;
            if (card_id >= 1 && card_id <= 110 && state.card_locations[card_id] == CardLocation::DISCARD_PILE) {
                state.card_locations[card_id] = CardData::get_card(card_id).one_time ? CardLocation::REMOVED_FROM_GAME : CardLocation::DISCARD_PILE;
                state.push_context();
                state.ctx().decision_player = Player::US;
                state.ctx().resolving_card = card_id;
                bool done = trigger_event(state, card_id, Player::US);
                if (done) state.pop_context();
                return done;
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
                case card_ids::SOCIALIST_GOVERNMENTS:
                    if (c_info.in_western_europe && state.countries[i].us_influence > 0 && state.ctx().node_counts[i] < 2) {
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
                case card_ids::TRUMAN_DOCTRINE:
                    if (c_info.region == Region::EUROPE && Scoring::get_country_control(state, i) == Player::NONE && state.countries[i].ussr_influence > 0) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::INDO_PAKISTANI_WAR:
                    if (i == countries::INDIA || i == countries::PAKISTAN) mask_out[i] = 1;
                    break;
                case card_ids::BRUSH_WAR:
                    if (c_info.stability <= 2) mask_out[i] = 1;
                    break;
                case card_ids::JUNTA:
                    if (c_info.region == Region::CENTRAL_AMERICA || c_info.region == Region::SOUTH_AMERICA) mask_out[i] = 1;
                    break;
                case card_ids::COLONIAL_REAR_GUARDS:
                    if (c_info.region == Region::AFRICA || c_info.in_southeast_asia) mask_out[i] = 1;
                    break;
                case card_ids::PUPPET_GOVERNMENTS:
                    if (state.countries[i].us_influence == 0 && state.countries[i].ussr_influence == 0) mask_out[i] = 1;
                    break;
                case card_ids::OAS_FOUNDED:
                    if ((c_info.region == Region::CENTRAL_AMERICA || c_info.region == Region::SOUTH_AMERICA) && state.ctx().node_counts[i] < 2) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::THE_VOICE_OF_AMERICA:
                    if (c_info.region != Region::EUROPE && state.countries[i].ussr_influence > 0 && state.ctx().node_counts[i] < 2) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::LIBERATION_THEOLOGY:
                    if (c_info.region == Region::CENTRAL_AMERICA && state.ctx().node_counts[i] < 2) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::USSURI_RIVER_SKIRMISH:
                    if (c_info.region == Region::ASIA && state.ctx().node_counts[i] < 2) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::THE_REFORMER:
                    if (c_info.region == Region::EUROPE && state.ctx().node_counts[i] < 2) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::MARINE_BARRACKS_BOMBING:
                    if (c_info.region == Region::MIDDLE_EAST && state.countries[i].us_influence > 0 && state.ctx().node_counts[i] < 2) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::PERSHING_II_DEPLOYED:
                    if (c_info.in_western_europe && state.countries[i].us_influence > 0 && !state.ctx().is_visited(i)) {
                        mask_out[i] = 1;
                    }
                    break;
                case card_ids::IRAN_IRAQ_WAR:
                    if (i == countries::IRAN || i == countries::IRAQ) mask_out[i] = 1;
                    break;
                default:
                    mask_out[i] = 1;
                    break;
            }
        }
    } else if (dt == DecisionType::SELECT_CARD) {
        *out_size = 112;
        for (uint8_t i = 0; i < 112; ++i) mask_out[i] = 0;

        switch (card) {
            case card_ids::SALT_NEGOTIATIONS:
            case card_ids::STAR_WARS:
                for (uint8_t i = 1; i <= 110; ++i) {
                    if (state.card_locations[i] == CardLocation::DISCARD_PILE && !CardData::is_scoring_card(i)) {
                        mask_out[i] = 1;
                    }
                }
                break;
            case card_ids::OUR_MAN_IN_TEHRAN:
                for (uint8_t k = 0; k < state.ctx().temp_card_cnt; ++k) {
                    uint8_t c = state.ctx().temp_cards[k];
                    if (c >= 1 && c <= 110) mask_out[c] = 1;
                }
                break;
            case card_ids::BLOCKADE:
                for (uint8_t i = 1; i <= 110; ++i) {
                    if (state.card_locations[i] == CardLocation::HAND_US && CardData::get_card(i).ops >= 3) {
                        mask_out[i] = 1;
                    }
                }
                break;
            default:
                CardLocation loc = (p == Player::US) ? CardLocation::HAND_US : CardLocation::HAND_USSR;
                for (uint8_t i = 1; i <= 110; ++i) {
                    if (state.card_locations[i] == loc) mask_out[i] = 1;
                }
                break;
        }
    } else if (dt == DecisionType::CHOOSE_BRANCH) {
        *out_size = 8;
        for (uint8_t i = 0; i < 8; ++i) mask_out[i] = 0;
        mask_out[0] = 1;
        mask_out[1] = 1;
        if (card == card_ids::HOW_I_LEARNED_TO_STOP_WORRYING) {
            mask_out[2] = 1; mask_out[3] = 1; mask_out[4] = 1;
        }
    }
}

} // namespace ts
