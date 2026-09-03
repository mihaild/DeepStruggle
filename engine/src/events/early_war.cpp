#include "ts/card_handlers.hpp"
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

bool trigger_duck_and_cover(GameState& state, Player p) noexcept {
    if (state.defcon > 1) {
        state.defcon--;
        if (state.defcon == 2) state.defcon_dropped_to_2 = 1;
    }
    // Check DEFCON suicide
    if (state.defcon == 1) {
        resolve_defcon_one_loss(state, p);
        return true;
    }
    int16_t vp_gain = 5 - state.defcon;
    if (vp_gain > 0) {
        state.victory_points = static_cast<int8_t>(std::min(20, state.victory_points + vp_gain));
        if (state.victory_points >= 20) state.current_phase = Phase::GAME_OVER;
    }
    return true;
}

bool trigger_five_year_plan(GameState& state, Player p) noexcept {
    // USSR randomly discards a card
    uint8_t ussr_cards[111];
    uint8_t count = 0;
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == CardLocation::HAND_USSR) {
            ussr_cards[count++] = i;
        }
    }
    if (count == 0) return true;

    uint32_t chosen_idx = Prng::random_index(state.rng_state, count);
    uint8_t chosen_card = ussr_cards[chosen_idx];

    const auto& c_info = CardData::get_card(chosen_card);
    if (c_info.side == Player::US) {
        // Discard card, but trigger event!
        state.card_locations[chosen_card] = c_info.one_time ? CardLocation::REMOVED_FROM_GAME : CardLocation::DISCARD_PILE;
        state.push_context();
        state.ctx().decision_player = Player::US;
        state.ctx().resolving_card = chosen_card;
        bool done = CardHandlers::trigger_event(state, chosen_card, Player::US);
        if (done) {
            state.pop_context();
        }
        return done;
    } else {
        // Discard without event
        state.card_locations[chosen_card] = CardLocation::DISCARD_PILE;
        return true;
    }
}

bool trigger_socialist_governments(GameState& state, Player p) noexcept {
    if (state.has_flag(effect_bits::IRON_LADY_PLAYED)) return true;

    state.ctx().decision_player = Player::USSR;
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().remaining_steps = 3;
    state.ctx().max_per_country = 2;
    state.ctx().allow_early_stop = 1;
    state.ctx().resolving_card = card_ids::SOCIALIST_GOVERNMENTS;
    return false;
}

bool trigger_fidel(GameState& state, Player p) noexcept {
    state.countries[countries::CUBA].us_influence = 0;
    uint8_t cur_ussr = state.countries[countries::CUBA].ussr_influence;
    if (cur_ussr < 3) {
        state.countries[countries::CUBA].ussr_influence = 3;
    }
    return true;
}

bool trigger_vietnam_revolts(GameState& state, Player p) noexcept {
    state.countries[countries::VIETNAM].add_influence(Player::USSR, 2);
    state.set_flag(effect_bits::VIETNAM_REVOLTS_ACTIVE);
    return true;
}

bool trigger_blockade(GameState& state, Player p) noexcept {
    // Check if US has any card with Ops >= 3 to discard
    bool has_3ops = false;
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == CardLocation::HAND_US) {
            // Effective Ops, as the discard itself is judged: Containment's +1 can make a
            // printed 2 qualify, and this test decides whether the US is asked at all.
            if (Operations::get_effective_ops(state, i, Player::US) >= 3) {
                has_3ops = true;
                break;
            }
        }
    }
    if (!has_3ops) {
        // Remove all US influence from West Germany
        state.countries[countries::WEST_GERMANY].us_influence = 0;
        return true;
    }

    // US chooses card to discard
    state.ctx().decision_player = Player::US;
    state.ctx().decision_type = DecisionType::SELECT_CARD;
    state.ctx().remaining_steps = 1;
    state.ctx().allow_early_stop = 1;
    state.ctx().resolving_card = card_ids::BLOCKADE;
    return false;
}

bool trigger_korean_war(GameState& state, Player p, uint8_t forced_roll) noexcept {
    return war_helpers::trigger_war(state, card_ids::KOREAN_WAR, p, forced_roll);
}

bool trigger_romanian_abdication(GameState& state, Player p) noexcept {
    state.countries[countries::ROMANIA].us_influence = 0;
    state.countries[countries::ROMANIA].ussr_influence = std::max(state.countries[countries::ROMANIA].ussr_influence, static_cast<uint8_t>(3));
    return true;
}

bool trigger_arab_israeli_war(GameState& state, Player p, uint8_t forced_roll) noexcept {
    return war_helpers::trigger_war(state, card_ids::ARAB_ISRAELI_WAR, p, forced_roll);
}

bool trigger_comecon(GameState& state, Player p) noexcept {
    state.ctx().decision_player = Player::USSR;
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().remaining_steps = 4;
    state.ctx().max_per_country = 1;
    state.ctx().allow_early_stop = 1;
    state.ctx().resolving_card = card_ids::COMECON;
    return false;
}

bool trigger_nasser(GameState& state, Player p) noexcept {
    state.countries[countries::EGYPT].add_influence(Player::USSR, 2);
    uint8_t us_inf = state.countries[countries::EGYPT].us_influence;
    uint8_t remove_amt = (us_inf + 1) / 2; // Half rounded up
    state.countries[countries::EGYPT].remove_influence(Player::US, remove_amt);
    return true;
}

bool trigger_warsaw_pact(GameState& state, Player p) noexcept {
    state.set_flag(effect_bits::WARSAW_PACT_PLAYED);
    // USSR chooses branch: 0 = Remove all US influence from 4 countries in Eastern Europe, 1 = Add 5 USSR influence in EE (max 2 per country)
    state.ctx().decision_player = Player::USSR;
    state.ctx().decision_type = DecisionType::CHOOSE_BRANCH;
    state.ctx().resolving_card = card_ids::WARSAW_PACT;
    return false;
}

bool trigger_de_gaulle(GameState& state, Player p) noexcept {
    state.countries[countries::FRANCE].remove_influence(Player::US, 2);
    state.countries[countries::FRANCE].add_influence(Player::USSR, 1);
    state.set_flag(effect_bits::NATO_CANCELED_FRANCE);
    return true;
}

bool trigger_captured_nazi_scientist(GameState& state, Player p) noexcept {
    uint8_t& cur_track = (p == Player::US) ? state.us_space_track : state.ussr_space_track;
    uint8_t opp_track = (p == Player::US) ? state.ussr_space_track : state.us_space_track;
    if (cur_track < 8) {
        cur_track++;
        const auto& box = SpaceRace::get_box_info(cur_track);
        uint8_t vp = (cur_track > opp_track) ? box.vp_first : box.vp_second;
        if (vp > 0) {
            int32_t vp_delta = (p == Player::US) ? vp : -static_cast<int32_t>(vp);
            int32_t new_vp = static_cast<int32_t>(state.victory_points) + vp_delta;
            new_vp = std::clamp(new_vp, -20, 20);
            state.victory_points = static_cast<int8_t>(new_vp);
            if (state.victory_points >= 20 || state.victory_points <= -20) {
                state.current_phase = Phase::GAME_OVER;
            }
        }
    }
    return true;
}

bool trigger_truman_doctrine(GameState& state, Player p) noexcept {
    state.ctx().decision_player = Player::US;
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().remaining_steps = 1;
    state.ctx().allow_early_stop = 0;   // removal is mandatory; see trigger_war
    state.ctx().resolving_card = card_ids::TRUMAN_DOCTRINE;
    return false;
}

bool trigger_olympic_games(GameState& state, Player p) noexcept {
    Player opp = get_opponent(p);
    // Opponent chooses: Branch 0 = Participate, Branch 1 = Boycott
    state.ctx().decision_player = opp;
    state.ctx().decision_type = DecisionType::CHOOSE_BRANCH;
    state.ctx().resolving_card = card_ids::OLYMPIC_GAMES;
    return false;
}

bool trigger_nato(GameState& state, Player p) noexcept {
    if (state.has_flag(effect_bits::MARSHALL_PLAN_PLAYED) || state.has_flag(effect_bits::WARSAW_PACT_PLAYED)) {
        state.set_flag(effect_bits::NATO_ACTIVE);
    }
    return true;
}

bool trigger_independent_reds(GameState& state, Player p) noexcept {
    // Independent Reds: Add US Influence to Yugoslavia, Romania, Bulgaria, Hungary, or Czechoslovakia
    // equal to that country's USSR Influence.
    bool any_valid = false;
    uint8_t targets[] = { countries::YUGOSLAVIA, countries::ROMANIA, countries::BULGARIA, countries::HUNGARY, countries::CZECHOSLOVAKIA };
    for (uint8_t cid : targets) {
        if (state.countries[cid].ussr_influence > 0) {
            any_valid = true;
            break;
        }
    }
    if (!any_valid) {
        return true; // No valid target country has USSR influence -> event finishes immediately without effect
    }

    state.ctx().decision_player = Player::US;
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().remaining_steps = 1;
    state.ctx().allow_early_stop = 0;   // the no-target case returned above
    state.ctx().resolving_card = card_ids::INDEPENDENT_REDS;
    return false;
}

bool trigger_marshall_plan(GameState& state, Player p) noexcept {
    state.set_flag(effect_bits::MARSHALL_PLAN_PLAYED);
    state.ctx().decision_player = Player::US;
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().remaining_steps = 7;
    state.ctx().max_per_country = 1;
    state.ctx().allow_early_stop = 1;
    state.ctx().resolving_card = card_ids::MARSHALL_PLAN;
    return false;
}

bool trigger_indo_pakistani_war(GameState& state, Player p) noexcept {
    return war_helpers::trigger_war(state, card_ids::INDO_PAKISTANI_WAR, p);
}

bool trigger_containment(GameState& state, Player p) noexcept {
    state.set_flag(effect_bits::CONTAINMENT_ACTIVE);
    return true;
}

bool trigger_cia_created(GameState& state, Player p) noexcept {
    // US conducts Operations using card Ops value (1 Op base)
    state.ctx().decision_player = Player::US;
    state.ctx().pending_op_card = card_ids::CIA_CREATED;
    state.ctx().pending_ops_value = Operations::grant_ops(state, 1, Player::US);
    state.ctx().decision_type = DecisionType::SELECT_OP_MODE;
    state.ctx().resolving_card = 0;
    return false;
}

bool trigger_us_japan_pact(GameState& state, Player p) noexcept {
    state.set_flag(effect_bits::US_JAPAN_PACT_ACTIVE);
    uint8_t required_us = static_cast<uint8_t>(state.countries[countries::JAPAN].ussr_influence + 4);
    state.countries[countries::JAPAN].us_influence = std::max(state.countries[countries::JAPAN].us_influence, required_us);
    return true;
}

bool trigger_suez_crisis(GameState& state, Player p) noexcept {
    state.ctx().decision_player = Player::USSR;
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().remaining_steps = 4;
    state.ctx().max_per_country = 2;
    state.ctx().allow_early_stop = 1;
    state.ctx().resolving_card = card_ids::SUEZ_CRISIS;
    return false;
}

bool trigger_east_european_unrest(GameState& state, Player p) noexcept {
    state.ctx().decision_player = Player::US;
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().remaining_steps = 3;
    state.ctx().max_per_country = 1;
    state.ctx().allow_early_stop = 1;
    state.ctx().resolving_card = card_ids::EAST_EUROPEAN_UNREST;
    return false;
}

bool trigger_decolonization(GameState& state, Player p) noexcept {
    state.ctx().decision_player = Player::USSR;
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().remaining_steps = 4;
    state.ctx().max_per_country = 1;
    state.ctx().allow_early_stop = 1;
    state.ctx().resolving_card = card_ids::DECOLONIZATION;
    return false;
}

bool trigger_red_scare_purge(GameState& state, Player p) noexcept {
    if (p == Player::US) {
        state.set_flag(effect_bits::PURGE_USSR_ACTIVE);
    } else {
        state.set_flag(effect_bits::PURGE_US_ACTIVE);
    }
    return true;
}

bool trigger_de_stalinization(GameState& state, Player p) noexcept {
    // USSR removes up to 4 influence from countries and places in non-US controlled countries (max 2 per country)
    state.ctx().decision_player = Player::USSR;
    state.ctx().decision_type = DecisionType::POINT_NODE; // Stage 1: Removal (up to 4)
    state.ctx().remaining_steps = 4;
    state.ctx().max_per_country = 0; // Stage 1 removal flag
    state.ctx().allow_early_stop = 1;
    state.ctx().resolving_card = card_ids::DE_STALINIZATION;
    state.ctx().temp_card_cnt = 0; // Counts total removed
    return false;
}

bool trigger_nuclear_test_ban(GameState& state, Player p) noexcept {
    int16_t vp = state.defcon - 2;
    if (vp > 0) {
        int32_t vp_delta = (p == Player::US) ? vp : -vp;
        int32_t new_vp = static_cast<int32_t>(state.victory_points) + vp_delta;
        state.victory_points = static_cast<int8_t>(std::clamp(new_vp, -20, 20));
        if (state.victory_points >= 20 || state.victory_points <= -20) {
            state.current_phase = Phase::GAME_OVER;
        }
    }
    state.defcon = static_cast<uint8_t>(std::min(5, static_cast<int>(state.defcon) + 2));
    return true;
}

bool trigger_formosan_resolution(GameState& state, Player p) noexcept {
    state.set_flag(effect_bits::FORMOSAN_RESOLUTION_ACTIVE);
    return true;
}

bool trigger_defectors(GameState& state, Player p) noexcept {
    if (state.current_phase == Phase::HEADLINE) {
        if (state.headline_ussr_card > 0) {
            state.headline_ussr_card = 0;
        }
    } else if (state.current_phase == Phase::ACTION_ROUND) {
        // If played by USSR during an Action Round, US gains 1 VP
        if (state.phasing_player == Player::USSR || (state.phasing_player == Player::NONE && p == Player::USSR)) {
            state.victory_points = static_cast<int8_t>(std::min(20, state.victory_points + 1));
            if (state.victory_points >= 20) state.current_phase = Phase::GAME_OVER;
        }
    }
    return true;
}

bool trigger_cambridge_five(GameState& state, Player p) noexcept {
    if (state.turn >= 8) return true; // Cannot be played in Late War
    // Check if US has scoring cards
    uint8_t score_cards[16];
    uint8_t cnt = 0;
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == CardLocation::HAND_US && CardData::is_scoring_card(i)) {
            if (cnt < 16) score_cards[cnt++] = i;
        }
    }
    if (cnt == 0) return true;

    // USSR chooses region named on one of revealed scoring cards to add 1 influence
    state.ctx().decision_player = Player::USSR;
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().remaining_steps = 1;
    state.ctx().allow_early_stop = 0;   // the no-scoring-card case returned above
    state.ctx().resolving_card = card_ids::THE_CAMBRIDGE_FIVE;
    uint8_t stored_cnt = static_cast<uint8_t>(std::min<size_t>(cnt, state.ctx().temp_cards.size()));
    for (uint8_t k = 0; k < stored_cnt; ++k) state.ctx().temp_cards[k] = score_cards[k];
    state.ctx().temp_card_cnt = stored_cnt;
    return false;
}

bool trigger_special_relationship(GameState& state, Player p) noexcept {
    if (!Scoring::is_controlled_by(state, countries::UNITED_KINGDOM, Player::US)) return true;

    if (!state.has_flag(effect_bits::NATO_ACTIVE)) {
        // Add 1 US influence adjacent to UK
        state.ctx().decision_player = Player::US;
        state.ctx().decision_type = DecisionType::POINT_NODE;
        state.ctx().remaining_steps = 1;
        state.ctx().allow_early_stop = 0;   // placement is mandatory; see trigger_war
        state.ctx().resolving_card = card_ids::SPECIAL_RELATIONSHIP;
        return false;
    } else {
        // Add 2 US influence in one Western European country + 2 VP
        state.victory_points = static_cast<int8_t>(std::min(20, state.victory_points + 2));
        if (state.victory_points >= 20) state.current_phase = Phase::GAME_OVER;
        state.ctx().decision_player = Player::US;
        state.ctx().decision_type = DecisionType::POINT_NODE;
        state.ctx().remaining_steps = 1;
        state.ctx().allow_early_stop = 0;   // placement is mandatory; see trigger_war
        state.ctx().resolving_card = card_ids::SPECIAL_RELATIONSHIP;
        return false;
    }
}

bool trigger_norad(GameState& state, Player p) noexcept {
    state.set_flag(effect_bits::NORAD_ACTIVE);
    return true;
}

bool trigger_un_intervention(GameState& state, Player p) noexcept {
    Player opp = get_opponent(p);
    bool has_opp_card = false;
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == ((p == Player::US) ? CardLocation::HAND_US : CardLocation::HAND_USSR)) {
            if (CardData::get_card(i).side == opp && !CardData::is_scoring_card(i)) {
                has_opp_card = true;
                break;
            }
        }
    }
    if (!has_opp_card) return true;

    state.ctx().decision_player = p;
    state.ctx().decision_type = DecisionType::SELECT_CARD;
    state.ctx().resolving_card = card_ids::UN_INTERVENTION;
    return false;
}

} // namespace early_war

} // namespace ts
