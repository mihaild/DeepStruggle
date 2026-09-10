#pragma once
#include "types.hpp"
#include "game_state.hpp"
#include "constants.hpp"
#include "map_data.hpp"
#include "scoring.hpp"
#include "prng.hpp"
#include "micro_action.hpp"
#include <algorithm>

namespace ts {
namespace war_helpers {

[[nodiscard]] inline bool is_war_card(uint8_t card_id) noexcept {
    return card_id == card_ids::KOREAN_WAR ||
           card_id == card_ids::ARAB_ISRAELI_WAR ||
           card_id == card_ids::INDO_PAKISTANI_WAR ||
           card_id == card_ids::BRUSH_WAR ||
           card_id == card_ids::IRAN_IRAQ_WAR;
}

inline bool has_fixed_target(uint8_t card_id, uint8_t& out_target) noexcept {
    if (card_id == card_ids::KOREAN_WAR) {
        out_target = countries::SOUTH_KOREA;
        return true;
    }
    if (card_id == card_ids::ARAB_ISRAELI_WAR) {
        out_target = countries::ISRAEL;
        return true;
    }
    out_target = 255;
    return false;
}

[[nodiscard]] inline bool is_valid_war_target(const GameState& state, uint8_t card_id, Player p, uint8_t target) noexcept {
    if (target >= 84) return false;
    if (card_id == card_ids::INDO_PAKISTANI_WAR) {
        return target == countries::INDIA || target == countries::PAKISTAN;
    }
    if (card_id == card_ids::IRAN_IRAQ_WAR) {
        return target == countries::IRAN || target == countries::IRAQ;
    }
    if (card_id == card_ids::BRUSH_WAR) {
        const auto& c_info = MapData::get_country(target);
        if (c_info.stability > 2) return false;
        bool nato_canceled_for_country = (target == countries::WEST_GERMANY && state.has_flag(effect_bits::NATO_CANCELED_WEST_GERMANY)) ||
                                         (target == countries::FRANCE && state.has_flag(effect_bits::NATO_CANCELED_FRANCE));
        if (p == Player::USSR && state.has_flag(effect_bits::NATO_ACTIVE) &&
            c_info.region == Region::EUROPE && Scoring::is_controlled_by(state, target, Player::US) &&
            !nato_canceled_for_country) {
            return false;
        }
        return true;
    }
    return false;
}

inline bool trigger_war(GameState& state, uint8_t card_id, Player player, uint8_t forced_roll = 0) noexcept {
    if (card_id == card_ids::ARAB_ISRAELI_WAR && state.has_flag(effect_bits::CAMP_DAVID_PLAYED)) {
        return true;
    }

    uint8_t fixed_target = 255;
    if (has_fixed_target(card_id, fixed_target)) {
        state.ctx().decision_player = Player::NONE;
        state.ctx().decision_type = DecisionType::ROLL_DIE;
        state.ctx().resolving_card = card_id;
        state.ctx().pending_roll = RollType::WAR_EVENT;
        state.ctx().roll_target = fixed_target;
        state.ctx().roll_actor = player;
        // `forced_roll` is not stored. A war with a fixed target rolls inside the event with
        // nothing to choose, so the caller that wants a specific die passes it on the ROLL_DIE
        // action; see the read below.
        (void)forced_roll;
        return false;
    }

    // Choice target wars: Indo-Pakistani, Brush, Iran-Iraq
    state.ctx().decision_player = player;
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().remaining_steps = 1;
    // A war is fought against a legal target or not at all; declining is not one of its
    // options. Set explicitly rather than inherited: at turn 5's headline of ts-replayer game
    // 170 SALT Negotiations and Missile Envy resolve first and leave allow_early_stop set, so
    // Brush War's target choice offered a decline alongside its 52 targets that
    // handle_war_step has no case for -- and the war silently never happened.
    state.ctx().allow_early_stop = 0;
    state.ctx().resolving_card = card_id;
    return false;
}

inline void get_war_target_mask(const GameState& state, uint8_t card_id, Player p, uint8_t* mask_out) noexcept {
    for (uint8_t i = 0; i < 84; ++i) {
        if (is_valid_war_target(state, card_id, p, i)) {
            mask_out[i] = 1;
        }
    }
}

inline bool handle_war_step(GameState& state, uint8_t card_id, Player p, const MicroAction& action) noexcept {
    if (state.ctx().decision_type == DecisionType::POINT_NODE) {
        uint8_t target = action.primary_id;
        if (!is_valid_war_target(state, card_id, p, target)) {
            return false;
        }
        state.ctx().pending_roll = RollType::WAR_EVENT;
        state.ctx().roll_target = target;
        state.ctx().roll_actor = p;
        state.ctx().decision_player = Player::NONE;
        state.ctx().decision_type = DecisionType::ROLL_DIE;
        return false;
    }

    if (state.ctx().decision_type == DecisionType::ROLL_DIE) {
        uint8_t target = state.ctx().roll_target;
        Player roller = (state.ctx().roll_actor != Player::NONE) ? state.ctx().roll_actor : p;
        Player opp = get_opponent(roller);

        uint8_t mil_ops = (card_id == card_ids::BRUSH_WAR) ? 3 : 2;
        int8_t vp_award = (card_id == card_ids::BRUSH_WAR) ? 1 : 2;
        int8_t threshold = (card_id == card_ids::BRUSH_WAR) ? 3 : 4;
        bool check_target_control = (card_id == card_ids::ARAB_ISRAELI_WAR);

        // 1. Give Military Operations
        if (roller == Player::US) {
            state.us_mil_ops = static_cast<uint8_t>(std::min(5, static_cast<int>(state.us_mil_ops) + mil_ops));
        } else {
            state.ussr_mil_ops = static_cast<uint8_t>(std::min(5, static_cast<int>(state.ussr_mil_ops) + mil_ops));
        }

        // 2. Calculate Modifiers (-1 per opponent-controlled neighbor, plus target if Arab-Israeli)
        int16_t mod = 0;
        if (check_target_control) {
            if (Scoring::is_controlled_by(state, target, opp)) mod--;
        }
        const auto& c_info = MapData::get_country(target);
        for (uint8_t i = 0; i < c_info.num_neighbors; ++i) {
            if (Scoring::is_controlled_by(state, c_info.neighbors[i], opp)) mod--;
        }

        // 3. Roll Die & Record DieRollRecord
        uint8_t forced_roll = action.primary_id;
        uint8_t roll = (forced_roll >= 1 && forced_roll <= 6) ? forced_roll : Prng::roll_d6(state.rng_state);
        state.last_die_roll = roll;
        bool success = (roll + mod >= threshold);

        state.last_roll = DieRollRecord{
            .type = RollType::WAR_EVENT,
            .roller = roller,
            .card_id = card_id,
            .country_id = target,
            .roll1 = roll,
            .mod1 = static_cast<int8_t>(mod),
            .roll2 = 0,
            .mod2 = threshold,
            .success = success,
            .net_delta = static_cast<int8_t>(success ? vp_award : 0)
        };

        // 4. On Success: Award VPs and Switch Influence
        if (success) {
            int32_t vp_delta = (roller == Player::US) ? vp_award : -vp_award;
            state.victory_points = static_cast<int8_t>(std::clamp(static_cast<int32_t>(state.victory_points) + vp_delta, -20, 20));
            uint8_t opp_inf = state.countries[target].get_influence(opp);
            state.countries[target].remove_influence(opp, opp_inf);
            state.countries[target].add_influence(roller, opp_inf);
            if (state.victory_points >= 20 || state.victory_points <= -20) {
                state.current_phase = Phase::GAME_OVER;
            }
        }

        // 5. Flower Power Penalty (If US played war event while Flower Power active)
        if (roller == Player::US && state.has_flag(effect_bits::FLOWER_POWER_ACTIVE)) {
            state.victory_points = static_cast<int8_t>(std::max(-20, state.victory_points - 2));
            if (state.victory_points <= -20) {
                state.current_phase = Phase::GAME_OVER;
            }
        }

        state.ctx().resolving_card = 0;
        return true;
    }

    return true;
}

} // namespace war_helpers
} // namespace ts
