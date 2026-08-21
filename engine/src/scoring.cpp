#include "ts/scoring.hpp"
#include "ts/map_data.hpp"
#include "ts/constants.hpp"
#include <algorithm>

namespace ts {

Player Scoring::get_country_control(const GameState& state, uint8_t country_id) noexcept {
    if (country_id >= 84) return Player::NONE;
    const auto& c_info = MapData::get_country(country_id);
    uint8_t stab = c_info.stability;
    uint8_t us_inf = state.countries[country_id].us_influence;
    uint8_t ussr_inf = state.countries[country_id].ussr_influence;

    if (us_inf >= stab && (us_inf - ussr_inf) >= stab) {
        return Player::US;
    }
    if (ussr_inf >= stab && (ussr_inf - us_inf) >= stab) {
        return Player::USSR;
    }
    return Player::NONE;
}

bool Scoring::is_controlled_by(const GameState& state, uint8_t country_id, Player p) noexcept {
    return get_country_control(state, country_id) == p;
}

RegionScoreSummary Scoring::evaluate_region(const GameState& state, Region r) noexcept {
    RegionScoreSummary summary{};
    if (r == Region::NONE_REGION) return summary;

    bool taiwan_is_bg = (r == Region::ASIA && state.has_flag(effect_bits::FORMOSAN_RESOLUTION_ACTIVE) &&
                         is_controlled_by(state, countries::TAIWAN, Player::US));

    uint8_t total_bg = MapData::get_region_battleground_count(r) + (taiwan_is_bg ? 1 : 0);

    for (uint8_t cid = 0; cid < 84; ++cid) {
        const auto& c_info = MapData::get_country(cid);
        if (c_info.region != r) continue;

        Player ctrl = get_country_control(state, cid);
        bool is_bg = c_info.battleground || (cid == countries::TAIWAN && taiwan_is_bg);

        if (ctrl == Player::US) {
            summary.us_countries++;
            if (is_bg) summary.us_battlegrounds++;
            if (c_info.superpower_adjacent == Player::USSR) {
                summary.us_superpower_adjacent++;
            }
        } else if (ctrl == Player::USSR) {
            summary.ussr_countries++;
            if (is_bg) summary.ussr_battlegrounds++;
            if (c_info.superpower_adjacent == Player::US) {
                summary.ussr_superpower_adjacent++;
            }
        }
    }

    uint8_t effective_ussr_bg = summary.ussr_battlegrounds;
    if ((r == Region::MIDDLE_EAST || r == Region::ASIA) && state.has_flag(effect_bits::SHUTTLE_DIPLOMACY_ACTIVE)) {
        if (effective_ussr_bg > 0) effective_ussr_bg--;
    }

    uint8_t us_non_bg = (summary.us_countries >= summary.us_battlegrounds) ? (summary.us_countries - summary.us_battlegrounds) : 0;
    uint8_t ussr_non_bg = (summary.ussr_countries >= effective_ussr_bg) ? (summary.ussr_countries - effective_ussr_bg) : 0;

    // Evaluate US Status
    if (summary.us_countries > summary.ussr_countries && summary.us_battlegrounds == total_bg) {
        summary.us_status = RegionalStatus::CONTROL;
    } else if (summary.us_countries > summary.ussr_countries &&
               summary.us_battlegrounds > effective_ussr_bg &&
               summary.us_battlegrounds >= 1 &&
               us_non_bg >= 1) {
        summary.us_status = RegionalStatus::DOMINATION;
    } else if (summary.us_countries >= 1) {
        summary.us_status = RegionalStatus::PRESENCE;
    } else {
        summary.us_status = RegionalStatus::NONE;
    }

    // Evaluate USSR Status
    if (summary.ussr_countries > summary.us_countries && effective_ussr_bg == total_bg) {
        summary.ussr_status = RegionalStatus::CONTROL;
    } else if (summary.ussr_countries > summary.us_countries &&
               effective_ussr_bg > summary.us_battlegrounds &&
               effective_ussr_bg >= 1 &&
               ussr_non_bg >= 1) {
        summary.ussr_status = RegionalStatus::DOMINATION;
    } else if (summary.ussr_countries >= 1) {
        summary.ussr_status = RegionalStatus::PRESENCE;
    } else {
        summary.ussr_status = RegionalStatus::NONE;
    }

    // Base VP lookup per region
    int16_t presence_vp = 0;
    int16_t domination_vp = 0;
    int16_t control_vp = 0;

    switch (r) {
        case Region::EUROPE:
            presence_vp = 3; domination_vp = 7; control_vp = 0; // Control is instant victory!
            break;
        case Region::ASIA:
            presence_vp = 3; domination_vp = 7; control_vp = 9;
            break;
        case Region::MIDDLE_EAST:
            presence_vp = 3; domination_vp = 5; control_vp = 7;
            break;
        case Region::AFRICA:
            presence_vp = 1; domination_vp = 4; control_vp = 6;
            break;
        case Region::CENTRAL_AMERICA:
            presence_vp = 1; domination_vp = 3; control_vp = 5;
            break;
        case Region::SOUTH_AMERICA:
            presence_vp = 2; domination_vp = 5; control_vp = 6;
            break;
        default:
            break;
    }

    auto get_base_vp = [&](RegionalStatus st) -> int16_t {
        switch (st) {
            case RegionalStatus::PRESENCE: return presence_vp;
            case RegionalStatus::DOMINATION: return domination_vp;
            case RegionalStatus::CONTROL: return control_vp;
            default: return 0;
        }
    };

    summary.us_score = get_base_vp(summary.us_status) + summary.us_battlegrounds + summary.us_superpower_adjacent;
    summary.ussr_score = get_base_vp(summary.ussr_status) + effective_ussr_bg + summary.ussr_superpower_adjacent;
    summary.net_delta = summary.us_score - summary.ussr_score;

    return summary;
}

void Scoring::score_region(GameState& state, Region r) noexcept {
    if (r == Region::NONE_REGION) return;

    auto summary = evaluate_region(state, r);

    // Europe Control Instant Victory check
    if (r == Region::EUROPE) {
        if (summary.us_status == RegionalStatus::CONTROL) {
            state.victory_points = 20;
            state.current_phase = Phase::GAME_OVER;
            return;
        }
        if (summary.ussr_status == RegionalStatus::CONTROL) {
            state.victory_points = -20;
            state.current_phase = Phase::GAME_OVER;
            return;
        }
    }

    // If Shuttle Diplomacy was active and this was ME or Asia, clear flag
    if ((r == Region::MIDDLE_EAST || r == Region::ASIA) && state.has_flag(effect_bits::SHUTTLE_DIPLOMACY_ACTIVE)) {
        state.clear_flag(effect_bits::SHUTTLE_DIPLOMACY_ACTIVE);
    }

    int32_t new_vp = static_cast<int32_t>(state.victory_points) + summary.net_delta;
    new_vp = std::clamp(new_vp, -20, 20);
    state.victory_points = static_cast<int8_t>(new_vp);

    if (state.victory_points >= 20 || state.victory_points <= -20) {
        state.current_phase = Phase::GAME_OVER;
    }
}

void Scoring::score_southeast_asia(GameState& state) noexcept {
    // 1 VP each for Burma, Cambodia/Laos, Vietnam, Malaysia, Indonesia, Philippines. 2 VP for Thailand.
    int16_t us_pts = 0;
    int16_t ussr_pts = 0;

    constexpr std::array<uint8_t, 6> SE_COUNTRIES = {
        countries::BURMA, countries::LAOS_CAMBODIA, countries::VIETNAM,
        countries::MALAYSIA, countries::INDONESIA, countries::PHILIPPINES
    };

    for (uint8_t cid : SE_COUNTRIES) {
        Player ctrl = get_country_control(state, cid);
        if (ctrl == Player::US) us_pts += 1;
        else if (ctrl == Player::USSR) ussr_pts += 1;
    }

    Player thai_ctrl = get_country_control(state, countries::THAILAND);
    if (thai_ctrl == Player::US) us_pts += 2;
    else if (thai_ctrl == Player::USSR) ussr_pts += 2;

    int32_t new_vp = static_cast<int32_t>(state.victory_points) + (us_pts - ussr_pts);
    new_vp = std::clamp(new_vp, -20, 20);
    state.victory_points = static_cast<int8_t>(new_vp);

    if (state.victory_points >= 20 || state.victory_points <= -20) {
        state.current_phase = Phase::GAME_OVER;
    }
}

void Scoring::evaluate_military_ops(GameState& state) noexcept {
    uint8_t req = state.defcon;
    uint8_t us_def = (state.us_mil_ops >= req) ? 0 : (req - state.us_mil_ops);
    uint8_t ussr_def = (state.ussr_mil_ops >= req) ? 0 : (req - state.ussr_mil_ops);

    // US gains VP if USSR has deficit; USSR gains VP (negative VP delta) if US has deficit
    int16_t delta = static_cast<int16_t>(ussr_def) - static_cast<int16_t>(us_def);
    int32_t new_vp = static_cast<int32_t>(state.victory_points) + delta;
    new_vp = std::clamp(new_vp, -20, 20);
    state.victory_points = static_cast<int8_t>(new_vp);

    state.us_mil_ops = 0;
    state.ussr_mil_ops = 0;

    if (state.victory_points >= 20 || state.victory_points <= -20) {
        state.current_phase = Phase::GAME_OVER;
    }
}

void Scoring::execute_final_scoring(GameState& state) noexcept {
    if (state.current_phase == Phase::GAME_OVER) return;

    // Score all 6 regions in standard order
    score_region(state, Region::EUROPE);
    if (state.current_phase == Phase::GAME_OVER) return;

    score_region(state, Region::ASIA);
    if (state.current_phase == Phase::GAME_OVER) return;

    score_region(state, Region::MIDDLE_EAST);
    if (state.current_phase == Phase::GAME_OVER) return;

    score_region(state, Region::AFRICA);
    if (state.current_phase == Phase::GAME_OVER) return;

    score_region(state, Region::CENTRAL_AMERICA);
    if (state.current_phase == Phase::GAME_OVER) return;

    score_region(state, Region::SOUTH_AMERICA);
    if (state.current_phase == Phase::GAME_OVER) return;

    // China card bonus (+1 VP to holder)
    if (state.china_card_holder == Player::US) {
        state.victory_points = static_cast<int8_t>(std::min(20, state.victory_points + 1));
    } else if (state.china_card_holder == Player::USSR) {
        state.victory_points = static_cast<int8_t>(std::max(-20, state.victory_points - 1));
    }

    state.current_phase = Phase::GAME_OVER;
}

} // namespace ts
