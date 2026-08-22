#include "ts/observation.hpp"
#include "ts/map_data.hpp"
#include "ts/card_data.hpp"
#include "ts/scoring.hpp"
#include "ts/ops.hpp"
#include <cstring>
#include <algorithm>

namespace ts {

void Observation::extract(const GameState& state, Player perspective, ObservationBuffer* out_buf) noexcept {
    if (!out_buf) return;
    std::memset(out_buf, 0, sizeof(ObservationBuffer));

    float sign = (perspective == Player::US) ? 1.0f : -1.0f;

    // 1. Board features (84 * 28)
    for (uint8_t i = 0; i < 84; ++i) {
        const auto& c_info = MapData::get_country(i);
        size_t offset = i * 28;

        float us_inf = static_cast<float>(state.countries[i].us_influence);
        float ussr_inf = static_cast<float>(state.countries[i].ussr_influence);

        out_buf->board_features[offset + 0] = us_inf / 10.0f;
        out_buf->board_features[offset + 1] = ussr_inf / 10.0f;
        out_buf->board_features[offset + 2] = (us_inf - ussr_inf) / 10.0f;
        out_buf->board_features[offset + 3] = static_cast<float>(c_info.stability) / 5.0f;
        out_buf->board_features[offset + 4] = c_info.battleground ? 1.0f : 0.0f;

        Player ctrl = Scoring::get_country_control(state, i);
        out_buf->board_features[offset + 5] = (ctrl == Player::US) ? 1.0f : 0.0f;
        out_buf->board_features[offset + 6] = (ctrl == Player::USSR) ? 1.0f : 0.0f;
        out_buf->board_features[offset + 7] = (ctrl == Player::NONE) ? 1.0f : 0.0f;

        out_buf->board_features[offset + 8] = (c_info.superpower_adjacent == Player::US) ? 1.0f : 0.0f;
        out_buf->board_features[offset + 9] = (c_info.superpower_adjacent == Player::USSR) ? 1.0f : 0.0f;

        size_t r_idx = static_cast<size_t>(c_info.region);
        if (r_idx < 6) {
            out_buf->board_features[offset + 10 + r_idx] = 1.0f;
        }

        out_buf->board_features[offset + 16] = c_info.in_western_europe ? 1.0f : 0.0f;
        out_buf->board_features[offset + 17] = c_info.in_eastern_europe ? 1.0f : 0.0f;
        out_buf->board_features[offset + 18] = c_info.in_southeast_asia ? 1.0f : 0.0f;

        out_buf->board_features[offset + 19] = Operations::can_place_influence(state, Player::US, i) ? 1.0f : 0.0f;
        out_buf->board_features[offset + 20] = Operations::can_place_influence(state, Player::USSR, i) ? 1.0f : 0.0f;
        out_buf->board_features[offset + 21] = Operations::can_coup(state, Player::US, i) ? 1.0f : 0.0f;
        out_buf->board_features[offset + 22] = Operations::can_coup(state, Player::USSR, i) ? 1.0f : 0.0f;
        out_buf->board_features[offset + 23] = Operations::can_realign(state, Player::US, i) ? 1.0f : 0.0f;
        out_buf->board_features[offset + 24] = Operations::can_realign(state, Player::USSR, i) ? 1.0f : 0.0f;

        out_buf->board_features[offset + 25] = state.ctx().is_visited(i) ? 1.0f : 0.0f;
        out_buf->board_features[offset + 26] = static_cast<float>(state.ctx().node_counts[i]) / 5.0f;
        out_buf->board_features[offset + 27] = (us_inf >= c_info.stability) ? 1.0f : 0.0f;
    }

    // 2. Card features (110 * 12)
    for (uint8_t i = 1; i <= 110; ++i) {
        const auto& c_info = CardData::get_card(i);
        size_t offset = (i - 1) * 12;

        CardLocation loc = state.card_locations[i];
        size_t loc_idx = static_cast<size_t>(loc);
        if (loc_idx < 7) {
            out_buf->card_features[offset + loc_idx] = 1.0f;
        }

        out_buf->card_features[offset + 7] = static_cast<float>(c_info.ops) / 4.0f;
        out_buf->card_features[offset + 8] = (c_info.side == Player::US) ? 1.0f : ((c_info.side == Player::USSR) ? -1.0f : 0.0f);
        out_buf->card_features[offset + 9] = static_cast<float>(c_info.era) / 2.0f;
        out_buf->card_features[offset + 10] = c_info.one_time ? 1.0f : 0.0f;
        out_buf->card_features[offset + 11] = c_info.is_scoring ? 1.0f : 0.0f;
    }

    // 3. Global features (76)
    out_buf->global_features[0] = static_cast<float>(state.victory_points) / 20.0f;
    out_buf->global_features[1] = static_cast<float>(state.defcon) / 5.0f;
    out_buf->global_features[2] = static_cast<float>(state.us_mil_ops) / 5.0f;
    out_buf->global_features[3] = static_cast<float>(state.ussr_mil_ops) / 5.0f;
    out_buf->global_features[4] = static_cast<float>(state.us_space_track) / 8.0f;
    out_buf->global_features[5] = static_cast<float>(state.ussr_space_track) / 8.0f;
    out_buf->global_features[6] = static_cast<float>(state.turn) / 10.0f;
    out_buf->global_features[7] = static_cast<float>(state.action_round) / 8.0f;
    out_buf->global_features[8] = (state.phasing_player == Player::US) ? 1.0f : -1.0f;
    out_buf->global_features[9] = static_cast<float>(state.current_phase) / 6.0f;
    out_buf->global_features[10] = (state.china_card_holder == Player::US) ? 1.0f : -1.0f;
    out_buf->global_features[11] = state.china_card_playable ? 1.0f : 0.0f;

    for (size_t b = 0; b < 47; ++b) {
        out_buf->global_features[12 + b] = ((state.persistent_effects & (1ULL << b)) != 0) ? 1.0f : 0.0f;
    }

    out_buf->global_features[59] = static_cast<float>(state.get_space_turns_used(Player::US)) / 2.0f;
    out_buf->global_features[60] = static_cast<float>(state.get_space_turns_used(Player::USSR)) / 2.0f;
    out_buf->global_features[57] = state.defcon_dropped_to_2_in_ar ? 1.0f : 0.0f;
    out_buf->global_features[58] = static_cast<float>(state.ctx_stack_depth) / 3.0f;

    // 4. Turn aggregates (32)
    for (size_t p = 0; p < 2; ++p) {
        for (size_t r = 0; r < 6; ++r) {
            out_buf->turn_aggregates[p * 6 + r] = static_cast<float>(state.turn_aggregates.ops_spent_by_region[p][r]) / 10.0f;
            out_buf->turn_aggregates[12 + p * 6 + r] = static_cast<float>(state.turn_aggregates.coups_by_region[p][r]) / 5.0f;
        }
        out_buf->turn_aggregates[24 + p] = static_cast<float>(state.turn_aggregates.headlines_played[p]);
        out_buf->turn_aggregates[26 + p] = static_cast<float>(state.turn_aggregates.space_attempts[p]) / 2.0f;
    }

    // 5. History sequence (16 * 32)
    for (size_t h = 0; h < 16; ++h) {
        const auto& tok = state.action_history.ring_buffer[h];
        size_t h_off = h * 32;
        out_buf->history_sequence[h_off + 0] = (tok.acting_player == Player::US) ? 1.0f : ((tok.acting_player == Player::USSR) ? -1.0f : 0.0f);
        out_buf->history_sequence[h_off + 1] = static_cast<float>(tok.action_type) / 7.0f;
        out_buf->history_sequence[h_off + 2] = static_cast<float>(tok.card_id) / 110.0f;
        out_buf->history_sequence[h_off + 3] = static_cast<float>(tok.target_id) / 84.0f;
        out_buf->history_sequence[h_off + 4] = static_cast<float>(tok.ops_value) / 5.0f;
        out_buf->history_sequence[h_off + 5] = static_cast<float>(tok.die_roll) / 6.0f;
        out_buf->history_sequence[h_off + 6] = static_cast<float>(tok.us_inf_delta) / 5.0f;
        out_buf->history_sequence[h_off + 7] = static_cast<float>(tok.ussr_inf_delta) / 5.0f;
        out_buf->history_sequence[h_off + 8] = static_cast<float>(tok.defcon_after) / 5.0f;
    }

    out_buf->active_player = sign;
}

void extract_observation(const GameState& state, Player perspective, ObservationBuffer* out_buf) noexcept {
    Observation::extract(state, perspective, out_buf);
}

} // namespace ts
