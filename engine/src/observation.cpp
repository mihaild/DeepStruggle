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

    Player my_player = perspective;
    if (my_player == Player::NONE) {
        my_player = (state.ctx().decision_player != Player::NONE) 
            ? state.ctx().decision_player : state.phasing_player;
        if (my_player == Player::NONE) my_player = Player::US;
    }
    Player opp_player = (my_player == Player::US) ? Player::USSR : Player::US;
    float side_sign = (my_player == Player::US) ? 1.0f : -1.0f;

    // 1. Board features (84 * 28) - Canonical (Myself vs Opponent)
    for (uint8_t i = 0; i < 84; ++i) {
        const auto& c_info = MapData::get_country(i);
        size_t offset = i * 28;

        float my_inf = static_cast<float>(my_player == Player::US ? state.countries[i].us_influence : state.countries[i].ussr_influence);
        float opp_inf = static_cast<float>(my_player == Player::US ? state.countries[i].ussr_influence : state.countries[i].us_influence);

        out_buf->board_features[offset + 0] = my_inf / 10.0f;
        out_buf->board_features[offset + 1] = opp_inf / 10.0f;
        out_buf->board_features[offset + 2] = (my_inf - opp_inf) / 10.0f; // Net margin from MY perspective
        out_buf->board_features[offset + 3] = static_cast<float>(c_info.stability) / 5.0f;
        out_buf->board_features[offset + 4] = c_info.battleground ? 1.0f : 0.0f;

        Player ctrl = Scoring::get_country_control(state, i);
        out_buf->board_features[offset + 5] = (ctrl == my_player) ? 1.0f : 0.0f;  // MY Control
        out_buf->board_features[offset + 6] = (ctrl == opp_player) ? 1.0f : 0.0f; // OPPONENT Control
        out_buf->board_features[offset + 7] = (ctrl == Player::NONE) ? 1.0f : 0.0f;

        out_buf->board_features[offset + 8] = (c_info.superpower_adjacent == my_player) ? 1.0f : 0.0f;  // Adjacent to MY Superpower
        out_buf->board_features[offset + 9] = (c_info.superpower_adjacent == opp_player) ? 1.0f : 0.0f; // Adjacent to OPP Superpower

        size_t r_idx = static_cast<size_t>(c_info.region);
        if (r_idx < 6) {
            out_buf->board_features[offset + 10 + r_idx] = 1.0f;
        }

        out_buf->board_features[offset + 16] = c_info.in_western_europe ? 1.0f : 0.0f;
        out_buf->board_features[offset + 17] = c_info.in_eastern_europe ? 1.0f : 0.0f;
        out_buf->board_features[offset + 18] = c_info.in_southeast_asia ? 1.0f : 0.0f;

        out_buf->board_features[offset + 19] = Operations::can_place_influence(state, my_player, i) ? 1.0f : 0.0f;
        out_buf->board_features[offset + 20] = Operations::can_place_influence(state, opp_player, i) ? 1.0f : 0.0f;
        out_buf->board_features[offset + 21] = Operations::can_coup(state, my_player, i) ? 1.0f : 0.0f;
        out_buf->board_features[offset + 22] = Operations::can_coup(state, opp_player, i) ? 1.0f : 0.0f;
        out_buf->board_features[offset + 23] = Operations::can_realign(state, my_player, i) ? 1.0f : 0.0f;
        out_buf->board_features[offset + 24] = Operations::can_realign(state, opp_player, i) ? 1.0f : 0.0f;

        out_buf->board_features[offset + 25] = state.ctx().is_visited(i) ? 1.0f : 0.0f;
        out_buf->board_features[offset + 26] = static_cast<float>(state.ctx().node_counts[i]) / 5.0f;
        out_buf->board_features[offset + 27] = (my_inf >= c_info.stability) ? 1.0f : 0.0f;
    }

    // 2. Card features (110 * 12) - Canonical (Myself vs Opponent)
    for (uint8_t i = 1; i <= 110; ++i) {
        const auto& c_info = CardData::get_card(i);
        size_t offset = (i - 1) * 12;

        CardLocation loc = state.card_locations[i];
        size_t canon_loc = 0;
        if (loc == CardLocation::DRAW_DECK || loc == CardLocation::UNAVAILABLE) {
            canon_loc = 0;
        } else if ((loc == CardLocation::HAND_US && my_player == Player::US) ||
                   (loc == CardLocation::HAND_USSR && my_player == Player::USSR)) {
            canon_loc = 1; // MY_HAND
        } else if ((loc == CardLocation::HAND_US && my_player == Player::USSR) ||
                   (loc == CardLocation::HAND_USSR && my_player == Player::US)) {
            canon_loc = 2; // OPPONENT_HAND
        } else if (loc == CardLocation::DISCARD_PILE) {
            canon_loc = 3;
        } else if (loc == CardLocation::REMOVED_FROM_GAME) {
            canon_loc = 4;
        } else if (loc == CardLocation::ONGOING_EVENT) {
            canon_loc = 5;
        } else if (loc == CardLocation::PEEKED_TEMP) {
            canon_loc = 6;
        }

        if (canon_loc < 7) {
            out_buf->card_features[offset + canon_loc] = 1.0f;
        }

        out_buf->card_features[offset + 7] = static_cast<float>(c_info.ops) / 4.0f;
        // Card side relative to myself: +1.0 = FRIENDLY, -1.0 = HOSTILE, 0.0 = NEUTRAL
        float rel_side = (c_info.side == my_player) ? 1.0f : ((c_info.side == opp_player) ? -1.0f : 0.0f);
        out_buf->card_features[offset + 8] = rel_side;
        out_buf->card_features[offset + 9] = static_cast<float>(c_info.era) / 2.0f;
        out_buf->card_features[offset + 10] = c_info.one_time ? 1.0f : 0.0f;
        out_buf->card_features[offset + 11] = c_info.is_scoring ? 1.0f : 0.0f;
    }

    // 3. Global features (76) - Canonical (Myself vs Opponent)
    float my_vp = (my_player == Player::US) ? static_cast<float>(state.victory_points) : -static_cast<float>(state.victory_points);
    out_buf->global_features[0] = my_vp / 20.0f; // +1.0 = I am at +20 VP, -1.0 = I am at -20 VP
    out_buf->global_features[1] = static_cast<float>(state.defcon) / 5.0f;
    
    float my_mil_ops = static_cast<float>(my_player == Player::US ? state.us_mil_ops : state.ussr_mil_ops);
    float opp_mil_ops = static_cast<float>(my_player == Player::US ? state.ussr_mil_ops : state.us_mil_ops);
    out_buf->global_features[2] = my_mil_ops / 5.0f;
    out_buf->global_features[3] = opp_mil_ops / 5.0f;

    float my_space = static_cast<float>(my_player == Player::US ? state.us_space_track : state.ussr_space_track);
    float opp_space = static_cast<float>(my_player == Player::US ? state.ussr_space_track : state.us_space_track);
    out_buf->global_features[4] = my_space / 8.0f;
    out_buf->global_features[5] = opp_space / 8.0f;

    out_buf->global_features[6] = static_cast<float>(state.turn) / 10.0f;
    out_buf->global_features[7] = static_cast<float>(state.action_round) / 8.0f;
    out_buf->global_features[8] = (state.phasing_player == my_player) ? 1.0f : -1.0f;
    out_buf->global_features[9] = static_cast<float>(state.current_phase) / 6.0f;
    out_buf->global_features[10] = (state.china_card_holder == my_player) ? 1.0f : -1.0f;
    out_buf->global_features[11] = state.china_card_playable ? 1.0f : 0.0f;

    for (size_t b = 0; b < 47; ++b) {
        out_buf->global_features[12 + b] = ((state.persistent_effects & (1ULL << b)) != 0) ? 1.0f : 0.0f;
    }

    out_buf->global_features[57] = state.defcon_dropped_to_2_in_ar ? 1.0f : 0.0f;
    out_buf->global_features[58] = static_cast<float>(state.ctx_stack_depth) / 3.0f;
    out_buf->global_features[59] = static_cast<float>(state.get_space_turns_used(my_player)) / 2.0f;
    out_buf->global_features[60] = static_cast<float>(state.get_space_turns_used(opp_player)) / 2.0f;

    // Explicit Side Identity Flags (Model knows exactly which side it is playing)
    out_buf->global_features[61] = (my_player == Player::US) ? 1.0f : 0.0f;   // I_AM_US
    out_buf->global_features[62] = (my_player == Player::USSR) ? 1.0f : 0.0f; // I_AM_USSR
    out_buf->global_features[63] = side_sign;                                 // SIDE_SIGN (+1.0 US, -1.0 USSR)

    // 4. Turn aggregates (32) - Canonical (Myself vs Opponent)
    size_t my_p_idx = (my_player == Player::US) ? 0 : 1;
    size_t opp_p_idx = (my_player == Player::US) ? 1 : 0;

    for (size_t r = 0; r < 6; ++r) {
        out_buf->turn_aggregates[0 + r] = static_cast<float>(state.turn_aggregates.ops_spent_by_region[my_p_idx][r]) / 10.0f;
        out_buf->turn_aggregates[6 + r] = static_cast<float>(state.turn_aggregates.ops_spent_by_region[opp_p_idx][r]) / 10.0f;
        out_buf->turn_aggregates[12 + r] = static_cast<float>(state.turn_aggregates.coups_by_region[my_p_idx][r]) / 5.0f;
        out_buf->turn_aggregates[18 + r] = static_cast<float>(state.turn_aggregates.coups_by_region[opp_p_idx][r]) / 5.0f;
    }
    out_buf->turn_aggregates[24] = static_cast<float>(state.turn_aggregates.headlines_played[my_p_idx]);
    out_buf->turn_aggregates[25] = static_cast<float>(state.turn_aggregates.headlines_played[opp_p_idx]);
    out_buf->turn_aggregates[26] = static_cast<float>(state.turn_aggregates.space_attempts[my_p_idx]) / 2.0f;
    out_buf->turn_aggregates[27] = static_cast<float>(state.turn_aggregates.space_attempts[opp_p_idx]) / 2.0f;

    // 5. History sequence (16 * 32) - Canonical (Myself vs Opponent)
    for (size_t h = 0; h < 16; ++h) {
        const auto& tok = state.action_history.ring_buffer[h];
        size_t h_off = h * 32;
        float tok_acting = (tok.acting_player == my_player) ? 1.0f : ((tok.acting_player == opp_player) ? -1.0f : 0.0f);
        out_buf->history_sequence[h_off + 0] = tok_acting;
        out_buf->history_sequence[h_off + 1] = static_cast<float>(tok.action_type) / 7.0f;
        out_buf->history_sequence[h_off + 2] = static_cast<float>(tok.card_id) / 110.0f;
        out_buf->history_sequence[h_off + 3] = static_cast<float>(tok.target_id) / 84.0f;
        out_buf->history_sequence[h_off + 4] = static_cast<float>(tok.ops_value) / 5.0f;
        out_buf->history_sequence[h_off + 5] = static_cast<float>(tok.die_roll) / 6.0f;

        float my_delta = static_cast<float>(my_player == Player::US ? tok.us_inf_delta : tok.ussr_inf_delta);
        float opp_delta = static_cast<float>(my_player == Player::US ? tok.ussr_inf_delta : tok.us_inf_delta);
        out_buf->history_sequence[h_off + 6] = my_delta / 5.0f;
        out_buf->history_sequence[h_off + 7] = opp_delta / 5.0f;
        out_buf->history_sequence[h_off + 8] = static_cast<float>(tok.defcon_after) / 5.0f;
    }

    out_buf->active_player = side_sign;
}

void extract_observation(const GameState& state, Player perspective, ObservationBuffer* out_buf) noexcept {
    Observation::extract(state, perspective, out_buf);
}

} // namespace ts
