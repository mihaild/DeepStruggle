#include "ts/observation.hpp"
#include "ts/map_data.hpp"
#include "ts/card_data.hpp"
#include "ts/scoring.hpp"
#include "ts/ops.hpp"
#include <cstring>
#include <algorithm>

namespace ts {

namespace {

int16_t compute_net_realign_mod(const GameState& state, Player p, uint8_t country_id) noexcept {
    if (country_id >= 84 || p == Player::NONE) return 0;
    const auto& c_info = MapData::get_country(country_id);
    Player opp = (p == Player::US) ? Player::USSR : Player::US;

    int16_t my_mod = 0;
    int16_t opp_mod = 0;

    for (uint8_t n = 0; n < c_info.num_neighbors; ++n) {
        uint8_t n_id = c_info.neighbors[n];
        if (Scoring::is_controlled_by(state, n_id, p)) my_mod++;
        if (Scoring::is_controlled_by(state, n_id, opp)) opp_mod++;
    }

    uint8_t my_inf = state.countries[country_id].get_influence(p);
    uint8_t opp_inf = state.countries[country_id].get_influence(opp);
    if (my_inf > opp_inf) my_mod++;
    else if (opp_inf > my_inf) opp_mod++;

    if (c_info.superpower_adjacent == p) my_mod++;
    if (c_info.superpower_adjacent == opp) opp_mod++;

    if (p == Player::US && state.has_flag(effect_bits::IRAN_CONTRA_ACTIVE)) my_mod -= 1;
    if (opp == Player::US && state.has_flag(effect_bits::IRAN_CONTRA_ACTIVE)) opp_mod -= 1;

    return my_mod - opp_mod;
}

bool is_coup_nuclear_hazard(const GameState& state, Player p, uint8_t country_id) noexcept {
    if (state.defcon > 2 || country_id >= 84) return false;
    const auto& c_info = MapData::get_country(country_id);
    if (!c_info.battleground) return false;
    
    // In TS, couping a battleground reduces DEFCON by 1.
    // If DEFCON is 2 and country is legal to coup, couping causes DEFCON 1 Nuclear Loss!
    return Operations::can_coup(state, p, country_id);
}

} // anonymous namespace

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
        
        // 2. Net Realignment Modifier (Dice advantage normalized by 5.0)
        int16_t net_realign = compute_net_realign_mod(state, my_player, i);
        out_buf->board_features[offset + 2] = std::clamp(static_cast<float>(net_realign) / 5.0f, -1.0f, 1.0f);
        
        out_buf->board_features[offset + 3] = static_cast<float>(c_info.stability) / 5.0f;
        out_buf->board_features[offset + 4] = c_info.battleground ? 1.0f : 0.0f;

        Player ctrl = Scoring::get_country_control(state, i);
        out_buf->board_features[offset + 5] = (ctrl == my_player) ? 1.0f : 0.0f;  // MY Control
        out_buf->board_features[offset + 6] = (ctrl == opp_player) ? 1.0f : 0.0f; // OPPONENT Control
        
        // 7. Coup Nuclear Hazard: 1.0 if couping here causes DEFCON 1 Nuclear Loss
        out_buf->board_features[offset + 7] = is_coup_nuclear_hazard(state, my_player, i) ? 1.0f : 0.0f;

        out_buf->board_features[offset + 8] = (c_info.superpower_adjacent == my_player) ? 1.0f : 0.0f;  // Adjacent to MY Superpower
        out_buf->board_features[offset + 9] = (c_info.superpower_adjacent == opp_player) ? 1.0f : 0.0f; // Adjacent to OPP Superpower

        size_t r_idx = static_cast<size_t>(c_info.region);
        if (r_idx < 6) {
            out_buf->board_features[offset + 10 + r_idx] = 1.0f;
        }

        out_buf->board_features[offset + 16] = c_info.in_western_europe ? 1.0f : 0.0f;
        out_buf->board_features[offset + 17] = c_info.in_eastern_europe ? 1.0f : 0.0f;
        out_buf->board_features[offset + 18] = c_info.in_southeast_asia ? 1.0f : 0.0f;

        bool can_my_place = false;
        bool can_opp_place = false;
        if (state.current_phase == Phase::SETUP) {
            auto check_setup_place = [&](Player pl) -> bool {
                if (pl == Player::USSR) return c_info.in_eastern_europe;
                if (state.ctx().pending_ops_value == 0) return c_info.in_western_europe;
                return state.countries[i].us_influence > 0;
            };
            can_my_place = check_setup_place(my_player);
            can_opp_place = check_setup_place(opp_player);
        } else {
            can_my_place = Operations::can_place_influence(state, my_player, i);
            can_opp_place = Operations::can_place_influence(state, opp_player, i);
        }
        out_buf->board_features[offset + 19] = can_my_place ? 1.0f : 0.0f;
        out_buf->board_features[offset + 20] = can_opp_place ? 1.0f : 0.0f;

        bool can_my_coup = (state.current_phase != Phase::SETUP) && Operations::can_coup(state, my_player, i);
        bool can_opp_coup = (state.current_phase != Phase::SETUP) && Operations::can_coup(state, opp_player, i);
        out_buf->board_features[offset + 21] = can_my_coup ? 1.0f : 0.0f;
        out_buf->board_features[offset + 22] = can_opp_coup ? 1.0f : 0.0f;

        bool can_my_realign = (state.current_phase != Phase::SETUP) && Operations::can_realign(state, my_player, i);
        bool can_opp_realign = (state.current_phase != Phase::SETUP) && Operations::can_realign(state, opp_player, i);
        out_buf->board_features[offset + 23] = can_my_realign ? 1.0f : 0.0f;
        out_buf->board_features[offset + 24] = can_opp_realign ? 1.0f : 0.0f;

        out_buf->board_features[offset + 25] = static_cast<float>(state.ctx().node_counts[i]) / 5.0f;
        
        // 26. Influence Deficit to My Control: How many Ops needed to achieve control
        int my_def_stab = std::max(0, static_cast<int>(c_info.stability) - static_cast<int>(my_inf));
        int my_def_margin = std::max(0, static_cast<int>(opp_inf) + static_cast<int>(c_info.stability) - static_cast<int>(my_inf));
        int my_deficit = std::max(my_def_stab, my_def_margin);
        out_buf->board_features[offset + 26] = std::min(static_cast<float>(my_deficit) / 5.0f, 2.0f);
        
        // 27. Influence Deficit to Opponent Control
        int opp_def_stab = std::max(0, static_cast<int>(c_info.stability) - static_cast<int>(opp_inf));
        int opp_def_margin = std::max(0, static_cast<int>(my_inf) + static_cast<int>(c_info.stability) - static_cast<int>(opp_inf));
        int opp_deficit = std::max(opp_def_stab, opp_def_margin);
        out_buf->board_features[offset + 27] = std::min(static_cast<float>(opp_deficit) / 5.0f, 2.0f);
    }

    // 2. Card features (110 * 12) - Canonical (Myself vs Opponent)
    uint8_t draw_pile_cnt = 0;
    uint8_t discard_pile_cnt = 0;
    uint8_t my_hand_cnt = 0;
    uint8_t opp_hand_cnt = 0;

    for (uint8_t i = 1; i <= 110; ++i) {
        const auto& c_info = CardData::get_card(i);
        size_t offset = (i - 1) * 12;

        CardLocation loc = state.card_locations[i];
        size_t canon_loc = 0;
        if (loc == CardLocation::DRAW_DECK || loc == CardLocation::UNAVAILABLE) {
            canon_loc = 0;
            if (loc == CardLocation::DRAW_DECK) draw_pile_cnt++;
        } else if (in_hand_of(loc, my_player)) {
            canon_loc = 1; // MY_HAND -- both hand variants; I always know my own hand
            my_hand_cnt++;
        } else if (in_hand_of(loc, opp_player)) {
            canon_loc = 0; // OPPONENT_HAND is hidden: fold into slot 0 (UNKNOWN/UNAVAILABLE).
            // The legacy layout predates knowledge tracking and folds the known variant here
            // too, on purpose: this function must reproduce the old observation exactly.
            opp_hand_cnt++;
        } else if (loc == CardLocation::DISCARD_PILE) {
            canon_loc = 3;
            discard_pile_cnt++;
        } else if (loc == CardLocation::REMOVED_FROM_GAME) {
            canon_loc = 4;
        } else if (loc == CardLocation::ONGOING_EVENT) {
            canon_loc = 5;
        } else if (loc == CardLocation::HEADLINE_COMMITTED) {
            // Face down and known only to its owner, but it has left the hand; treated as
            // out of play for the hand features, which is what the readers of those care
            // about. It is public again the moment it resolves.
        } else if (loc == CardLocation::PEEKED_TEMP) {
            canon_loc = 6;
        }

        if (canon_loc < 7) {
            out_buf->card_features[offset + canon_loc] = 1.0f;
        }

        out_buf->card_features[offset + 7] = static_cast<float>(c_info.ops) / 4.0f;
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

    // Bits 0..44 only. The span is global_features[12..56]: 57 and 58 are defcon_dropped_to_2
    // and ctx_stack_depth, assigned just below, so a loop to 47 wrote bits 45 and 46
    // (SPACE_USSR_ATTEMPT_1/2) and had them overwritten on the next two lines. Their information
    // reaches the model through global_features[59]/[60] instead. Anything added at bit >= 45
    // needs its own feature; it will not appear here.
    for (size_t b = 0; b < 45; ++b) {
        out_buf->global_features[12 + b] = ((state.persistent_effects & (1ULL << b)) != 0) ? 1.0f : 0.0f;
    }

    out_buf->global_features[57] = state.defcon_dropped_to_2 ? 1.0f : 0.0f;
    out_buf->global_features[58] = static_cast<float>(state.ctx_stack_depth) / 3.0f;
    out_buf->global_features[59] = static_cast<float>(state.get_space_turns_used(my_player)) / 2.0f;
    out_buf->global_features[60] = static_cast<float>(state.get_space_turns_used(opp_player)) / 2.0f;

    // Side Identity Flag
    out_buf->global_features[61] = (my_player == Player::US) ? 1.0f : 0.0f; // I_AM_US
    
    // Deck Tracking Features
    out_buf->global_features[62] = static_cast<float>(draw_pile_cnt) / 100.0f;
    out_buf->global_features[63] = static_cast<float>(discard_pile_cnt) / 100.0f;

    // Real-Time Regional Scoring VP Differentials for the 6 Regions
    for (size_t r = 0; r < 6; ++r) {
        auto summary = Scoring::evaluate_region(state, static_cast<Region>(r));
        float my_region_vp = (my_player == Player::US) ? static_cast<float>(summary.net_delta) : -static_cast<float>(summary.net_delta);
        out_buf->global_features[64 + r] = std::clamp(my_region_vp / 20.0f, -1.0f, 1.0f);
    }

    out_buf->global_features[70] = static_cast<float>(opp_hand_cnt) / 10.0f;
    out_buf->global_features[71] = static_cast<float>(my_hand_cnt) / 10.0f;

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
        size_t ring_idx = (state.action_history.head_idx + h) & 15;
        const auto& tok = state.action_history.ring_buffer[ring_idx];
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

void Observation::extract_v21(const GameState& state, Player perspective,
                             ObservationBufferV21* out_buf) noexcept {
    if (!out_buf) return;

    // Everything except the card block is copied from the legacy extraction rather than
    // recomputed, so the two layouts cannot drift apart in the sections that are supposed to
    // be the same. Only the card features below are v2.1's own work.
    ObservationBuffer legacy;
    Observation::extract(state, perspective, &legacy);

    std::memset(out_buf, 0, sizeof(ObservationBufferV21));
    std::memcpy(out_buf->board_features, legacy.board_features, sizeof(legacy.board_features));
    std::memcpy(out_buf->global_features, legacy.global_features, sizeof(legacy.global_features));
    // history_sequence is deliberately not carried over; see ObservationBufferV21.
    std::memcpy(out_buf->turn_aggregates, legacy.turn_aggregates, sizeof(legacy.turn_aggregates));
    out_buf->active_player = legacy.active_player;

    Player my_player = perspective;
    if (my_player == Player::NONE) {
        my_player = (state.ctx().decision_player != Player::NONE)
            ? state.ctx().decision_player : state.phasing_player;
        if (my_player == Player::NONE) my_player = Player::US;
    }
    const Player opp_player = (my_player == Player::US) ? Player::USSR : Player::US;

    for (uint8_t i = 1; i <= 110; ++i) {
        const auto& c_info = CardData::get_card(i);
        const size_t offset = static_cast<size_t>(i - 1) * card_slots::V21_FEATURES;
        const CardLocation loc = state.card_locations[i];

        // Defaults to DECK_OR_HIDDEN, which is what the legacy layout does with anything its
        // chain does not claim -- in practice HEADLINE_COMMITTED, a card face down in the
        // headline. v2.1's scope is splitting the known-opponent-hand and not-in-game cases out
        // of slot 0; how a headline card is represented is a separate question and is left
        // exactly as it was, so the two layouts cannot disagree about it.
        size_t slot = card_slots::DECK_OR_HIDDEN;
        if (loc == CardLocation::UNAVAILABLE) {
            slot = card_slots::NOT_IN_GAME;
        } else if (loc == CardLocation::DRAW_DECK) {
            slot = card_slots::DECK_OR_HIDDEN;
        } else if (in_hand_of(loc, my_player)) {
            slot = card_slots::MY_HAND;
        } else if (in_hand_of(loc, opp_player)) {
            // The whole point of the split, and the whole point of *not* splitting further:
            // a card the opponent holds that I have seen is public knowledge and gets its own
            // slot, and one I have not seen goes back in with the draw deck, because from here
            // those two are the same thing. Anything else would leak the hand.
            slot = known_to_opponent(loc) ? card_slots::KNOWN_OPPONENT_HAND
                                          : card_slots::DECK_OR_HIDDEN;
        } else if (loc == CardLocation::DISCARD_PILE) {
            slot = card_slots::DISCARD;
        } else if (loc == CardLocation::REMOVED_FROM_GAME) {
            slot = card_slots::REMOVED;
        } else if (loc == CardLocation::ONGOING_EVENT) {
            slot = card_slots::ONGOING;
        } else if (loc == CardLocation::PEEKED_TEMP) {
            slot = card_slots::PEEKED;
        }

        // The China Card is not in card_locations and must not be. It is fixed at ONGOING_EVENT
        // for the whole game because the cards that scan a hand -- Grain Sales To Soviets, Five
        // Year Plan, Missile Envy, The Cambridge Five, Terrorism -- test membership through
        // card_locations, and a hand variant there would let it be stolen, discarded or forced,
        // which the rules forbid. So the observation reads china_card_holder directly, which is
        // public knowledge: both players always know who holds it.
        //
        // Without this the card block showed it as an ongoing event in every position, to holder
        // and opponent alike, and the card branch never saw the one card that is always safe to
        // play -- 4 Ops, no event of its own, so it can never fire an opponent event and can never
        // move DEFCON. Whether it is playable *this* turn stays in global_features[11]; being face
        // down does not take it out of the hand.
        if (i == card_ids::THE_CHINA_CARD) {
            slot = (state.china_card_holder == my_player) ? card_slots::MY_HAND
                 : (state.china_card_holder == opp_player) ? card_slots::KNOWN_OPPONENT_HAND
                 : card_slots::DECK_OR_HIDDEN;
        }

        out_buf->card_features[offset + slot] = 1.0f;

        const size_t base = card_slots::V21_PROPERTY_BASE;
        out_buf->card_features[offset + base + 0] = static_cast<float>(c_info.ops) / 4.0f;
        const float rel_side = (c_info.side == my_player) ? 1.0f
                             : ((c_info.side == opp_player) ? -1.0f : 0.0f);
        out_buf->card_features[offset + base + 1] = rel_side;
        out_buf->card_features[offset + base + 2] = static_cast<float>(c_info.era) / 2.0f;
        out_buf->card_features[offset + base + 3] = c_info.one_time ? 1.0f : 0.0f;
        out_buf->card_features[offset + base + 4] = c_info.is_scoring ? 1.0f : 0.0f;
    }
}

void Observation::extract_v22(const GameState& state, Player perspective,
                              ObservationBufferV22* out_buf) noexcept {
    // Built on v2.1 for the same reason v2.1 is built on legacy: the sections that are supposed
    // to be identical are identical by construction, not by inspection. Only the card block's
    // extra feature and the global block's extra twenty are v2.2's own work.
    ObservationBufferV21 v21;
    Observation::extract_v21(state, perspective, &v21);

    std::memset(out_buf, 0, sizeof(ObservationBufferV22));

    // The board is restrided rather than copied: v2.2 drops features 23 and 24 (can_my_realign,
    // can_opp_realign) out of the middle of each country's row, so what follows them shifts down
    // by two. Copied in two pieces per country so the surviving columns keep their meanings.
    for (size_t c = 0; c < 84; ++c) {
        const float* src = &v21.board_features[c * 28];
        float* dst = &out_buf->board_features[c * V22_BOARD_FEATURES];
        std::memcpy(dst, src, 23 * sizeof(float));            // 0..22 unchanged
        std::memcpy(dst + 23, src + 25, 3 * sizeof(float));   // 25..27 -> 23..25
    }
    std::memcpy(out_buf->global_features, v21.global_features, sizeof(v21.global_features));
    // turn_aggregates and active_player are deliberately not carried over; see
    // ObservationBufferV22.

    // Card block: v2.1's 13 features per card, restrided to 14, plus the active-card marker.
    const auto& ctx = state.ctx();
    for (size_t i = 0; i < 110; ++i) {
        std::memcpy(&out_buf->card_features[i * card_slots::V22_FEATURES],
                    &v21.card_features[i * card_slots::V21_FEATURES],
                    card_slots::V21_FEATURES * sizeof(float));

        // The card this decision belongs to. resolving_card is the card whose event is
        // executing; pending_op_card is the one whose Ops are being spent. They are usually the
        // same card and are both marked, because "which card am I in the middle of" is the
        // question this feature answers. Without it the network was asked to place a point with
        // no indication of what it was spending.
        const uint8_t card_id = static_cast<uint8_t>(i + 1);
        const bool active = (ctx.resolving_card == card_id) || (ctx.pending_op_card == card_id);
        out_buf->card_features[i * card_slots::V22_FEATURES + card_slots::ACTIVE_CARD] =
            active ? 1.0f : 0.0f;
    }

    // Decision context. Every field here is a pure function of the state, so an observation
    // remains reproducible from a GameState alone -- which is what lets a run branch from a
    // snapshot, and what any decision-time search will need.
    const size_t dt = static_cast<size_t>(ctx.decision_type);
    if (dt < 8) out_buf->global_features[ctx_slots::DECISION_TYPE + dt] = 1.0f;

    // op_mode is only meaningful while an Op is being spent; SELECT_OP_MODE is where it is
    // chosen, and before that the field holds whatever the last Op left. Gate on there being a
    // pending Op so it reads as "no mode" rather than as a stale one.
    if (ctx.pending_op_card != 0 || ctx.decision_type == DecisionType::POINT_NODE) {
        const size_t om = static_cast<size_t>(ctx.op_mode);
        if (om < 3) out_buf->global_features[ctx_slots::OP_MODE + om] = 1.0f;
    }

    out_buf->global_features[ctx_slots::REMAINING_STEPS]    = static_cast<float>(ctx.remaining_steps) / 7.0f;
    out_buf->global_features[ctx_slots::PENDING_OPS_VALUE]  = static_cast<float>(ctx.pending_ops_value) / 5.0f;
    out_buf->global_features[ctx_slots::MAX_PER_COUNTRY]    = static_cast<float>(ctx.max_per_country) / 5.0f;
    out_buf->global_features[ctx_slots::ALLOW_EARLY_STOP]   = ctx.allow_early_stop ? 1.0f : 0.0f;
    // 255 means "no branch chosen", which is neither of these.
    out_buf->global_features[ctx_slots::TIMING_OPS_FIRST]   = (ctx.timing_branch == 0) ? 1.0f : 0.0f;
    out_buf->global_features[ctx_slots::TIMING_EVENT_FIRST] = (ctx.timing_branch == 1) ? 1.0f : 0.0f;
    out_buf->global_features[ctx_slots::EVENT_GRANTED_OPS]  = ctx.event_granted_ops ? 1.0f : 0.0f;
    out_buf->global_features[ctx_slots::SUPPRESS_OP_EVENT]  = ctx.suppress_op_card_event ? 1.0f : 0.0f;
    out_buf->global_features[ctx_slots::TEMP_CARD_COUNT]    = static_cast<float>(ctx.temp_card_cnt) / 8.0f;
}

void extract_observation(const GameState& state, Player perspective, ObservationBuffer* out_buf) noexcept {
    Observation::extract(state, perspective, out_buf);
}

void extract_observation_v21(const GameState& state, Player perspective,
                            ObservationBufferV21* out_buf) noexcept {
    Observation::extract_v21(state, perspective, out_buf);
}

void extract_observation_v22(const GameState& state, Player perspective,
                            ObservationBufferV22* out_buf) noexcept {
    Observation::extract_v22(state, perspective, out_buf);
}

} // namespace ts
