#include "ts/action_mask.hpp"
#include "ts/map_data.hpp"
#include "ts/card_data.hpp"
#include "ts/card_handlers.hpp"
#include "ts/ops.hpp"
#include "ts/space_race.hpp"
#include <cstring>

namespace ts {

void ActionMask::generate_mask(const GameState& state, uint8_t* mask_out, size_t* out_size) noexcept {
    if (!mask_out || !out_size) return;

    if (state.current_phase == Phase::GAME_OVER) {
        *out_size = 0;
        return;
    }

    const auto& ctx = state.ctx();
    Player p = (ctx.decision_player != Player::NONE) ? ctx.decision_player : state.phasing_player;

    switch (ctx.decision_type) {
        case DecisionType::NONE:
            *out_size = 1;
            mask_out[0] = 1;
            break;

        case DecisionType::SELECT_CARD: {
            *out_size = 112;
            std::memset(mask_out, 0, 112);

            // 0. Space Walk (Box 6) end-of-turn discard
            if (ctx.resolving_card == card_ids::SPACE_WALK_DISCARD) {
                CardLocation loc = (p == Player::US) ? CardLocation::HAND_US : CardLocation::HAND_USSR;
                for (uint8_t i = 1; i <= 110; ++i) {
                    if (state.card_locations[i] == loc) {
                        mask_out[i] = 1;
                    }
                }
                mask_out[0] = 1; // Allow early stop / pass
                return;
            }

            // 1. If currently inside an active event's sub-decision:
            if (ctx.resolving_card != 0) {
                CardHandlers::get_event_action_mask(state, mask_out, out_size);
                bool any_legal = false;
                for (size_t i = 0; i < *out_size; ++i) {
                    if (mask_out[i]) { any_legal = true; break; }
                }
                if (!any_legal || ctx.allow_early_stop) {
                    mask_out[0] = 1;
                }
                return;
            }

            // 2. UN Intervention (#32) companion card selection
            if (ctx.pending_op_card == card_ids::UN_INTERVENTION) {
                CardLocation loc = (p == Player::US) ? CardLocation::HAND_US : CardLocation::HAND_USSR;
                for (uint8_t i = 1; i <= 110; ++i) {
                    if (state.card_locations[i] == loc && CardData::is_opponent_card(i, p)) {
                        mask_out[i] = 1;
                    }
                }
                bool any = false;
                for (size_t i = 0; i < 112; ++i) if (mask_out[i]) { any = true; break; }
                if (!any) mask_out[0] = 1;
                return;
            }

            // 3. Forced play (Missile Envy)
            if (state.forced_card_player == p && state.forced_card_id != 0) {
                CardLocation loc = (p == Player::US) ? CardLocation::HAND_US : CardLocation::HAND_USSR;
                if (state.card_locations[state.forced_card_id] == loc) {
                    mask_out[state.forced_card_id] = 1;
                    return;
                }
            }

            // 4. Quagmire / Bear Trap
            bool trapped = (p == Player::US && state.has_flag(effect_bits::QUAGMIRE_ACTIVE)) ||
                           (p == Player::USSR && state.has_flag(effect_bits::BEAR_TRAP_ACTIVE));

            if (trapped) {
                CardLocation loc = (p == Player::US) ? CardLocation::HAND_US : CardLocation::HAND_USSR;
                bool has_2ops = false;
                for (uint8_t i = 1; i <= 110; ++i) {
                    if (state.card_locations[i] == loc && CardData::get_card(i).ops >= 2) {
                        has_2ops = true;
                        mask_out[i] = 1;
                    }
                }
                if (!has_2ops) {
                    // Must play scoring cards if holding any
                    for (uint8_t i = 1; i <= 110; ++i) {
                        if (state.card_locations[i] == loc && CardData::is_scoring_card(i)) {
                            mask_out[i] = 1;
                        }
                    }
                }
                bool any = false;
                for (size_t i = 0; i < 112; ++i) if (mask_out[i]) { any = true; break; }
                if (!any) mask_out[0] = 1;
                return;
            }

            // 5. Standard Hand Selection
            CardLocation loc = (p == Player::US) ? CardLocation::HAND_US : CardLocation::HAND_USSR;
            for (uint8_t i = 1; i <= 110; ++i) {
                if (i == card_ids::THE_CHINA_CARD) continue; // China card handled below
                if (state.current_phase == Phase::HEADLINE && i == card_ids::UN_INTERVENTION) continue;
                if (state.card_locations[i] == loc) {
                    mask_out[i] = 1;
                }
            }

            // The China Card
            if (state.china_card_holder == p && state.china_card_playable && state.current_phase != Phase::HEADLINE) {
                mask_out[card_ids::THE_CHINA_CARD] = 1;
            }

            bool any = false;
            for (size_t i = 0; i < 112; ++i) if (mask_out[i]) { any = true; break; }
            if (!any) mask_out[0] = 1;
            break;
        }

        case DecisionType::SELECT_PLAY_MODE: {
            *out_size = 4;
            std::memset(mask_out, 0, 4);
            uint8_t card = ctx.pending_op_card;

            if (CardData::is_scoring_card(card)) {
                mask_out[static_cast<size_t>(PlayMode::EVENT)] = 1;
                return;
            }

            // China Card can ONLY be played for Operations (no Event, no Space Race)
            if (card == card_ids::THE_CHINA_CARD) {
                mask_out[static_cast<size_t>(PlayMode::OPS)] = 1;
                return;
            }

            // Forced play (Missile Envy recipient must play for Operations)
            if (state.forced_card_player == p && (state.forced_card_id == card || state.forced_card_id == card_ids::MISSILE_ENVY)) {
                mask_out[static_cast<size_t>(PlayMode::OPS)] = 1;
                return;
            }

            // Defectors cannot be played as an event by US during Action Round
            if (card == card_ids::DEFECTORS && p == Player::US && state.current_phase == Phase::ACTION_ROUND) {
                mask_out[static_cast<size_t>(PlayMode::OPS)] = 1;
                if (SpaceRace::can_attempt_space(state, p, card)) {
                    mask_out[static_cast<size_t>(PlayMode::SPACE)] = 1;
                }
                return;
            }

            // Event play: legal ONLY for friendly or neutral cards (excluding China Card),
            // and only if event prerequisites are met.
            if (!CardData::is_opponent_card(card, p) && CardHandlers::can_trigger_event(state, card, p)) {
                mask_out[static_cast<size_t>(PlayMode::EVENT)] = 1;
            }

            // Ops play: always legal for non-scoring cards
            mask_out[static_cast<size_t>(PlayMode::OPS)] = 1;

            // Space play: legal if prerequisites are met and card is NOT China Card
            if (SpaceRace::can_attempt_space(state, p, card)) {
                mask_out[static_cast<size_t>(PlayMode::SPACE)] = 1;
            }
            break;
        }

        case DecisionType::CHOOSE_TIMING_BRANCH: {
            *out_size = 2;
            mask_out[0] = 1; // OPS_FIRST
            mask_out[1] = 1; // EVENT_FIRST
            break;
        }

        case DecisionType::SELECT_OP_MODE: {
            *out_size = 3;
            std::memset(mask_out, 0, 3);
            uint8_t ops = ctx.pending_ops_value;
            uint8_t op_card = ctx.pending_op_card;

            // Check if influence placement is possible (Junta & Tear Down This Wall forbid Influence)
            if (op_card != card_ids::JUNTA && op_card != card_ids::TEAR_DOWN_THIS_WALL) {
                uint8_t inf_mask[84];
                Operations::get_influence_placement_mask(state, p, ops, inf_mask);
                for (uint8_t i = 0; i < 84; ++i) {
                    if (inf_mask[i]) {
                        mask_out[static_cast<size_t>(OpMode::INFLUENCE)] = 1;
                        break;
                    }
                }
            } else {
                // Junta & Tear Down This Wall: bonus coup/realign is optional.
                // Selecting INFLUENCE mode acts as declining/skipping the bonus action.
                mask_out[static_cast<size_t>(OpMode::INFLUENCE)] = 1;
            }

            // Check if coup is possible (KAL-007 & Glasnost forbid Coups)
            if (op_card != card_ids::SOVIETS_SHOOT_DOWN_KAL_007 && op_card != card_ids::GLASNOST) {
                uint8_t coup_mask[84];
                Operations::get_coup_target_mask(state, p, coup_mask);
                for (uint8_t i = 0; i < 84; ++i) {
                    if (coup_mask[i]) {
                        if (op_card == card_ids::JUNTA) {
                            const auto& c = MapData::get_country(i);
                            if (c.region != Region::CENTRAL_AMERICA && c.region != Region::SOUTH_AMERICA) continue;
                        } else if (op_card == card_ids::TEAR_DOWN_THIS_WALL) {
                            if (MapData::get_country(i).region != Region::EUROPE) continue;
                        }
                        mask_out[static_cast<size_t>(OpMode::COUP)] = 1;
                        break;
                    }
                }
            }

            // Check if realign is possible
            uint8_t realign_mask[84];
            Operations::get_realign_target_mask(state, p, realign_mask);
            for (uint8_t i = 0; i < 84; ++i) {
                if (realign_mask[i]) {
                    if (op_card == card_ids::JUNTA) {
                        const auto& c = MapData::get_country(i);
                        if (c.region != Region::CENTRAL_AMERICA && c.region != Region::SOUTH_AMERICA) continue;
                    } else if (op_card == card_ids::TEAR_DOWN_THIS_WALL) {
                        if (MapData::get_country(i).region != Region::EUROPE) continue;
                    }
                    mask_out[static_cast<size_t>(OpMode::REALIGN)] = 1;
                    break;
                }
            }
            if (!mask_out[0] && !mask_out[1] && !mask_out[2]) {
                mask_out[0] = 1; // Allow skipping if no ops legal
            }
            break;
        }

        case DecisionType::POINT_NODE: {
            *out_size = 84;
            std::memset(mask_out, 0, 84);

            if (ctx.resolving_card != 0) {
                CardHandlers::get_event_action_mask(state, mask_out, out_size);
                *out_size = 84;
                return;
            }

            // Setup Phase placement
            if (state.current_phase == Phase::SETUP) {
                if (p == Player::USSR) {
                    for (uint8_t i = 0; i < 84; ++i) {
                        if (MapData::get_country(i).in_eastern_europe) mask_out[i] = 1;
                    }
                } else if (p == Player::US) {
                    if (ctx.pending_ops_value == 0) {
                        // Stage 0: 7 Western Europe placements
                        for (uint8_t i = 0; i < 84; ++i) {
                            if (MapData::get_country(i).in_western_europe) mask_out[i] = 1;
                        }
                    } else {
                        // Stage 1: 2 Bonus placements in countries with existing US presence
                        for (uint8_t i = 0; i < 84; ++i) {
                            if (state.countries[i].us_influence > 0) mask_out[i] = 1;
                        }
                    }
                }
                return;
            }

            // Standard Op Mode Node Selection
            if (ctx.op_mode == OpMode::COUP) {
                Operations::get_coup_target_mask(state, p, mask_out);
                if (ctx.pending_op_card == card_ids::JUNTA) {
                    for (uint8_t i = 0; i < 84; ++i) {
                        const auto& c = MapData::get_country(i);
                        if (c.region != Region::CENTRAL_AMERICA && c.region != Region::SOUTH_AMERICA) mask_out[i] = 0;
                    }
                } else if (ctx.pending_op_card == card_ids::TEAR_DOWN_THIS_WALL) {
                    for (uint8_t i = 0; i < 84; ++i) {
                        if (MapData::get_country(i).region != Region::EUROPE) mask_out[i] = 0;
                    }
                }
            } else if (ctx.op_mode == OpMode::REALIGN) {
                Operations::get_realign_target_mask(state, p, mask_out);
                if (ctx.pending_op_card == card_ids::JUNTA) {
                    for (uint8_t i = 0; i < 84; ++i) {
                        const auto& c = MapData::get_country(i);
                        if (c.region != Region::CENTRAL_AMERICA && c.region != Region::SOUTH_AMERICA) mask_out[i] = 0;
                    }
                } else if (ctx.pending_op_card == card_ids::TEAR_DOWN_THIS_WALL) {
                    for (uint8_t i = 0; i < 84; ++i) {
                        if (MapData::get_country(i).region != Region::EUROPE) mask_out[i] = 0;
                    }
                }
            } else {
                if (ctx.remaining_steps > 0) {
                    Operations::get_influence_placement_mask(state, p, ctx.remaining_steps, mask_out);
                    uint8_t op_card = ctx.pending_op_card;
                    uint8_t total_spent = ctx.pending_ops_value - ctx.remaining_steps;
                    if (op_card == card_ids::THE_CHINA_CARD) {
                        uint8_t non_asia_base = Operations::get_effective_ops(state, op_card, p, Region::NONE_REGION);
                        for (uint8_t i = 0; i < 84; ++i) {
                            if (mask_out[i] && MapData::get_country(i).region != Region::ASIA) {
                                uint8_t cost = Operations::get_influence_cost(state, p, i);
                                if (total_spent + cost > non_asia_base) {
                                    mask_out[i] = 0;
                                }
                            }
                        }
                    } else if (p == Player::USSR && state.has_flag(effect_bits::VIETNAM_REVOLTS_ACTIVE)) {
                        uint8_t non_se_base = Operations::get_effective_ops(state, op_card, p, Region::NONE_REGION);
                        for (uint8_t i = 0; i < 84; ++i) {
                            if (mask_out[i] && !MapData::get_country(i).in_southeast_asia) {
                                uint8_t cost = Operations::get_influence_cost(state, p, i);
                                if (total_spent + cost > non_se_base) {
                                    mask_out[i] = 0;
                                }
                            }
                        }
                    }
                }
            }
            *out_size = 84;
            break;
        }

        case DecisionType::CHOOSE_BRANCH: {
            *out_size = 8;
            std::memset(mask_out, 0, 8);
            if (ctx.resolving_card != 0) {
                CardHandlers::get_event_action_mask(state, mask_out, out_size);
                bool any_legal = false;
                for (size_t i = 0; i < *out_size; ++i) {
                    if (mask_out[i]) { any_legal = true; break; }
                }
                if (!any_legal) {
                    mask_out[0] = 1;
                }
            } else {
                mask_out[0] = 1;
                mask_out[1] = 1;
            }
            break;
        }

        case DecisionType::ROLL_DIE: {
            *out_size = 7;
            std::memset(mask_out, 0, 7);
            mask_out[0] = 1; // 0 = Auto-roll
            for (uint8_t r = 1; r <= 6; ++r) mask_out[r] = 1; // 1..6 = Forced roll
            break;
        }
    }
}


void ActionMask::generate_flat_mask_212(const GameState& state, uint8_t* mask_212) noexcept {
    if (!mask_212) return;
    std::memset(mask_212, 0, FLAT_ACTION_SPACE_SIZE);

    if (state.current_phase == Phase::GAME_OVER || state.victory_points >= 20 || state.victory_points <= -20) {
        return;
    }

    uint8_t temp_mask[128];
    size_t temp_size = 0;
    generate_mask(state, temp_mask, &temp_size);

    const auto& ctx = state.ctx();
    switch (ctx.decision_type) {
        case DecisionType::NONE:
            mask_212[211] = 1;
            break;

        case DecisionType::SELECT_CARD:
            for (size_t i = 1; i <= 110 && i < temp_size; ++i) {
                if (temp_mask[i]) {
                    mask_212[i - 1] = 1;
                }
            }
            if (temp_mask[0]) {
                mask_212[211] = 1; // Pass / early stop
            }
            break;

        case DecisionType::SELECT_PLAY_MODE:
            for (size_t i = 0; i < 4 && i < temp_size; ++i) {
                if (temp_mask[i]) {
                    mask_212[110 + i] = 1;
                }
            }
            break;

        case DecisionType::CHOOSE_TIMING_BRANCH:
            for (size_t i = 0; i < 2 && i < temp_size; ++i) {
                if (temp_mask[i]) {
                    mask_212[114 + i] = 1;
                }
            }
            break;

        case DecisionType::SELECT_OP_MODE:
            for (size_t i = 0; i < 3 && i < temp_size; ++i) {
                if (temp_mask[i]) {
                    mask_212[116 + i] = 1;
                }
            }
            break;

        case DecisionType::POINT_NODE:
            for (size_t i = 0; i < 84 && i < temp_size; ++i) {
                if (temp_mask[i]) {
                    mask_212[119 + i] = 1;
                }
            }
            if (ctx.allow_early_stop) {
                mask_212[211] = 1;
            }
            break;

        case DecisionType::CHOOSE_BRANCH:
            for (size_t i = 0; i < 8 && i < temp_size; ++i) {
                if (temp_mask[i]) {
                    mask_212[203 + i] = 1;
                }
            }
            if (ctx.allow_early_stop) {
                mask_212[211] = 1;
            }
            break;
    }

    // Safety guarantee: ensure at least one action is legal if game is active
    bool any_legal = false;
    for (size_t i = 0; i < FLAT_ACTION_SPACE_SIZE; ++i) {
        if (mask_212[i]) {
            any_legal = true;
            break;
        }
    }
    if (!any_legal) {
        mask_212[211] = 1; // Fallback confirm/done
    }
}

MicroAction ActionMask::decode_flat_action_212(const GameState& state, uint16_t action_idx) noexcept {
    const auto& ctx = state.ctx();

    if (action_idx == 211) {
        if (ctx.decision_type == DecisionType::SELECT_CARD) {
            return MicroAction{DecisionType::SELECT_CARD, 0, 0, 0};
        }
        return MicroAction{ctx.decision_type, 255, 0, action_flags::CONFIRM_DONE};
    }

    if (action_idx < 110) {
        return MicroAction{DecisionType::SELECT_CARD, static_cast<uint8_t>(action_idx + 1), 0, 0};
    }
    if (action_idx < 114) {
        return MicroAction{DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(action_idx - 110), 0, 0};
    }
    if (action_idx < 116) {
        return MicroAction{DecisionType::CHOOSE_TIMING_BRANCH, static_cast<uint8_t>(action_idx - 114), 0, 0};
    }
    if (action_idx < 119) {
        return MicroAction{DecisionType::SELECT_OP_MODE, static_cast<uint8_t>(action_idx - 116), 0, 0};
    }
    if (action_idx < 203) {
        return MicroAction{DecisionType::POINT_NODE, static_cast<uint8_t>(action_idx - 119), 0, 0};
    }
    if (action_idx < 211) {
        return MicroAction{DecisionType::CHOOSE_BRANCH, static_cast<uint8_t>(action_idx - 203), 0, 0};
    }

    return MicroAction{ctx.decision_type, 255, 0, action_flags::CONFIRM_DONE};
}

int16_t ActionMask::encode_micro_action_212(const GameState& state, const MicroAction& action) noexcept {
    (void)state;
    if (action.is_confirm_done() || action.primary_id == 255) {
        return 211;
    }

    switch (action.decision_type) {
        case DecisionType::SELECT_CARD:
            if (action.primary_id == 0) return 211;
            if (action.primary_id >= 1 && action.primary_id <= 110) {
                return static_cast<int16_t>(action.primary_id - 1);
            }
            return 211;

        case DecisionType::SELECT_PLAY_MODE:
            if (action.primary_id < 4) return static_cast<int16_t>(110 + action.primary_id);
            return 211;

        case DecisionType::CHOOSE_TIMING_BRANCH:
            if (action.primary_id < 2) return static_cast<int16_t>(114 + action.primary_id);
            return 211;

        case DecisionType::SELECT_OP_MODE:
            if (action.primary_id < 3) return static_cast<int16_t>(116 + action.primary_id);
            return 211;

        case DecisionType::POINT_NODE:
            if (action.primary_id < 84) return static_cast<int16_t>(119 + action.primary_id);
            return 211;

        case DecisionType::CHOOSE_BRANCH:
            if (action.primary_id < 8) return static_cast<int16_t>(203 + action.primary_id);
            return 211;

        case DecisionType::NONE:
        default:
            return 211;
    }
}

} // namespace ts
