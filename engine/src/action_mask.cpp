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
    Player p = ctx.decision_player;

    switch (ctx.decision_type) {
        case DecisionType::NONE:
            *out_size = 0;
            break;

        case DecisionType::SELECT_CARD: {
            *out_size = 112;
            std::memset(mask_out, 0, 112);

            // 1. If currently inside an active event's sub-decision:
            if (ctx.resolving_card != 0) {
                CardHandlers::get_event_action_mask(state, mask_out, out_size);
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
                return;
            }

            // 3. Forced play (Missile Envy)
            if (state.forced_card_player == p && state.forced_card_id != 0) {
                mask_out[state.forced_card_id] = 1;
                return;
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
                return;
            }

            // 5. Standard Hand Selection
            CardLocation loc = (p == Player::US) ? CardLocation::HAND_US : CardLocation::HAND_USSR;
            for (uint8_t i = 1; i <= 110; ++i) {
                if (state.card_locations[i] == loc) {
                    mask_out[i] = 1;
                }
            }

            // The China Card
            if (state.china_card_holder == p && state.china_card_playable && state.current_phase != Phase::HEADLINE) {
                mask_out[card_ids::THE_CHINA_CARD] = 1;
            }
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

            // Event play
            if (CardHandlers::can_trigger_event(state, card, p)) {
                mask_out[static_cast<size_t>(PlayMode::EVENT)] = 1;
            }

            // Ops play
            mask_out[static_cast<size_t>(PlayMode::OPS)] = 1;

            // Space play
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

            // Check if influence placement is possible
            uint8_t inf_mask[84];
            Operations::get_influence_placement_mask(state, p, ops, inf_mask);
            for (uint8_t i = 0; i < 84; ++i) {
                if (inf_mask[i]) {
                    mask_out[static_cast<size_t>(OpMode::INFLUENCE)] = 1;
                    break;
                }
            }

            // Check if coup is possible
            uint8_t coup_mask[84];
            Operations::get_coup_target_mask(state, p, coup_mask);
            for (uint8_t i = 0; i < 84; ++i) {
                if (coup_mask[i]) {
                    mask_out[static_cast<size_t>(OpMode::COUP)] = 1;
                    break;
                }
            }

            // Check if realign is possible
            uint8_t realign_mask[84];
            Operations::get_realign_target_mask(state, p, realign_mask);
            for (uint8_t i = 0; i < 84; ++i) {
                if (realign_mask[i]) {
                    mask_out[static_cast<size_t>(OpMode::REALIGN)] = 1;
                    break;
                }
            }
            break;
        }

        case DecisionType::POINT_NODE: {
            *out_size = 84;
            std::memset(mask_out, 0, 84);

            if (ctx.resolving_card != 0) {
                CardHandlers::get_event_action_mask(state, mask_out, out_size);
                return;
            }

            // Setup Phase placement
            if (state.current_phase == Phase::SETUP) {
                if (p == Player::USSR) {
                    for (uint8_t i = 0; i < 84; ++i) {
                        if (MapData::get_country(i).in_eastern_europe) mask_out[i] = 1;
                    }
                } else if (p == Player::US) {
                    for (uint8_t i = 0; i < 84; ++i) {
                        if (MapData::get_country(i).in_western_europe) mask_out[i] = 1;
                    }
                }
                return;
            }

            // Standard Op Mode Node Selection
            if (ctx.op_mode == OpMode::COUP) {
                Operations::get_coup_target_mask(state, p, mask_out);
            } else if (ctx.op_mode == OpMode::REALIGN) {
                Operations::get_realign_target_mask(state, p, mask_out);
            } else {
                if (ctx.remaining_steps > 0) {
                    Operations::get_influence_placement_mask(state, p, ctx.remaining_steps, mask_out);
                }
            }
            break;
        }

        case DecisionType::CHOOSE_BRANCH: {
            *out_size = 8;
            std::memset(mask_out, 0, 8);
            if (ctx.resolving_card != 0) {
                CardHandlers::get_event_action_mask(state, mask_out, out_size);
            } else {
                mask_out[0] = 1;
                mask_out[1] = 1;
            }
            break;
        }
    }
}

} // namespace ts
