#include "ts/engine.hpp"
#include "ts/state_machine.hpp"
#include "ts/action_mask.hpp"
#include "ts/serialization.hpp"
#include "ts/card_data.hpp"

namespace ts {

void Engine::init_game(GameState& state, uint64_t seed) noexcept {
    StateMachine::init_new_game(state, seed);
}

void Engine::get_legal_action_mask(const GameState& state, uint8_t* mask_out, size_t* out_size) noexcept {
    ActionMask::generate_mask(state, mask_out, out_size);
}

void Engine::get_flat_action_mask(const GameState& state, uint8_t* mask_212) noexcept {
    ActionMask::generate_flat_mask_212(state, mask_212);
}

bool Engine::step(GameState& state, const MicroAction& action) noexcept {
    return StateMachine::step(state, action);
}

bool Engine::step_flat(GameState& state, uint16_t action_idx) noexcept {
    MicroAction action = ActionMask::decode_flat_action_212(state, action_idx);
    return StateMachine::step(state, action);
}

bool Engine::is_terminal(const GameState& state) noexcept {
    return state.current_phase == Phase::GAME_OVER || state.victory_points >= 20 || state.victory_points <= -20;
}

float Engine::get_terminal_utility(const GameState& state) noexcept {
    if (state.victory_points >= 20) return 1.0f;
    if (state.victory_points <= -20) return -1.0f;
    if (state.victory_points > 0) return 1.0f;
    if (state.victory_points < 0) return -1.0f;
    return 0.0f;
}

bool Engine::has_held_scoring_card(const GameState& state, Player p) noexcept {
    CardLocation target_loc = (p == Player::US) ? CardLocation::HAND_US :
                              ((p == Player::USSR) ? CardLocation::HAND_USSR : CardLocation::UNAVAILABLE);
    if (target_loc == CardLocation::UNAVAILABLE) return false;
    for (uint8_t i = 1; i <= 110; ++i) {
        if (CardData::is_scoring_card(i) && state.card_locations[i] == target_loc) {
            return true;
        }
    }
    return false;
}

bool Engine::is_held_scoring_game_over(const GameState& state) noexcept {
    if (state.current_phase != Phase::GAME_OVER) return false;
    if (state.defcon <= 1) return false;
    uint8_t max_ar = (state.turn <= 3) ? 6 : 7;
    if (state.action_round <= max_ar) return false;
    return has_held_scoring_card(state, Player::US) || has_held_scoring_card(state, Player::USSR);
}

bool Engine::is_held_scoring_loss(const GameState& state, Player p) noexcept {
    if (!is_held_scoring_game_over(state)) return false;
    return has_held_scoring_card(state, p);
}

void Engine::serialize(const GameState& state, uint8_t* out_bytes, size_t max_bytes) noexcept {
    Serializer::serialize_binary(state, out_bytes, max_bytes);
}

void Engine::deserialize(GameState& state, const uint8_t* in_bytes, size_t in_size) noexcept {
    Serializer::deserialize_binary(state, in_bytes, in_size);
}

std::string Engine::to_json(const GameState& state) {
    return Serializer::to_json(state);
}

} // namespace ts
