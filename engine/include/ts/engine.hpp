#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include "types.hpp"
#include "game_state.hpp"
#include "micro_action.hpp"

namespace ts {

class Engine {
public:
    // Initialize standard game state (shuffles Early War deck, deals hands, sets USSR China Card face up)
    static void init_game(GameState& state, uint64_t seed) noexcept;

    // Writes binary mask of valid actions into mask_out buffer.
    // Max buffer required: 112 bytes for SELECT_CARD, 84 bytes for POINT_NODE.
    static void get_legal_action_mask(const GameState& state, uint8_t* mask_out, size_t* out_size) noexcept;

    // Advances game state by 1 validated micro-action. Returns true on success.
    static bool step(GameState& state, const MicroAction& action) noexcept;

    // Fast terminal state evaluation
    static bool is_terminal(const GameState& state) noexcept;
    static float get_terminal_utility(const GameState& state) noexcept; // +1.0 (US), -1.0 (USSR), 0.0 (Draw)

    // Serialization & Inspection
    static void serialize(const GameState& state, uint8_t* out_bytes, size_t max_bytes) noexcept;
    static void deserialize(GameState& state, const uint8_t* in_bytes, size_t in_size) noexcept;
    static std::string to_json(const GameState& state);
};

} // namespace ts
