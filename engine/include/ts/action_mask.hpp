#pragma once
#include <cstdint>
#include <cstddef>
#include "types.hpp"
#include "game_state.hpp"
#include "micro_action.hpp"

namespace ts {

constexpr size_t FLAT_ACTION_SPACE_SIZE = 212;

class ActionMask {
public:
    // Generates the legal action mask for the active decision context in state.
    // Writes flags (1 for legal, 0 for illegal) into mask_out.
    // Sets *out_size to the number of elements written.
    static void generate_mask(const GameState& state, uint8_t* mask_out, size_t* out_size) noexcept;

    // Generates the unified 212-dimensional legal action mask for neural network consumption.
    // Writes exactly 212 uint8 flags (1 = legal, 0 = illegal) into mask_212.
    static void generate_flat_mask_212(const GameState& state, uint8_t* mask_212) noexcept;

    // Decodes a flat action index [0..211] into a concrete MicroAction
    static MicroAction decode_flat_action_212(const GameState& state, uint16_t action_idx) noexcept;

    // Encodes a MicroAction into its flat action index [0..211]
    static int16_t encode_micro_action_212(const GameState& state, const MicroAction& action) noexcept;
};

} // namespace ts
