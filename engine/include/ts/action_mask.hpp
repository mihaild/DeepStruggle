#pragma once
#include <cstdint>
#include <cstddef>
#include "types.hpp"
#include "game_state.hpp"

namespace ts {

class ActionMask {
public:
    // Generates the legal action mask for the active decision context in state.
    // Writes flags (1 for legal, 0 for illegal) into mask_out.
    // Sets *out_size to the number of elements written.
    static void generate_mask(const GameState& state, uint8_t* mask_out, size_t* out_size) noexcept;
};

} // namespace ts
