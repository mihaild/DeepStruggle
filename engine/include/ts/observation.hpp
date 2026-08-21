#pragma once
#include <cstdint>
#include "types.hpp"
#include "game_state.hpp"

namespace ts {

class Observation {
public:
    // Extracts neural network observation vector from perspective of player
    static void extract(const GameState& state, Player perspective, ObservationBuffer* out_buf) noexcept;
};

void extract_observation(const GameState& state, Player perspective, ObservationBuffer* out_buf) noexcept;

} // namespace ts
