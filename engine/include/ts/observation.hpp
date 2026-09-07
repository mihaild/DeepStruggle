#pragma once
#include <cstdint>
#include "types.hpp"
#include "game_state.hpp"

namespace ts {

class Observation {
public:
    // Extracts neural network observation vector from perspective of player
    static void extract(const GameState& state, Player perspective, ObservationBuffer* out_buf) noexcept;

    // Layout v2: the same observation with a wider card block, splitting "the opponent is known
    // to hold this" and "this card is not in the game yet" out of the old catch-all slot 0.
    //
    // Implemented by running `extract` and rewriting only the card features, so every other
    // section is identical to the legacy layout by construction rather than by inspection.
    static void extract_v2(const GameState& state, Player perspective,
                           ObservationBufferV2* out_buf) noexcept;
};

void extract_observation(const GameState& state, Player perspective, ObservationBuffer* out_buf) noexcept;
void extract_observation_v2(const GameState& state, Player perspective,
                            ObservationBufferV2* out_buf) noexcept;

} // namespace ts
