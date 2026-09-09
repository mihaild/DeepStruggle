#pragma once
#include <cstdint>
#include "types.hpp"
#include "game_state.hpp"

namespace ts {

class Observation {
public:
    // Extracts neural network observation vector from perspective of player
    static void extract(const GameState& state, Player perspective, ObservationBuffer* out_buf) noexcept;

    // Layout v2.1: the same observation with a wider card block, splitting "the opponent is known
    // to hold this" and "this card is not in the game yet" out of the old catch-all slot 0.
    //
    // Implemented by running `extract` and rewriting only the card features, so every other
    // section is identical to the legacy layout by construction rather than by inspection.
    static void extract_v21(const GameState& state, Player perspective,
                           ObservationBufferV21* out_buf) noexcept;

    // Layout v2.2: v2.1 plus the decision context, minus the blocks nothing read.
    //
    // Built on top of `extract_v21` for the same reason it is built on `extract` -- the sections
    // that are meant to be identical are identical by construction. What v2.2 adds is a card
    // feature marking the card currently being played, and twenty global scalars describing the
    // decision being asked. What it drops is turn_aggregates and active_player, which the network
    // never sliced.
    static void extract_v22(const GameState& state, Player perspective,
                           ObservationBufferV22* out_buf) noexcept;
};

void extract_observation(const GameState& state, Player perspective, ObservationBuffer* out_buf) noexcept;
void extract_observation_v21(const GameState& state, Player perspective,
                            ObservationBufferV21* out_buf) noexcept;
void extract_observation_v22(const GameState& state, Player perspective,
                            ObservationBufferV22* out_buf) noexcept;

} // namespace ts
