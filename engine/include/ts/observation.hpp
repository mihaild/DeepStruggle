#pragma once
#include <cstdint>
#include "types.hpp"
#include "game_state.hpp"

namespace ts {

class Observation {
public:
    // Extracts the observation from `perspective`'s point of view -- layout v2.3, and the only
    // layout there is.
    //
    // This used to be a chain: v2.3 was built on v2.1, which was built on the legacy extractor,
    // each stage copying the sections that were meant to be identical so they could not drift.
    // That was the right shape while three layouts had to agree. With one layout the chain is
    // just two intermediate buffers and a restride, so it is written out directly.
    static void extract(const GameState& state, Player perspective,
                        ObservationBufferV23* out_buf) noexcept;
};

void extract_observation(const GameState& state, Player perspective,
                         ObservationBufferV23* out_buf) noexcept;

} // namespace ts
