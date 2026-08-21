#pragma once
#include <cstdint>
#include "types.hpp"
#include "constants.hpp"

namespace ts {

// Compact 4-byte micro-action passed to Engine::step()
struct alignas(4) MicroAction {
    DecisionType decision_type; // Active decision type being answered
    uint8_t      primary_id;    // Card ID (1..110), Country ID (0..83), or Branch (0..7)
    uint8_t      secondary_id;  // Optional sub-index / mode (0 if unused)
    uint8_t      flags;         // Bit 7 (0x80): CONFIRM_DONE / Early stop flag

    constexpr bool is_confirm_done() const noexcept {
        return (flags & action_flags::CONFIRM_DONE) != 0;
    }
};
static_assert(sizeof(MicroAction) == 4, "MicroAction must be exactly 4 bytes");

} // namespace ts
