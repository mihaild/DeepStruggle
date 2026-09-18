#pragma once
#include <cstdint>
#include <cstddef>
#include "types.hpp"
#include "game_state.hpp"
#include "micro_action.hpp"

namespace ts {

// The flat action layout, in one place. These were written out as 119, 203 and 211 at eight
// sites in action_mask.cpp; a layout change with them spelled by hand fails silently, decoding an
// action as the wrong country instead of raising. The Python side takes the same numbers from
// bindings/action_encoder.py, and tests/bindings/test_flat_reserved_indices.py checks the two
// agree.
namespace flat_slots {
    constexpr uint16_t CARD          = 0;    // 110 wide: card id - 1
    constexpr uint16_t RESOLUTION    = 110;  //   5 wide: EVENT, SPACE, OPS_{INFLUENCE,COUP,REALIGN}
    constexpr uint16_t OP_MODE       = 112;  //   3 wide: the deferred Ops choice, sharing OPS_*
    constexpr uint16_t ROLL_DIE      = 115;  //   1
    constexpr uint16_t NODE          = 116;  //  84 wide: country id
    constexpr uint16_t BRANCH        = 200;  //   8 wide
    constexpr uint16_t CONFIRM_DONE  = 208;  //   1: the single "nothing happens" index
    constexpr uint16_t DEFCON_VALUE  = 209;  //   5 wide: set DEFCON to 1..5 (Summit, How I Learned)
    constexpr uint16_t REGION        = 214;  //   6 wide: Europe..South America (Chernobyl)
    constexpr uint16_t SIZE          = 220;

    constexpr uint16_t CARD_COUNT    = 110;
    constexpr uint16_t NODE_COUNT    = 84;
    constexpr uint16_t OP_MODE_COUNT = 3;
    constexpr uint16_t BRANCH_COUNT  = 8;
    constexpr uint16_t DEFCON_COUNT  = 5;
    constexpr uint16_t REGION_COUNT  = 6;
}

constexpr size_t FLAT_ACTION_SPACE_SIZE = flat_slots::SIZE;

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
