#pragma once
#include <cstdint>
#include "types.hpp"
#include "game_state.hpp"
#include "micro_action.hpp"

namespace ts {

class CardHandlers {
public:
    // Initiates execution of an event.
    // Returns true if event is fully finished immediately (no sub-decisions),
    // or false if it transitioned into interactive sub-decisions in ctx().
    static bool trigger_event(GameState& state, uint8_t card_id, Player player) noexcept;

    // Handles a sub-decision for an ongoing card event.
    // Returns true when the entire event finishes and pops from context stack.
    static bool handle_event_step(GameState& state, const MicroAction& action) noexcept;

    // Fills legal action mask for the current event step
    static void get_event_action_mask(const GameState& state, uint8_t* mask_out, size_t* out_size) noexcept;

    // Prerequisite checks (e.g. NATO requires Marshall or Warsaw)
    static bool can_trigger_event(const GameState& state, uint8_t card_id, Player player) noexcept;
};

} // namespace ts
