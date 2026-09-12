#pragma once
#include <cstdint>
#include "types.hpp"
#include "game_state.hpp"

namespace ts {

struct SpaceBoxInfo {
    uint8_t min_ops;
    uint8_t max_roll; // Roll <= max_roll is success
    uint8_t vp_first;
    uint8_t vp_second;
};

class SpaceRace {
public:
    static const SpaceBoxInfo& get_box_info(uint8_t box) noexcept;

    // Checks if player can attempt space race with given card
    static bool can_attempt_space(const GameState& state, Player p, uint8_t card_id) noexcept;

    // Resolves a space race attempt
    // Returns true if roll succeeded and track advanced
    static bool attempt_space(GameState& state, Player p, uint8_t card_id, uint8_t forced_roll = 0) noexcept;

    // Ability checks
    static bool has_animal_in_space(const GameState& state, Player p) noexcept;  // 2 space attempts / turn
    static bool has_man_in_space(const GameState& state, Player p) noexcept;     // Opponent reveals headline first
    static bool has_space_walk(const GameState& state, Player p) noexcept;       // Discard held card
    static bool has_space_station_ar8(const GameState& state, Player p) noexcept;// 8 ARs
};

} // namespace ts
