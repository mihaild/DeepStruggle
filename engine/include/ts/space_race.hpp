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

    // Moves `p`'s marker to `box` (1..8) and pays that box's VP: the first award when `p` is now
    // ahead of the opponent, the second otherwise. The one implementation behind a successful
    // attempt, Captured Nazi Scientist and One Small Step.
    static void advance_to(GameState& state, Player p, uint8_t box) noexcept;

    // Ability checks
    static bool has_animal_in_space(const GameState& state, Player p) noexcept;  // 2 space attempts / turn
    static bool has_man_in_earth_orbit(const GameState& state, Player p) noexcept; // Opponent reveals headline first
    static bool has_eagle_bear_landed(const GameState& state, Player p) noexcept;  // Discard held card
    static bool has_space_station_ar8(const GameState& state, Player p) noexcept;// 8 ARs

    // The opposing headline `p` has seen, else 0: with Man in Earth Orbit held by `p` alone the opponent
    // chooses first and reveals it (StateMachine::start_turn asks it first), so while `p` chooses
    // its own headline the opponent's card is known to it.
    static uint8_t revealed_headline(const GameState& state, Player p) noexcept;
};

} // namespace ts
