#pragma once
#include <cstdint>
#include "types.hpp"
#include "game_state.hpp"

namespace ts {

struct CoupResult {
    bool    success;
    uint8_t die_roll;
    int16_t margin;
    uint8_t opp_inf_removed;
    uint8_t player_inf_added;
    bool    defcon_degraded;
    bool    caused_defcon_suicide;
};

struct RealignResult {
    uint8_t us_roll;
    uint8_t ussr_roll;
    int16_t us_mod;
    int16_t ussr_mod;
    int16_t us_total;
    int16_t ussr_total;
    Player  winner;
    uint8_t inf_removed;
};

class Operations {
public:
    static uint8_t get_modified_ops(const GameState& state, uint8_t base_ops, Player player, Region target_region = Region::NONE_REGION) noexcept;
    static uint8_t get_effective_ops(const GameState& state, uint8_t card_id, Player player, Region target_region = Region::NONE_REGION) noexcept;

    // The Ops budget to hand a player who is about to conduct Operations, whether from playing
    // a card for Ops or from an Event that grants them. Conditional region bonuses -- Vietnam
    // Revolts, and the China Card's Asia bonus -- are granted up front, and the budget ladder
    // in the state machine withdraws them if the player spends outside the region. Every grant
    // goes through these so none can quietly omit the region: passing the default
    // NONE_REGION instead left UN Intervention's named card worth only its printed Ops, and at
    // turn 2 AR4 of ts-replayer game 108 the USSR could afford one Southeast Asian placement
    // where the human made two.
    static uint8_t grant_ops_for_card(const GameState& state, uint8_t card_id, Player player) noexcept;
    static uint8_t grant_ops(const GameState& state, uint8_t base_ops, Player player) noexcept;

    // Influence Placement
    static bool can_place_influence(const GameState& state, Player p, uint8_t country_id) noexcept;
    static uint8_t get_influence_cost(const GameState& state, Player p, uint8_t country_id) noexcept;
    static bool place_influence(GameState& state, Player p, uint8_t country_id) noexcept;

    // Coup Attempts
    static bool can_coup(const GameState& state, Player p, uint8_t country_id) noexcept;
    static CoupResult execute_coup(GameState& state, Player p, uint8_t country_id, uint8_t ops_value, uint8_t forced_roll = 0) noexcept;

    // Realignment Rolls
    static bool can_realign(const GameState& state, Player p, uint8_t country_id) noexcept;
    static RealignResult execute_realign(GameState& state, Player p, uint8_t country_id, uint8_t forced_us_roll = 0, uint8_t forced_ussr_roll = 0) noexcept;

    // Shared Coup/Realignment Restrictions
    static bool can_coup_or_realign(const GameState& state, Player p, uint8_t country_id) noexcept;
    static bool can_coup_or_realign(const GameState& state, Player p) noexcept;

    // Fast bitmask queries for legal targets
    static void get_influence_placement_mask(const GameState& state, Player p, uint8_t ops_available, uint8_t* out_mask_84) noexcept;
    static void get_coup_target_mask(const GameState& state, Player p, uint8_t* out_mask_84) noexcept;
    static void get_realign_target_mask(const GameState& state, Player p, uint8_t* out_mask_84) noexcept;
};

} // namespace ts
