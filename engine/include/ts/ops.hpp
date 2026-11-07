#pragma once
#include <array>
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
    // The card's Ops for an action aimed at one country: the regional bonuses exactly as that
    // country earns them (Asia for the China Card, Southeast Asia for Vietnam Revolts).
    static uint8_t get_effective_ops_in(const GameState& state, uint8_t card_id, Player player, uint8_t country_id) noexcept;

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

    // The same two grants, written into the current frame together with the unbonused base the
    // region ladder falls back to (DecisionContext::ops_plain). Every site that hands a player
    // Ops goes through one of these.
    static void grant_card_ops_to_ctx(GameState& state, uint8_t card_id, Player player) noexcept;
    static void grant_event_ops_to_ctx(GameState& state, uint8_t base_ops, Player player) noexcept;

    // What the budget in the current frame is worth outside its bonus regions.
    static uint8_t plain_budget(const GameState& state, uint8_t op_card, Player player) noexcept;

    // The region-bonus ladder: the budget the current frame keeps once an Op is spent in
    // `country_id`. All Ops in Southeast Asia keeps both bonuses; elsewhere in Asia forfeits
    // Vietnam Revolts; outside Asia forfeits the China Card's as well. The current budget where
    // no bonus is in play, or where the frame keeps no budget (pending_ops_value 0).
    static uint8_t budget_if_spent_in(const GameState& state, Player player, uint8_t country_id) noexcept;

    // Clears every target in `mask_84` whose cost -- its Influence cost, or 1 for a
    // realignment roll -- does not fit the budget that spending there keeps. Withdrawing a bonus
    // only after the Op was spent let the last one go anywhere: with the China Card under
    // Vietnam Revolts, five Ops in Southeast Asia and the sixth in India, or the final bonus
    // Op's realignment roll in Iran.
    static void mask_region_budget(const GameState& state, Player player, bool influence, uint8_t* mask_84) noexcept;

    // Influence Placement
    static bool can_place_influence(const GameState& state, Player p, uint8_t country_id) noexcept;
    static uint8_t get_influence_cost(const GameState& state, Player p, uint8_t country_id) noexcept;
    static bool place_influence(GameState& state, Player p, uint8_t country_id) noexcept;

    // Coup Attempts
    static bool can_coup(const GameState& state, Player p, uint8_t country_id) noexcept;

    // can_place_influence and can_coup for all 84 countries at once, as masks (country i is bit
    // i % 64 of word i / 64): the same functions, called in a loop in their own translation unit,
    // where the compiler can inline them and lift what does not depend on the country. For the
    // observation, which asks both for every country from both sides.
    static std::array<uint64_t, 2> placeable_countries(const GameState& state, Player p) noexcept;
    static std::array<uint64_t, 2> coupable_countries(const GameState& state, Player p) noexcept;
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
