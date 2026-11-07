#pragma once
// P43: canonical placement order (owner, 2026-11-07). A seat playing obs_features::CANONICAL_FIRST or
// CANONICAL_LAST places each placement run -- the setup's three, an Ops influence play, the verified
// event placement and removal runs below -- in non-decreasing rank of its order (most important
// first, or last; MapData::placement_rank is the important-last rank). Every board a run can reach
// then has one action sequence. A run that would strand points (points left, free order would have a
// target, none at or above the floor) releases its floor: the mask falls back to free order for that
// choice, and the run continues canonically from the country chosen. So the boards a run can reach
// are exactly free order's, never more and never fewer.
#include "ts/constants.hpp"
#include "ts/game_state.hpp"
#include "ts/map_data.hpp"

#include <cstdint>

namespace ts::canonical {

// The event cards whose placement or removal runs follow the canonical order: each verified to reach
// exactly free order's boards (tools probe; engine/tests/test_canonical_placement.cpp). Any other
// event's runs stay in free order.
inline constexpr uint8_t EVENT_CARDS[] = {
    card_ids::SOCIALIST_GOVERNMENTS, card_ids::COMECON, card_ids::WARSAW_PACT_FORMED,
    card_ids::MARSHALL_PLAN, card_ids::SUEZ_CRISIS, card_ids::EAST_EUROPEAN_UNREST,
    card_ids::DECOLONIZATION, card_ids::DE_STALINIZATION, card_ids::COLONIAL_REAR_GUARDS,
    card_ids::OAS_FOUNDED, card_ids::THE_VOICE_OF_AMERICA, card_ids::LIBERATION_THEOLOGY,
};

inline bool event_covered(uint8_t card) noexcept {
    for (uint8_t c : EVENT_CARDS) {
        if (c == card) return true;
    }
    return false;
}

// 0: free order; otherwise the rule bit the seat plays.
inline uint32_t seat_order(const GameState& state, Player p) noexcept {
    if (seat_plays_rule(state, p, obs_features::CANONICAL_FIRST)) return obs_features::CANONICAL_FIRST;
    if (seat_plays_rule(state, p, obs_features::CANONICAL_LAST)) return obs_features::CANONICAL_LAST;
    return 0;
}

// A country's place in the seat's order, 0..83.
inline uint8_t rank_in(uint32_t order, uint8_t country) noexcept {
    const uint8_t last = MapData::placement_rank(country);
    return order == obs_features::CANONICAL_FIRST ? static_cast<uint8_t>(83 - last) : last;
}

// Is the decision pending in `state` a covered placement run, and in which order (0 when not)?
inline uint32_t run_order(const GameState& state) noexcept {
    const DecisionContext& c = state.ctx();
    if (c.decision_type != DecisionType::POINT_NODE) return 0;
    const uint32_t order = seat_order(state, c.decision_player);
    if (order == 0) return 0;
    if (state.current_phase == Phase::SETUP) return order;
    if (c.is_ops_influence_play()) return order;
    if (c.resolving_card != 0 && event_covered(c.resolving_card)) return order;
    return 0;
}

// What identifies one run: constant through it, different from the run before and after. The Ops
// budget (pending_ops_value) moves inside an Ops run when a region bonus is withdrawn, so it is part
// of the tag only in the setup, where it is the stage.
inline uint16_t run_tag(const GameState& state) noexcept {
    const DecisionContext& c = state.ctx();
    uint32_t h = static_cast<uint32_t>(state.ctx_stack_depth);
    h = h * 31u + static_cast<uint32_t>(c.decision_player);
    h = h * 31u + static_cast<uint32_t>(c.op_mode);
    h = h * 131u + c.resolving_card;
    h = h * 131u + c.pending_op_card;
    h = h * 31u + c.event_stage;
    h = h * 31u + (state.current_phase == Phase::SETUP ? 1u + c.pending_ops_value : 0u);
    return static_cast<uint16_t>((h ^ (h >> 16)) | 1u);   // never 0
}

// The floor the pending run has set, as a rank in its order (-1 for none): the rank of the last country
// placed in this run. A floor belongs to its run: the tag and the points left must still match.
inline int floor_rank(const GameState& state) noexcept {
    const DecisionContext& c = state.ctx();
    if (c.canon_floor == 0 || c.canon_tag != run_tag(state) || c.canon_left != c.remaining_steps) return -1;
    return static_cast<int>(c.canon_floor) - 1;
}

}  // namespace ts::canonical
