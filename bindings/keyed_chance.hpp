// Keyed chance: a world's randomness tied to the situation it is drawn in, not to how many draws
// came before (the playout batches' `keyed` option).
//
// Every random draw the engine makes -- a coup or realignment die, a war or space-race roll, a
// random discard, the next turn's deal -- comes from `state.rng_state` (SplitMix64). Sharing one
// stream between two continuations of a decision (the rollout root's `shared_dice`, the labeller's
// sibling slice) keeps them paired only until they consume it differently, which is almost at once:
// measured on P34's sibling slice, two candidates' results in one world correlate at 0.23, all of it
// from the shared deal. Here the stream is reset before every engine step to
// hash(world key, situation): two continuations that reach the same situation -- the same turn,
// action round, phasing side, phase, decision, staged roll with its target and roller, steps left
// and resolving card -- draw the same dice and the same cards, however they got there.
//
// Each world's draws are still a fresh random stream (the world key is), so a single continuation
// plays the same game distribution as before; only the coupling between continuations changes.
// No engine change: the engine reads rng_state, and this sets it.
#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>

#include "ts/card_data.hpp"
#include "ts/constants.hpp"
#include "ts/engine.hpp"
#include "ts/game_state.hpp"
#include "ts/prng.hpp"

namespace ts::keyed {

inline uint64_t mix(uint64_t h, uint64_t v) noexcept {
    uint64_t z = h ^ (v + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2));
    return Prng::next_u64(z);
}

// The situation a step is taken in, hashed with the world's key.
inline uint64_t situation(const GameState& s, uint64_t world_key) noexcept {
    const auto& c = s.ctx();
    uint64_t h = world_key;
    h = mix(h, (uint64_t{s.turn} << 24) | (uint64_t{s.action_round} << 16) |
               (uint64_t(static_cast<uint8_t>(s.phasing_player)) << 8) | uint64_t(s.current_phase));
    // Player::USSR is -1: narrow to its byte before widening so it cannot fill adjacent fields.
    h = mix(h, (uint64_t(c.decision_type) << 40) | (uint64_t(static_cast<uint8_t>(c.decision_player)) << 32) |
               (uint64_t(c.pending_roll) << 24) | (uint64_t{c.roll_target} << 16) |
               (uint64_t(static_cast<uint8_t>(c.roll_actor)) << 8) | uint64_t{c.remaining_steps});
    h = mix(h, (uint64_t{c.resolving_card} << 24) | (uint64_t{c.pending_op_card} << 16) |
               (uint64_t{s.ctx_stack_depth} << 8) | uint64_t{s.headline_stage});
    return h;
}

// The turn-end chance node, whose step deals the next turn's cards (rollout::at_turn_cleanup's test).
inline bool at_turn_end(const GameState& s) noexcept {
    return s.current_phase != Phase::GAME_OVER && s.victory_points < 20 && s.victory_points > -20 &&
           s.ctx().decision_player == Player::NONE && s.ctx().decision_type == DecisionType::ROLL_DIE &&
           s.ctx().pending_roll == RollType::TURN_CLEANUP;
}

// The next deal from one deck order per world: each card a fixed priority hash(world key, card), and
// each player dealt the deck's cards in priority order (the USSR first, as the engine deals) through
// the engine's forced-deal slots. Keyed draws alone pick by index among the deck's cards, so two
// decks differing by one card deal differently from there on; by priority, they deal the same cards
// except where that card is concerned. The deck is predicted as the deal will see it: today's deck,
// plus the era's cards at turns 4 and 8 (StateMachine::add_era_cards_to_deck). A deal the deck cannot
// cover -- a reshuffle mid-deal -- is left to the keyed stream.
inline void force_deal(GameState& s, uint64_t world_key) noexcept {
    const int next = s.turn + 1;
    if (next > 10) return;
    const WarEra era_added = next == 4 ? WarEra::MID : next == 8 ? WarEra::LATE : WarEra::EARLY;
    uint8_t deck[111];
    uint64_t prio[111];
    size_t n = 0;
    int hand_us = 0, hand_ussr = 0;
    for (uint8_t i = 1; i <= 110; ++i) {
        const CardLocation loc = s.card_locations[i];
        hand_us += in_hand_of(loc, Player::US) ? 1 : 0;
        hand_ussr += in_hand_of(loc, Player::USSR) ? 1 : 0;
        const bool joins = era_added != WarEra::EARLY && i != card_ids::THE_CHINA_CARD &&
                           CardData::get_card(i).era == era_added;
        if (loc == CardLocation::DRAW_DECK || joins) {
            deck[n] = i;
            prio[n] = mix(world_key ^ 0xDEC0DEDULL, i);
            ++n;
        }
    }
    // One card more than each hand lacks, as a reserve: the Space Walk privilege may discard a card
    // after this, before the deal, and that hand then takes one more -- from its own reserve, never
    // from a card meant for the other hand. An unused reserve stays in the deck.
    const int target = next <= 3 ? 8 : 9;
    const int need_ussr = std::min<int>(GameState::FORCED_DEAL_MAX, std::max(0, target - hand_ussr) + 1);
    const int need_us = std::min<int>(GameState::FORCED_DEAL_MAX, std::max(0, target - hand_us) + 1);
    if (need_ussr + need_us > static_cast<int>(n)) return;
    for (size_t a = 1; a < n; ++a) {          // insertion sort by priority: n <= 110, once a turn
        for (size_t b = a; b > 0 && prio[b] < prio[b - 1]; --b) {
            std::swap(prio[b], prio[b - 1]);
            std::swap(deck[b], deck[b - 1]);
        }
    }
    for (int k = 0; k < need_ussr; ++k) s.forced_deal_ussr[k] = deck[k];
    for (int k = 0; k < need_us; ++k) s.forced_deal_us[k] = deck[need_ussr + k];
    s.forced_deal_ussr_count = static_cast<uint8_t>(need_ussr);
    s.forced_deal_us_count = static_cast<uint8_t>(need_us);
    s.forced_deal_ussr_pos = 0;
    s.forced_deal_us_pos = 0;
}

// Key the stream for the step about to be taken from `s` (and, at the turn's end, the deal).
inline void key(GameState& s, uint64_t world_key) noexcept {
    s.rng_state = situation(s, world_key);
    if (at_turn_end(s)) force_deal(s, world_key);
}

// Engine::auto_advance_step, one micro-step at a time with the stream keyed before each: the
// position the full drain would reach, every draw on the way keyed. Returns the steps taken.
inline size_t advance(GameState& s, uint64_t world_key) noexcept {
    size_t n = 0;
    for (; n < 4096; ++n) {
        if (Engine::is_terminal(s)) break;
        key(s, world_key);
        if (Engine::auto_advance_step(s, 1) == 0) break;
    }
    return n;
}

}  // namespace ts::keyed
