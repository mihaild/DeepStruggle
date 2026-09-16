#pragma once
#include <cstdint>
#include "types.hpp"
#include "game_state.hpp"
#include "micro_action.hpp"

namespace ts {

class StateMachine {
public:
    static void init_new_game(GameState& state, uint64_t seed) noexcept;
    [[nodiscard]] static bool step(GameState& state, const MicroAction& action) noexcept;

    // Phase Transitions
    static void start_turn(GameState& state) noexcept;
    static void advance_headline_step(GameState& state) noexcept;
    static void advance_after_ops(GameState& state) noexcept;
    static void advance_after_action_round(GameState& state) noexcept;
    static void offer_cuban_missile_payoff(GameState& state) noexcept;
    // Parks the turn's cleanup on a chance node so it resolves as a step of its own. See
    // RollType::TURN_CLEANUP.
    static void begin_turn_cleanup(GameState& state) noexcept;
    static void end_turn(GameState& state) noexcept;
    static void finish_end_turn(GameState& state) noexcept;

    // Card Deck Management
    static void add_era_cards_to_deck(GameState& state, WarEra era) noexcept;
    static void deal_cards_to_hands(GameState& state) noexcept;
    static void reshuffle_discard_into_draw(GameState& state) noexcept;
};

} // namespace ts
