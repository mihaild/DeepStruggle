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
    // The same, with the legality check read from `legal` -- the flat mask of exactly this state,
    // as ActionMask::generate_flat_mask_212 wrote it -- instead of a mask generated here. Nothing
    // else differs. For callers that already hold that mask (CachedState, auto_advance_step), so
    // the check costs a lookup instead of a second generation. nullptr generates it, as above.
    [[nodiscard]] static bool step(GameState& state, const MicroAction& action,
                                   const uint8_t* legal) noexcept;

    // Phase Transitions
    static void start_turn(GameState& state) noexcept;
    static void advance_headline_step(GameState& state) noexcept;
    static void advance_after_ops(GameState& state) noexcept;
    static void advance_after_action_round(GameState& state) noexcept;
    static void finish_play_without_ops(GameState& state) noexcept;
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
