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
    // For display only, never in play: while `pause_between_headlines` is set, the headline phase
    // stops where the first headline is done and the second not begun, and `paused_between_headlines`
    // says it did. A copy of the game stepped this way shows what the first headline did alone --
    // its VP, which can end the game on their own (20 VP) before the second resolves. Off, the
    // engine is unchanged. Only the browser build sets it (ts_first_headline_json); per thread, so a
    // native harness stepping games on many threads never sees another's.
    static thread_local bool pause_between_headlines;
    static thread_local bool paused_between_headlines;
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

private:
    // The step itself; `step` wraps it with P43's canonical-run bookkeeping (ts/canonical.hpp).
    [[nodiscard]] static bool step_impl(GameState& state, const MicroAction& action, const uint8_t* legal) noexcept;
};

} // namespace ts
