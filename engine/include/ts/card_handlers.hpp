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
    static bool trigger_event(GameState& state, uint8_t card_id, Player player, uint8_t forced_roll = 0) noexcept;

    // Handles a sub-decision for an ongoing card event.
    // Returns true when the entire event finishes and pops from context stack.
    static bool handle_event_step(GameState& state, const MicroAction& action) noexcept;

    // Fills legal action mask for the current event step
    static void get_event_action_mask(const GameState& state, uint8_t* mask_out, size_t* out_size) noexcept;

    // Prerequisite checks (e.g. NATO requires Marshall or Warsaw)
    static bool can_trigger_event(const GameState& state, uint8_t card_id, Player player) noexcept;

    // Would this card's Event actually do anything if it were fired right now?
    //
    // Distinct from can_trigger_event, which is the *mask* question -- may this card be chosen
    // as an Event -- and is answered for a card the player is about to select. This is the
    // *removal* question: a starred card leaves the game only when its Event is implemented
    // (rules.md 280-282), and an Event that is fired but cannot occur has not been implemented.
    // The two differ because some conditions live inside the handler rather than in the
    // prerequisite table; those handlers call this so the condition has one definition and
    // cannot drift from the removal decision that depends on it.
    //
    // Must be evaluated BEFORE trigger_event: an Event that fires sets flags, and asking
    // afterwards can read the answer the Event itself just produced.
    static bool event_has_effect(const GameState& state, uint8_t card_id, Player player) noexcept;

    // Whether this cleanup may sweep out a card an Event has just placed into a hand -- in
    // practice Missile Envy, which lives in the recipient's hand until their next action round.
    // Not a tidy Event/Operations split: every Event site and the headline's Ops cleanup respect
    // the handover, while the action round's Ops cleanup ignores it, because there the card
    // being spent *is* Missile Envy.
    enum class Handover : uint8_t { Respect, Ignore };

    // Where a card goes once it has been played.
    //
    // One implementation for what used to be eleven copies that had drifted apart. A starred
    // card leaves the game only when its Event is *implemented* (rules.md 280-282), so
    // `event_occurred` -- which must come from event_has_effect, evaluated BEFORE the Event runs
    // -- is what separates removal from a discard. Kitchen Debates is skipped because its
    // handler sets its own location, the outcome being known only there.
    static void relocate_played_card(GameState& state, uint8_t card, bool event_occurred,
                                     Handover handover) noexcept;
};

} // namespace ts
