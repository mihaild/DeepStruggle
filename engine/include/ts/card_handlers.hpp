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
    static bool trigger_event(GameState& state, uint8_t card_id, Player player) noexcept;

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

    // The set of cards Missile Envy may take from `giver`: the non-scoring ones of highest
    // Ops, valued as the giver would play them (Containment, Brezhnev Doctrine and Red
    // Scare/Purge count; Vietnam Revolts does not). A property of that hand, recomputed wherever it is asked -- the trigger, the legal
    // mask and the validation all use these -- so there is no stored list to fall out of step
    // with the hand it describes. `highest_takeable_ops` returns 0 when there is nothing to
    // take. Both skip ctx().resolving_card, since the engine leaves an Ops card in its owner's
    // hand until the play finishes and the event would otherwise find the card in front of it.
    static uint8_t highest_takeable_ops(const GameState& state, Player giver) noexcept;
    static bool missile_envy_may_take(const GameState& state, uint8_t card, Player giver,
                                      uint8_t best_ops) noexcept;

    // Where a card goes once it has been played.
    //
    // One implementation for what used to be eleven copies that had drifted apart. A starred
    // card leaves the game only when its Event is *implemented* (rules.md 280-282), so
    // `event_occurred` -- which must come from event_has_effect, evaluated BEFORE the Event runs
    // -- is what separates removal from a discard. Kitchen Debates is skipped because its
    // handler sets its own location, the outcome being known only there.
    static void relocate_played_card(GameState& state, uint8_t card, bool event_occurred,
                                     Handover handover) noexcept;

    // Flower Power: "The USSR receives 2 VP for every US played War card, used for Operations or
    // an Event". Called from commit_card_to_ops, where the US spends a card's own Operations --
    // a play for Ops, event-first or ops-first -- so the charge has one definition. A war card
    // lent to UN Intervention is not charged (see the UN Intervention handler), and a war card
    // played as its Event is charged where the war is rolled. Returns false if the charge ended
    // the game, and the caller must stop.
    static bool charge_flower_power_for_ops(GameState& state, Player p, uint8_t card) noexcept;

    // UN Intervention was played as an Event: U-2 Incident's rider, "if UN Intervention is
    // played later this turn, either by the US or the USSR, the USSR receives an additional 1
    // VP". Every route that plays UN Intervention calls this. Returns false if the game ended.
    static bool pay_u2_incident_rider(GameState& state) noexcept;

private:
    // The two entry points above without the record: each public one runs its counterpart and
    // adds what it did to VP and DEFCON to state.last_event_tracks.
    static bool run_trigger_event(GameState& state, uint8_t card_id, Player player) noexcept;
    static bool run_event_step(GameState& state, const MicroAction& action) noexcept;
};

} // namespace ts
