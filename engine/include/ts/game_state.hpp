#pragma once
#include <cstdint>
#include <array>
#include <string_view>
#include <type_traits>
#include "types.hpp"
#include "constants.hpp"

namespace ts {

enum class RollType : uint8_t {
    NONE = 0,
    COUP = 1,
    REALIGNMENT = 2,
    SPACE_RACE = 3,
    WAR_EVENT = 4,
    OLYMPIC_GAMES = 5,
    SUMMIT = 6,
    TRAP_ESCAPE = 7,
    // Not a die at all: the turn's cleanup, parked on the chance-node machinery so it becomes a
    // step of its own rather than happening inside the action round that ended the turn. Two
    // reasons. The score the log narrates for a turn's last action round is the score *before*
    // cleanup pays the Military Operations deficit, and with cleanup folded into the same step
    // there was no moment at which the engine held that number, so the comparison had to be
    // skipped -- one entry in eight going unchecked. And a game decided during an action round
    // must not have cleanup applied to it afterwards: awarding the deficit at 20 VP moved the
    // score back off 20 and un-won a won game.
    TURN_CLEANUP = 8
};

// Structured record of a die roll event occurring within a step (16 bytes)
struct alignas(2) DieRollRecord {
    RollType type = RollType::NONE;
    Player roller = Player::NONE;
    uint8_t card_id = 0;
    uint8_t country_id = 255;  // 255 if not country-targeted
    uint8_t roll1 = 0;         // Primary / Active / US / Sponsor roll (1..6)
    int8_t mod1 = 0;           // Primary modifier (Ops / adjacency / dom)
    uint8_t roll2 = 0;         // Opponent / USSR roll (1..6, 0 if none)
    int8_t mod2 = 0;           // Opponent modifier
    bool success = false;
    int8_t net_delta = 0;      // Influence removed or VP awarded
    uint8_t padding[6] = {0};
};
static_assert(sizeof(DieRollRecord) == 16, "DieRollRecord must be exactly 16 bytes");
static_assert(std::is_trivially_copyable_v<DieRollRecord>, "DieRollRecord must be trivially copyable");

// Single Micro-Action Token in rolling history (16 bytes, 16-byte aligned)
struct alignas(16) ActionToken {
    Player     acting_player;        // US or USSR
    ActionType action_type;          // Micro-action executed
    uint8_t    card_id;              // Primary card associated with action (0 if N/A)
    uint8_t    secondary_card_id;    // Paired card (e.g. UN Intervention paired card) (0 if N/A)
    uint8_t    target_id;            // Target country index (255 if N/A)
    uint8_t    ops_value;            // Effective Ops used (0..5)
    uint8_t    die_roll;             // Die roll result (0 if no roll)
    int8_t     us_inf_delta;         // Net US influence delta at target
    int8_t     ussr_inf_delta;       // Net USSR influence delta at target
    uint8_t    defcon_after;         // Global DEFCON level after action
    uint8_t    padding[5];
};
static_assert(sizeof(ActionToken) == 16, "ActionToken must be exactly 16 bytes");

// Rolling 16-step action history ring buffer
struct ActionHistoryBuffer {
    std::array<ActionToken, 16> ring_buffer;
    uint8_t head_idx;                // Current insertion index (0..15)
    uint8_t total_count;             // Cumulative actions recorded in match

    inline void record(const ActionToken& token) noexcept {
        ring_buffer[head_idx] = token;
        head_idx = (head_idx + 1) & 15;
        if (total_count < 255) total_count++;
    }
};

// Aggregated operational metrics accumulated during current turn
struct TurnAggregates {
    // Index 0 = US, Index 1 = USSR; 6 regions (Europe, Asia, ME, Africa, CA, SA)
    std::array<std::array<uint8_t, 6>, 2> ops_spent_by_region;
    std::array<std::array<uint8_t, 6>, 2> coups_by_region;
    std::array<std::array<uint8_t, 6>, 2> realignments_by_region;

    std::array<uint8_t, 2> headlines_played;       // [0]=US Headline, [1]=USSR Headline
    std::array<uint8_t, 2> space_attempts;         // Space attempts executed this turn

    inline void clear() noexcept {
        ops_spent_by_region = {};
        coups_by_region = {};
        realignments_by_region = {};
        headlines_played = {};
        space_attempts = {};
    }
};

// Influence state per country node (84 countries)
struct alignas(2) CountryState {
    uint8_t us_influence;
    uint8_t ussr_influence;

    inline uint8_t get_influence(Player p) const noexcept {
        if (p == Player::US) return us_influence;
        if (p == Player::USSR) return ussr_influence;
        return 0;
    }

    inline void add_influence(Player p, uint8_t amt) noexcept {
        if (p == Player::US) {
            uint32_t sum = static_cast<uint32_t>(us_influence) + amt;
            us_influence = static_cast<uint8_t>(sum > 255 ? 255 : sum);
        } else if (p == Player::USSR) {
            uint32_t sum = static_cast<uint32_t>(ussr_influence) + amt;
            ussr_influence = static_cast<uint8_t>(sum > 255 ? 255 : sum);
        }
    }

    inline void remove_influence(Player p, uint8_t amt) noexcept {
        if (p == Player::US) {
            us_influence = (amt >= us_influence) ? 0 : static_cast<uint8_t>(us_influence - amt);
        } else if (p == Player::USSR) {
            ussr_influence = (amt >= ussr_influence) ? 0 : static_cast<uint8_t>(ussr_influence - amt);
        }
    }
};

// Single Frame of the Decision State Machine
struct alignas(64) DecisionContext {
    Player       decision_player;       // Player making current micro-decision
    DecisionType decision_type;         // Active primitive decision type
    OpMode       op_mode;               // Active OpMode (INFLUENCE=0, COUP=1, REALIGN=2)
    uint8_t      pending_op_card;       // Card ID pending Op resolution
    uint8_t      pending_ops_value;     // Effective Ops available for pending Op mode
    uint8_t      remaining_steps;       // Ops points or node selections remaining in sequence (0..7)
    uint8_t      max_per_country;       // Maximum influence placements/removals allowed per country
    uint8_t      allow_early_stop;      // 1 if CONFIRM_DONE (0x80) is legal, 0 otherwise
    uint8_t      resolving_card;        // Card ID currently executing (1..110)
    uint8_t      timing_branch;         // 0 = OPS_FIRST, 1 = EVENT_FIRST, 255 = NONE

    // Transient tracking across sub-actions (cleared when resolving_card finishes)
    std::array<uint64_t, 2> start_influence_nodes; // Bitmask of countries with friendly influence at start of Op
    std::array<uint64_t, 2> visited_nodes;         // 128-bit bitmask of nodes already modified
    std::array<uint8_t, 84> node_counts;           // Placements/removals per node during this event
    uint8_t                 suppress_op_card_event; // 1 = do not fire pending_op_card's event
    uint8_t                 event_granted_ops;      // 1 = Ops came from an event, not a card play

    // The chance node about to resolve, named rather than packed into temp_cards.
    //
    // Forced dice deliberately do NOT live here. A forced roll is a test and replayer
    // affordance, not game state, and the ROLL_DIE action already carries two: `primary_id` is
    // the acting player's die and `secondary_id` the opponent's. Storing them per-side here and
    // reading them per-actor at the chance node is what made a USSR realignment read its two
    // dice swapped; with nothing stored there is nothing to swap.
    RollType pending_roll;    // NONE when no chance node is staged
    uint8_t  roll_target;     // country id; a card id for SPACE_RACE and TRAP_ESCAPE
    Player   roll_actor;      // who rolls; NONE means the phasing player

    // Which stage a multi-stage event is in: CHE's second coup, De-Stalinization's placement
    // phase. Both used to fake this -- CHE by storing `country + 1` in a card slot,
    // De-Stalinization by reading `max_per_country == 0` -- for want of a field to put it in.
    uint8_t  event_stage;

    uint8_t                 pad[1];

    inline void set_start_influence(uint8_t node) noexcept {
        if (node < 64) start_influence_nodes[0] |= (1ULL << node);
        else if (node < 84) start_influence_nodes[1] |= (1ULL << (node - 64));
    }

    inline bool has_start_influence(uint8_t node) const noexcept {
        if (node < 64) return (start_influence_nodes[0] & (1ULL << node)) != 0;
        if (node < 84) return (start_influence_nodes[1] & (1ULL << (node - 64))) != 0;
        return false;
    }

    inline void mark_visited(uint8_t node) noexcept {
        if (node < 64) {
            visited_nodes[0] |= (1ULL << node);
        } else if (node < 84) {
            visited_nodes[1] |= (1ULL << (node - 64));
        }
    }

    inline bool is_visited(uint8_t node) const noexcept {
        if (node < 64) {
            return (visited_nodes[0] & (1ULL << node)) != 0;
        } else if (node < 84) {
            return (visited_nodes[1] & (1ULL << (node - 64))) != 0;
        }
        return false;
    }
};

// =============================================================================
// Complete Unified Game State
// =============================================================================
struct alignas(64) GameState {
    // -------------------------------------------------------------------------
    // 1. Global Tracks
    // -------------------------------------------------------------------------
    int8_t  victory_points;            // Range: [-20, +20] (+20 = US win, -20 = USSR win)
    uint8_t defcon;                    // Range: [1, 5]
    uint8_t us_mil_ops;                // Range: [0, 5]
    uint8_t ussr_mil_ops;              // Range: [0, 5]
    uint8_t us_space_track;            // Range: [0, 8]
    uint8_t ussr_space_track;          // Range: [0, 8]
    uint8_t turn;                      // Range: [1, 10]
    uint8_t action_round;              // Range: [1, 8] (8 with North Sea Oil)

    // -------------------------------------------------------------------------
    // 2. Phasing, Priority & Forced Plays
    // -------------------------------------------------------------------------
    Player  phasing_player;            // Active player executing the action round
    uint8_t headline_us_card;          // Stored US headline card ID (1..110, 0 if NONE)
    uint8_t headline_ussr_card;        // Stored USSR headline card ID (1..110, 0 if NONE)
    uint8_t headline_first_card;       // Card ID resolving 1st in headline (1..110, 0 if NONE)
    uint8_t headline_second_card;      // Card ID resolving 2nd in headline (1..110, 0 if NONE)
    uint8_t headline_stage;            // 0 = choosing, 1 = resolving 1st, 2 = resolving 2nd, 3 = done
    Player  headline_first_owner;      // Player who selected 1st headline card
    Player  headline_second_owner;     // Player who selected 2nd headline card
    Phase   current_phase;             // Active game phase
    Player  forced_card_player;        // Player forced to play specific card (Missile Envy)
    uint8_t forced_card_id;            // Card ID forced on next AR (49 for Missile Envy, 0 if NONE)
    uint8_t defcon_dropped_to_2; // 1 if DEFCON has reached 2 since the phase began (NORAD).
                                 // Cleared when an action round ends and when the headline
                                 // does, so a headline's drop cannot claim a later round.
    DieRollRecord last_roll;           // Structured record of die roll event occurring in current step
    uint8_t last_die_roll;             // Backwards-compat: result of most recent die roll (1..6, 0 if none)
    uint8_t last_opp_die_roll;         // Backwards-compat: opponent die roll

    // -------------------------------------------------------------------------
    // 3. China Card Status
    // -------------------------------------------------------------------------
    Player  china_card_holder;         // US or USSR
    uint8_t china_card_playable;       // 1 if face up / playable, 0 if face down

    // -------------------------------------------------------------------------
    // 5. Persistent Board Flags & Parametric Effects (64-bit bitfield)
    // -------------------------------------------------------------------------
    uint64_t persistent_effects;

    // -------------------------------------------------------------------------
    // 6. Card Registry Locations (Index 0 unused, 1..110)
    // -------------------------------------------------------------------------
    std::array<CardLocation, 111> card_locations;

    // -------------------------------------------------------------------------
    // 7. Board Graph Nodes (84 Countries)
    // -------------------------------------------------------------------------
    std::array<CountryState, 84> countries;

    // -------------------------------------------------------------------------
    // 8. Re-entrant Decision State Machine Stack (Max Depth 3)
    // -------------------------------------------------------------------------
    std::array<DecisionContext, 3> ctx_stack;
    uint8_t                       ctx_stack_depth; // 0 = base context, 1..2 = nested

    inline DecisionContext& ctx() noexcept {
        return ctx_stack[ctx_stack_depth];
    }
    inline const DecisionContext& ctx() const noexcept {
        return ctx_stack[ctx_stack_depth];
    }
    inline void push_context() noexcept {
        if (ctx_stack_depth < 2) {
            ctx_stack_depth++;
            ctx_stack[ctx_stack_depth] = DecisionContext{};
        }
    }
    inline void pop_context() noexcept {
        if (ctx_stack_depth > 0) {
            ctx_stack_depth--;
        }
    }

    // -------------------------------------------------------------------------
    // 9. Play History & Turn Aggregates
    // -------------------------------------------------------------------------
    ActionHistoryBuffer action_history;
    TurnAggregates      turn_aggregates;

    // -------------------------------------------------------------------------
    // 10. Deterministic PRNG State (SplitMix64)
    // -------------------------------------------------------------------------
    uint64_t rng_state;

    // Fast helper methods
    inline bool has_flag(uint64_t flag) const noexcept {
        return (persistent_effects & flag) != 0;
    }
    inline void set_flag(uint64_t flag) noexcept {
        persistent_effects |= flag;
    }
    inline void clear_flag(uint64_t flag) noexcept {
        persistent_effects &= ~flag;
    }

    // Space race attempts state tracking (stored in persistent_effects bits 43..46)
    inline uint8_t get_space_turns_used(Player p) const noexcept {
        if (p == Player::US) {
            if (has_flag(effect_bits::SPACE_US_ATTEMPT_2)) return 2;
            if (has_flag(effect_bits::SPACE_US_ATTEMPT_1)) return 1;
            return 0;
        } else if (p == Player::USSR) {
            if (has_flag(effect_bits::SPACE_USSR_ATTEMPT_2)) return 2;
            if (has_flag(effect_bits::SPACE_USSR_ATTEMPT_1)) return 1;
            return 0;
        }
        return 0;
    }

    inline void record_space_attempt(Player p) noexcept {
        if (p == Player::US) {
            if (!has_flag(effect_bits::SPACE_US_ATTEMPT_1)) {
                set_flag(effect_bits::SPACE_US_ATTEMPT_1);
            } else {
                set_flag(effect_bits::SPACE_US_ATTEMPT_2);
            }
        } else if (p == Player::USSR) {
            if (!has_flag(effect_bits::SPACE_USSR_ATTEMPT_1)) {
                set_flag(effect_bits::SPACE_USSR_ATTEMPT_1);
            } else {
                set_flag(effect_bits::SPACE_USSR_ATTEMPT_2);
            }
        }
    }

    inline void set_space_turns_used(Player p, uint8_t count) noexcept {
        if (p == Player::US) {
            clear_flag(effect_bits::SPACE_US_ATTEMPT_1 | effect_bits::SPACE_US_ATTEMPT_2);
            if (count >= 1) set_flag(effect_bits::SPACE_US_ATTEMPT_1);
            if (count >= 2) set_flag(effect_bits::SPACE_US_ATTEMPT_2);
        } else if (p == Player::USSR) {
            clear_flag(effect_bits::SPACE_USSR_ATTEMPT_1 | effect_bits::SPACE_USSR_ATTEMPT_2);
            if (count >= 1) set_flag(effect_bits::SPACE_USSR_ATTEMPT_1);
            if (count >= 2) set_flag(effect_bits::SPACE_USSR_ATTEMPT_2);
        }
    }
};

static_assert(std::is_trivially_copyable_v<GameState>, "GameState must be trivially copyable");
static_assert(sizeof(GameState) <= 4096, "GameState exceeds 4 KB L1/L2 footprint limit");

// Neural Network / RL Observation Buffer -- the legacy layout, 4293 floats.
//
// Kept exactly as it was so checkpoints trained against it keep loading. Card slot 0 merges the
// draw deck, cards not yet in the game, and the whole of the opponent's hand; nothing in it can
// express what the opponent is known to hold. ObservationBufferV21 below is the layout that can.
struct alignas(64) ObservationBuffer {
    float board_features[84 * 28];     // 84 countries x 28 node features
    float card_features[110 * 12];     // 110 cards x 12 status features
    float global_features[76];         // Global tracks, turn, AR, flags
    float history_sequence[16 * 32];   // 16-step action history projection
    float turn_aggregates[32];         // Operational counts by region & turn
    float active_player;               // +1.0 (US), -1.0 (USSR)
};

// Observation layout v2.1 -- 3891 floats. Identical to the legacy buffer except that each card
// carries 13 status features instead of 12, splitting two things the old slot 0 could not:
//
//   slot 2  the opponent holds this card *and I know it* -- previously indistinguishable from
//           a card sitting in the deck, though the engine has always known the difference;
//   slot 7  the card is not in the game yet (a later era, or an unused optional), previously
//           merged with the draw deck even though which one it is has never been a secret.
//
// Slot 0 still merges the draw deck with the *unknown* part of the opponent's hand, and that is
// not an oversight: those two are exactly what the observer cannot tell apart, and separating
// them would hand the network the hidden information the game is played to discover.
struct alignas(64) ObservationBufferV21 {
    float board_features[84 * 28];
    float card_features[110 * 13];
    float global_features[76];
    // No history_sequence. ActionHistoryBuffer::record() is called nowhere, so in the legacy
    // layout those 512 floats are a constant zero vector that the network still spends a Conv1d,
    // a 512->128 projection and 128 of its 768 fusion inputs encoding -- 4.17% of its parameters
    // learning a bias. The buffer stays in GameState for now; only the observation drops it.
    float turn_aggregates[32];
    float active_player;
};

struct alignas(64) ObservationBufferV22 {
    // 26, not 28: can_my_realign and can_opp_realign are dropped. can_realign is exactly
    // can_coup_or_realign, and can_coup is that plus "The Reformer blocks USSR coups in Europe",
    // so the two pairs are identical outside that one late-war card -- 168 floats for a case that
    // effectively does not arise. can_my_coup / can_opp_coup carry the whole signal.
    float board_features[84 * 26];
    float card_features[110 * 14];   // v2.1's 13, plus "this is the card being played"
    float global_features[101];      // v2.1's 72 written ones, plus the 29-float context
    // No turn_aggregates and no active_player. Twenty of those 32 floats had no writer anywhere
    // in the engine, and ColdWarNetV2 never sliced any of them -- its forward pass stops at the
    // global block -- so the 12 that were written never reached a network either. Which side is
    // to move is already in global_features[61] (I_AM_US) and in every perspective-relative
    // feature around it.
};

// Card status slots, shared by both layouts where they overlap. The v2.1 names are the authority;
// the legacy layout uses 0..6 with the same meanings, lacks KNOWN_OPPONENT_HAND and UNAVAILABLE,
// and starts its property block at 7 rather than 8.
namespace card_slots {
    constexpr size_t DECK_OR_HIDDEN       = 0; // draw deck, or an opponent card I have not seen
    constexpr size_t MY_HAND              = 1;
    constexpr size_t KNOWN_OPPONENT_HAND  = 2; // v2.1 only
    constexpr size_t DISCARD              = 3;
    constexpr size_t REMOVED              = 4;
    constexpr size_t ONGOING              = 5;
    constexpr size_t PEEKED               = 6;
    constexpr size_t NOT_IN_GAME          = 7; // v2.1 only
    constexpr size_t ACTIVE_CARD          = 13; // v2.2 only: resolving_card or pending_op_card
    constexpr size_t LEGACY_FEATURES      = 12;
    constexpr size_t V21_FEATURES          = 13;
    constexpr size_t V22_FEATURES          = 14;
    constexpr size_t LEGACY_PROPERTY_BASE = 7;
    constexpr size_t V21_PROPERTY_BASE     = 8;
}

// The number of floats a consumer reads, which is NOT sizeof(buffer)/sizeof(float): both
// buffers are alignas(64) and so are padded past their last member. Everything that copies an
// observation out copies exactly this many floats and must never use sizeof for it.
constexpr size_t OBS_SIZE_LEGACY = 84 * 28 + 110 * card_slots::LEGACY_FEATURES + 76
                                 + 16 * 32 + 32 + 1;
constexpr size_t OBS_SIZE_V21 = 84 * 28 + 110 * card_slots::V21_FEATURES + 76 + 32 + 1;
constexpr size_t OBS_SIZE_V22 = 84 * 26 + 110 * card_slots::V22_FEATURES + 101;
constexpr size_t V22_BOARD_FEATURES = 26;

// Where the decision context sits inside v2.2's global block. Named because an off-by-one here is
// invisible: every one of these is a legitimate 0.0 most of the time.
// Engine feature flags. A run records which of these it trained under, and anything missing from
// a record is false -- so an old checkpoint keeps the behaviour it learned without anyone having
// to remember what that was. Width is not a version number: two layouts of the same width can
// differ in content, and that difference is invisible to every check we have.
namespace obs_flags {
    constexpr uint32_t NONE          = 0u;
    //: Retired, and kept only so the bit is not reused with a different meaning while
    //: checkpoints recording it by name still exist. It made a card being decided about visible
    //: to the decider, which Grain Sales needed because it showed a card without moving it. The
    //: card now goes to PEEKED_TEMP and reaches the observation through the ordinary location
    //: chain, so there is nothing left for the flag to do.
    constexpr uint32_t STAGED_CARDS  = 1u << 0;
}

namespace ctx_slots {
    // 72, not 76: legacy reserved 76 globals and only ever wrote 0..71, so v2.2 reclaims
    // the four it left blank rather than carrying them forward as padding.
    constexpr size_t BASE                 = 72;
    constexpr size_t DECISION_TYPE        = BASE +  0; // 8 one-hot: NONE..ROLL_DIE
    constexpr size_t OP_MODE              = BASE +  8; // 3 one-hot: INFLUENCE, COUP, REALIGN
    constexpr size_t REMAINING_STEPS      = BASE + 11;
    constexpr size_t PENDING_OPS_VALUE    = BASE + 12;
    constexpr size_t MAX_PER_COUNTRY      = BASE + 13;
    constexpr size_t ALLOW_EARLY_STOP     = BASE + 14;
    constexpr size_t TIMING_OPS_FIRST     = BASE + 15;
    constexpr size_t TIMING_EVENT_FIRST   = BASE + 16;
    constexpr size_t EVENT_GRANTED_OPS    = BASE + 17;
    constexpr size_t SUPPRESS_OP_EVENT    = BASE + 18;
    constexpr size_t TEMP_CARD_COUNT      = BASE + 19;
    // Headline: which stage, and whose card resolves when. Space box 4 makes the order a
    // mechanic, and none of headline_stage / headline_first_owner was read before.
    constexpr size_t HEADLINE_STAGE       = BASE + 20;
    constexpr size_t HEADLINE_FIRST_MINE  = BASE + 21;
    constexpr size_t HEADLINE_SECOND_MINE = BASE + 22;
    // Chernobyl's forbidden region as a one-hot, all zero when it is not in play. The three raw
    // bits at 40..42 encode the same index in binary and stay in the effect dump; this is the
    // form the network can separate.
    constexpr size_t CHERNOBYL_REGION     = BASE + 23; // 6 wide
    constexpr size_t COUNT                = 29;
}
static_assert(OBS_SIZE_LEGACY == 4293, "the legacy observation width is a checkpoint contract");
static_assert(OBS_SIZE_V21 == 3891,
              "v2.1 adds one card feature and drops the never-written history block");
static_assert(sizeof(ObservationBuffer) >= OBS_SIZE_LEGACY * sizeof(float),
              "the buffer must hold every float a reader will copy out of it");
static_assert(sizeof(ObservationBufferV21) >= OBS_SIZE_V21 * sizeof(float),
              "the buffer must hold every float a reader will copy out of it");
// v2.2 is the only layout still being changed -- the staged-card flag landed against it, and
// arms F and F2 were evaluated against a v2.2 that had gained a feature after they trained on
// it -- and it was the one layout with neither guard. Add or remove a feature and the width
// changes silently, every v2.2 checkpoint misreads its input, and the build stays green.
static_assert(OBS_SIZE_V22 == 3825,
              "v2.2 is a checkpoint contract: 84*26 board + 110*14 card + 101 global");
static_assert(sizeof(ObservationBufferV22) >= OBS_SIZE_V22 * sizeof(float),
              "the buffer must hold every float a reader will copy out of it");

// Missile Envy moves itself into the opponent's hand, who must play it on their next
// action round. The generic post-play cleanup would discard it straight back out of that
// hand, stranding forced_card_id on a card nobody holds -- and the action mask, which only
// forces a card that is actually in hand, then drops the forced play without a trace.
// When the opponent held nothing eligible the card was never handed over, so it discards
// normally; checking the location rather than the id alone keeps both cases right.
inline bool keeps_own_card_location(const GameState& state, uint8_t card) noexcept {
    if (card != card_ids::MISSILE_ENVY) return false;
    return is_in_any_hand(state.card_locations[card]);
}

// Everything `p` is holding becomes public to the opponent.
//
// Only the cards held *now*. Knowledge does not attach to the player, it attaches to the cards:
// what the opponent has seen, they have seen, and anything drawn afterwards is hidden again --
// which falls out of this touching locations rather than setting a flag. Cards leaving the hand
// clear themselves, since the new location overwrites the old one.
inline void reveal_hand(GameState& state, Player p) noexcept {
    if (p == Player::NONE) return;
    for (uint8_t i = 1; i <= 110; ++i) {
        if (in_hand_of(state.card_locations[i], p)) {
            state.card_locations[i] = revealed(state.card_locations[i]);
        }
    }
}

// Both hands become public. Used where the deduction is symmetric -- see the draw deck running
// out in StateMachine::deal_cards_to_hands.
inline void reveal_both_hands(GameState& state) noexcept {
    reveal_hand(state, Player::US);
    reveal_hand(state, Player::USSR);
}

// Cuban Missile Crisis can be paid off at any time. Modelled as the head of the payer's own
// action round, and as part of a coup they make -- the two moments a player is doing something
// anyway -- rather than as a standing option on every decision in the game.
inline bool can_pay_off_cuban_missile_crisis(const GameState& state, Player p) noexcept {
    if (p == Player::US) {
        return state.has_flag(effect_bits::CMC_ACTIVE_USSR) &&
               (state.countries[countries::WEST_GERMANY].us_influence >= 2 ||
                state.countries[countries::TURKEY].us_influence >= 2);
    }
    if (p == Player::USSR) {
        return state.has_flag(effect_bits::CMC_ACTIVE_US) &&
               state.countries[countries::CUBA].ussr_influence >= 2;
    }
    return false;
}

} // namespace ts
