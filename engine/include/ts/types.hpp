#pragma once
#include <cstdint>
#include <cstddef>
#include <type_traits>

namespace ts {

// Superpower identification
enum class Player : int8_t {
    USSR = -1,
    NONE = 0,
    US   = 1
};

inline constexpr Player get_opponent(Player p) noexcept {
    return static_cast<Player>(-static_cast<int8_t>(p));
}

inline constexpr size_t player_to_index(Player p) noexcept {
    // Returns 0 for US, 1 for USSR, 2 for NONE
    if (p == Player::US) return 0;
    if (p == Player::USSR) return 1;
    return 2;
}

inline constexpr Player index_to_player(size_t idx) noexcept {
    if (idx == 0) return Player::US;
    if (idx == 1) return Player::USSR;
    return Player::NONE;
}

// Major Game Phases
enum class Phase : uint8_t {
    SETUP,
    HEADLINE,
    ACTION_ROUND,
    INTERRUPT,
    DISCARD,
    END_TURN,
    GAME_OVER
};

// War Eras for deck additions and scoring rules
enum class WarEra : uint8_t {
    EARLY = 0,
    MID   = 1,
    LATE  = 2
};

// Card physical location registry.
//
// A hand is split four ways rather than two: whether the *other* player knows the card is
// there is part of the position, and it is not recoverable from anything else. A card cannot
// leave a hand secretly, so once its presence is public it stays public until it is played or
// discarded -- at which point its new location is public anyway. Storing that as a location
// rather than as a parallel "known" bitset means every one of the ~105 sites that moves a card
// destroys the knowledge automatically, instead of each having to remember to clear a bit.
//
// The holder always knows their own hand, so KNOWN can only mean "known to the non-holder"
// and one value per (holder, known) pair is enough. There is no second observer to track.
//
// The bare HAND_US / HAND_USSR names are deliberately gone. Every site that compared against
// them has to decide which variants it means, and deleting the names makes the compiler
// enumerate them instead of leaving thirty silent rules bugs -- see keeps_own_card_location
// in game_state.hpp for what one of those looks like in practice. Use in_hand_of(),
// known_to_opponent() and hand_of() below.
enum class CardLocation : uint8_t {
    UNAVAILABLE        = 0, // Not yet in the deck (future era or unintroduced optional card)
    DRAW_DECK          = 1,
    HAND_US_UNKNOWN    = 2, // In the US hand; the USSR has not seen it
    HAND_US_KNOWN      = 3, // In the US hand and the USSR knows it
    HAND_USSR_UNKNOWN  = 4, // In the USSR hand; the US has not seen it
    HAND_USSR_KNOWN    = 5, // In the USSR hand and the US knows it
    DISCARD_PILE       = 6,
    REMOVED_FROM_GAME  = 7,
    ONGOING_EVENT      = 8,
    PEEKED_TEMP        = 9,
    // Committed to the headline, face down, and no longer in the hand it came from. Both
    // headlines are played at once and only then resolved in order, so neither card is
    // holdable while the other resolves -- a card that reads a hand must not find it there.
    // Overwritten by the ordinary post-resolution cleanup, which sets the card's real
    // destination once its event is done.
    HEADLINE_COMMITTED = 10
};

// True for any of the four hand variants.
constexpr bool is_in_any_hand(CardLocation loc) noexcept {
    return loc == CardLocation::HAND_US_UNKNOWN || loc == CardLocation::HAND_US_KNOWN
        || loc == CardLocation::HAND_USSR_UNKNOWN || loc == CardLocation::HAND_USSR_KNOWN;
}

// True when the card is in `p`'s hand, whether or not the opponent knows.
constexpr bool in_hand_of(CardLocation loc, Player p) noexcept {
    if (p == Player::US) {
        return loc == CardLocation::HAND_US_UNKNOWN || loc == CardLocation::HAND_US_KNOWN;
    }
    if (p == Player::USSR) {
        return loc == CardLocation::HAND_USSR_UNKNOWN || loc == CardLocation::HAND_USSR_KNOWN;
    }
    return false;
}

// Whether the player who is *not* holding the card knows it is in that hand.
constexpr bool known_to_opponent(CardLocation loc) noexcept {
    return loc == CardLocation::HAND_US_KNOWN || loc == CardLocation::HAND_USSR_KNOWN;
}

// The hand location for `p`. `known` defaults false: a card entering a hand is hidden unless
// something made it public, so the default is the conservative one.
constexpr CardLocation hand_of(Player p, bool known = false) noexcept {
    if (p == Player::US) {
        return known ? CardLocation::HAND_US_KNOWN : CardLocation::HAND_US_UNKNOWN;
    }
    return known ? CardLocation::HAND_USSR_KNOWN : CardLocation::HAND_USSR_UNKNOWN;
}

// Whose hand it is, or NONE when it is not in one.
constexpr Player hand_holder(CardLocation loc) noexcept {
    if (loc == CardLocation::HAND_US_UNKNOWN || loc == CardLocation::HAND_US_KNOWN) {
        return Player::US;
    }
    if (loc == CardLocation::HAND_USSR_UNKNOWN || loc == CardLocation::HAND_USSR_KNOWN) {
        return Player::USSR;
    }
    return Player::NONE;
}

// The same hand, marked public. Identity for anything not in a hand, so a caller that reveals
// "every card the opponent holds" can run over all 110 without a location test.
constexpr CardLocation revealed(CardLocation loc) noexcept {
    if (loc == CardLocation::HAND_US_UNKNOWN) return CardLocation::HAND_US_KNOWN;
    if (loc == CardLocation::HAND_USSR_UNKNOWN) return CardLocation::HAND_USSR_KNOWN;
    return loc;
}

// Primitive Micro-Decision Types
enum class DecisionType : uint8_t {
    NONE                 = 0,
    SELECT_CARD          = 1, // Select card ID (Hand play, Discard, Escape roll, Search, UN paired)
    SELECT_PLAY_MODE     = 2, // Choose mode for selected card: EVENT(0), OPS(1), SPACE(2), PASS(3)
    CHOOSE_TIMING_BRANCH = 3, // Choose timing for opponent card: 0 = OPS_FIRST, 1 = EVENT_FIRST
    SELECT_OP_MODE       = 4, // Select Op usage type: 0 = INFLUENCE, 1 = COUP, 2 = REALIGN
    POINT_NODE           = 5, // Select single country node (Influence/Coup/Realign/Event target)
    CHOOSE_BRANCH        = 6, // Discrete branch (0..7) or CONFIRM_DONE (bit 7 set / 0x80)
    ROLL_DIE             = 7  // Stochastic Chance node / Nature micro-decision
};

// Micro-Action Types recorded in action history
enum class ActionType : uint8_t {
    HEADLINE           = 0,
    PLAY_EVENT         = 1,
    PLAY_OPS_INFLUENCE = 2,
    PLAY_OPS_COUP      = 3,
    PLAY_OPS_REALIGN   = 4,
    PLAY_SPACE         = 5,
    DISCARD_CARD       = 6,
    PASS               = 7
};

// Map Regions
enum class Region : uint8_t {
    EUROPE          = 0,
    ASIA            = 1,
    MIDDLE_EAST     = 2,
    AFRICA          = 3,
    CENTRAL_AMERICA = 4,
    SOUTH_AMERICA   = 5,
    NONE_REGION     = 255
};

// Sub-regions
enum class SubRegion : uint8_t {
    NONE_SUBREGION  = 0,
    WESTERN_EUROPE  = 1,
    EASTERN_EUROPE  = 2,
    SOUTHEAST_ASIA  = 3
};

// Play Modes
enum class PlayMode : uint8_t {
    EVENT = 0,
    OPS   = 1,
    SPACE = 2,
    PASS  = 3
};

// Timing Branches
enum class TimingBranch : uint8_t {
    OPS_FIRST   = 0,
    EVENT_FIRST = 1
};

// Op Modes
enum class OpMode : uint8_t {
    INFLUENCE = 0,
    COUP      = 1,
    REALIGN   = 2
};

} // namespace ts
