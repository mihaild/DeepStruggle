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

// Card physical location registry
enum class CardLocation : uint8_t {
    UNAVAILABLE        = 0, // Not yet in the deck (future era or unintroduced optional card)
    DRAW_DECK          = 1,
    HAND_US            = 2,
    HAND_USSR          = 3,
    DISCARD_PILE       = 4,
    REMOVED_FROM_GAME  = 5,
    ONGOING_EVENT      = 6,
    PEEKED_TEMP        = 7
};

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
