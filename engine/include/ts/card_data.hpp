#pragma once
#include <cstdint>
#include <string_view>
#include <array>
#include "types.hpp"
#include "constants.hpp"

namespace ts {

struct CardInfo {
    uint8_t          id;
    std::string_view name;
    uint8_t          ops;
    Player           side; // US, USSR, or NONE (Neutral)
    WarEra           era;
    bool             one_time;
    bool             is_scoring;
    bool             is_war_card;
    bool             requires_context_push;
    bool             switches_priority;
};

class CardData {
public:
    static const CardInfo& get_card(uint8_t id) noexcept;
    static std::string_view get_card_name(uint8_t id) noexcept;
    static uint8_t get_card_by_name(std::string_view name) noexcept;

    static bool is_scoring_card(uint8_t id) noexcept;
    static bool is_war_card(uint8_t id) noexcept;
    static bool is_opponent_card(uint8_t id, Player player) noexcept;
    static bool is_friendly_card(uint8_t id, Player player) noexcept;
    static bool is_neutral_card(uint8_t id) noexcept;
};

} // namespace ts
