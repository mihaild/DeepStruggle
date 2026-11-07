#pragma once
#include <cstdint>
#include <string_view>
#include <array>
#include "types.hpp"
#include "constants.hpp"

namespace ts {

struct CountryInfo {
    uint8_t          id;
    std::string_view name;
    uint8_t          stability;
    bool             battleground;
    Region           region;
    bool             in_western_europe;
    bool             in_eastern_europe;
    bool             in_southeast_asia;
    Player           superpower_adjacent; // US, USSR, or NONE
    uint8_t          num_neighbors;
    std::array<uint8_t, 6> neighbors;
    std::array<uint64_t, 2> neighbor_mask;
};

class MapData {
public:
    static const CountryInfo& get_country(uint8_t id) noexcept;
    static std::string_view get_country_name(uint8_t id) noexcept;
    static uint8_t get_country_by_name(std::string_view name) noexcept;

    // Bitmasks for quick set operations
    static const std::array<uint64_t, 2>& get_region_mask(Region r) noexcept;
    static const std::array<uint64_t, 2>& get_battleground_mask() noexcept;

    // P43: the static order a canonical placement run follows (ts/canonical.hpp), a permutation of
    // 0..83, most important last (CANONICAL_LAST; CANONICAL_FIRST reverses it):
    // regions from least to most scoring weight (Africa, South America, Central America, Asia, the
    // Middle East, Europe), non-battlegrounds before battlegrounds within each, Europe's battlegrounds
    // last as Italy, France, West Germany, East Germany, Poland; ties by country id.
    static uint8_t placement_rank(uint8_t id) noexcept;
    static const std::array<uint64_t, 2>& get_region_battleground_mask(Region r) noexcept;
    static const std::array<uint64_t, 2>& get_western_europe_mask() noexcept;
    static const std::array<uint64_t, 2>& get_eastern_europe_mask() noexcept;
    static const std::array<uint64_t, 2>& get_southeast_asia_mask() noexcept;
    static const std::array<uint64_t, 2>& get_superpower_adjacent_mask(Player p) noexcept;

    static uint8_t get_region_country_count(Region r) noexcept;
    static uint8_t get_region_battleground_count(Region r) noexcept;

    static bool is_adjacent(uint8_t c1, uint8_t c2) noexcept;
    static bool is_adjacent_to_superpower(uint8_t country_id, Player p) noexcept;
};

} // namespace ts
