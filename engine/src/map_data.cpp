#include "ts/map_data.hpp"
#include <cstring>

namespace ts {

namespace {

constexpr std::array<CountryInfo, 84> COUNTRIES = {{
    // 0: Canada
    {0, "Canada", 4, false, Region::EUROPE, true, false, false, Player::US, 1, {1,0,0,0,0,0}, {1ULL<<1, 0}},
    // 1: United Kingdom
    {1, "United Kingdom", 5, false, Region::EUROPE, true, false, false, Player::NONE, 4, {0,2,6,8,0,0}, {(1ULL<<0)|(1ULL<<2)|(1ULL<<6)|(1ULL<<8), 0}},
    // 2: Norway
    {2, "Norway", 4, false, Region::EUROPE, true, false, false, Player::NONE, 2, {1,3,0,0,0,0}, {(1ULL<<1)|(1ULL<<3), 0}},
    // 3: Sweden
    {3, "Sweden", 4, false, Region::EUROPE, true, false, false, Player::NONE, 3, {2,4,5,0,0,0}, {(1ULL<<2)|(1ULL<<4)|(1ULL<<5), 0}},
    // 4: Denmark
    {4, "Denmark", 3, false, Region::EUROPE, true, false, false, Player::NONE, 2, {3,7,0,0,0,0}, {(1ULL<<3)|(1ULL<<7), 0}},
    // 5: Finland
    {5, "Finland", 4, false, Region::EUROPE, true, true, false, Player::USSR, 1, {3,0,0,0,0,0}, {1ULL<<3, 0}},
    // 6: Benelux
    {6, "Benelux", 3, false, Region::EUROPE, true, false, false, Player::NONE, 2, {1,7,0,0,0,0}, {(1ULL<<1)|(1ULL<<7), 0}},
    // 7: West Germany
    {7, "West Germany", 4, true, Region::EUROPE, true, false, false, Player::NONE, 5, {6,4,8,13,14,0}, {(1ULL<<6)|(1ULL<<4)|(1ULL<<8)|(1ULL<<13)|(1ULL<<14), 0}},
    // 8: France
    {8, "France", 3, true, Region::EUROPE, true, false, false, Player::NONE, 5, {1,7,9,10,47,0}, {(1ULL<<1)|(1ULL<<7)|(1ULL<<9)|(1ULL<<10)|(1ULL<<47), 0}},
    // 9: Spain/Portugal
    {9, "Spain/Portugal", 2, false, Region::EUROPE, true, false, false, Player::NONE, 3, {8,10,46,0,0,0}, {(1ULL<<8)|(1ULL<<10)|(1ULL<<46), 0}},
    // 10: Italy
    {10, "Italy", 2, true, Region::EUROPE, true, false, false, Player::NONE, 5, {8,9,13,18,11,0}, {(1ULL<<8)|(1ULL<<9)|(1ULL<<13)|(1ULL<<18)|(1ULL<<11), 0}},
    // 11: Greece
    {11, "Greece", 2, false, Region::EUROPE, true, false, false, Player::NONE, 4, {10,18,20,12,0,0}, {(1ULL<<10)|(1ULL<<18)|(1ULL<<20)|(1ULL<<12), 0}},
    // 12: Turkey
    {12, "Turkey", 2, false, Region::EUROPE, true, false, false, Player::USSR, 4, {11,20,19,22,0,0}, {(1ULL<<11)|(1ULL<<20)|(1ULL<<19)|(1ULL<<22), 0}},
    // 13: Austria
    {13, "Austria", 4, false, Region::EUROPE, true, true, false, Player::NONE, 4, {7,14,10,17,0,0}, {(1ULL<<7)|(1ULL<<14)|(1ULL<<10)|(1ULL<<17), 0}},
    // 14: East Germany
    {14, "East Germany", 3, true, Region::EUROPE, false, true, false, Player::NONE, 4, {7,13,15,16,0,0}, {(1ULL<<7)|(1ULL<<13)|(1ULL<<15)|(1ULL<<16), 0}},
    // 15: Poland
    {15, "Poland", 3, true, Region::EUROPE, false, true, false, Player::USSR, 2, {14,16,0,0,0,0}, {(1ULL<<14)|(1ULL<<16), 0}},
    // 16: Czechoslovakia
    {16, "Czechoslovakia", 3, false, Region::EUROPE, false, true, false, Player::NONE, 3, {14,15,17,0,0,0}, {(1ULL<<14)|(1ULL<<15)|(1ULL<<17), 0}},
    // 17: Hungary
    {17, "Hungary", 3, false, Region::EUROPE, false, true, false, Player::NONE, 4, {13,16,18,19,0,0}, {(1ULL<<13)|(1ULL<<16)|(1ULL<<18)|(1ULL<<19), 0}},
    // 18: Yugoslavia
    {18, "Yugoslavia", 3, false, Region::EUROPE, false, true, false, Player::NONE, 4, {10,17,19,11,0,0}, {(1ULL<<10)|(1ULL<<17)|(1ULL<<19)|(1ULL<<11), 0}},
    // 19: Romania
    {19, "Romania", 3, false, Region::EUROPE, false, true, false, Player::USSR, 3, {17,18,12,0,0,0}, {(1ULL<<17)|(1ULL<<18)|(1ULL<<12), 0}},
    // 20: Bulgaria
    {20, "Bulgaria", 3, false, Region::EUROPE, false, true, false, Player::NONE, 2, {11,12,0,0,0,0}, {(1ULL<<11)|(1ULL<<12), 0}},

    // Middle East
    // 21: Lebanon
    {21, "Lebanon", 1, false, Region::MIDDLE_EAST, false, false, false, Player::NONE, 3, {22,23,26,0,0,0}, {(1ULL<<22)|(1ULL<<23)|(1ULL<<26), 0}},
    // 22: Syria
    {22, "Syria", 2, false, Region::MIDDLE_EAST, false, false, false, Player::NONE, 3, {12,21,23,0,0,0}, {(1ULL<<12)|(1ULL<<21)|(1ULL<<23), 0}},
    // 23: Israel
    {23, "Israel", 4, true, Region::MIDDLE_EAST, false, false, false, Player::NONE, 4, {21,22,26,29,0,0}, {(1ULL<<21)|(1ULL<<22)|(1ULL<<26)|(1ULL<<29), 0}},
    // 24: Iraq
    {24, "Iraq", 3, true, Region::MIDDLE_EAST, false, false, false, Player::NONE, 4, {26,28,27,25,0,0}, {(1ULL<<26)|(1ULL<<28)|(1ULL<<27)|(1ULL<<25), 0}},
    // 25: Iran
    {25, "Iran", 2, true, Region::MIDDLE_EAST, false, false, false, Player::NONE, 3, {24,31,32,0,0,0}, {(1ULL<<24)|(1ULL<<31)|(1ULL<<32), 0}},
    // 26: Jordan
    {26, "Jordan", 2, false, Region::MIDDLE_EAST, false, false, false, Player::NONE, 4, {23,21,24,28,0,0}, {(1ULL<<23)|(1ULL<<21)|(1ULL<<24)|(1ULL<<28), 0}},
    // 27: Gulf States
    {27, "Gulf States", 3, false, Region::MIDDLE_EAST, false, false, false, Player::NONE, 2, {24,28,0,0,0,0}, {(1ULL<<24)|(1ULL<<28), 0}},
    // 28: Saudi Arabia
    {28, "Saudi Arabia", 3, true, Region::MIDDLE_EAST, false, false, false, Player::NONE, 3, {26,24,27,0,0,0}, {(1ULL<<26)|(1ULL<<24)|(1ULL<<27), 0}},
    // 29: Egypt
    {29, "Egypt", 2, true, Region::MIDDLE_EAST, false, false, false, Player::NONE, 3, {23,30,51,0,0,0}, {(1ULL<<23)|(1ULL<<30)|(1ULL<<51), 0}},
    // 30: Libya
    {30, "Libya", 2, true, Region::MIDDLE_EAST, false, false, false, Player::NONE, 2, {29,48,0,0,0,0}, {(1ULL<<29)|(1ULL<<48), 0}},

    // Asia
    // 31: Afghanistan
    {31, "Afghanistan", 2, false, Region::ASIA, false, false, false, Player::USSR, 2, {25,32,0,0,0,0}, {(1ULL<<25)|(1ULL<<32), 0}},
    // 32: Pakistan
    {32, "Pakistan", 2, true, Region::ASIA, false, false, false, Player::NONE, 3, {25,31,33,0,0,0}, {(1ULL<<25)|(1ULL<<31)|(1ULL<<33), 0}},
    // 33: India
    {33, "India", 3, true, Region::ASIA, false, false, false, Player::NONE, 2, {32,34,0,0,0,0}, {(1ULL<<32)|(1ULL<<34), 0}},
    // 34: Burma
    {34, "Burma", 2, false, Region::ASIA, false, false, true, Player::NONE, 2, {33,35,0,0,0,0}, {(1ULL<<33)|(1ULL<<35), 0}},
    // 35: Laos/Cambodia
    {35, "Laos/Cambodia", 1, false, Region::ASIA, false, false, true, Player::NONE, 3, {34,36,37,0,0,0}, {(1ULL<<34)|(1ULL<<36)|(1ULL<<37), 0}},
    // 36: Thailand
    {36, "Thailand", 2, true, Region::ASIA, false, false, true, Player::NONE, 3, {35,37,38,0,0,0}, {(1ULL<<35)|(1ULL<<37)|(1ULL<<38), 0}},
    // 37: Vietnam
    {37, "Vietnam", 1, false, Region::ASIA, false, false, true, Player::NONE, 2, {35,36,0,0,0,0}, {(1ULL<<35)|(1ULL<<36), 0}},
    // 38: Malaysia
    {38, "Malaysia", 2, false, Region::ASIA, false, false, true, Player::NONE, 3, {36,39,41,0,0,0}, {(1ULL<<36)|(1ULL<<39)|(1ULL<<41), 0}},
    // 39: Indonesia
    {39, "Indonesia", 1, false, Region::ASIA, false, false, true, Player::NONE, 2, {38,40,0,0,0,0}, {(1ULL<<38)|(1ULL<<40), 0}},
    // 40: Philippines
    {40, "Philippines", 2, false, Region::ASIA, false, false, true, Player::NONE, 2, {39,45,0,0,0,0}, {(1ULL<<39)|(1ULL<<45), 0}},
    // 41: Australia
    {41, "Australia", 4, false, Region::ASIA, false, false, false, Player::NONE, 1, {38,0,0,0,0,0}, {1ULL<<38, 0}},
    // 42: Taiwan
    {42, "Taiwan", 3, false, Region::ASIA, false, false, false, Player::NONE, 2, {45,44,0,0,0,0}, {(1ULL<<45)|(1ULL<<44), 0}},
    // 43: North Korea
    {43, "North Korea", 3, true, Region::ASIA, false, false, false, Player::USSR, 1, {44,0,0,0,0,0}, {1ULL<<44, 0}},
    // 44: South Korea
    {44, "South Korea", 3, true, Region::ASIA, false, false, false, Player::NONE, 3, {43,45,42,0,0,0}, {(1ULL<<43)|(1ULL<<45)|(1ULL<<42), 0}},
    // 45: Japan
    {45, "Japan", 4, true, Region::ASIA, false, false, false, Player::US, 3, {44,42,40,0,0,0}, {(1ULL<<44)|(1ULL<<42)|(1ULL<<40), 0}},

    // Africa
    // 46: Morocco
    {46, "Morocco", 3, false, Region::AFRICA, false, false, false, Player::NONE, 3, {9,47,49,0,0,0}, {(1ULL<<9)|(1ULL<<47)|(1ULL<<49), 0}},
    // 47: Algeria
    {47, "Algeria", 2, true, Region::AFRICA, false, false, false, Player::NONE, 4, {8,46,48,50,0,0}, {(1ULL<<8)|(1ULL<<46)|(1ULL<<48)|(1ULL<<50), 0}},
    // 48: Tunisia
    {48, "Tunisia", 2, false, Region::AFRICA, false, false, false, Player::NONE, 2, {30,47,0,0,0,0}, {(1ULL<<30)|(1ULL<<47), 0}},
    // 49: West African States
    {49, "West African States", 2, false, Region::AFRICA, false, false, false, Player::NONE, 2, {46,52,0,0,0,0}, {(1ULL<<46)|(1ULL<<52), 0}},
    // 50: Saharan States
    {50, "Saharan States", 1, false, Region::AFRICA, false, false, false, Player::NONE, 2, {47,53,0,0,0,0}, {(1ULL<<47)|(1ULL<<53), 0}},
    // 51: Sudan
    {51, "Sudan", 1, false, Region::AFRICA, false, false, false, Player::NONE, 2, {29,54,0,0,0,0}, {(1ULL<<29)|(1ULL<<54), 0}},
    // 52: Ivory Coast
    {52, "Ivory Coast", 2, false, Region::AFRICA, false, false, false, Player::NONE, 2, {49,53,0,0,0,0}, {(1ULL<<49)|(1ULL<<53), 0}},
    // 53: Nigeria
    {53, "Nigeria", 1, true, Region::AFRICA, false, false, false, Player::NONE, 3, {50,52,56,0,0,0}, {(1ULL<<50)|(1ULL<<52)|(1ULL<<56), 0}},
    // 54: Ethiopia
    {54, "Ethiopia", 1, false, Region::AFRICA, false, false, false, Player::NONE, 2, {51,55,0,0,0,0}, {(1ULL<<51)|(1ULL<<55), 0}},
    // 55: Somalia
    {55, "Somalia", 2, false, Region::AFRICA, false, false, false, Player::NONE, 2, {54,58,0,0,0,0}, {(1ULL<<54)|(1ULL<<58), 0}},
    // 56: Cameroon
    {56, "Cameroon", 1, false, Region::AFRICA, false, false, false, Player::NONE, 2, {53,57,0,0,0,0}, {(1ULL<<53)|(1ULL<<57), 0}},
    // 57: Zaire
    {57, "Zaire", 1, true, Region::AFRICA, false, false, false, Player::NONE, 3, {56,59,60,0,0,0}, {(1ULL<<56)|(1ULL<<59)|(1ULL<<60), 0}},
    // 58: Kenya
    {58, "Kenya", 2, false, Region::AFRICA, false, false, false, Player::NONE, 2, {55,61,0,0,0,0}, {(1ULL<<55)|(1ULL<<61), 0}},
    // 59: Angola
    {59, "Angola", 1, true, Region::AFRICA, false, false, false, Player::NONE, 3, {57,62,63,0,0,0}, {(1ULL<<57)|(1ULL<<62)|(1ULL<<63), 0}},
    // 60: Zimbabwe
    {60, "Zimbabwe", 1, false, Region::AFRICA, false, false, false, Player::NONE, 3, {57,61,62,0,0,0}, {(1ULL<<57)|(1ULL<<61)|(1ULL<<62), 0}},
    // 61: SE African States
    {61, "SE African States", 1, false, Region::AFRICA, false, false, false, Player::NONE, 2, {58,60,0,0,0,0}, {(1ULL<<58)|(1ULL<<60), 0}},
    // 62: Botswana
    {62, "Botswana", 2, false, Region::AFRICA, false, false, false, Player::NONE, 3, {59,60,63,0,0,0}, {(1ULL<<59)|(1ULL<<60)|(1ULL<<63), 0}},
    // 63: South Africa
    {63, "South Africa", 3, true, Region::AFRICA, false, false, false, Player::NONE, 2, {59,62,0,0,0,0}, {(1ULL<<59)|(1ULL<<62), 0}},

    // Central America
    // 64: Mexico
    {64, "Mexico", 2, true, Region::CENTRAL_AMERICA, false, false, false, Player::US, 1, {65,0,0,0,0,0}, {0, 1ULL<<(65-64)}},
    // 65: Guatemala
    {65, "Guatemala", 1, false, Region::CENTRAL_AMERICA, false, false, false, Player::NONE, 3, {64,66,67,0,0,0}, {0, (1ULL<<(64-64))|(1ULL<<(66-64))|(1ULL<<(67-64))}},
    // 66: El Salvador
    {66, "El Salvador", 1, false, Region::CENTRAL_AMERICA, false, false, false, Player::NONE, 2, {65,67,0,0,0,0}, {0, (1ULL<<(65-64))|(1ULL<<(67-64))}},
    // 67: Honduras
    {67, "Honduras", 2, false, Region::CENTRAL_AMERICA, false, false, false, Player::NONE, 4, {65,66,69,68,0,0}, {0, (1ULL<<(65-64))|(1ULL<<(66-64))|(1ULL<<(69-64))|(1ULL<<(68-64))}},
    // 68: Costa Rica
    {68, "Costa Rica", 3, false, Region::CENTRAL_AMERICA, false, false, false, Player::NONE, 3, {67,69,70,0,0,0}, {0, (1ULL<<(67-64))|(1ULL<<(69-64))|(1ULL<<(70-64))}},
    // 69: Nicaragua
    {69, "Nicaragua", 1, false, Region::CENTRAL_AMERICA, false, false, false, Player::NONE, 3, {67,68,71,0,0,0}, {0, (1ULL<<(67-64))|(1ULL<<(68-64))|(1ULL<<(71-64))}},
    // 70: Panama
    {70, "Panama", 2, true, Region::CENTRAL_AMERICA, false, false, false, Player::NONE, 2, {68,74,0,0,0,0}, {0, (1ULL<<(68-64))|(1ULL<<(74-64))}},
    // 71: Cuba
    {71, "Cuba", 3, true, Region::CENTRAL_AMERICA, false, false, false, Player::US, 2, {69,72,0,0,0,0}, {0, (1ULL<<(69-64))|(1ULL<<(72-64))}},
    // 72: Haiti
    {72, "Haiti", 1, false, Region::CENTRAL_AMERICA, false, false, false, Player::NONE, 2, {71,73,0,0,0,0}, {0, (1ULL<<(71-64))|(1ULL<<(73-64))}},
    // 73: Dominican Rep
    {73, "Dominican Rep", 1, false, Region::CENTRAL_AMERICA, false, false, false, Player::NONE, 1, {72,0,0,0,0,0}, {0, 1ULL<<(72-64)}},

    // South America
    // 74: Colombia
    {74, "Colombia", 1, false, Region::SOUTH_AMERICA, false, false, false, Player::NONE, 3, {70,75,77,0,0,0}, {0, (1ULL<<(70-64))|(1ULL<<(75-64))|(1ULL<<(77-64))}},
    // 75: Ecuador
    {75, "Ecuador", 2, false, Region::SOUTH_AMERICA, false, false, false, Player::NONE, 2, {74,76,0,0,0,0}, {0, (1ULL<<(74-64))|(1ULL<<(76-64))}},
    // 76: Peru
    {76, "Peru", 2, false, Region::SOUTH_AMERICA, false, false, false, Player::NONE, 3, {75,81,78,0,0,0}, {0, (1ULL<<(75-64))|(1ULL<<(81-64))|(1ULL<<(78-64))}},
    // 77: Venezuela
    {77, "Venezuela", 2, true, Region::SOUTH_AMERICA, false, false, false, Player::NONE, 2, {74,79,0,0,0,0}, {0, (1ULL<<(74-64))|(1ULL<<(79-64))}},
    // 78: Bolivia
    {78, "Bolivia", 2, false, Region::SOUTH_AMERICA, false, false, false, Player::NONE, 2, {76,80,0,0,0,0}, {0, (1ULL<<(76-64))|(1ULL<<(80-64))}},
    // 79: Brazil
    {79, "Brazil", 2, true, Region::SOUTH_AMERICA, false, false, false, Player::NONE, 2, {77,83,0,0,0,0}, {0, (1ULL<<(77-64))|(1ULL<<(83-64))}},
    // 80: Paraguay
    {80, "Paraguay", 2, false, Region::SOUTH_AMERICA, false, false, false, Player::NONE, 3, {78,82,83,0,0,0}, {0, (1ULL<<(78-64))|(1ULL<<(82-64))|(1ULL<<(83-64))}},
    // 81: Chile
    {81, "Chile", 3, true, Region::SOUTH_AMERICA, false, false, false, Player::NONE, 2, {76,82,0,0,0,0}, {0, (1ULL<<(76-64))|(1ULL<<(82-64))}},
    // 82: Argentina
    {82, "Argentina", 2, true, Region::SOUTH_AMERICA, false, false, false, Player::NONE, 3, {81,80,83,0,0,0}, {0, (1ULL<<(81-64))|(1ULL<<(80-64))|(1ULL<<(83-64))}},
    // 83: Uruguay
    {83, "Uruguay", 2, false, Region::SOUTH_AMERICA, false, false, false, Player::NONE, 3, {79,80,82,0,0,0}, {0, (1ULL<<(79-64))|(1ULL<<(80-64))|(1ULL<<(82-64))}}
}};

// Region Masks
constexpr std::array<std::array<uint64_t, 2>, 6> REGION_MASKS = []() {
    std::array<std::array<uint64_t, 2>, 6> masks{};
    for (size_t i = 0; i < 84; ++i) {
        size_t r = static_cast<size_t>(COUNTRIES[i].region);
        if (i < 64) {
            masks[r][0] |= (1ULL << i);
        } else {
            masks[r][1] |= (1ULL << (i - 64));
        }
    }
    return masks;
}();

constexpr std::array<uint64_t, 2> BATTLEGROUND_MASK = []() {
    std::array<uint64_t, 2> mask{};
    for (size_t i = 0; i < 84; ++i) {
        if (COUNTRIES[i].battleground) {
            if (i < 64) {
                mask[0] |= (1ULL << i);
            } else {
                mask[1] |= (1ULL << (i - 64));
            }
        }
    }
    return mask;
}();

constexpr std::array<std::array<uint64_t, 2>, 6> REGION_BG_MASKS = []() {
    std::array<std::array<uint64_t, 2>, 6> masks{};
    for (size_t i = 0; i < 6; ++i) {
        masks[i][0] = REGION_MASKS[i][0] & BATTLEGROUND_MASK[0];
        masks[i][1] = REGION_MASKS[i][1] & BATTLEGROUND_MASK[1];
    }
    return masks;
}();

constexpr std::array<uint64_t, 2> WESTERN_EUROPE_MASK = []() {
    std::array<uint64_t, 2> mask{};
    for (size_t i = 0; i < 84; ++i) {
        if (COUNTRIES[i].in_western_europe) {
            if (i < 64) mask[0] |= (1ULL << i);
            else mask[1] |= (1ULL << (i - 64));
        }
    }
    return mask;
}();

constexpr std::array<uint64_t, 2> EASTERN_EUROPE_MASK = []() {
    std::array<uint64_t, 2> mask{};
    for (size_t i = 0; i < 84; ++i) {
        if (COUNTRIES[i].in_eastern_europe) {
            if (i < 64) mask[0] |= (1ULL << i);
            else mask[1] |= (1ULL << (i - 64));
        }
    }
    return mask;
}();

constexpr std::array<uint64_t, 2> SOUTHEAST_ASIA_MASK = []() {
    std::array<uint64_t, 2> mask{};
    for (size_t i = 0; i < 84; ++i) {
        if (COUNTRIES[i].in_southeast_asia) {
            if (i < 64) mask[0] |= (1ULL << i);
            else mask[1] |= (1ULL << (i - 64));
        }
    }
    return mask;
}();

constexpr std::array<std::array<uint64_t, 2>, 2> SUPERPOWER_ADJ_MASKS = []() {
    std::array<std::array<uint64_t, 2>, 2> masks{};
    for (size_t i = 0; i < 84; ++i) {
        if (COUNTRIES[i].superpower_adjacent == Player::US) {
            if (i < 64) masks[0][0] |= (1ULL << i);
            else masks[0][1] |= (1ULL << (i - 64));
        } else if (COUNTRIES[i].superpower_adjacent == Player::USSR) {
            if (i < 64) masks[1][0] |= (1ULL << i);
            else masks[1][1] |= (1ULL << (i - 64));
        }
    }
    return masks;
}();

} // namespace

const CountryInfo& MapData::get_country(uint8_t id) noexcept {
    return (id < 84) ? COUNTRIES[id] : COUNTRIES[0];
}

std::string_view MapData::get_country_name(uint8_t id) noexcept {
    return (id < 84) ? COUNTRIES[id].name : "Unknown";
}

uint8_t MapData::get_country_by_name(std::string_view name) noexcept {
    for (uint8_t i = 0; i < 84; ++i) {
        if (COUNTRIES[i].name == name) return i;
    }
    return 255;
}

const std::array<uint64_t, 2>& MapData::get_region_mask(Region r) noexcept {
    size_t idx = static_cast<size_t>(r);
    return (idx < 6) ? REGION_MASKS[idx] : REGION_MASKS[0];
}

const std::array<uint64_t, 2>& MapData::get_battleground_mask() noexcept {
    return BATTLEGROUND_MASK;
}

const std::array<uint64_t, 2>& MapData::get_region_battleground_mask(Region r) noexcept {
    size_t idx = static_cast<size_t>(r);
    return (idx < 6) ? REGION_BG_MASKS[idx] : REGION_BG_MASKS[0];
}

const std::array<uint64_t, 2>& MapData::get_western_europe_mask() noexcept {
    return WESTERN_EUROPE_MASK;
}

const std::array<uint64_t, 2>& MapData::get_eastern_europe_mask() noexcept {
    return EASTERN_EUROPE_MASK;
}

const std::array<uint64_t, 2>& MapData::get_southeast_asia_mask() noexcept {
    return SOUTHEAST_ASIA_MASK;
}

const std::array<uint64_t, 2>& MapData::get_superpower_adjacent_mask(Player p) noexcept {
    size_t idx = (p == Player::US) ? 0 : 1;
    return SUPERPOWER_ADJ_MASKS[idx];
}

uint8_t MapData::get_region_country_count(Region r) noexcept {
    switch (r) {
        case Region::EUROPE: return 21;
        case Region::ASIA: return 15;
        case Region::MIDDLE_EAST: return 10;
        case Region::AFRICA: return 18;
        case Region::CENTRAL_AMERICA: return 10;
        case Region::SOUTH_AMERICA: return 10;
        default: return 0;
    }
}

uint8_t MapData::get_region_battleground_count(Region r) noexcept {
    switch (r) {
        case Region::EUROPE: return 5;
        case Region::ASIA: return 6;
        case Region::MIDDLE_EAST: return 6;
        case Region::AFRICA: return 5;
        case Region::CENTRAL_AMERICA: return 3;
        case Region::SOUTH_AMERICA: return 4;
        default: return 0;
    }
}

bool MapData::is_adjacent(uint8_t c1, uint8_t c2) noexcept {
    if (c1 >= 84 || c2 >= 84) return false;
    if (c2 < 64) {
        return (COUNTRIES[c1].neighbor_mask[0] & (1ULL << c2)) != 0;
    } else {
        return (COUNTRIES[c1].neighbor_mask[1] & (1ULL << (c2 - 64))) != 0;
    }
}

bool MapData::is_adjacent_to_superpower(uint8_t country_id, Player p) noexcept {
    if (country_id >= 84) return false;
    return COUNTRIES[country_id].superpower_adjacent == p;
}

} // namespace ts
