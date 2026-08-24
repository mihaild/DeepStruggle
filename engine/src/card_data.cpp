#include "ts/card_data.hpp"

namespace ts {

namespace {

constexpr std::array<CardInfo, 111> CARDS = {{
    // 0: Dummy entry
    {0, "None", 0, Player::NONE, WarEra::EARLY, false, false, false, false, false},

    // Early War (1..35, 103..106)
    {1, "Asia Scoring", 0, Player::NONE, WarEra::EARLY, false, true, false, false, false},
    {2, "Europe Scoring", 0, Player::NONE, WarEra::EARLY, false, true, false, false, false},
    {3, "Middle East Scoring", 0, Player::NONE, WarEra::EARLY, false, true, false, false, false},
    {4, "Duck and Cover", 3, Player::US, WarEra::EARLY, false, false, false, false, false},
    {5, "Five Year Plan", 3, Player::US, WarEra::EARLY, false, false, false, true, false},
    {6, "The China Card", 4, Player::NONE, WarEra::EARLY, false, false, false, false, false},
    {7, "Socialist Governments", 3, Player::USSR, WarEra::EARLY, false, false, false, false, false},
    {8, "Fidel", 2, Player::USSR, WarEra::EARLY, true, false, false, false, false},
    {9, "Vietnam Revolts", 2, Player::USSR, WarEra::EARLY, true, false, false, false, false},
    {10, "Blockade", 1, Player::USSR, WarEra::EARLY, true, false, false, false, false},
    {11, "Korean War", 2, Player::USSR, WarEra::EARLY, true, false, true, false, false},
    {12, "Romanian Abdication", 1, Player::USSR, WarEra::EARLY, true, false, false, false, false},
    {13, "Arab-Israeli War", 2, Player::USSR, WarEra::EARLY, false, false, true, false, false},
    {14, "COMECON", 3, Player::USSR, WarEra::EARLY, true, false, false, false, false},
    {15, "Nasser", 1, Player::USSR, WarEra::EARLY, true, false, false, false, false},
    {16, "Warsaw Pact Formed", 3, Player::USSR, WarEra::EARLY, true, false, false, false, false},
    {17, "De Gaulle Leads France", 3, Player::USSR, WarEra::EARLY, true, false, false, false, false},
    {18, "Captured Nazi Scientist", 1, Player::NONE, WarEra::EARLY, true, false, false, false, false},
    {19, "Truman Doctrine", 1, Player::US, WarEra::EARLY, true, false, false, false, false},
    {20, "Olympic Games", 2, Player::NONE, WarEra::EARLY, false, false, false, false, false},
    {21, "NATO", 4, Player::US, WarEra::EARLY, true, false, false, false, false},
    {22, "Independent Reds", 2, Player::US, WarEra::EARLY, true, false, false, false, false},
    {23, "Marshall Plan", 4, Player::US, WarEra::EARLY, true, false, false, false, false},
    {24, "Indo-Pakistani War", 2, Player::NONE, WarEra::EARLY, false, false, true, false, false},
    {25, "Containment", 3, Player::US, WarEra::EARLY, true, false, false, false, false},
    {26, "CIA Created", 1, Player::US, WarEra::EARLY, true, false, false, false, false},
    {27, "US/Japan Mutual Defense Pact", 4, Player::US, WarEra::EARLY, true, false, false, false, false},
    {28, "Suez Crisis", 3, Player::USSR, WarEra::EARLY, true, false, false, false, false},
    {29, "East European Unrest", 3, Player::US, WarEra::EARLY, false, false, false, false, false},
    {30, "Decolonization", 2, Player::USSR, WarEra::EARLY, false, false, false, false, false},
    {31, "Red Scare/Purge", 4, Player::NONE, WarEra::EARLY, false, false, false, false, false},
    {32, "UN Intervention", 1, Player::NONE, WarEra::EARLY, false, false, false, false, false},
    {33, "De-Stalinization", 3, Player::USSR, WarEra::EARLY, true, false, false, false, false},
    {34, "Nuclear Test Ban", 4, Player::NONE, WarEra::EARLY, false, false, false, false, false},
    {35, "Formosan Resolution", 2, Player::US, WarEra::EARLY, true, false, false, false, false},

    // Mid War (36..81, 107..108)
    {36, "Brush War", 3, Player::NONE, WarEra::MID, false, false, true, false, false},
    {37, "Central America Scoring", 0, Player::NONE, WarEra::MID, false, true, false, false, false},
    {38, "Southeast Asia Scoring", 0, Player::NONE, WarEra::MID, true, true, false, false, false},
    {39, "Arms Race", 3, Player::NONE, WarEra::MID, false, false, false, false, false},
    {40, "Cuban Missile Crisis", 3, Player::NONE, WarEra::MID, true, false, false, false, false},
    {41, "Nuclear Subs", 2, Player::US, WarEra::MID, true, false, false, false, false},
    {42, "Quagmire", 3, Player::USSR, WarEra::MID, true, false, false, false, false},
    {43, "SALT Negotiations", 3, Player::NONE, WarEra::MID, true, false, false, true, false},
    {44, "Bear Trap", 3, Player::US, WarEra::MID, true, false, false, false, false},
    {45, "Summit", 1, Player::NONE, WarEra::MID, false, false, false, false, false},
    {46, "How I Learned to Stop Worrying", 2, Player::NONE, WarEra::MID, true, false, false, false, false},
    {47, "Junta", 2, Player::NONE, WarEra::MID, false, false, false, false, false},
    {48, "Kitchen Debates", 1, Player::US, WarEra::MID, true, false, false, false, false},
    {49, "Missile Envy", 2, Player::NONE, WarEra::MID, false, false, false, true, true},
    {50, "“We Will Bury You”", 4, Player::USSR, WarEra::MID, true, false, false, false, false},
    {51, "Brezhnev Doctrine", 3, Player::USSR, WarEra::MID, true, false, false, false, false},
    {52, "Portuguese Empire Crumbles", 2, Player::USSR, WarEra::MID, true, false, false, false, false},
    {53, "South African Unrest", 2, Player::USSR, WarEra::MID, false, false, false, false, false},
    {54, "Allende", 1, Player::USSR, WarEra::MID, true, false, false, false, false},
    {55, "Willy Brandt", 2, Player::USSR, WarEra::MID, true, false, false, false, false},
    {56, "Muslim Revolution", 4, Player::USSR, WarEra::MID, false, false, false, false, false},
    {57, "ABM Treaty", 4, Player::NONE, WarEra::MID, false, false, false, false, false},
    {58, "Cultural Revolution", 3, Player::USSR, WarEra::MID, true, false, false, false, false},
    {59, "Flower Power", 4, Player::USSR, WarEra::MID, true, false, false, false, false},
    {60, "U-2 Incident", 3, Player::USSR, WarEra::MID, true, false, false, false, false},
    {61, "OPEC", 3, Player::USSR, WarEra::MID, false, false, false, false, false},
    {62, "“Lone Gunman”", 1, Player::USSR, WarEra::MID, true, false, false, false, false},
    {63, "Colonial Rear Guards", 2, Player::US, WarEra::MID, false, false, false, false, false},
    {64, "Panama Canal Returned", 1, Player::US, WarEra::MID, true, false, false, false, false},
    {65, "Camp David Accords", 2, Player::US, WarEra::MID, true, false, false, false, false},
    {66, "Puppet Governments", 2, Player::US, WarEra::MID, true, false, false, false, false},
    {67, "Grain Sales to Soviets", 2, Player::US, WarEra::MID, false, false, false, true, false},
    {68, "John Paul II Elected Pope", 2, Player::US, WarEra::MID, true, false, false, false, false},
    {69, "Latin American Death Squads", 2, Player::NONE, WarEra::MID, false, false, false, false, false},
    {70, "OAS Founded", 1, Player::US, WarEra::MID, true, false, false, false, false},
    {71, "Nixon Plays the China Card", 2, Player::US, WarEra::MID, true, false, false, false, false},
    {72, "Sadat Expels Soviets", 1, Player::US, WarEra::MID, true, false, false, false, false},
    {73, "Shuttle Diplomacy", 3, Player::US, WarEra::MID, false, false, false, false, false},
    {74, "The Voice of America", 2, Player::US, WarEra::MID, false, false, false, false, false},
    {75, "Liberation Theology", 2, Player::USSR, WarEra::MID, false, false, false, false, false},
    {76, "Ussuri River Skirmish", 3, Player::US, WarEra::MID, true, false, false, false, false},
    {77, "“Ask Not What Your Country Can Do For You…”", 3, Player::US, WarEra::MID, true, false, false, false, false},
    {78, "Alliance for Progress", 3, Player::US, WarEra::MID, true, false, false, false, false},
    {79, "Africa Scoring", 0, Player::NONE, WarEra::MID, false, true, false, false, false},
    {80, "“One Small Step…”", 2, Player::NONE, WarEra::MID, false, false, false, false, false},
    {81, "South America Scoring", 0, Player::NONE, WarEra::MID, false, true, false, false, false},

    // Late War (82..102, 109..110)
    {82, "Iranian Hostage Crisis", 3, Player::USSR, WarEra::LATE, true, false, false, false, false},
    {83, "The Iron Lady", 3, Player::US, WarEra::LATE, true, false, false, false, false},
    {84, "Reagan Bombs Libya", 2, Player::US, WarEra::LATE, true, false, false, false, false},
    {85, "Star Wars", 2, Player::US, WarEra::LATE, true, false, false, true, false},
    {86, "North Sea Oil", 3, Player::US, WarEra::LATE, true, false, false, false, false},
    {87, "The Reformer", 3, Player::USSR, WarEra::LATE, true, false, false, false, false},
    {88, "Marine Barracks Bombing", 2, Player::USSR, WarEra::LATE, true, false, false, false, false},
    {89, "Soviets Shoot Down KAL-007", 4, Player::US, WarEra::LATE, true, false, false, false, false},
    {90, "Glasnost", 4, Player::USSR, WarEra::LATE, true, false, false, false, false},
    {91, "Ortega Elected in Nicaragua", 2, Player::USSR, WarEra::LATE, true, false, false, false, false},
    {92, "Terrorism", 2, Player::NONE, WarEra::LATE, false, false, false, false, false},
    {93, "Iran-Contra Scandal", 2, Player::USSR, WarEra::LATE, true, false, false, false, false},
    {94, "Chernobyl", 3, Player::US, WarEra::LATE, true, false, false, false, false},
    {95, "Latin American Debt Crisis", 2, Player::USSR, WarEra::LATE, false, false, false, false, false},
    {96, "Tear Down this Wall", 3, Player::US, WarEra::LATE, true, false, false, false, false},
    {97, "“An Evil Empire”", 3, Player::US, WarEra::LATE, true, false, true, false, false},
    {98, "Aldrich Ames Remix", 3, Player::USSR, WarEra::LATE, true, false, false, false, false},
    {99, "Pershing II Deployed", 3, Player::USSR, WarEra::LATE, true, false, false, false, false},
    {100, "Wargames", 4, Player::NONE, WarEra::LATE, true, false, false, false, false},
    {101, "Solidarity", 2, Player::US, WarEra::LATE, true, false, false, false, false},
    {102, "Iran-Iraq War", 2, Player::NONE, WarEra::LATE, true, false, true, false, false},

    // Extra Early War Cards
    {103, "Defectors", 2, Player::US, WarEra::EARLY, false, false, false, false, false},
    {104, "The Cambridge Five", 2, Player::USSR, WarEra::EARLY, false, false, false, false, false},
    {105, "Special Relationship", 2, Player::US, WarEra::EARLY, false, false, false, false, false},
    {106, "NORAD", 3, Player::US, WarEra::EARLY, true, false, false, false, false},

    // Extra Mid War Cards
    {107, "Che", 3, Player::USSR, WarEra::MID, false, false, false, false, false},
    {108, "Our Man in Tehran", 2, Player::US, WarEra::MID, true, false, false, false, false},

    // Extra Late War Cards
    {109, "Yuri and Samantha", 2, Player::USSR, WarEra::LATE, true, false, false, false, false},
    {110, "AWACS Sale to Saudis", 3, Player::US, WarEra::LATE, true, false, false, false, false}
}};

} // namespace

const CardInfo& CardData::get_card(uint8_t id) noexcept {
    return (id >= 1 && id <= 110) ? CARDS[id] : CARDS[0];
}

std::string_view CardData::get_card_name(uint8_t id) noexcept {
    return (id >= 1 && id <= 110) ? CARDS[id].name : "Unknown Card";
}

uint8_t CardData::get_card_by_name(std::string_view name) noexcept {
    for (uint8_t i = 1; i <= 110; ++i) {
        if (CARDS[i].name == name) return i;
    }
    return 0;
}

bool CardData::is_scoring_card(uint8_t id) noexcept {
    return (id >= 1 && id <= 110) && CARDS[id].is_scoring;
}

bool CardData::is_war_card(uint8_t id) noexcept {
    return (id >= 1 && id <= 110) && CARDS[id].is_war_card;
}

bool CardData::is_opponent_card(uint8_t id, Player player) noexcept {
    if (id < 1 || id > 110 || player == Player::NONE) return false;
    return CARDS[id].side == get_opponent(player);
}

bool CardData::is_friendly_card(uint8_t id, Player player) noexcept {
    if (id < 1 || id > 110 || player == Player::NONE) return false;
    return CARDS[id].side == player;
}

bool CardData::is_neutral_card(uint8_t id) noexcept {
    if (id < 1 || id > 110) return false;
    return CARDS[id].side == Player::NONE;
}

} // namespace ts
