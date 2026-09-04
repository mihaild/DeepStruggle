#include "test_framework.hpp"
#include "ts/scoring.hpp"
#include "ts/constants.hpp"

TEST(ScoringTest, CountryControlFormula) {
    ts::GameState state{};

    // Canada (Stability 4): US=4, USSR=0 -> US controls
    state.countries[ts::countries::CANADA].us_influence = 4;
    state.countries[ts::countries::CANADA].ussr_influence = 0;
    ASSERT_EQ(ts::Scoring::get_country_control(state, ts::countries::CANADA), ts::Player::US);

    // Canada: US=4, USSR=1 -> diff is 3 (< 4) -> Uncontrolled!
    state.countries[ts::countries::CANADA].ussr_influence = 1;
    ASSERT_EQ(ts::Scoring::get_country_control(state, ts::countries::CANADA), ts::Player::NONE);

    // Canada: US=5, USSR=1 -> diff is 4 (>= 4) -> US controls!
    state.countries[ts::countries::CANADA].us_influence = 5;
    ASSERT_EQ(ts::Scoring::get_country_control(state, ts::countries::CANADA), ts::Player::US);
}

TEST(ScoringTest, RegionalDominationAndControl) {
    ts::GameState state{};
    // Middle East: 10 countries, 6 Battlegrounds (Israel, Iraq, Iran, Saudi Arabia, Egypt, Libya)

    // US controls Israel(4), Egypt(2), Lebanon(1) [2 BGs, 1 non-BG = 3 total]
    state.countries[ts::countries::ISRAEL].us_influence = 4;
    state.countries[ts::countries::EGYPT].us_influence = 2;
    state.countries[ts::countries::LEBANON].us_influence = 1;

    // USSR controls Iraq(3) [1 BG = 1 total]
    state.countries[ts::countries::IRAQ].ussr_influence = 3;

    auto summary = ts::Scoring::evaluate_region(state, ts::Region::MIDDLE_EAST);
    ASSERT_EQ(summary.us_status, ts::RegionalStatus::DOMINATION);
    ASSERT_EQ(summary.ussr_status, ts::RegionalStatus::PRESENCE);

    // Domination base VP in ME is 5, plus 2 BGs = 7 for US.
    // Presence base VP in ME is 3, plus 1 BG = 4 for USSR.
    // Net delta = 7 - 4 = +3 VP.
    ASSERT_EQ(summary.net_delta, 3);
}

TEST(ScoringTest, EuropeControlInstantVictory) {
    ts::GameState state{};
    state.current_phase = ts::Phase::ACTION_ROUND;
    // Europe has 5 battlegrounds: West Germany(7), France(8), Italy(10), East Germany(14), Poland(15)
    // Total 21 countries.
    // US controls all 5 BGs and 6 total countries (more than USSR 0).
    state.countries[ts::countries::WEST_GERMANY].us_influence = 4;
    state.countries[ts::countries::FRANCE].us_influence = 3;
    state.countries[ts::countries::ITALY].us_influence = 2;
    state.countries[ts::countries::EAST_GERMANY].us_influence = 3;
    state.countries[ts::countries::POLAND].us_influence = 3;
    state.countries[ts::countries::UNITED_KINGDOM].us_influence = 5;

    ts::Scoring::score_region(state, ts::Region::EUROPE);
    ASSERT_EQ(state.victory_points, 20);
    ASSERT_EQ(state.current_phase, ts::Phase::GAME_OVER);
}

TEST(ScoringTest, FinalScoringAccumulationWithIntermediateOver20) {
    ts::GameState state{};
    state.current_phase = ts::Phase::END_TURN;
    state.turn = 10;
    state.victory_points = 15; // Starting at +15 VP

    // 1. Europe: US has Domination (West Germany, Italy, UK) vs USSR None
    state.countries[ts::countries::WEST_GERMANY].us_influence = 4;
    state.countries[ts::countries::ITALY].us_influence = 3;
    state.countries[ts::countries::UNITED_KINGDOM].us_influence = 5;
    // US: Domination in Europe = 7 base + 2 BGs = +9 VP. (Intermediate VP: 15 + 9 = +24 VP)

    // 2. Asia: USSR has Domination (North Korea, Pakistan, Vietnam) vs US Presence (Japan)
    state.countries[ts::countries::NORTH_KOREA].ussr_influence = 3;
    state.countries[ts::countries::PAKISTAN].ussr_influence = 3;
    state.countries[ts::countries::VIETNAM].ussr_influence = 2;
    state.countries[ts::countries::JAPAN].us_influence = 4;
    // USSR: Domination in Asia = 7 base + 2 BGs = 9 VP.
    // US: Presence in Asia = 3 base + 1 BG = 4 VP.
    // Asia net delta = 4 - 9 = -5 VP. (Intermediate VP: 24 - 5 = +19 VP)

    // China Card held by USSR (-1 VP)
    state.china_card_holder = ts::Player::USSR;

    // Execute Final Scoring
    ts::Scoring::execute_final_scoring(state);

    // Final VP must be exactly 15 + 9 - 5 - 1 = +18 VP (all regions counted, not clamped at 20 midway!)
    ASSERT_EQ(state.victory_points, 18);
    ASSERT_EQ(state.current_phase, ts::Phase::GAME_OVER);
}

// Shuttle Diplomacy subtracts one battleground *country* from the USSR's total in Asia or the
// Middle East. It comes off the country count as well as the battleground count, and both
// matter: Domination and Control are decided by who holds more countries, so taking only the
// battleground could never flip the status the card exists to flip.
TEST(ScoringTest, ShuttleDiplomacyRemovesACountryNotJustABattleground) {
    // The Middle East of ts-replayer game 121 at turn 6 AR5: four countries and three
    // battlegrounds each, so both sides are on Presence and the net is nothing.
    ts::GameState state{};
    for (uint8_t i = 0; i < 84; ++i) { state.countries[i].us_influence = 0;
                                       state.countries[i].ussr_influence = 0; }
    state.countries[ts::countries::LEBANON].us_influence = 1;       // US, non-battleground
    state.countries[ts::countries::ISRAEL].us_influence = 4;        // US battleground
    state.countries[ts::countries::EGYPT].us_influence = 3;         // US battleground
    state.countries[ts::countries::LIBYA].us_influence = 2;         // US battleground
    state.countries[ts::countries::SYRIA].ussr_influence = 2;       // USSR, non-battleground
    state.countries[ts::countries::IRAQ].ussr_influence = 3;        // USSR battleground
    state.countries[ts::countries::IRAN].ussr_influence = 2;        // USSR battleground
    state.countries[ts::countries::SAUDI_ARABIA].ussr_influence = 3;// USSR battleground

    auto level = ts::Scoring::evaluate_region(state, ts::Region::MIDDLE_EAST);
    ASSERT_EQ(level.us_status, ts::RegionalStatus::PRESENCE);
    ASSERT_EQ(level.ussr_status, ts::RegionalStatus::PRESENCE);
    ASSERT_EQ(level.net_delta, 0);

    state.set_flag(ts::effect_bits::SHUTTLE_DIPLOMACY_ACTIVE);
    auto shuttled = ts::Scoring::evaluate_region(state, ts::Region::MIDDLE_EAST);
    // Four countries against three makes the US dominant: 5 + 3 battlegrounds against the
    // USSR's 3 + 2.
    ASSERT_EQ(shuttled.us_status, ts::RegionalStatus::DOMINATION);
    ASSERT_EQ(shuttled.ussr_status, ts::RegionalStatus::PRESENCE);
    ASSERT_EQ(shuttled.us_score, 8);
    ASSERT_EQ(shuttled.ussr_score, 5);
    ASSERT_EQ(shuttled.net_delta, 3);
}

// In Asia the battleground it removes may be Japan, which borders the United States, so the
// USSR's bonus for a country adjacent to the enemy superpower goes with it.
TEST(ScoringTest, ShuttleDiplomacyAlsoRemovesTheAsianAdjacencyBonus) {
    ts::GameState state{};
    for (uint8_t i = 0; i < 84; ++i) { state.countries[i].us_influence = 0;
                                       state.countries[i].ussr_influence = 0; }
    state.countries[ts::countries::JAPAN].ussr_influence = 4;       // battleground, borders the US
    state.countries[ts::countries::NORTH_KOREA].ussr_influence = 3; // battleground
    state.countries[ts::countries::LAOS_CAMBODIA].ussr_influence = 1;

    auto plain = ts::Scoring::evaluate_region(state, ts::Region::ASIA);
    ASSERT_EQ(plain.ussr_superpower_adjacent, 1);

    state.set_flag(ts::effect_bits::SHUTTLE_DIPLOMACY_ACTIVE);
    auto shuttled = ts::Scoring::evaluate_region(state, ts::Region::ASIA);
    // The USSR keeps Domination here -- it still holds more countries than the US -- so what
    // changes is the battleground it loses and the adjacency bonus that went with it.
    ASSERT_EQ(shuttled.ussr_status, ts::RegionalStatus::DOMINATION);
    ASSERT_EQ(shuttled.ussr_score, plain.ussr_score - 2);
}

// Final scoring is untouched: the card is spent during the game, not at the end of it.
TEST(ScoringTest, ShuttleDiplomacyDoesNotApplyToFinalScoring) {
    ts::GameState state{};
    for (uint8_t i = 0; i < 84; ++i) { state.countries[i].us_influence = 0;
                                       state.countries[i].ussr_influence = 0; }
    state.countries[ts::countries::IRAQ].ussr_influence = 3;
    state.countries[ts::countries::IRAN].ussr_influence = 2;
    state.set_flag(ts::effect_bits::SHUTTLE_DIPLOMACY_ACTIVE);

    auto during = ts::Scoring::evaluate_region(state, ts::Region::MIDDLE_EAST, false);
    auto final_ = ts::Scoring::evaluate_region(state, ts::Region::MIDDLE_EAST, true);
    ASSERT_LT(during.ussr_score, final_.ussr_score);
}
