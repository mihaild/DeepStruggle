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
