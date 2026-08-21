#include "test_framework.hpp"
#include "ts/ops.hpp"
#include "ts/constants.hpp"
#include "ts/map_data.hpp"

TEST(OpsTest, InfluenceCostDynamicTransition) {
    ts::GameState state{};
    // East Germany (stab 3): USSR has 3 inf -> USSR controls.
    state.countries[ts::countries::EAST_GERMANY].ussr_influence = 3;

    // US cost in enemy controlled country is 2 Ops
    ASSERT_EQ(ts::Operations::get_influence_cost(state, ts::Player::US, ts::countries::EAST_GERMANY), 2);

    // If US places 1 influence, USSR still controls (3 vs 1, stab 3, diff is 2 < 3) -> uncontrolled!
    state.countries[ts::countries::EAST_GERMANY].us_influence = 1;
    // Now uncontrolled -> cost drops to 1 Op!
    ASSERT_EQ(ts::Operations::get_influence_cost(state, ts::Player::US, ts::countries::EAST_GERMANY), 1);
}

TEST(OpsTest, CoupDefconDegradationAndMilOps) {
    ts::GameState state{};
    state.defcon = 5;
    state.us_mil_ops = 0;
    state.countries[ts::countries::EGYPT].ussr_influence = 2; // BG country, stab 2

    // US coups Egypt with 3 Ops card and rolls 4
    auto res = ts::Operations::execute_coup(state, ts::Player::US, ts::countries::EGYPT, 3, 4);

    ASSERT_TRUE(res.success);
    ASSERT_TRUE(res.defcon_degraded);
    ASSERT_EQ(state.defcon, 4);
    ASSERT_EQ(state.us_mil_ops, 3);
    // Total = 4 + 3 = 7. Margin = 7 - 2*2 = 3.
    // Removes 2 USSR influence and adds 1 US influence!
    ASSERT_EQ(state.countries[ts::countries::EGYPT].ussr_influence, 0);
    ASSERT_EQ(state.countries[ts::countries::EGYPT].us_influence, 1);
}

TEST(OpsTest, NATOProtectionAgainstCoup) {
    ts::GameState state{};
    state.set_flag(ts::effect_bits::NATO_ACTIVE);
    state.defcon = 5;

    // US controls France (8) (stab 3: 4 US vs 1 USSR)
    state.countries[ts::countries::FRANCE].us_influence = 4;
    state.countries[ts::countries::FRANCE].ussr_influence = 1;

    // USSR cannot coup France under NATO
    ASSERT_TRUE(!ts::Operations::can_coup(state, ts::Player::USSR, ts::countries::FRANCE));

    // If De Gaulle cancels NATO for France, USSR can coup
    state.set_flag(ts::effect_bits::NATO_CANCELED_FRANCE);
    ASSERT_TRUE(ts::Operations::can_coup(state, ts::Player::USSR, ts::countries::FRANCE));
}
