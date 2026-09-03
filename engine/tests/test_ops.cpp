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

TEST(OpsTest, RealignmentDefconRestrictions) {
    ts::GameState state{};
    // Give opponent influence in each region
    state.countries[ts::countries::WEST_GERMANY].us_influence = 2; // Europe
    state.countries[ts::countries::JAPAN].us_influence = 2;        // Asia
    state.countries[ts::countries::EGYPT].us_influence = 2;        // Middle East
    state.countries[ts::countries::ANGOLA].us_influence = 2;       // Africa
    state.countries[ts::countries::CHILE].us_influence = 2;        // South America
    state.countries[ts::countries::CUBA].us_influence = 2;         // Central America

    // DEFCON 5: All regions allowed for coup and realign
    state.defcon = 5;
    ASSERT_TRUE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::WEST_GERMANY));
    ASSERT_TRUE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::JAPAN));
    ASSERT_TRUE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::EGYPT));
    ASSERT_TRUE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::ANGOLA));
    ASSERT_TRUE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::CHILE));
    ASSERT_TRUE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::CUBA));

    // DEFCON 4: Europe restricted
    state.defcon = 4;
    ASSERT_FALSE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::WEST_GERMANY));
    ASSERT_FALSE(ts::Operations::can_coup(state, ts::Player::USSR, ts::countries::WEST_GERMANY));
    ASSERT_TRUE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::JAPAN));
    ASSERT_TRUE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::EGYPT));
    ASSERT_TRUE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::ANGOLA));

    // DEFCON 3: Europe and Asia restricted
    state.defcon = 3;
    ASSERT_FALSE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::WEST_GERMANY));
    ASSERT_FALSE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::JAPAN));
    ASSERT_FALSE(ts::Operations::can_coup(state, ts::Player::USSR, ts::countries::JAPAN));
    ASSERT_TRUE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::EGYPT));
    ASSERT_TRUE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::ANGOLA));

    // DEFCON 2: Europe, Asia, and Middle East restricted
    state.defcon = 2;
    ASSERT_FALSE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::WEST_GERMANY));
    ASSERT_FALSE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::JAPAN));
    ASSERT_FALSE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::EGYPT));
    ASSERT_FALSE(ts::Operations::can_coup(state, ts::Player::USSR, ts::countries::EGYPT));
    ASSERT_TRUE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::ANGOLA));
    ASSERT_TRUE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::CHILE));
    ASSERT_TRUE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::CUBA));
}

TEST(OpsTest, TearDownThisWallDefconExemption) {
    ts::GameState state{};
    state.defcon = 2;
    state.countries[ts::countries::WEST_GERMANY].ussr_influence = 2;
    state.countries[ts::countries::EGYPT].ussr_influence = 2;

    // Normal play at DEFCON 2: US cannot realign or coup in West Germany or Egypt
    ASSERT_FALSE(ts::Operations::can_realign(state, ts::Player::US, ts::countries::WEST_GERMANY));
    ASSERT_FALSE(ts::Operations::can_coup(state, ts::Player::US, ts::countries::WEST_GERMANY));

    // When resolving Tear Down This Wall (#96) as US. The exemption belongs to the free action
    // the event grants, so it is the flag and not the card that turns it on.
    state.ctx().pending_op_card = ts::card_ids::TEAR_DOWN_THIS_WALL;
    state.ctx().event_granted_ops = 1;
    ASSERT_TRUE(ts::Operations::can_realign(state, ts::Player::US, ts::countries::WEST_GERMANY));
    ASSERT_TRUE(ts::Operations::can_coup(state, ts::Player::US, ts::countries::WEST_GERMANY));

    // The same card's own Ops -- or its Ops borrowed by UN Intervention, which never triggers
    // the event at all -- are ordinary Ops and obey DEFCON like anyone else's.
    state.ctx().event_granted_ops = 0;
    ASSERT_FALSE(ts::Operations::can_realign(state, ts::Player::US, ts::countries::WEST_GERMANY));
    ASSERT_FALSE(ts::Operations::can_coup(state, ts::Player::US, ts::countries::WEST_GERMANY));
    state.ctx().event_granted_ops = 1;

    // But Middle East is not Europe, so still restricted at DEFCON 2
    ASSERT_FALSE(ts::Operations::can_realign(state, ts::Player::US, ts::countries::EGYPT));
    ASSERT_FALSE(ts::Operations::can_coup(state, ts::Player::US, ts::countries::EGYPT));

    // And USSR is not exempt under Tear Down This Wall
    state.countries[ts::countries::WEST_GERMANY].us_influence = 2;
    ASSERT_FALSE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::WEST_GERMANY));
}

TEST(OpsTest, ReformerRestrictsCoupNotRealign) {
    ts::GameState state{};
    state.defcon = 5;
    state.countries[ts::countries::WEST_GERMANY].us_influence = 2;

    // Before Reformer at DEFCON 5: USSR can coup and realign in West Germany
    ASSERT_TRUE(ts::Operations::can_coup(state, ts::Player::USSR, ts::countries::WEST_GERMANY));
    ASSERT_TRUE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::WEST_GERMANY));

    // Set THE_REFORMER_PLAYED
    state.set_flag(ts::effect_bits::THE_REFORMER_PLAYED);

    // Coup in Europe is blocked by Reformer
    ASSERT_FALSE(ts::Operations::can_coup(state, ts::Player::USSR, ts::countries::WEST_GERMANY));
    // Realignment in Europe is NOT blocked by Reformer
    ASSERT_TRUE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::WEST_GERMANY));
}

TEST(OpsTest, CanCoupOrRealignSideQuery) {
    ts::GameState state{};
    state.defcon = 2;
    // With empty board, no opponent influence anywhere -> false
    ASSERT_FALSE(ts::Operations::can_coup_or_realign(state, ts::Player::US));

    // Add US influence only in Europe (West Germany) at DEFCON 2 -> Europe restricted -> false for USSR
    state.countries[ts::countries::WEST_GERMANY].us_influence = 2;
    ASSERT_FALSE(ts::Operations::can_coup_or_realign(state, ts::Player::USSR));

    // Add US influence in Angola (Africa, never restricted by DEFCON) -> true for USSR
    state.countries[ts::countries::ANGOLA].us_influence = 1;
    ASSERT_TRUE(ts::Operations::can_coup_or_realign(state, ts::Player::USSR));
}
