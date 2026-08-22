#include "test_framework.hpp"
#include "game_test_wrapper.hpp"
#include "ts/constants.hpp"
#include "ts/scoring.hpp"
#include "ts/card_data.hpp"
#include "ts/space_race.hpp"

// =============================================================================
// Full Game Simulation Test Suite using GameTestWrapper
// =============================================================================

TEST(FullGameTest, FullGame_Turn1ToFinalScoring_DominationAndControlRequirements) {
    // Run a full game from Turn 1 to Final Scoring with Seed 44
    ts::GameTestWrapper wrapper(44);
    auto policy = ts::GameTestWrapper::create_balanced_policy();

    size_t steps_executed = wrapper.run_to_completion(policy, 5000);

    // 1. Terminal state & game completion checks
    ASSERT_TRUE(wrapper.is_terminal());
    ASSERT_EQ(wrapper.get_phase(), ts::Phase::GAME_OVER);
    ASSERT_EQ(wrapper.get_turn(), 11); // Turn 10 completed -> final scoring executed -> turn 11
    ASSERT_GT(steps_executed, 200);
    ASSERT_EQ(wrapper.step_count, steps_executed);
    ASSERT_EQ(wrapper.action_history.size(), steps_executed);

    // 2. VP & DEFCON safety checks (no premature victory or DEFCON suicide)
    ASSERT_GE(wrapper.get_vp(), -20);
    ASSERT_LE(wrapper.get_vp(), 20);
    ASSERT_GE(wrapper.get_defcon(), 2);
    ASSERT_LE(wrapper.get_defcon(), 5);

    // 3. Domination, Control, and Presence requirements
    ASSERT_GE(wrapper.total_dominations(), 1);
    ASSERT_GE(wrapper.total_controls(), 1);
    ASSERT_GE(wrapper.total_presences(), 1);

    // Verify regional status specifics
    auto ca_sum = wrapper.get_region_summary(ts::Region::CENTRAL_AMERICA);
    ASSERT_EQ(ca_sum.ussr_status, ts::RegionalStatus::CONTROL); // Central America controlled by USSR

    auto me_sum = wrapper.get_region_summary(ts::Region::MIDDLE_EAST);
    ASSERT_EQ(me_sum.us_status, ts::RegionalStatus::DOMINATION); // Middle East dominated by US

    auto eu_sum = wrapper.get_region_summary(ts::Region::EUROPE);
    ASSERT_EQ(eu_sum.us_status, ts::RegionalStatus::PRESENCE); // Europe balanced presence
    ASSERT_EQ(eu_sum.ussr_status, ts::RegionalStatus::PRESENCE);

    auto asia_sum = wrapper.get_region_summary(ts::Region::ASIA);
    ASSERT_EQ(asia_sum.us_status, ts::RegionalStatus::PRESENCE); // Asia presence
}

TEST(FullGameTest, Wrapper_TurnByTurn_ExecutionAndStateInspection) {
    ts::GameTestWrapper wrapper(44);
    auto policy = ts::GameTestWrapper::create_balanced_policy();

    // Verify Turn 1 starts in SETUP phase
    ASSERT_EQ(wrapper.get_turn(), 1);
    ASSERT_EQ(wrapper.get_phase(), ts::Phase::SETUP);
    ASSERT_FALSE(wrapper.is_terminal());

    // Step through turns 1 to 10
    for (uint8_t expected_turn = 1; expected_turn <= 10; ++expected_turn) {
        ASSERT_EQ(wrapper.get_turn(), expected_turn);
        size_t steps_in_turn = wrapper.run_turn(policy, 1000);
        ASSERT_GT(steps_in_turn, 0);

        // Turn invariants
        ASSERT_GE(wrapper.get_vp(), -20);
        ASSERT_LE(wrapper.get_vp(), 20);
        ASSERT_GE(wrapper.get_defcon(), 2);
    }

    // After Turn 10 finishes, final scoring runs and game terminates cleanly
    ASSERT_TRUE(wrapper.is_terminal());
    ASSERT_EQ(wrapper.get_phase(), ts::Phase::GAME_OVER);
    ASSERT_EQ(wrapper.get_turn(), 11);
}

TEST(FullGameTest, Wrapper_ScoringEvents_LogInspection) {
    ts::GameTestWrapper wrapper(44);
    auto policy = ts::GameTestWrapper::create_balanced_policy();

    wrapper.run_to_completion(policy, 5000);

    ASSERT_GT(wrapper.scoring_events.size(), 0);

    bool has_domination = false;
    bool has_control = false;
    bool has_presence = false;

    for (const auto& ev : wrapper.scoring_events) {
        if (ev.us_status == ts::RegionalStatus::DOMINATION || ev.ussr_status == ts::RegionalStatus::DOMINATION) {
            has_domination = true;
        }
        if (ev.us_status == ts::RegionalStatus::CONTROL || ev.ussr_status == ts::RegionalStatus::CONTROL) {
            has_control = true;
        }
        if (ev.us_status == ts::RegionalStatus::PRESENCE || ev.ussr_status == ts::RegionalStatus::PRESENCE) {
            has_presence = true;
        }
    }

    ASSERT_TRUE(has_domination);
    ASSERT_TRUE(has_control);
    ASSERT_TRUE(has_presence);
}
