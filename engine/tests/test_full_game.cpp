#include "ts/prng.hpp"
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
    uint64_t working_seed = 0;
    auto policy = ts::GameTestWrapper::create_balanced_policy();
    for (uint64_t s = 1; s <= 200; ++s) {
        ts::GameTestWrapper w(s);
        w.run_to_completion(policy, 5000);
        if (w.get_turn() == 11 && w.total_dominations() >= 1 && w.total_controls() >= 1 && w.total_presences() >= 1) {
            working_seed = s;
            std::cout << "FOUND SEED: " << s << std::endl;
            break;
        }
    }
    ASSERT_GT(working_seed, 0);
    ts::GameTestWrapper wrapper(working_seed);
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
}

TEST(FullGameTest, Wrapper_TurnByTurn_ExecutionAndStateInspection) {
    ts::GameTestWrapper wrapper(5);
    auto policy = ts::GameTestWrapper::create_balanced_policy();

    // Verify Turn 1 starts in SETUP phase
    ASSERT_EQ(wrapper.get_turn(), 1);
    ASSERT_EQ(wrapper.get_phase(), ts::Phase::SETUP);
    ASSERT_FALSE(wrapper.is_terminal());

    // Step through turns 1 to 10
    for (uint8_t expected_turn = 1; expected_turn <= 10; ++expected_turn) {
        if (wrapper.is_terminal()) break;
        ASSERT_EQ(wrapper.get_turn(), expected_turn);
        size_t steps_in_turn = wrapper.run_turn(policy, 1000);
        ASSERT_GT(steps_in_turn, 0);

        // Turn invariants
        ASSERT_GE(wrapper.get_vp(), -20);
        ASSERT_LE(wrapper.get_vp(), 20);
        if (wrapper.get_defcon() < 2) {
            std::cout << "DEFCON dropped to " << (int)wrapper.get_defcon() << " on turn " << (int)expected_turn << " phase: " << (int)wrapper.get_phase() << std::endl;
        }
        ASSERT_GE(wrapper.get_defcon(), 2);
    }

    // The game must terminate cleanly. It may end early on a 20 VP win rather than running
    // to final scoring -- with this policy and seed the USSR now reaches -20 on turn 10 --
    // so assert the termination is well formed rather than that the game lasted a fixed
    // number of turns, which is a property of the scripted policy and not of the rules.
    ASSERT_TRUE(wrapper.is_terminal());
    ASSERT_EQ(wrapper.get_phase(), ts::Phase::GAME_OVER);
    ASSERT_LE(wrapper.get_turn(), 11);
    const bool reached_final_scoring = (wrapper.get_turn() == 11);
    const bool won_on_vp = (wrapper.get_vp() == 20 || wrapper.get_vp() == -20);
    ASSERT_TRUE(reached_final_scoring || won_on_vp);
}

// What this test checks is how scoring classifies domination, control and presence, so it needs
// a game that plays scoring cards and reaches all three. The scripted policy reaches a 20 VP win
// with no scoring at all on many seeds, and *which* seeds do that moves with any change to the
// action space or to the order things resolve in -- a property of the policy, not of the rules.
// A pinned seed therefore fails for reasons that have nothing to do with scoring: seed 6 stopped
// producing scoring events when headline Ops began discarding the card they were spent on. So
// the seed is searched for, the way FullGame_Turn1ToFinalScoring does above, and the search
// itself asserts that such a game exists.
TEST(FullGameTest, Wrapper_ScoringEvents_LogInspection) {
    auto policy = ts::GameTestWrapper::create_balanced_policy();
    uint64_t working_seed = 0;
    for (uint64_t s = 1; s <= 200; ++s) {
        ts::GameTestWrapper w(s);
        w.run_to_completion(policy, 5000);
        if (w.total_dominations() >= 1 && w.total_controls() >= 1 && w.total_presences() >= 1) {
            working_seed = s;
            break;
        }
    }
    // No seed in 1..200 reached all three regional statuses.
    ASSERT_GT(working_seed, 0);

    ts::GameTestWrapper wrapper(working_seed);
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

TEST(FullGameTest, EventBiasedFuzzing_HighEventProbability_MaintainsInvariantsAcrossManyGames) {
    uint64_t fuzzer_prng = 2026;
    uint32_t total_games = 0;
    uint32_t total_events = 0;
    uint32_t target_games = 200;

    uint8_t mask_buf[128];
    size_t mask_size = 0;

    while (total_games < target_games) {
        ts::GameState state{};
        ts::Engine::init_game(state, ts::Prng::next_u64(fuzzer_prng));

        while (!ts::Engine::is_terminal(state)) {
            ts::Engine::get_legal_action_mask(state, mask_buf, &mask_size);
            ASSERT_GT(mask_size, 0);

            std::vector<uint8_t> legal_indices;
            for (size_t i = 0; i < mask_size; ++i) {
                if (mask_buf[i]) legal_indices.push_back(static_cast<uint8_t>(i));
            }

            if (legal_indices.empty()) {
                if (state.ctx().allow_early_stop || state.ctx().decision_type == ts::DecisionType::POINT_NODE) {
                    ts::MicroAction action{};
                    action.decision_type = state.ctx().decision_type;
                    action.flags = ts::action_flags::CONFIRM_DONE;
                    ts::Engine::step(state, action);
                    continue;
                }
                ASSERT_TRUE(false);
            }

            uint8_t chosen_action_id = legal_indices[0];
            ts::DecisionType d_type = state.ctx().decision_type;

            if (d_type == ts::DecisionType::SELECT_PLAY_MODE) {
                bool event_legal = (std::find(legal_indices.begin(), legal_indices.end(), 0) != legal_indices.end());
                uint32_t roll = ts::Prng::random_index(fuzzer_prng, 100);
                if (event_legal && roll < 95) {
                    chosen_action_id = 0;
                    total_events++;
                } else {
                    uint32_t idx = ts::Prng::random_index(fuzzer_prng, static_cast<uint32_t>(legal_indices.size()));
                    chosen_action_id = legal_indices[idx];
                }
            } else if (d_type == ts::DecisionType::CHOOSE_TIMING_BRANCH) {
                bool event_first_legal = (std::find(legal_indices.begin(), legal_indices.end(), 1) != legal_indices.end());
                uint32_t roll = ts::Prng::random_index(fuzzer_prng, 100);
                if (event_first_legal && roll < 95) {
                    chosen_action_id = 1;
                    total_events++;
                } else {
                    uint32_t idx = ts::Prng::random_index(fuzzer_prng, static_cast<uint32_t>(legal_indices.size()));
                    chosen_action_id = legal_indices[idx];
                }
            } else if (d_type == ts::DecisionType::SELECT_CARD && state.ctx().resolving_card == 0) {
                std::vector<uint8_t> event_cards;
                for (uint8_t cid : legal_indices) {
                    if (cid >= 1 && cid <= 110) {
                        const auto& c_info = ts::CardData::get_card(cid);
                        if (!c_info.is_scoring) {
                            event_cards.push_back(cid);
                        }
                    }
                }
                uint32_t roll = ts::Prng::random_index(fuzzer_prng, 100);
                if (!event_cards.empty() && roll < 90) {
                    uint32_t idx = ts::Prng::random_index(fuzzer_prng, static_cast<uint32_t>(event_cards.size()));
                    chosen_action_id = event_cards[idx];
                } else {
                    uint32_t idx = ts::Prng::random_index(fuzzer_prng, static_cast<uint32_t>(legal_indices.size()));
                    chosen_action_id = legal_indices[idx];
                }
            } else {
                uint32_t chosen_idx = ts::Prng::random_index(fuzzer_prng, static_cast<uint32_t>(legal_indices.size()));
                chosen_action_id = legal_indices[chosen_idx];
            }

            ts::MicroAction action{};
            action.decision_type = d_type;
            action.primary_id = chosen_action_id;

            ts::Engine::step(state, action);

            ASSERT_GE(state.victory_points, -20);
            ASSERT_LE(state.victory_points, 20);
            ASSERT_GE(state.defcon, 1);
            ASSERT_LE(state.defcon, 5);
            ASSERT_LE(state.us_mil_ops, 5);
            ASSERT_LE(state.ussr_mil_ops, 5);
            ASSERT_LE(state.us_space_track, 8);
            ASSERT_LE(state.ussr_space_track, 8);
            ASSERT_LE(state.ctx_stack_depth, 2);
        }
        total_games++;
    }

    ASSERT_EQ(total_games, target_games);
    ASSERT_GT(total_events, 500);
}
