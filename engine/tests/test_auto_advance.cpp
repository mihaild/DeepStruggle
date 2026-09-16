#include "test_framework.hpp"
#include "ts/engine.hpp"
#include "ts/map_data.hpp"
#include "ts/state_machine.hpp"
#include "ts/card_data.hpp"
#include "ts/card_handlers.hpp"
#include "ts/constants.hpp"
#include "ts/scoring.hpp"
#include "ts/ops.hpp"
#include "ts/action_mask.hpp"
#include <vector>

TEST(AutoAdvanceTest, SingleChoiceOpMode_AutoAdvancesToPointNode) {
    ts::GameState state{};
    ts::Engine::init_game(state, 12345);
   state.current_phase = ts::Phase::ACTION_ROUND;
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.turn = 1;
    state.action_round = 1;
    state.phasing_player = ts::Player::US;
    state.defcon = 2; // Coups banned in Europe, Asia, ME

    // Clear USSR influence in LatAm and Africa so no coups/realignments are legal there either
    for (uint8_t c = 0; c < 84; ++c) {
        auto reg = ts::MapData::get_country(c).region;
        if (reg == ts::Region::CENTRAL_AMERICA || reg == ts::Region::SOUTH_AMERICA || reg == ts::Region::AFRICA) {
            state.countries[c].ussr_influence = 0;
            state.countries[c].us_influence = 0;
        }
    }

    // Set US decision to SELECT_OP_MODE with 4 Ops
    state.ctx().decision_player = ts::Player::US;
    state.ctx().decision_type = ts::DecisionType::SELECT_OP_MODE;
    state.ctx().pending_ops_value = 4;
    state.ctx().pending_op_card = 0;

    // Check mask: only INFLUENCE (flat action 116) should be legal
    uint8_t mask[212];
    ts::ActionMask::generate_flat_mask_212(state, mask);
    ASSERT_EQ(mask[116], 1);
    ASSERT_EQ(mask[117], 0); // Coup
    ASSERT_EQ(mask[118], 0); // Realign

    // Calling auto_advance_step directly should advance to POINT_NODE
    size_t advanced = ts::Engine::auto_advance_step(state);
    ASSERT_GE(advanced, 1u);
    ASSERT_EQ(static_cast<int>(state.ctx().decision_type), static_cast<int>(ts::DecisionType::POINT_NODE));
    ASSERT_EQ(static_cast<int>(state.ctx().op_mode), static_cast<int>(ts::OpMode::INFLUENCE));
}

TEST(AutoAdvanceTest, SuezCrisis_Le4_AutoResolvesCompletely) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
   state.current_phase = ts::Phase::ACTION_ROUND;
    state.current_phase = ts::Phase::ACTION_ROUND;

    // Set UK=2, France=1, Israel=0 (Sum = 3 <= 4)
    state.countries[ts::countries::UNITED_KINGDOM].us_influence = 2;
    state.countries[ts::countries::FRANCE].us_influence = 1;
    state.countries[ts::countries::ISRAEL].us_influence = 0;

    ts::CardHandlers::trigger_event(state, ts::card_ids::SUEZ_CRISIS, ts::Player::USSR);
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::SUEZ_CRISIS);
    ASSERT_EQ(state.ctx().remaining_steps, 4);

    // Auto advance should resolve all 3 points and finish the event
    size_t adv = ts::Engine::auto_advance_step(state);
    ASSERT_GE(adv, 3u);
    ASSERT_EQ(state.ctx().resolving_card, 0);
    ASSERT_EQ(state.countries[ts::countries::UNITED_KINGDOM].us_influence, 0);
    ASSERT_EQ(state.countries[ts::countries::FRANCE].us_influence, 0);
    ASSERT_EQ(state.countries[ts::countries::ISRAEL].us_influence, 0);
}

TEST(AutoAdvanceTest, SuezCrisis_Gt4_DoesNotAutoResolve) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
   state.current_phase = ts::Phase::ACTION_ROUND;
    state.current_phase = ts::Phase::ACTION_ROUND;

    // Set UK=2, France=2, Israel=2 (Sum = 6 > 4)
    state.countries[ts::countries::UNITED_KINGDOM].us_influence = 2;
    state.countries[ts::countries::FRANCE].us_influence = 2;
    state.countries[ts::countries::ISRAEL].us_influence = 2;

    ts::CardHandlers::trigger_event(state, ts::card_ids::SUEZ_CRISIS, ts::Player::USSR);
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::SUEZ_CRISIS);

    // Auto advance should NOT wipe them because USSR has a choice
    size_t adv = ts::Engine::auto_advance_step(state);
    ASSERT_EQ(adv, 0u);
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::SUEZ_CRISIS);
    ASSERT_EQ(state.countries[ts::countries::UNITED_KINGDOM].us_influence, 2);
}

TEST(AutoAdvanceTest, MuslimRevolution_Le2_AutoResolvesCompletely) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);

    state.current_phase = ts::Phase::ACTION_ROUND;

    // Set Iran=2, Egypt=1, all other 6 eligible countries=0
    uint8_t mr_targets[] = { ts::countries::SUDAN, ts::countries::IRAN, ts::countries::IRAQ,
                             ts::countries::EGYPT, ts::countries::LIBYA, ts::countries::SAUDI_ARABIA,
                             ts::countries::SYRIA, ts::countries::JORDAN };
    for (uint8_t c : mr_targets) state.countries[c].us_influence = 0;
    state.countries[ts::countries::IRAN].us_influence = 2;
    state.countries[ts::countries::EGYPT].us_influence = 1;

    ts::CardHandlers::trigger_event(state, ts::card_ids::MUSLIM_REVOLUTION, ts::Player::USSR);
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::MUSLIM_REVOLUTION);

    size_t adv = ts::Engine::auto_advance_step(state);
    ASSERT_GE(adv, 2u);
    ASSERT_EQ(state.ctx().resolving_card, 0);
    ASSERT_EQ(state.countries[ts::countries::IRAN].us_influence, 0);
    ASSERT_EQ(state.countries[ts::countries::EGYPT].us_influence, 0);
}

TEST(AutoAdvanceTest, MuslimRevolution_Gt2_DoesNotAutoResolve) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
   state.current_phase = ts::Phase::ACTION_ROUND;
    state.current_phase = ts::Phase::ACTION_ROUND;

    uint8_t mr_targets[] = { ts::countries::SUDAN, ts::countries::IRAN, ts::countries::IRAQ,
                             ts::countries::EGYPT, ts::countries::LIBYA, ts::countries::SAUDI_ARABIA,
                             ts::countries::SYRIA, ts::countries::JORDAN };
    for (uint8_t c : mr_targets) state.countries[c].us_influence = 0;
    state.countries[ts::countries::IRAN].us_influence = 2;
    state.countries[ts::countries::EGYPT].us_influence = 1;
    state.countries[ts::countries::SAUDI_ARABIA].us_influence = 1; // 3 countries

    ts::CardHandlers::trigger_event(state, ts::card_ids::MUSLIM_REVOLUTION, ts::Player::USSR);
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::MUSLIM_REVOLUTION);

    size_t adv = ts::Engine::auto_advance_step(state);
    ASSERT_EQ(adv, 0u);
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::MUSLIM_REVOLUTION);
}

TEST(AutoAdvanceTest, EastEuropeanUnrest_Le3_AutoResolvesCompletely) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
   state.current_phase = ts::Phase::ACTION_ROUND;
    state.current_phase = ts::Phase::ACTION_ROUND;

    uint8_t eeu_targets[] = { ts::countries::FINLAND, ts::countries::AUSTRIA, ts::countries::EAST_GERMANY,
                              ts::countries::POLAND, ts::countries::CZECHOSLOVAKIA, ts::countries::HUNGARY,
                              ts::countries::YUGOSLAVIA, ts::countries::ROMANIA, ts::countries::BULGARIA };
    for (uint8_t c : eeu_targets) state.countries[c].ussr_influence = 0;
    state.countries[ts::countries::POLAND].ussr_influence = 3;
    state.countries[ts::countries::EAST_GERMANY].ussr_influence = 2;

    ts::CardHandlers::trigger_event(state, ts::card_ids::EAST_EUROPEAN_UNREST, ts::Player::US);
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::EAST_EUROPEAN_UNREST);

    size_t adv = ts::Engine::auto_advance_step(state);
    ASSERT_GE(adv, 2u);
    ASSERT_EQ(state.ctx().resolving_card, 0);
    ASSERT_EQ(state.countries[ts::countries::POLAND].ussr_influence, 2);
    ASSERT_EQ(state.countries[ts::countries::EAST_GERMANY].ussr_influence, 1);
}

TEST(AutoAdvanceTest, TrumanDoctrine_1Country_AutoResolves) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
   state.current_phase = ts::Phase::ACTION_ROUND;
    state.current_phase = ts::Phase::ACTION_ROUND;

    // Clear USSR influence across Europe except Austria (uncontrolled)
    for (uint8_t c = 0; c < 84; ++c) {
        if (ts::MapData::get_country(c).region == ts::Region::EUROPE) {
            state.countries[c].ussr_influence = 0;
            state.countries[c].us_influence = 0;
        }
    }
    state.countries[ts::countries::AUSTRIA].ussr_influence = 3;
    state.countries[ts::countries::AUSTRIA].us_influence = 0;

    ts::CardHandlers::trigger_event(state, ts::card_ids::TRUMAN_DOCTRINE, ts::Player::US);
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::TRUMAN_DOCTRINE);

    size_t adv = ts::Engine::auto_advance_step(state);
    ASSERT_EQ(adv, 1u);
    ASSERT_EQ(state.ctx().resolving_card, 0);
    ASSERT_EQ(state.countries[ts::countries::AUSTRIA].ussr_influence, 0);
}

TEST(AutoAdvanceTest, IndependentReds_1Country_AutoResolves) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
   state.current_phase = ts::Phase::ACTION_ROUND;
    state.current_phase = ts::Phase::ACTION_ROUND;

    uint8_t ir_targets[] = { ts::countries::YUGOSLAVIA, ts::countries::ROMANIA, ts::countries::BULGARIA,
                             ts::countries::HUNGARY, ts::countries::CZECHOSLOVAKIA };
    for (uint8_t c : ir_targets) {
        state.countries[c].ussr_influence = 0;
        state.countries[c].us_influence = 0;
    }
    state.countries[ts::countries::YUGOSLAVIA].ussr_influence = 3;

    ts::CardHandlers::trigger_event(state, ts::card_ids::INDEPENDENT_REDS, ts::Player::US);
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::INDEPENDENT_REDS);

    size_t adv = ts::Engine::auto_advance_step(state);
    ASSERT_EQ(adv, 1u);
    ASSERT_EQ(state.ctx().resolving_card, 0);
    ASSERT_EQ(state.countries[ts::countries::YUGOSLAVIA].us_influence, 3);
}

static uint16_t pick_deterministic_action(const uint8_t* mask) {
    for (int i = 0; i < 211; ++i) {
        if (mask[i]) return static_cast<uint16_t>(i);
    }
    return 211;
}

TEST(AutoAdvanceTest, DeterministicEquivalence_FullGame) {
    const uint64_t seeds[] = { 1, 42, 100, 777, 10042 };
    for (uint64_t seed : seeds) {
        // Run 1: without auto-advance
        ts::GameState s1{};
        ts::Engine::init_game(s1, seed);

        int max_steps = 1500;
        int step_cnt1 = 0;
        while (!ts::Engine::is_terminal(s1) && step_cnt1 < max_steps) {
            uint8_t mask[212];
            ts::ActionMask::generate_flat_mask_212(s1, mask);
            uint16_t chosen = pick_deterministic_action(mask);
            ASSERT_TRUE(ts::Engine::step_flat(s1, chosen, false));
            step_cnt1++;
        }

        // Run 2: with auto-advance
        ts::GameState s2{};
        ts::Engine::init_game(s2, seed);
        ts::Engine::auto_advance_step(s2);

        int step_cnt2 = 0;
        while (!ts::Engine::is_terminal(s2) && step_cnt2 < max_steps) {
            uint8_t mask[212];
            ts::ActionMask::generate_flat_mask_212(s2, mask);
            uint16_t chosen = pick_deterministic_action(mask);
            ASSERT_TRUE(ts::Engine::step_flat(s2, chosen, true));
            step_cnt2++;
        }

        // Both runs must reach the EXACT same final board state
        ASSERT_EQ(s1.victory_points, s2.victory_points);
        ASSERT_EQ(s1.defcon, s2.defcon);
        ASSERT_EQ(s1.turn, s2.turn);
        ASSERT_EQ(s1.action_round, s2.action_round);
        ASSERT_EQ(s1.current_phase, s2.current_phase);
        ASSERT_EQ(ts::Engine::get_terminal_utility(s1), ts::Engine::get_terminal_utility(s2));

        for (uint8_t c = 0; c < 84; ++c) {
            ASSERT_EQ(s1.countries[c].us_influence, s2.countries[c].us_influence);
            ASSERT_EQ(s1.countries[c].ussr_influence, s2.countries[c].ussr_influence);
        }
        for (uint8_t card = 1; card <= 110; ++card) {
            ASSERT_EQ(s1.card_locations[card], s2.card_locations[card]);
        }
        ASSERT_EQ(s1.persistent_effects, s2.persistent_effects);

        // Auto-advance must take strictly fewer micro-actions
        ASSERT_LT(step_cnt2, step_cnt1);
    }
}
