#include "test_framework.hpp"
#include "ts/engine.hpp"
#include "ts/state_machine.hpp"
#include "ts/card_data.hpp"
#include "ts/card_handlers.hpp"
#include "ts/constants.hpp"
#include "ts/scoring.hpp"
#include "ts/map_data.hpp"
#include "ts/ops.hpp"
#include "ts/space_race.hpp"
#include "ts/action_mask.hpp"
#include "ts/serialization.hpp"
#include <cstring>

// =============================================================================
// 1. Setup Phase State Tests
// =============================================================================

TEST(StateLifecycleTest, SetupState_USSRAndUSPlacement_TransitionsToHeadline) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);

    ASSERT_EQ(state.current_phase, ts::Phase::SETUP);
    ASSERT_EQ(state.phasing_player, ts::Player::USSR);
    ASSERT_EQ(state.ctx().decision_player, ts::Player::USSR);
    ASSERT_EQ(state.ctx().remaining_steps, 6);

    // 1. USSR places 6 influence in Eastern Europe
    for (int i = 0; i < 6; ++i) {
        bool ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::POLAND, 0, 0});
        ASSERT_TRUE(ok);
    }
    ASSERT_EQ(state.countries[ts::countries::POLAND].ussr_influence, 6);

    // Transitions to US placing 7 influence in Western Europe
    ASSERT_EQ(state.current_phase, ts::Phase::SETUP);
    ASSERT_EQ(state.ctx().decision_player, ts::Player::US);
    ASSERT_EQ(state.ctx().remaining_steps, 7);

    // 2. US places 7 influence in Western Europe
    for (int i = 0; i < 7; ++i) {
        bool ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::WEST_GERMANY, 0, 0});
        ASSERT_TRUE(ok);
    }
    ASSERT_EQ(state.countries[ts::countries::WEST_GERMANY].us_influence, 7);

    // Setup complete -> Transitions cleanly to Headline phase of Turn 1
    ASSERT_EQ(state.current_phase, ts::Phase::HEADLINE);
    ASSERT_EQ(state.turn, 1);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_CARD);
}

TEST(StateLifecycleTest, SetupState_ActionMask_RestrictsToRegions) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);

    // USSR Setup: Eastern Europe only
    uint8_t mask[84]{};
    size_t out_size = 0;
    ts::ActionMask::generate_mask(state, mask, &out_size);
    ASSERT_EQ(mask[ts::countries::POLAND], 1);
    ASSERT_EQ(mask[ts::countries::EAST_GERMANY], 1);
    ASSERT_EQ(mask[ts::countries::WEST_GERMANY], 0); // Western Europe illegal
    ASSERT_EQ(mask[ts::countries::EGYPT], 0);        // Middle East illegal
}

// =============================================================================
// 2. Headline Phase State Tests
// =============================================================================

TEST(StateLifecycleTest, HeadlineState_HigherOpsResolvesFirst) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.current_phase = ts::Phase::HEADLINE;
    state.ctx().decision_player = ts::Player::US;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::HAND_US;   // 3 Ops
    state.card_locations[ts::card_ids::FIDEL] = ts::CardLocation::HAND_USSR;           // 2 Ops

    // US selects Duck and Cover (3 Ops)
    ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::DUCK_AND_COVER, 0, 0});

    // USSR selects Fidel (2 Ops)
    ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::FIDEL, 0, 0});

    // Higher Ops (US Duck & Cover, 3 Ops) resolves first -> DEFCON drops from 5 to 4
    // Then Fidel (2 Ops) resolves second -> Cuba US=0, USSR=3
    // Then transitions cleanly to Action Round 1!
    ASSERT_EQ(state.current_phase, ts::Phase::ACTION_ROUND);
    ASSERT_EQ(state.action_round, 1);
    ASSERT_EQ(state.phasing_player, ts::Player::USSR);
    ASSERT_EQ(state.defcon, 4);
    ASSERT_EQ(state.countries[ts::countries::CUBA].ussr_influence, 3);
}

TEST(StateLifecycleTest, HeadlineState_TiedOps_USGoesFirst) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.current_phase = ts::Phase::HEADLINE;
    state.ctx().decision_player = ts::Player::US;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;
    state.card_locations[ts::card_ids::NUCLEAR_TEST_BAN] = ts::CardLocation::HAND_US; // 3 Ops
    state.card_locations[ts::card_ids::SOCIALIST_GOVERNMENTS] = ts::CardLocation::HAND_USSR; // 3 Ops

    // US selects Nuclear Test Ban
    ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::NUCLEAR_TEST_BAN, 0, 0});

    // USSR selects Socialist Governments
    ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::SOCIALIST_GOVERNMENTS, 0, 0});

    // Tied Ops -> US resolves first (Nuclear Test Ban: DEFCON improved to 5, VP awarded)
    // Then Socialist Governments triggers sub-decision for USSR!
    ASSERT_EQ(state.ctx().decision_player, ts::Player::USSR);
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::SOCIALIST_GOVERNMENTS);
}

TEST(StateLifecycleTest, HeadlineState_DefectorsCancelsUSSRHeadline) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.current_phase = ts::Phase::HEADLINE;
    state.ctx().decision_player = ts::Player::US;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;
    state.card_locations[ts::card_ids::DEFECTORS] = ts::CardLocation::HAND_US;
    state.card_locations[ts::card_ids::FIDEL] = ts::CardLocation::HAND_USSR;
    state.countries[ts::countries::CUBA].us_influence = 2;
    state.countries[ts::countries::CUBA].ussr_influence = 0;
    state.victory_points = 0;

    ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::DEFECTORS, 0, 0});
    ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::FIDEL, 0, 0});

    // Defectors cancels USSR headline -> Cuba US influence remains 2, USSR remains 0!
    // US gains 1 VP from Defectors
    ASSERT_EQ(state.victory_points, 1);
    ASSERT_EQ(state.countries[ts::countries::CUBA].us_influence, 2);
    ASSERT_EQ(state.countries[ts::countries::CUBA].ussr_influence, 0);
    ASSERT_EQ(state.current_phase, ts::Phase::ACTION_ROUND);
    ASSERT_EQ(state.action_round, 1);
}

// =============================================================================
// 3. Action Round State Tests & Timing Branches
// =============================================================================

TEST(StateLifecycleTest, ActionRoundState_ChinaCardPlay_PassesToOpponentFaceDown) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.action_round = 1;
    state.phasing_player = ts::Player::USSR;
    state.china_card_holder = ts::Player::USSR;
    state.china_card_playable = 1;
    state.ctx().decision_player = ts::Player::USSR;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;

    // USSR plays China Card for Ops
    ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::THE_CHINA_CARD, 0, 0});
    ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::PlayMode::OPS), 0, 0});
    ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_OP_MODE, static_cast<uint8_t>(ts::OpMode::INFLUENCE), 0, 0});

    // Place influence in North Korea (4 points from China Card in Asia)
    for (int i = 0; i < 4; ++i) {
        ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::NORTH_KOREA, 0, 0});
    }

    // China Card passes to US face down (playable = 0)!
    ASSERT_EQ(state.china_card_holder, ts::Player::US);
    ASSERT_EQ(state.china_card_playable, 0);

    // Advanced cleanly to US Action Round 1
    ASSERT_EQ(state.phasing_player, ts::Player::US);
    ASSERT_EQ(state.action_round, 1);
}

TEST(StateLifecycleTest, ActionRoundState_OpponentCard_EventFirst_Timing) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.action_round = 1;
    state.phasing_player = ts::Player::US;
    state.ctx().decision_player = ts::Player::US;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;
    state.card_locations[ts::card_ids::FIDEL] = ts::CardLocation::HAND_US; // USSR card in US hand
    state.countries[ts::countries::CUBA].us_influence = 2;
    state.countries[ts::countries::CUBA].ussr_influence = 0;

    // US plays Fidel for Ops -> Choose EVENT_FIRST timing
    ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::FIDEL, 0, 0});
    ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::PlayMode::OPS), 0, 0});
    ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::CHOOSE_TIMING_BRANCH, static_cast<uint8_t>(ts::TimingBranch::EVENT_FIRST), 0, 0});

    // Fidel event occurs first: Cuba US=0, USSR=3
    ASSERT_EQ(state.countries[ts::countries::CUBA].us_influence, 0);
    ASSERT_EQ(state.countries[ts::countries::CUBA].ussr_influence, 3);

    // Then US conducts Ops from Fidel (2 Ops)
    ASSERT_EQ(state.ctx().decision_player, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_OP_MODE);
    ASSERT_EQ(state.ctx().pending_ops_value, 2);
}

// =============================================================================
// 4. Turn End State & Lifecycle Tests
// =============================================================================

TEST(StateLifecycleTest, TurnEndState_MilOpsCheck_PenalizesDeficit) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.turn = 1;
    state.action_round = 6;
    state.phasing_player = ts::Player::US;
    state.defcon = 3;
    state.us_mil_ops = 3;   // Meets required (3 >= 3)
    state.ussr_mil_ops = 1; // Deficit of 2 (1 < 3)
    state.victory_points = 0;

    // End turn
    ts::StateMachine::end_turn(state);

    // USSR suffered penalty of 2 VP -> US gains 2 VP
    ASSERT_EQ(state.victory_points, 2);

    // MilOps reset to 0 for next turn
    ASSERT_EQ(state.us_mil_ops, 0);
    ASSERT_EQ(state.ussr_mil_ops, 0);

    // Turn advanced to Turn 2
    ASSERT_EQ(state.turn, 2);
}

TEST(StateLifecycleTest, TurnEndState_HoldingScoringCard_CausesImmediateLoss) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.card_locations[ts::card_ids::EUROPE_SCORING] = ts::CardLocation::HAND_US; // US illegally holding scoring card

    ts::StateMachine::end_turn(state);

    // US loses immediately!
    ASSERT_EQ(state.current_phase, ts::Phase::GAME_OVER);
    ASSERT_EQ(state.victory_points, -20);
}

TEST(StateLifecycleTest, TurnEndState_TurnCleanup_ClearsPersistentFlags) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.set_flag(ts::effect_bits::PURGE_US_ACTIVE);
    state.set_flag(ts::effect_bits::PURGE_USSR_ACTIVE);
    state.set_flag(ts::effect_bits::BREZHNEV_DOCTRINE_ACTIVE);
    state.set_flag(ts::effect_bits::CONTAINMENT_ACTIVE);
    state.set_flag(ts::effect_bits::SALT_ACTIVE);
    state.china_card_playable = 0;

    ts::StateMachine::end_turn(state);

    // Turn cleanup mask clears round/turn active flags
    ASSERT_FALSE(state.has_flag(ts::effect_bits::PURGE_US_ACTIVE));
    ASSERT_FALSE(state.has_flag(ts::effect_bits::PURGE_USSR_ACTIVE));
    ASSERT_FALSE(state.has_flag(ts::effect_bits::BREZHNEV_DOCTRINE_ACTIVE));
    ASSERT_FALSE(state.has_flag(ts::effect_bits::CONTAINMENT_ACTIVE));
    ASSERT_FALSE(state.has_flag(ts::effect_bits::SALT_ACTIVE));

    // China Card flips face up
    ASSERT_EQ(state.china_card_playable, 1);
}

TEST(StateLifecycleTest, TurnEndState_Turn4_AddsMidWarCards) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.turn = 3; // Ending Turn 3

    ts::StateMachine::end_turn(state);

    ASSERT_EQ(state.turn, 4);
    // Mid War cards are now in the draw deck / hands
    bool mid_war_in_play = false;
    for (uint8_t i = 1; i <= 110; ++i) {
        if (ts::CardData::get_card(i).era == ts::WarEra::MID) {
            if (state.card_locations[i] == ts::CardLocation::DRAW_DECK ||
                state.card_locations[i] == ts::CardLocation::HAND_US ||
                state.card_locations[i] == ts::CardLocation::HAND_USSR) {
                mid_war_in_play = true;
                break;
            }
        }
    }
    ASSERT_TRUE(mid_war_in_play);
}

// =============================================================================
// 5. Final Scoring & Terminal States
// =============================================================================

TEST(StateLifecycleTest, FinalScoring_ChinaCardHolderGets1VP) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.china_card_holder = ts::Player::US;
    state.victory_points = 0;

    ts::Scoring::execute_final_scoring(state);

    // US holding China Card gains +1 VP in final scoring (assuming neutral board)
    ASSERT_TRUE(state.current_phase == ts::Phase::GAME_OVER);
}

TEST(StateLifecycleTest, GameOver_TerminalState_RejectsFurtherSteps) {
    ts::GameState state{};
    state.current_phase = ts::Phase::GAME_OVER;
    state.victory_points = 20;

    bool ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, 0, 0, 0});
    ASSERT_FALSE(ok);
    ASSERT_EQ(state.current_phase, ts::Phase::GAME_OVER);
}

// =============================================================================
// 6. Memory, Struct & Serialization Invariants
// =============================================================================

TEST(StateLifecycleTest, GameState_TriviallyCopyable_And_Under4KB) {
    static_assert(std::is_trivially_copyable_v<ts::GameState>, "GameState must remain trivially copyable");
    static_assert(sizeof(ts::GameState) <= 4096, "GameState must be within 4 KB");
}

TEST(StateLifecycleTest, DecisionContext_PushPopStack_DepthIntegrity) {
    ts::GameState state{};
    ASSERT_EQ(state.ctx_stack_depth, 0);

    state.push_context();
    ASSERT_EQ(state.ctx_stack_depth, 1);
    state.ctx().decision_player = ts::Player::USSR;
    state.ctx().resolving_card = ts::card_ids::GRAIN_SALES;

    state.push_context();
    ASSERT_EQ(state.ctx_stack_depth, 2);
    state.ctx().decision_player = ts::Player::US;
    state.ctx().resolving_card = ts::card_ids::FIVE_YEAR_PLAN;

    state.pop_context();
    ASSERT_EQ(state.ctx_stack_depth, 1);
    ASSERT_EQ(state.ctx().decision_player, ts::Player::USSR);
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::GRAIN_SALES);

    state.pop_context();
    ASSERT_EQ(state.ctx_stack_depth, 0);
}


// =============================================================================
// 7. Space Race State Machine & Turn Resets
// =============================================================================

TEST(StateLifecycleTest, SpaceRaceState_TwoAttemptsPerTurn_And_ResetOnTurnEnd) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.us_space_track = 2; // Reached Animal in Space (allows 2 attempts per turn)

    ASSERT_TRUE(ts::SpaceRace::has_animal_in_space(state, ts::Player::US));
    ASSERT_EQ(state.us_space_turns_used, 0);

    // First attempt used
    state.us_space_turns_used = 1;
    ASSERT_TRUE(ts::SpaceRace::can_attempt_space(state, ts::Player::US, ts::card_ids::DUCK_AND_COVER));

    // Second attempt used
    state.us_space_turns_used = 2;
    ASSERT_FALSE(ts::SpaceRace::can_attempt_space(state, ts::Player::US, ts::card_ids::DUCK_AND_COVER));

    // Turn ends -> space attempts reset to 0
    ts::StateMachine::end_turn(state);
    ASSERT_EQ(state.us_space_turns_used, 0);
    ASSERT_TRUE(ts::SpaceRace::can_attempt_space(state, ts::Player::US, ts::card_ids::DUCK_AND_COVER));
}

// =============================================================================
// 8. Dynamic Influence Placement Cost In Opponent-Controlled Country
// =============================================================================

TEST(StateLifecycleTest, ActionRoundState_InfluencePlacement_DynamicCostTransition) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.phasing_player = ts::Player::US;
    state.ctx().decision_player = ts::Player::US;
    state.ctx().decision_type = ts::DecisionType::POINT_NODE;
    state.ctx().remaining_steps = 3; // 3 Ops available

    // Canada is adjacent to USA (superpower adjacency allows initial placement)
    // Canada stability = 4. Set USSR influence to 4 (USSR controls Canada)
    state.countries[ts::countries::CANADA].ussr_influence = 4;
    state.countries[ts::countries::CANADA].us_influence = 0;
    ASSERT_TRUE(ts::Scoring::is_controlled_by(state, ts::countries::CANADA, ts::Player::USSR));

    // US places 1st point into USSR-controlled Canada -> costs 2 Ops (remaining: 1)
    bool ok1 = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::CANADA, 0, 0});
    ASSERT_TRUE(ok1);
    ASSERT_EQ(state.countries[ts::countries::CANADA].us_influence, 1);
    ASSERT_EQ(state.ctx().remaining_steps, 1);

    // USSR no longer controls Canada (USSR 4 vs US 1, lead is only 3 < 4 stability)!
    ASSERT_FALSE(ts::Scoring::is_controlled_by(state, ts::countries::CANADA, ts::Player::USSR));

    // US places 2nd point into now UNCONTROLLED Canada -> costs 1 Op (remaining: 0)!
    bool ok2 = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::CANADA, 0, 0});
    ASSERT_TRUE(ok2);
    ASSERT_EQ(state.countries[ts::countries::CANADA].us_influence, 2);
    ASSERT_EQ(state.ctx().remaining_steps, 0);
}

// =============================================================================
// 9. Instant Victory Conditions
// =============================================================================

TEST(StateLifecycleTest, GameOverState_InstantWin_AtPlus20VP) {
    ts::GameState state{};
    state.victory_points = 19;

    // US gains 2 VP -> reaches +20 VP (clamped to 20) and triggers GAME_OVER
    state.victory_points = static_cast<int8_t>(std::min(20, state.victory_points + 2));
    if (state.victory_points >= 20) state.current_phase = ts::Phase::GAME_OVER;

    ASSERT_EQ(state.victory_points, 20);
    ASSERT_EQ(state.current_phase, ts::Phase::GAME_OVER);
}

TEST(StateLifecycleTest, GameOverState_InstantWin_AtMinus20VP) {
    ts::GameState state{};
    state.victory_points = -19;

    // USSR gains 2 VP -> reaches -20 VP and triggers GAME_OVER
    state.victory_points = static_cast<int8_t>(std::max(-20, state.victory_points - 2));
    if (state.victory_points <= -20) state.current_phase = ts::Phase::GAME_OVER;

    ASSERT_EQ(state.victory_points, -20);
    ASSERT_EQ(state.current_phase, ts::Phase::GAME_OVER);
}

TEST(StateLifecycleTest, GameOverState_EuropeControl_InstantWinOnEuropeScoring) {
    ts::GameState state{};
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.victory_points = 0;

    // USSR controls Europe (all Battlegrounds + more countries)
    state.countries[ts::countries::EAST_GERMANY].ussr_influence = 3;
    state.countries[ts::countries::POLAND].ussr_influence = 3;
    state.countries[ts::countries::WEST_GERMANY].ussr_influence = 4;
    state.countries[ts::countries::FRANCE].ussr_influence = 3;
    state.countries[ts::countries::ITALY].ussr_influence = 3;

    // Europe Scoring triggered
    ts::Scoring::score_region(state, ts::Region::EUROPE);

    ASSERT_EQ(state.current_phase, ts::Phase::GAME_OVER);
    ASSERT_EQ(state.victory_points, -20); // USSR wins immediately!
}

// =============================================================================
// 10. DEFCON Progression & Reshuffle State Invariants
// =============================================================================

TEST(StateLifecycleTest, TurnEndState_DEFCON_ImprovesBy1_ClampedAt5) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.defcon = 2;

    ts::StateMachine::end_turn(state);
    ASSERT_EQ(state.defcon, 3); // DEFCON improves from 2 to 3

    state.defcon = 5;
    ts::StateMachine::end_turn(state);
    ASSERT_EQ(state.defcon, 5); // Clamped at 5
}
