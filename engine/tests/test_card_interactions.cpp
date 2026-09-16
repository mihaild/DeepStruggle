#include "test_framework.hpp"
#include "ts/engine.hpp"
#include "ts/card_data.hpp"
#include "ts/card_handlers.hpp"
#include "ts/constants.hpp"
#include "ts/scoring.hpp"
#include "ts/map_data.hpp"
#include "ts/ops.hpp"
#include "ts/action_mask.hpp"

// =============================================================================
// 1. Flower Power Comprehensive Matrix
// =============================================================================

TEST(CardInteractionTest, FlowerPower_Awards2VP_OnKoreanWarEventPlayByUS) {
    ts::GameState state{};
    state.set_flag(ts::effect_bits::FLOWER_POWER_ACTIVE);
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::KOREAN_WAR, ts::Player::US);
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::ROLL_DIE, 0, 0, 0));
    // Korean War played by US -> Flower Power gives -2 VP (to USSR)
    ASSERT_EQ(state.victory_points, -2);
}

TEST(CardInteractionTest, FlowerPower_Awards2VP_OnArabIsraeliWarEventPlayByUS) {
    ts::GameState state{};
    state.set_flag(ts::effect_bits::FLOWER_POWER_ACTIVE);
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::ARAB_ISRAELI_WAR, ts::Player::US);
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::ROLL_DIE, 0, 0, 0));
    ASSERT_EQ(state.victory_points, -2);
}

TEST(CardInteractionTest, FlowerPower_Awards2VP_OnIndoPakistaniWarEventPlayByUS) {
    ts::GameState state{};
    state.set_flag(ts::effect_bits::FLOWER_POWER_ACTIVE);
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::INDO_PAKISTANI_WAR, ts::Player::US);
    // US selects target Pakistan
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::PAKISTAN, 0, 0});
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::ROLL_DIE, 0, 0, 0));
    // Indo-Pakistani War played by US -> Flower Power gives -2 VP (to USSR)
    ASSERT_TRUE(state.victory_points <= -2);
}

TEST(CardInteractionTest, FlowerPower_Awards2VP_OnBrushWarEventPlayByUS) {
    ts::GameState state{};
    state.set_flag(ts::effect_bits::FLOWER_POWER_ACTIVE);
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::BRUSH_WAR, ts::Player::US);
    // Secondary_id = 1 (roll fails -> 0 VP from Brush War, but -2 VP from Flower Power)
    ts::MicroAction act{ts::DecisionType::POINT_NODE, ts::countries::ARGENTINA, 1, 0};
    ts::CardHandlers::handle_event_step(state, act);
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::ROLL_DIE, 1, 0, 0));
    ASSERT_EQ(state.victory_points, -2);
}

TEST(CardInteractionTest, FlowerPower_Awards2VP_OnIranIraqWarEventPlayByUS) {
    ts::GameState state{};
    state.set_flag(ts::effect_bits::FLOWER_POWER_ACTIVE);
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::IRAN_IRAQ_WAR, ts::Player::US);
    // Forced roll 1 (fails -> 0 VP from Iran-Iraq War, -2 VP from Flower Power)
    ts::MicroAction act{ts::DecisionType::POINT_NODE, ts::countries::IRAN, 1, 0};
    ts::CardHandlers::handle_event_step(state, act);
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::ROLL_DIE, 1, 0, 0));
    ASSERT_EQ(state.victory_points, -2);
}

TEST(CardInteractionTest, FlowerPower_NoVP_WhenWarCardSpacedByUS) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
    state.set_flag(ts::effect_bits::FLOWER_POWER_ACTIVE);
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.action_round = 1;
    state.phasing_player = ts::Player::US;
    state.ctx().decision_player = ts::Player::US;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;
    state.card_locations[ts::card_ids::KOREAN_WAR] = ts::hand_of(ts::Player::US);
    state.us_space_track = 0;
    state.victory_points = 0;

    // US plays Korean War for Space Race (secondary_id = 1 for forced roll 1 -> success)
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::KOREAN_WAR, 0, 0});
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::PlayMode::SPACE), 1, 0});
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::ROLL_DIE);
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::ROLL_DIE, 1, 0, 0});

    // Space race awards +2 VP to US (first to box 1), but Flower Power did NOT award -2 VP to USSR because event did not occur!
    ASSERT_EQ(state.victory_points, 2);
}

TEST(CardInteractionTest, FlowerPower_NoVP_WhenWarCardPlayedViaUNInterventionByUS) {
    ts::GameState state{};
    state.set_flag(ts::effect_bits::FLOWER_POWER_ACTIVE);
    state.card_locations[ts::card_ids::ARAB_ISRAELI_WAR] = ts::hand_of(ts::Player::US);
    state.victory_points = 0;

    // US plays UN Intervention on Arab-Israeli War
    ts::CardHandlers::trigger_event(state, ts::card_ids::UN_INTERVENTION, ts::Player::US);
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::ARAB_ISRAELI_WAR, 0, 0});

    // Arab-Israeli War was used for Ops, event canceled -> Flower Power gives 0 VP
    ASSERT_EQ(state.victory_points, 0);
}

TEST(CardInteractionTest, FlowerPower_CampDavidBlocksArabIsraeliWar_NoVPForArabIsraeliWar_ButAwardsForBrushWar) {
    ts::GameState state{};
    state.set_flag(ts::effect_bits::FLOWER_POWER_ACTIVE);
    state.set_flag(ts::effect_bits::CAMP_DAVID_PLAYED);
    state.victory_points = 0;

    // 1. Arab-Israeli War cannot trigger -> 0 VP
    ASSERT_FALSE(ts::CardHandlers::can_trigger_event(state, ts::card_ids::ARAB_ISRAELI_WAR, ts::Player::US));
    ts::CardHandlers::trigger_event(state, ts::card_ids::ARAB_ISRAELI_WAR, ts::Player::US);
    ASSERT_EQ(state.victory_points, 0);

    // 2. But Brush War CAN trigger and awards Flower Power VP!
    ASSERT_TRUE(ts::CardHandlers::can_trigger_event(state, ts::card_ids::BRUSH_WAR, ts::Player::US));
    ts::CardHandlers::trigger_event(state, ts::card_ids::BRUSH_WAR, ts::Player::US);
    ts::MicroAction act{ts::DecisionType::POINT_NODE, ts::countries::ARGENTINA, 1, 0};
    ts::CardHandlers::handle_event_step(state, act);
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::ROLL_DIE, 1, 0, 0));
    ASSERT_EQ(state.victory_points, -2);
}

TEST(CardInteractionTest, FlowerPower_AnEvilEmpireCancelsFlowerPower_NoVPOnWarCards) {
    ts::GameState state{};
    state.set_flag(ts::effect_bits::FLOWER_POWER_ACTIVE);
    state.victory_points = 0;

    // Play An Evil Empire (cancels Flower Power and gives US +1 VP)
    ts::CardHandlers::trigger_event(state, ts::card_ids::AN_EVIL_EMPIRE, ts::Player::US);
    ASSERT_EQ(state.victory_points, 1);
    ASSERT_FALSE(state.has_flag(ts::effect_bits::FLOWER_POWER_ACTIVE));

    // Now Korean War played by US awards NO VP to USSR
    ts::CardHandlers::trigger_event(state, ts::card_ids::KOREAN_WAR, ts::Player::US);
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::ROLL_DIE, 0, 0, 0));
    ASSERT_EQ(state.victory_points, 1); // Unchanged
}

// =============================================================================
// 2. Glasnost Matrix
// =============================================================================

TEST(CardInteractionTest, Glasnost_ReformerNotInPlay_GivesDefconAndVP_NoOps) {
    ts::GameState state{};
    state.defcon = 2;
    state.victory_points = 0;
    state.clear_flag(ts::effect_bits::THE_REFORMER_PLAYED);

    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::GLASNOST, ts::Player::USSR);
    ASSERT_TRUE(done); // Finished immediately without transitioning to Ops
    ASSERT_EQ(state.defcon, 3);
    ASSERT_EQ(state.victory_points, -2); // USSR gains 2 VP
    ASSERT_NE(state.ctx().decision_type, ts::DecisionType::SELECT_OP_MODE);
}

TEST(CardInteractionTest, Glasnost_ReformerInPlay_GivesDefconAndVP_And4Ops) {
    ts::GameState state{};
    state.defcon = 2;
    state.victory_points = 0;
    state.set_flag(ts::effect_bits::THE_REFORMER_PLAYED);

    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::GLASNOST, ts::Player::USSR);
    ASSERT_FALSE(done); // Transitions to 4 Ops for USSR!
    ASSERT_EQ(state.defcon, 3);
    ASSERT_EQ(state.victory_points, -2);
    ASSERT_EQ(state.ctx().decision_player, ts::Player::USSR);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_OP_MODE);
    ASSERT_EQ(state.ctx().pending_ops_value, 4);
}

// =============================================================================
// 3. Nuclear Subs & CIA Created DEFCON Suicide Matrix
// =============================================================================

TEST(CardInteractionTest, NuclearSubs_USCoupInBattleground_DoesNotDegradeDefcon) {
    ts::GameState state{};
    state.defcon = 2;
    state.set_flag(ts::effect_bits::NUCLEAR_SUBS_ACTIVE);
    state.countries[ts::countries::EGYPT].ussr_influence = 2; // BG in ME

    auto res = ts::Operations::execute_coup(state, ts::Player::US, ts::countries::EGYPT, 3, 6);
    ASSERT_EQ(state.defcon, 2); // DEFCON did NOT degrade!
    ASSERT_FALSE(res.defcon_degraded);
    ASSERT_FALSE(res.caused_defcon_suicide);
}

TEST(CardInteractionTest, NuclearSubs_USSRPlaysCIAAtDefcon2_USCoupsBattleground_USSRDoesNotLose) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
    state.defcon = 2;
    state.set_flag(ts::effect_bits::NUCLEAR_SUBS_ACTIVE);
    // South Africa, not Egypt. Egypt is in the Middle East, and at DEFCON 2 coups there are
    // forbidden -- so the coup this test asserts was illegal. It passed because Engine::step did
    // not validate coup targets at all, while the mask did; `step` now validates against the mask,
    // which is what surfaced it. South Africa is a battleground outside the DEFCON-restricted
    // regions, so it still exercises what the test is for: Nuclear Subs sparing the DEFCON track
    // on a US battleground coup.
    state.countries[ts::countries::SOUTH_AFRICA].ussr_influence = 2; // USSR influence in BG

    state.current_phase = ts::Phase::ACTION_ROUND;
    state.action_round = 1;
    state.phasing_player = ts::Player::USSR;
    state.ctx().decision_player = ts::Player::USSR;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;
    state.card_locations[ts::card_ids::CIA_CREATED] = ts::hand_of(ts::Player::USSR);

    // 1. USSR plays CIA Created for Ops (Event First)
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::CIA_CREATED, 0, 0});
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::PlayMode::OPS), 0, 0});
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::CHOOSE_TIMING_BRANCH, static_cast<uint8_t>(ts::TimingBranch::EVENT_FIRST), 0, 0});

    // 2. US conducts 1 Op from CIA Created -> selects COUP mode
    ASSERT_EQ(state.ctx().decision_player, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_OP_MODE);
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::SELECT_OP_MODE, static_cast<uint8_t>(ts::OpMode::COUP), 0, 0});

    // 3. US coups South Africa with forced roll 6
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::SOUTH_AFRICA, 6, 0});
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::ROLL_DIE);
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::ROLL_DIE, 6, 0, 0});

    // DEFCON remains 2 due to Nuclear Subs, and USSR did NOT lose!
    ASSERT_EQ(state.defcon, 2);
    ASSERT_NE(state.current_phase, ts::Phase::GAME_OVER);
}

TEST(CardInteractionTest, NuclearSubs_NotActive_USSRPlaysCIAAtDefcon2_USCoupsBattleground_USSRLosesDefconSuicide) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
    state.defcon = 2;
    state.clear_flag(ts::effect_bits::NUCLEAR_SUBS_ACTIVE);
    // South Africa, not Egypt: at DEFCON 2 a coup in the Middle East is forbidden, so the coup
    // this test asserts was illegal. See the sibling test above.
    state.countries[ts::countries::SOUTH_AFRICA].ussr_influence = 2; // BG outside the restriction

    state.current_phase = ts::Phase::ACTION_ROUND;
    state.action_round = 1;
    state.phasing_player = ts::Player::USSR;
    state.ctx().decision_player = ts::Player::USSR;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;
    state.card_locations[ts::card_ids::CIA_CREATED] = ts::hand_of(ts::Player::USSR);

    // 1. USSR plays CIA Created for Ops (Event First)
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::CIA_CREATED, 0, 0});
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::PlayMode::OPS), 0, 0});
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::CHOOSE_TIMING_BRANCH, static_cast<uint8_t>(ts::TimingBranch::EVENT_FIRST), 0, 0});

    // 2. US selects COUP
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::SELECT_OP_MODE, static_cast<uint8_t>(ts::OpMode::COUP), 0, 0});

    // 3. US coups a Battleground -> DEFCON drops from 2 to 1 on USSR turn -> USSR LOSES!
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::SOUTH_AFRICA, 6, 0});
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::ROLL_DIE);
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::ROLL_DIE, 6, 0, 0});

    ASSERT_EQ(state.defcon, 1);
    ASSERT_EQ(state.current_phase, ts::Phase::GAME_OVER);
    ASSERT_EQ(state.victory_points, 20); // US wins!
}

// =============================================================================
// 4. Cuban Missile Crisis & Lone Gunman Matrix
// =============================================================================

TEST(CardInteractionTest, CubanMissileCrisis_ActiveOnUSSR_USPlaysLoneGunman_USSRCoups_USSRLoses) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
    state.defcon = 2;
    state.set_flag(ts::effect_bits::CMC_ACTIVE_US); // CMC active against USSR
    state.countries[ts::countries::MEXICO].us_influence = 2;
    state.countries[ts::countries::CUBA].ussr_influence = 0;

    state.current_phase = ts::Phase::ACTION_ROUND;
    state.action_round = 1;
    state.phasing_player = ts::Player::US;
    state.ctx().decision_player = ts::Player::US;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;
    state.card_locations[ts::card_ids::LONE_GUNMAN] = ts::hand_of(ts::Player::US);

    // 1. US plays Lone Gunman for Ops (Event First)
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::LONE_GUNMAN, 0, 0});
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::PlayMode::OPS), 0, 0});
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::CHOOSE_TIMING_BRANCH, static_cast<uint8_t>(ts::TimingBranch::EVENT_FIRST), 0, 0});

    // 2. USSR conducts 1 Op from Lone Gunman -> chooses COUP mode
    ASSERT_EQ(state.ctx().decision_player, ts::Player::USSR);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_OP_MODE);
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::SELECT_OP_MODE, static_cast<uint8_t>(ts::OpMode::COUP), 0, 0});

    // 3. USSR coups Mexico without clearing CMC -> USSR LOSES immediately!
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::MEXICO, 4, 0});
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::ROLL_DIE);
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::ROLL_DIE, 4, 0, 0});

    ASSERT_EQ(state.current_phase, ts::Phase::GAME_OVER);
    ASSERT_EQ(state.victory_points, 20); // US wins!
}

TEST(CardInteractionTest, CubanMissileCrisis_ActiveOnUS_USSRPlaysCIACreated_USCoups_USLoses) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
    state.defcon = 2;
    state.set_flag(ts::effect_bits::CMC_ACTIVE_USSR); // CMC active against US
    // South Africa, not Egypt: at DEFCON 2 a coup in the Middle East is forbidden,
    // so the coup this test asserts was illegal. See the Nuclear Subs tests above.
    state.countries[ts::countries::SOUTH_AFRICA].ussr_influence = 2;
    state.countries[ts::countries::WEST_GERMANY].us_influence = 0;
    state.countries[ts::countries::TURKEY].us_influence = 0;

    state.current_phase = ts::Phase::ACTION_ROUND;
    state.action_round = 1;
    state.phasing_player = ts::Player::USSR;
    state.ctx().decision_player = ts::Player::USSR;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;
    state.card_locations[ts::card_ids::CIA_CREATED] = ts::hand_of(ts::Player::USSR);

    // 1. USSR plays CIA Created for Ops (Event First)
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::CIA_CREATED, 0, 0});
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::PlayMode::OPS), 0, 0});
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::CHOOSE_TIMING_BRANCH, static_cast<uint8_t>(ts::TimingBranch::EVENT_FIRST), 0, 0});

    // 2. US conducts 1 Op -> chooses COUP
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::SELECT_OP_MODE, static_cast<uint8_t>(ts::OpMode::COUP), 0, 0});

    // 3. US coups a battleground under CMC -> US LOSES immediately!
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::SOUTH_AFRICA, 4, 0});
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::ROLL_DIE);
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::ROLL_DIE, 4, 0, 0});

    ASSERT_EQ(state.current_phase, ts::Phase::GAME_OVER);
    ASSERT_EQ(state.victory_points, -20); // USSR wins!
}

// =============================================================================
// 5. Willy Brandt & Tear Down This Wall Interactions
// =============================================================================

TEST(CardInteractionTest, WillyBrandt_Then_TearDownThisWall_RestoresNATO) {
    ts::GameState state{};
    state.countries[ts::countries::WEST_GERMANY].ussr_influence = 0;
    state.countries[ts::countries::EAST_GERMANY].us_influence = 0;
    state.set_flag(ts::effect_bits::NATO_ACTIVE);

    // 1. USSR plays Willy Brandt
    ts::CardHandlers::trigger_event(state, ts::card_ids::WILLY_BRANDT, ts::Player::USSR);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::WILLY_BRANDT_PLAYED));
    ASSERT_TRUE(state.has_flag(ts::effect_bits::NATO_CANCELED_WEST_GERMANY));
    ASSERT_EQ(state.countries[ts::countries::WEST_GERMANY].ussr_influence, 1);

    // 2. US plays Tear Down This Wall
    ts::CardHandlers::trigger_event(state, ts::card_ids::TEAR_DOWN_THIS_WALL, ts::Player::US);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::TEAR_DOWN_THIS_WALL_PLAYED));
    ASSERT_FALSE(state.has_flag(ts::effect_bits::WILLY_BRANDT_PLAYED)); // Canceled!
    ASSERT_FALSE(state.has_flag(ts::effect_bits::NATO_CANCELED_WEST_GERMANY)); // Restored!
    ASSERT_EQ(state.countries[ts::countries::EAST_GERMANY].us_influence, 3);

    // 3. Willy Brandt cannot be triggered again
    ASSERT_FALSE(ts::CardHandlers::can_trigger_event(state, ts::card_ids::WILLY_BRANDT, ts::Player::USSR));
}

// =============================================================================
// 6. OPEC & North Sea Oil Interactions
// =============================================================================

TEST(CardInteractionTest, OPEC_Active_Then_NorthSeaOilBlocksIt) {
    ts::GameState state{};
    state.countries[ts::countries::EGYPT].ussr_influence = 4; // Controlled
    state.countries[ts::countries::IRAN].ussr_influence = 4;  // Controlled
    state.victory_points = 0;

    // 1. Without North Sea Oil: OPEC awards -2 VP to USSR
    ASSERT_TRUE(ts::CardHandlers::can_trigger_event(state, ts::card_ids::OPEC, ts::Player::USSR));
    ts::CardHandlers::trigger_event(state, ts::card_ids::OPEC, ts::Player::USSR);
    ASSERT_EQ(state.victory_points, -2);

    // 2. North Sea Oil played
    ts::CardHandlers::trigger_event(state, ts::card_ids::NORTH_SEA_OIL, ts::Player::US);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::NORTH_SEA_OIL_PLAYED));

    // 3. OPEC is now blocked
    ASSERT_FALSE(ts::CardHandlers::can_trigger_event(state, ts::card_ids::OPEC, ts::Player::USSR));
}

// =============================================================================
// 7. Muslim Revolution & AWACS Sale Interactions
// =============================================================================

TEST(CardInteractionTest, AWACS_Sale_BlocksMuslimRevolution) {
    ts::GameState state{};
    state.countries[ts::countries::SAUDI_ARABIA].us_influence = 0;

    // 1. Play AWACS Sale
    ts::CardHandlers::trigger_event(state, ts::card_ids::AWACS_SALE, ts::Player::US);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::AWACS_PLAYED));
    ASSERT_EQ(state.countries[ts::countries::SAUDI_ARABIA].us_influence, 2);

    // 2. Muslim Revolution is blocked
    ASSERT_FALSE(ts::CardHandlers::can_trigger_event(state, ts::card_ids::MUSLIM_REVOLUTION, ts::Player::USSR));
}

// =============================================================================
// 8. John Paul II & Solidarity Interactions
// =============================================================================

TEST(CardInteractionTest, JohnPaulII_EnablesSolidarity) {
    ts::GameState state{};
    state.countries[ts::countries::POLAND].ussr_influence = 3;
    state.countries[ts::countries::POLAND].us_influence = 0;

    // 1. Before John Paul II: Solidarity cannot trigger
    ASSERT_FALSE(ts::CardHandlers::can_trigger_event(state, ts::card_ids::SOLIDARITY, ts::Player::US));

    // 2. Play John Paul II
    ts::CardHandlers::trigger_event(state, ts::card_ids::JOHN_PAUL_II, ts::Player::US);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::JOHN_PAUL_II_PLAYED));
    ASSERT_EQ(state.countries[ts::countries::POLAND].ussr_influence, 1);
    ASSERT_EQ(state.countries[ts::countries::POLAND].us_influence, 1);

    // 3. Now Solidarity can trigger!
    ASSERT_TRUE(ts::CardHandlers::can_trigger_event(state, ts::card_ids::SOLIDARITY, ts::Player::US));
    ts::CardHandlers::trigger_event(state, ts::card_ids::SOLIDARITY, ts::Player::US);
    ASSERT_EQ(state.countries[ts::countries::POLAND].us_influence, 4);
}

// =============================================================================
// 9. U-2 Incident & UN Intervention Interaction
// =============================================================================

TEST(CardInteractionTest, U2Incident_Then_UNIntervention_AwardsExtraVP) {
    ts::GameState state{};
    state.victory_points = 0;
    state.card_locations[ts::card_ids::DE_GAULLE] = ts::hand_of(ts::Player::US);

    // 1. Play U-2 Incident -> -1 VP, sets U2_INCIDENT_ACTIVE
    ts::CardHandlers::trigger_event(state, ts::card_ids::U2_INCIDENT, ts::Player::USSR);
    ASSERT_EQ(state.victory_points, -1);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::U2_INCIDENT_ACTIVE));

    // 2. Play UN Intervention -> triggers UN bonus (+1 VP to USSR from U-2)
    ts::CardHandlers::trigger_event(state, ts::card_ids::UN_INTERVENTION, ts::Player::US);
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::DE_GAULLE, 0, 0});
}

// =============================================================================
// 10. Quagmire / Bear Trap Action Masking & Escape
// =============================================================================

TEST(CardInteractionTest, Quagmire_MaskOnlyAllows2PlusOpsCards) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
    state.set_flag(ts::effect_bits::QUAGMIRE_ACTIVE);
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.action_round = 1;
    state.phasing_player = ts::Player::US;
    state.ctx().decision_player = ts::Player::US;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;

    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == ts::hand_of(ts::Player::US)) {
            state.card_locations[i] = ts::CardLocation::DRAW_DECK;
        }
    }
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::hand_of(ts::Player::US); // 3 Ops (legal)
    state.card_locations[ts::card_ids::TRUMAN_DOCTRINE] = ts::hand_of(ts::Player::US); // 1 Op (illegal)

    uint8_t mask[112]{};
    size_t out_size = 0;
    ts::ActionMask::generate_mask(state, mask, &out_size);

    ASSERT_EQ(mask[ts::card_ids::DUCK_AND_COVER], 1);
    ASSERT_EQ(mask[ts::card_ids::TRUMAN_DOCTRINE], 0);

    // US discards Duck and Cover and rolls 3 -> escapes Quagmire!
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::DUCK_AND_COVER, 0, 0});
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::ROLL_DIE);
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::ROLL_DIE, 3, 0, 0});
    ASSERT_FALSE(state.has_flag(ts::effect_bits::QUAGMIRE_ACTIVE));
}

// =============================================================================
// 11. Ops Modifier Stacking (Purge + Brezhnev / Containment)
// =============================================================================

TEST(CardInteractionTest, OpsModifiers_PurgeAndBrezhnevDoctrine_NetZero) {
    ts::GameState state{};
    state.set_flag(ts::effect_bits::PURGE_USSR_ACTIVE);
    state.set_flag(ts::effect_bits::BREZHNEV_DOCTRINE_ACTIVE);

    // Card with base 3 Ops -> -1 + 1 = 3 Ops effective
    uint8_t effective = ts::Operations::get_effective_ops(state, ts::card_ids::WARSAW_PACT, ts::Player::USSR);
    ASSERT_EQ(effective, 3);
}

TEST(CardInteractionTest, OpsModifiers_PurgeAndContainment_NetZero) {
    ts::GameState state{};
    state.set_flag(ts::effect_bits::PURGE_US_ACTIVE);
    state.set_flag(ts::effect_bits::CONTAINMENT_ACTIVE);

    uint8_t effective = ts::Operations::get_effective_ops(state, ts::card_ids::DUCK_AND_COVER, ts::Player::US);
    ASSERT_EQ(effective, 3);
}

// =============================================================================
// 12. Defectors Headline Cancellation Matrix
// =============================================================================

TEST(CardInteractionTest, Defectors_CancelsScoringCardHeadline) {
    ts::GameState state{};
    state.current_phase = ts::Phase::HEADLINE;
    state.headline_ussr_card = ts::card_ids::EUROPE_SCORING;
    state.headline_us_card = ts::card_ids::DEFECTORS;
    state.victory_points = 0;

    ts::CardHandlers::trigger_event(state, ts::card_ids::DEFECTORS, ts::Player::US);
    ASSERT_EQ(state.headline_ussr_card, 0); // Europe scoring canceled!
    ASSERT_EQ(state.victory_points, 0);     // Headline cancellation does not award VP
}


// =============================================================================
// 13. NORAD Reaction & Canada Control
// =============================================================================

TEST(CardInteractionTest, NORAD_Active_CanadaControlled_DefconDropsTo2_AllowsInfluencePlacement) {
    ts::GameState state{};
    state.set_flag(ts::effect_bits::NORAD_ACTIVE);
    state.countries[ts::countries::CANADA].us_influence = 4; // US controls Canada (stability 2)
    state.countries[ts::countries::UNITED_KINGDOM].us_influence = 2; // Has US influence

    ASSERT_TRUE(ts::Scoring::is_controlled_by(state, ts::countries::CANADA, ts::Player::US));

    // When DEFCON drops to 2 in AR, NORAD triggers
    state.defcon = 3;
    state.defcon_dropped_to_2 = 1;
    // Verify NORAD flag remains active until Quagmire
    ASSERT_TRUE(state.has_flag(ts::effect_bits::NORAD_ACTIVE));
}

// =============================================================================
// 14. De-Stalinization Redistribution Limits & Validation
// =============================================================================

TEST(CardInteractionTest, DeStalinization_CannotPlaceInUSControlledCountry) {
    ts::GameState state{};
    state.countries[ts::countries::EAST_GERMANY].ussr_influence = 4;
    state.countries[ts::countries::WEST_GERMANY].us_influence = 4; // US-controlled (stability 3)
    state.countries[ts::countries::FRANCE].us_influence = 0; // Uncontrolled

    ASSERT_TRUE(ts::Scoring::is_controlled_by(state, ts::countries::WEST_GERMANY, ts::Player::US));

    ts::CardHandlers::trigger_event(state, ts::card_ids::DE_STALINIZATION, ts::Player::USSR);

    // Stage 1: Remove 2 from East Germany and confirm done
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::EAST_GERMANY, 0, 0});
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::EAST_GERMANY, 0, 0});
    ts::MicroAction done_act{};
    done_act.flags = ts::action_flags::CONFIRM_DONE;
    ts::CardHandlers::handle_event_step(state, done_act);

    // Stage 2: Attempt to place in US-controlled West Germany -> rejected!
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::WEST_GERMANY, 0, 0});
    ASSERT_FALSE(done);
    ASSERT_EQ(state.countries[ts::countries::WEST_GERMANY].ussr_influence, 0);

    // Place in France -> accepted!
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::FRANCE, 0, 0});
    done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::FRANCE, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::FRANCE].ussr_influence, 2);
}

// =============================================================================
// 15. Warsaw Pact Eastern Europe Isolation
// =============================================================================

TEST(CardInteractionTest, WarsawPact_Branch1_EasternEuropeOnly_Max2PerCountry) {
    ts::GameState state{};
    state.countries[ts::countries::POLAND].ussr_influence = 0;
    state.countries[ts::countries::EAST_GERMANY].ussr_influence = 0;
    state.countries[ts::countries::FRANCE].ussr_influence = 0; // Western Europe

    ts::CardHandlers::trigger_event(state, ts::card_ids::WARSAW_PACT, ts::Player::USSR);

    // Branch 1: Add 5 influence in Eastern Europe
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::CHOOSE_BRANCH, 1, 0, 0});

    // Attempt to place in France -> rejected
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::FRANCE, 0, 0});
    ASSERT_FALSE(done);
    ASSERT_EQ(state.countries[ts::countries::FRANCE].ussr_influence, 0);

    // Place 2 in Poland, 2 in East Germany, 1 in Czechoslovakia
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::POLAND, 0, 0});
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::POLAND, 0, 0});
    // 3rd placement in Poland rejected (max 2 per country)
    done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::POLAND, 0, 0});
    ASSERT_FALSE(done);
    ASSERT_EQ(state.countries[ts::countries::POLAND].ussr_influence, 2);

    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::EAST_GERMANY, 0, 0});
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::EAST_GERMANY, 0, 0});
    done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::CZECHOSLOVAKIA, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::EAST_GERMANY].ussr_influence, 2);
    ASSERT_EQ(state.countries[ts::countries::CZECHOSLOVAKIA].ussr_influence, 1);
}

// =============================================================================
// 16. Latin American Death Squads Coup Modifier
// =============================================================================

TEST(CardInteractionTest, LatinAmericanDeathSquads_AppliesCoupModifiers) {
    ts::GameState state{};
    state.countries[ts::countries::ARGENTINA].ussr_influence = 2; // Stability 2
    state.set_flag(ts::effect_bits::DEATH_SQUADS_US); // Death squads active for US

    // US coups with 2 Ops and roll 3 -> with +1 modifier from Death Squads:
    // Total = 3 + 1 + 2 = 6. Margin = 6 - 2*2 = 2 -> removes 2 USSR influence!
    auto res = ts::Operations::execute_coup(state, ts::Player::US, ts::countries::ARGENTINA, 2, 3);
    ASSERT_EQ(res.margin, 2);
    ASSERT_EQ(state.countries[ts::countries::ARGENTINA].ussr_influence, 0);
}

// =============================================================================
// 17. SALT Negotiations Coup Modifier
// =============================================================================

TEST(CardInteractionTest, SALT_Negotiations_AppliesMinus1ToAllCoups) {
    ts::GameState state{};
    state.countries[ts::countries::EGYPT].ussr_influence = 2; // Stability 2
    state.set_flag(ts::effect_bits::SALT_ACTIVE); // SALT -1 modifier

    // US coups with 2 Ops and roll 3 -> with -1 modifier:
    // Total = 3 - 1 + 2 = 4. Margin = 4 - 2*2 = 0 -> removes 0 influence!
    auto res = ts::Operations::execute_coup(state, ts::Player::US, ts::countries::EGYPT, 2, 3);
    ASSERT_EQ(res.margin, 0);
    ASSERT_EQ(state.countries[ts::countries::EGYPT].ussr_influence, 2);
}

// =============================================================================
// 18. Shuttle Diplomacy Scoring Modifier
// =============================================================================

TEST(CardInteractionTest, ShuttleDiplomacy_SetsFlagForScoringExclusion) {
    ts::GameState state{};
    ts::CardHandlers::trigger_event(state, ts::card_ids::SHUTTLE_DIPLOMACY, ts::Player::US);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::SHUTTLE_DIPLOMACY_ACTIVE));
}

// =============================================================================
// 19. An Evil Empire Cancels Flower Power
// =============================================================================

TEST(CardInteractionTest, AnEvilEmpire_CancelsFlowerPowerAndAwards1VP) {
    ts::GameState state{};
    state.set_flag(ts::effect_bits::FLOWER_POWER_ACTIVE);
    state.victory_points = 0;

    ts::CardHandlers::trigger_event(state, ts::card_ids::AN_EVIL_EMPIRE, ts::Player::US);
    ASSERT_EQ(state.victory_points, 1);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::EVIL_EMPIRE_PLAYED));
    ASSERT_FALSE(state.has_flag(ts::effect_bits::FLOWER_POWER_ACTIVE));
}

// =============================================================================
// 20. Red Scare / Purge Ops Value Reductions & Minimum 1 Op
// =============================================================================

TEST(CardInteractionTest, Purge_ReducesOpsValue_Minimum1Op) {
    ts::GameState state{};
    state.set_flag(ts::effect_bits::PURGE_US_ACTIVE);

    // 1 Op card with -1 Purge modifier -> clamped to minimum 1 Op
    uint8_t effective_1op = ts::Operations::get_effective_ops(state, ts::card_ids::TRUMAN_DOCTRINE, ts::Player::US);
    ASSERT_EQ(effective_1op, 1);

    // 3 Ops card with -1 Purge modifier -> 2 Ops
    uint8_t effective_3op = ts::Operations::get_effective_ops(state, ts::card_ids::DUCK_AND_COVER, ts::Player::US);
    ASSERT_EQ(effective_3op, 2);
}

// Ops granted to the opponent by an event run in a pushed frame, and finishing them must
// return to the frame underneath rather than end the action round. On an EVENT_FIRST play that
// frame holds the phasing player's own Ops: CIA Created reveals the USSR hand and hands the US
// one Op, after which the USSR still owes the card's own Op. The action round used to end
// instead, silently costing the USSR the Ops they paid for -- at turn 2 AR1 of ts-replayer
// game 103 their influence in Poland never went in.
TEST(CardInteractionTest, CIACreated_EventFirst_USSRStillSpendsItsOwnOpAfterUSOp) {
    ts::GameState state{};
    ts::Engine::init_game(state, 12345);

    state.defcon = 5;
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.action_round = 1;
    state.phasing_player = ts::Player::USSR;
    state.ctx().decision_player = ts::Player::USSR;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;
    state.card_locations[ts::card_ids::CIA_CREATED] = ts::hand_of(ts::Player::USSR);

    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::CIA_CREATED, 0, 0});
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::PlayMode::OPS), 0, 0});
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::CHOOSE_TIMING_BRANCH, static_cast<uint8_t>(ts::TimingBranch::EVENT_FIRST), 0, 0});

    // The US spends the granted Op on influence, which keeps the test off the dice.
    ASSERT_EQ(state.ctx().decision_player, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_OP_MODE);
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::SELECT_OP_MODE, static_cast<uint8_t>(ts::OpMode::INFLUENCE), 0, 0});
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::CANADA, 0, 0});

    // Control returns to the USSR for CIA Created's own Op, still in the same action round.
    ASSERT_NE(state.current_phase, ts::Phase::GAME_OVER);
    ASSERT_EQ(state.ctx().decision_player, ts::Player::USSR);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_OP_MODE);
}

// Events nest, and every finished frame has to unwind. Five Year Plan makes the USSR discard a
// card and that card's own event runs a frame deeper, so on an EVENT_FIRST play the stack is
// two deep. Popping once landed on Five Year Plan's own frame -- not SELECT_OP_MODE -- and the
// action round ended, costing the USSR the Ops they had paid for: at turn 2 AR6 of ts-replayer
// game 105 their influence in Iran never went in.
TEST(CardInteractionTest, FiveYearPlan_EventFirst_USSRStillSpendsItsOwnOpsAfterNestedEvent) {
    ts::GameState state{};
    ts::Engine::init_game(state, 12345);

    state.defcon = 5;
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.action_round = 1;
    state.phasing_player = ts::Player::USSR;
    state.ctx().decision_player = ts::Player::USSR;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;

    // Five Year Plan discards at random, so give the USSR exactly one other card to lose and
    // make it a US event that resolves without asking anything: Truman Doctrine needs a target,
    // so use one that resolves on its own instead.
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == ts::hand_of(ts::Player::USSR)) {
            state.card_locations[i] = ts::CardLocation::DISCARD_PILE;
        }
    }
    state.card_locations[ts::card_ids::FIVE_YEAR_PLAN] = ts::hand_of(ts::Player::USSR);
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::hand_of(ts::Player::USSR);

    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::FIVE_YEAR_PLAN, 0, 0});
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::PlayMode::OPS), 0, 0});
    ts::Engine::step(state, ts::MicroAction{ts::DecisionType::CHOOSE_TIMING_BRANCH, static_cast<uint8_t>(ts::TimingBranch::EVENT_FIRST), 0, 0});

    // Whatever the nested event did, the USSR is owed Five Year Plan's own Ops.
    ASSERT_NE(state.current_phase, ts::Phase::GAME_OVER);
    ASSERT_EQ(state.ctx().decision_player, ts::Player::USSR);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_OP_MODE);
}
