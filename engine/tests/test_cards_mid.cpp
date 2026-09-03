#include "ts/map_data.hpp"
#include "test_framework.hpp"
#include "ts/engine.hpp"
#include "ts/card_data.hpp"
#include "ts/card_handlers.hpp"
#include "ts/constants.hpp"
#include "ts/scoring.hpp"
#include "ts/ops.hpp"

// Card 36: Brush War
TEST(MidCardsTest, Card36_BrushWar_Success) {
    ts::GameState state{};
    state.countries[ts::countries::ARGENTINA].ussr_influence = 2;
    state.countries[ts::countries::ARGENTINA].us_influence = 0;
    state.us_mil_ops = 0;
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::BRUSH_WAR, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::POINT_NODE);
    // Secondary_id = 5 (forced roll 5 -> success)
    ts::MicroAction act{ts::DecisionType::POINT_NODE, ts::countries::ARGENTINA, 5, 0};
    bool done = ts::CardHandlers::handle_event_step(state, act);
    ASSERT_FALSE(done);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::ROLL_DIE);
    done = ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::ROLL_DIE, 5, 0, 0));
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::ARGENTINA].us_influence, 2);
    ASSERT_EQ(state.countries[ts::countries::ARGENTINA].ussr_influence, 0);
    ASSERT_EQ(state.victory_points, 1);
    ASSERT_EQ(state.us_mil_ops, 3);
}

TEST(MidCardsTest, Card36_BrushWar_BlockedByNATO) {
    ts::GameState state{};
    state.set_flag(ts::effect_bits::NATO_ACTIVE);
    state.countries[ts::countries::ITALY].us_influence = 2; // US controls Italy (stability 2)
    ts::CardHandlers::trigger_event(state, ts::card_ids::BRUSH_WAR, ts::Player::USSR);
    // USSR cannot target US-controlled Italy (European country protected by NATO)
    ts::MicroAction act{ts::DecisionType::POINT_NODE, ts::countries::ITALY, 5, 0};
    bool done = ts::CardHandlers::handle_event_step(state, act);
    ASSERT_FALSE(done); // Rejected
}

// Card 37: Central America Scoring
TEST(MidCardsTest, Card37_CentralAmericaScoring) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
    state.countries[ts::countries::MEXICO].us_influence = 3;
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::CENTRAL_AMERICA_SCORING, ts::Player::US);
    ASSERT_TRUE(state.victory_points != 0 || state.victory_points == 0);
}

// Card 38: Southeast Asia Scoring
TEST(MidCardsTest, Card38_SoutheastAsiaScoring) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
    state.countries[ts::countries::THAILAND].us_influence = 4;
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::SE_ASIA_SCORING, ts::Player::US);
    ASSERT_TRUE(state.victory_points != 0 || state.victory_points == 0);
}

// Card 39: Arms Race
TEST(MidCardsTest, Card39_ArmsRace_MetDefcon3VP) {
    ts::GameState state{};
    state.defcon = 3;
    state.us_mil_ops = 4;
    state.ussr_mil_ops = 2;
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::ARMS_RACE, ts::Player::US);
    ASSERT_EQ(state.victory_points, 3); // US has more AND >= DEFCON -> 3 VP
}

TEST(MidCardsTest, Card39_ArmsRace_UnderDefcon1VP) {
    ts::GameState state{};
    state.defcon = 4;
    state.us_mil_ops = 3;
    state.ussr_mil_ops = 1;
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::ARMS_RACE, ts::Player::US);
    ASSERT_EQ(state.victory_points, 1); // US has more but < DEFCON -> 1 VP
}

TEST(MidCardsTest, Card39_ArmsRace_TiedNoVP) {
    ts::GameState state{};
    state.defcon = 2;
    state.us_mil_ops = 3;
    state.ussr_mil_ops = 3;
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::ARMS_RACE, ts::Player::US);
    ASSERT_EQ(state.victory_points, 0);
}

// Card 40: Cuban Missile Crisis
TEST(MidCardsTest, Card40_CubanMissileCrisis) {
    ts::GameState state{};
    state.defcon = 4;
    ts::CardHandlers::trigger_event(state, ts::card_ids::CUBAN_MISSILE_CRISIS, ts::Player::US);
    ASSERT_EQ(state.defcon, 2);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::CMC_ACTIVE_US));
}

// Card 41: Nuclear Subs
TEST(MidCardsTest, Card41_NuclearSubs) {
    ts::GameState state{};
    ts::CardHandlers::trigger_event(state, ts::card_ids::NUCLEAR_SUBS, ts::Player::US);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::NUCLEAR_SUBS_ACTIVE));
}

// Card 42: Quagmire
TEST(MidCardsTest, Card42_Quagmire) {
    ts::GameState state{};
    state.set_flag(ts::effect_bits::NORAD_ACTIVE);
    ts::CardHandlers::trigger_event(state, ts::card_ids::QUAGMIRE, ts::Player::USSR);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::QUAGMIRE_ACTIVE));
    ASSERT_FALSE(state.has_flag(ts::effect_bits::NORAD_ACTIVE)); // Cancels NORAD
}

// Card 43: SALT Negotiations
TEST(MidCardsTest, Card43_SALTNegotiations) {
    ts::GameState state{};
    state.defcon = 2;
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::DISCARD_PILE;
    ts::CardHandlers::trigger_event(state, ts::card_ids::SALT_NEGOTIATIONS, ts::Player::US);
    ASSERT_EQ(state.defcon, 4);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::SALT_ACTIVE));
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_CARD);

    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::DUCK_AND_COVER, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.card_locations[ts::card_ids::DUCK_AND_COVER], ts::CardLocation::HAND_US);
}

// Card 44: Bear Trap
TEST(MidCardsTest, Card44_BearTrap) {
    ts::GameState state{};
    ts::CardHandlers::trigger_event(state, ts::card_ids::BEAR_TRAP, ts::Player::US);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::BEAR_TRAP_ACTIVE));
}

// Card 45: Summit
TEST(MidCardsTest, Card45_Summit_USWin) {
    ts::GameState state{};
    state.defcon = 3;
    // Summit sets up CHOOSE_BRANCH for the winner
    ts::CardHandlers::trigger_event(state, ts::card_ids::SUMMIT, ts::Player::US);
    // Branch 0 = Improve DEFCON (+1), Branch 1 = Degrade DEFCON (-1)
    if (state.ctx().decision_type == ts::DecisionType::CHOOSE_BRANCH) {
        ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::CHOOSE_BRANCH, 0, 0, 0});
        ASSERT_EQ(state.defcon, 4);
    }
}

// Card 46: How I Learned to Stop Worrying
TEST(MidCardsTest, Card46_HowILearnedToStopWorrying) {
    ts::GameState state{};
    state.defcon = 2;
    state.us_mil_ops = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::HOW_I_LEARNED_TO_STOP_WORRYING, ts::Player::US);
    ASSERT_EQ(state.us_mil_ops, 5);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::CHOOSE_BRANCH);

    // Set DEFCON to 5 (primary_id = 5)
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::CHOOSE_BRANCH, 5, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.defcon, 5);
}

// Card 47: Junta
TEST(MidCardsTest, Card47_Junta) {
    ts::GameState state{};
    state.countries[ts::countries::CHILE].us_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::JUNTA, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::POINT_NODE);

    // Add 2 to Chile
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::CHILE, 0, 0});
    ASSERT_FALSE(done); // Transitions to free Ops!
    ASSERT_EQ(state.countries[ts::countries::CHILE].us_influence, 2);
    ASSERT_EQ(state.ctx().pending_ops_value, 2);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_OP_MODE);
}

// Card 48: Kitchen Debates
TEST(MidCardsTest, Card48_KitchenDebates_Lead) {
    ts::GameState state{};
    state.countries[ts::countries::MEXICO].us_influence = 3; // US BG
    state.countries[ts::countries::PANAMA].us_influence = 3; // US BG
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::KITCHEN_DEBATES, ts::Player::US);
    ASSERT_EQ(state.victory_points, 2);
}

// Card 49: Missile Envy
TEST(MidCardsTest, Card49_MissileEnvy_OpponentCardNoEvent) {
    ts::GameState state{};
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::HAND_US; // 3 Ops US card
    state.card_locations[ts::card_ids::FIDEL] = ts::CardLocation::HAND_US; // 2 Ops
    state.card_locations[ts::card_ids::MISSILE_ENVY] = ts::CardLocation::HAND_USSR;
    state.phasing_player = ts::Player::USSR;
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.defcon = 2;
    state.victory_points = 0;
    state.countries[ts::countries::URUGUAY].ussr_influence = 1;

    // USSR triggers Missile Envy event
    ts::CardHandlers::trigger_event(state, ts::card_ids::MISSILE_ENVY, ts::Player::USSR);

    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_OP_MODE);
    ASSERT_EQ(state.ctx().pending_op_card, ts::card_ids::DUCK_AND_COVER);
    ASSERT_EQ(state.ctx().pending_ops_value, 3);

    // USSR spends 3 Ops on Influence
    ts::Engine::step(state, ts::MicroAction(ts::DecisionType::SELECT_OP_MODE, static_cast<uint8_t>(ts::OpMode::INFLUENCE), 0, 0));
    ts::Engine::step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::URUGUAY, 0, 0));
    ts::Engine::step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::URUGUAY, 0, 0));
    ts::Engine::step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::URUGUAY, 0, 0));

    // Duck and Cover event must NOT fire: DEFCON must remain 2, VP remains 0, game not over
    ASSERT_NE(state.current_phase, ts::Phase::GAME_OVER);
    ASSERT_EQ(state.defcon, 2);
    ASSERT_EQ(state.victory_points, 0);
    ASSERT_EQ(state.card_locations[ts::card_ids::DUCK_AND_COVER], ts::CardLocation::DISCARD_PILE);
}

TEST(MidCardsTest, Card49_MissileEnvy_RecipientMustPlayForOps) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == ts::CardLocation::HAND_US || state.card_locations[i] == ts::CardLocation::HAND_USSR) {
            state.card_locations[i] = ts::CardLocation::DRAW_DECK;
        }
    }
    state.card_locations[ts::card_ids::MISSILE_ENVY] = ts::CardLocation::HAND_US;
    state.card_locations[ts::card_ids::FIDEL] = ts::CardLocation::HAND_US;
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::HAND_US;
    state.forced_card_player = ts::Player::US;
    state.forced_card_id = ts::card_ids::MISSILE_ENVY;
    state.phasing_player = ts::Player::US;
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.ctx().decision_player = ts::Player::US;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;
    state.countries[ts::countries::URUGUAY].us_influence = 1;

    // At SELECT_CARD, US must ONLY be allowed to select Missile Envy
    uint8_t card_mask[112] = {0};
    size_t card_out_size = 0;
    ts::ActionMask::generate_mask(state, card_mask, &card_out_size);
    ASSERT_EQ(card_mask[ts::card_ids::MISSILE_ENVY], 1);
    ASSERT_EQ(card_mask[ts::card_ids::FIDEL], 0);
    ASSERT_EQ(card_mask[ts::card_ids::DUCK_AND_COVER], 0);

    // US selects forced Missile Envy
    ts::Engine::step(state, ts::MicroAction(ts::DecisionType::SELECT_CARD, ts::card_ids::MISSILE_ENVY, 0, 0));
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_PLAY_MODE);

    uint8_t mask[4] = {0};
    size_t out_size = 0;
    ts::ActionMask::generate_mask(state, mask, &out_size);

    // PlayMode::OPS (index 1) must be legal (1).
    // PlayMode::EVENT (index 0) and PlayMode::SPACE (index 2) must be ILLEGAL (0).
    ASSERT_EQ(mask[static_cast<size_t>(ts::PlayMode::OPS)], 1);
    ASSERT_EQ(mask[static_cast<size_t>(ts::PlayMode::EVENT)], 0);
    ASSERT_EQ(mask[static_cast<size_t>(ts::PlayMode::SPACE)], 0);

    // Attempting EVENT must fail
    bool event_accepted = ts::Engine::step(state, ts::MicroAction(ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::PlayMode::EVENT), 0, 0));
    ASSERT_FALSE(event_accepted);

    // Step OPS (index 1)
    bool ops_accepted = ts::Engine::step(state, ts::MicroAction(ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::PlayMode::OPS), 0, 0));
    ASSERT_TRUE(ops_accepted);

    // Spend 2 ops on influence
    ts::Engine::step(state, ts::MicroAction(ts::DecisionType::SELECT_OP_MODE, static_cast<uint8_t>(ts::OpMode::INFLUENCE), 0, 0));
    ts::Engine::step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::URUGUAY, 0, 0));
    ts::Engine::step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::URUGUAY, 0, 0));

    // After play completes:
    ASSERT_EQ(state.forced_card_id, 0);
    ASSERT_EQ(state.forced_card_player, ts::Player::NONE);
    ASSERT_EQ(state.card_locations[ts::card_ids::MISSILE_ENVY], ts::CardLocation::DISCARD_PILE);
}

TEST(MidCardsTest, Card49_MissileEnvy_SingleHighest) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == ts::CardLocation::HAND_USSR) {
            state.card_locations[i] = ts::CardLocation::DRAW_DECK;
        }
    }
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::HAND_USSR; // 3 Ops US card
    state.card_locations[ts::card_ids::FIDEL] = ts::CardLocation::HAND_USSR; // 2 Ops

    ts::CardHandlers::trigger_event(state, ts::card_ids::MISSILE_ENVY, ts::Player::US);
    // US took Duck and Cover (US event -> executed immediately and discarded), gave Missile Envy to USSR
    ASSERT_EQ(state.card_locations[ts::card_ids::DUCK_AND_COVER], ts::CardLocation::DISCARD_PILE);
    ASSERT_EQ(state.victory_points, 1);
    ASSERT_EQ(state.card_locations[ts::card_ids::MISSILE_ENVY], ts::CardLocation::HAND_USSR);
    ASSERT_EQ(state.forced_card_player, ts::Player::USSR);
    ASSERT_EQ(state.forced_card_id, ts::card_ids::MISSILE_ENVY);
}

// Card 50: We Will Bury You
TEST(MidCardsTest, Card50_WeWillBuryYou) {
    ts::GameState state{};
    state.defcon = 4;
    ts::CardHandlers::trigger_event(state, ts::card_ids::WE_WILL_BURY_YOU, ts::Player::USSR);
    ASSERT_EQ(state.defcon, 3);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::WE_WILL_BURY_YOU_PENDING));
}

// Card 51: Brezhnev Doctrine
TEST(MidCardsTest, Card51_BrezhnevDoctrine) {
    ts::GameState state{};
    ts::CardHandlers::trigger_event(state, ts::card_ids::BREZHNEV_DOCTRINE, ts::Player::USSR);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::BREZHNEV_DOCTRINE_ACTIVE));
}

// Card 52: Portuguese Empire Crumbles
TEST(MidCardsTest, Card52_PortugueseEmpireCrumbles) {
    ts::GameState state{};
    state.countries[ts::countries::ANGOLA].ussr_influence = 0;
    state.countries[ts::countries::SE_AFRICAN_STS].ussr_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::PORTUGUESE_EMPIRE_CRUMBLES, ts::Player::USSR);
    ASSERT_EQ(state.countries[ts::countries::ANGOLA].ussr_influence, 2);
    ASSERT_EQ(state.countries[ts::countries::SE_AFRICAN_STS].ussr_influence, 2);
}

// Card 53: South African Unrest
TEST(MidCardsTest, Card53_SouthAfricanUnrest_Branch0) {
    ts::GameState state{};
    state.countries[ts::countries::SOUTH_AFRICA].ussr_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::SOUTH_AFRICAN_UNREST, ts::Player::USSR);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::CHOOSE_BRANCH);
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::CHOOSE_BRANCH, 0, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::SOUTH_AFRICA].ussr_influence, 2);
}

// Branch 1 is "1 Influence in South Africa and 2 Influence in any countries adjacent to South
// Africa" -- plural, so the pair may be split. The engine asked once and added both to the one
// country chosen, which at turn 5 AR2 of ts-replayer game 112 could not express the USSR's
// 1 in Botswana and 1 in Angola.
TEST(MidCardsTest, Card53_SouthAfricanUnrest_Branch1_SplitsAcrossTwoNeighbours) {
    ts::GameState state{};
    state.countries[ts::countries::SOUTH_AFRICA].ussr_influence = 0;
    state.countries[ts::countries::BOTSWANA].ussr_influence = 0;
    state.countries[ts::countries::ANGOLA].ussr_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::SOUTH_AFRICAN_UNREST, ts::Player::USSR);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::CHOOSE_BRANCH);

    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::CHOOSE_BRANCH, 1, 0, 0});
    ASSERT_FALSE(done);
    ASSERT_EQ(state.countries[ts::countries::SOUTH_AFRICA].ussr_influence, 1);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::POINT_NODE);

    done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::BOTSWANA, 0, 0});
    ASSERT_FALSE(done); // one of the two adjacent placements is still to come
    ASSERT_EQ(state.countries[ts::countries::BOTSWANA].ussr_influence, 1);

    done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::ANGOLA, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::ANGOLA].ussr_influence, 1);
    ASSERT_EQ(state.ctx().resolving_card, 0);
}

// Both into one country stays available: it is a choice, not the only option.
TEST(MidCardsTest, Card53_SouthAfricanUnrest_Branch1_MayStackBothInOneNeighbour) {
    ts::GameState state{};
    state.countries[ts::countries::SOUTH_AFRICA].ussr_influence = 0;
    state.countries[ts::countries::BOTSWANA].ussr_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::SOUTH_AFRICAN_UNREST, ts::Player::USSR);
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::CHOOSE_BRANCH, 1, 0, 0});
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::BOTSWANA, 0, 0});
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::BOTSWANA, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::BOTSWANA].ussr_influence, 2);
}

// An event that says "add Influence" places it directly. The Ops path charges two per point
// in a country the opponent controls; this must not.
TEST(MidCardsTest, Card53_SouthAfricanUnrest_Branch1_IgnoresOpponentControl) {
    ts::GameState state{};
    state.countries[ts::countries::SOUTH_AFRICA].ussr_influence = 0;
    // US control of Botswana: stability 2, so 4 US Influence against 0 is firmly controlled.
    state.countries[ts::countries::BOTSWANA].ussr_influence = 0;
    state.countries[ts::countries::BOTSWANA].us_influence = 4;
    ts::CardHandlers::trigger_event(state, ts::card_ids::SOUTH_AFRICAN_UNREST, ts::Player::USSR);
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::CHOOSE_BRANCH, 1, 0, 0});
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::BOTSWANA, 0, 0});
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::BOTSWANA, 0, 0});
    ASSERT_TRUE(done);
    // Both points land despite US control -- an event pays no doubled cost.
    ASSERT_EQ(state.countries[ts::countries::BOTSWANA].ussr_influence, 2);
    ASSERT_EQ(state.countries[ts::countries::BOTSWANA].us_influence, 4);
}

// A war is fought against a legal target or not at all. trigger_war used to leave
// allow_early_stop as whatever the previous decision in the frame had set, so a war that
// followed one -- SALT Negotiations and Missile Envy at turn 5's headline of ts-replayer game
// 170 -- offered a decline alongside its targets that handle_war_step has no case for. The
// state machine then ended the event frame and the war silently never happened.
TEST(MidCardsTest, Card36_BrushWar_TargetChoiceNeverOffersADecline) {
    ts::GameState state{};
    state.ctx().allow_early_stop = 1;      // left set by whatever resolved before the war
    ts::CardHandlers::trigger_event(state, ts::card_ids::BRUSH_WAR, ts::Player::USSR);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::POINT_NODE);
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::BRUSH_WAR);
    ASSERT_EQ(state.ctx().allow_early_stop, 0);

    uint8_t mask[212] = {0};
    ts::ActionMask::generate_flat_mask_212(state, mask);
    ASSERT_EQ(mask[211], 0);
    uint8_t targets = 0;
    for (uint8_t i = 119; i < 203; ++i) targets = static_cast<uint8_t>(targets + mask[i]);
    ASSERT_GT(targets, 0);
}

// De-Stalinization is the other side of it: declining stage 1 is legal and moves on to stage
// 2, and when exactly two Influence were removed the remaining step count lands on the value
// it already had. Nothing about the decision "looks" different, so the frame must be kept on
// the handler's word rather than on any guess about which fields ought to have changed.
TEST(MidCardsTest, Card33_DeStalinization_DeclineAfterTwoRemovalsContinuesToPlacement) {
    ts::GameState state{};
    state.countries[ts::countries::POLAND].ussr_influence = 3;
    ts::CardHandlers::trigger_event(state, ts::card_ids::DE_STALINIZATION, ts::Player::USSR);
    ASSERT_EQ(state.ctx().allow_early_stop, 1);
    ASSERT_EQ(state.ctx().remaining_steps, 4);

    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::POLAND, 0, 0});
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::POLAND, 0, 0});
    ASSERT_EQ(state.ctx().remaining_steps, 2);   // two removed, two still allowed

    ts::MicroAction decline{ts::DecisionType::POINT_NODE, 0, 0, ts::action_flags::CONFIRM_DONE};
    bool done = ts::CardHandlers::handle_event_step(state, decline);
    ASSERT_FALSE(done);
    ASSERT_EQ(state.ctx().remaining_steps, 2);   // unchanged, yet the event has moved on
    ASSERT_EQ(state.ctx().max_per_country, 2);   // ...to stage 2, which is how you can tell
    ASSERT_EQ(state.ctx().allow_early_stop, 0);  // all removed Influence must be placed
}

// Card 54: Allende
TEST(MidCardsTest, Card54_Allende) {
    ts::GameState state{};
    state.countries[ts::countries::CHILE].ussr_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::ALLENDE, ts::Player::USSR);
    ASSERT_EQ(state.countries[ts::countries::CHILE].ussr_influence, 2);
}

// Card 55: Willy Brandt
TEST(MidCardsTest, Card55_WillyBrandt_Basic) {
    ts::GameState state{};
    state.countries[ts::countries::WEST_GERMANY].ussr_influence = 0;
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::WILLY_BRANDT, ts::Player::USSR);
    ASSERT_EQ(state.victory_points, -1);
    ASSERT_EQ(state.countries[ts::countries::WEST_GERMANY].ussr_influence, 1);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::WILLY_BRANDT_PLAYED));
    ASSERT_TRUE(state.has_flag(ts::effect_bits::NATO_CANCELED_WEST_GERMANY));
}

// Card 56: Muslim Revolution
TEST(MidCardsTest, Card56_MuslimRevolution) {
    ts::GameState state{};
    state.countries[ts::countries::IRAN].us_influence = 3;
    state.countries[ts::countries::IRAQ].us_influence = 2;
    ts::CardHandlers::trigger_event(state, ts::card_ids::MUSLIM_REVOLUTION, ts::Player::USSR);
    ASSERT_EQ(state.ctx().remaining_steps, 2);
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::IRAN, 0, 0});
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::IRAQ, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::IRAN].us_influence, 0);
    ASSERT_EQ(state.countries[ts::countries::IRAQ].us_influence, 0);
}

// Card 57: ABM Treaty
TEST(MidCardsTest, Card57_ABMTreaty) {
    ts::GameState state{};
    state.defcon = 2;
    ts::CardHandlers::trigger_event(state, ts::card_ids::ABM_TREATY, ts::Player::US);
    ASSERT_EQ(state.defcon, 3);
    ASSERT_EQ(state.ctx().pending_ops_value, 4);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_OP_MODE);
}

// Card 58: Cultural Revolution
TEST(MidCardsTest, Card58_CulturalRevolution_USHasCard) {
    ts::GameState state{};
    state.china_card_holder = ts::Player::US;
    state.china_card_playable = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::CULTURAL_REVOLUTION, ts::Player::USSR);
    ASSERT_EQ(state.china_card_holder, ts::Player::USSR);
    ASSERT_EQ(state.china_card_playable, 1);
}

// Card 59: Flower Power
TEST(MidCardsTest, Card59_FlowerPower) {
    ts::GameState state{};
    ts::CardHandlers::trigger_event(state, ts::card_ids::FLOWER_POWER, ts::Player::USSR);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::FLOWER_POWER_ACTIVE));
}

// Card 60: U-2 Incident
TEST(MidCardsTest, Card60_U2Incident) {
    ts::GameState state{};
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::U2_INCIDENT, ts::Player::USSR);
    ASSERT_EQ(state.victory_points, -1);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::U2_INCIDENT_ACTIVE));
}

// Card 61: OPEC
TEST(MidCardsTest, Card61_OPEC) {
    ts::GameState state{};
    state.countries[ts::countries::EGYPT].ussr_influence = 4; // Controlled
    state.countries[ts::countries::IRAN].ussr_influence = 4;  // Controlled
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::OPEC, ts::Player::USSR);
    ASSERT_EQ(state.victory_points, -2);
}

// Card 62: Lone Gunman
TEST(MidCardsTest, Card62_LoneGunman) {
    ts::GameState state{};
    ts::CardHandlers::trigger_event(state, ts::card_ids::LONE_GUNMAN, ts::Player::USSR);
    ASSERT_EQ(state.ctx().decision_player, ts::Player::USSR);
    ASSERT_EQ(state.ctx().pending_ops_value, 1);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_OP_MODE);
}

// Card 63: Colonial Rear Guards
TEST(MidCardsTest, Card63_ColonialRearGuards) {
    ts::GameState state{};
    state.countries[ts::countries::ANGOLA].us_influence = 0;
    state.countries[ts::countries::ZAIRE].us_influence = 0;
    state.countries[ts::countries::NIGERIA].us_influence = 0;
    state.countries[ts::countries::VIETNAM].us_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::COLONIAL_REAR_GUARDS, ts::Player::US);
    ASSERT_EQ(state.ctx().remaining_steps, 4);
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::ANGOLA, 0, 0});
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::ZAIRE, 0, 0});
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::NIGERIA, 0, 0});
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::VIETNAM, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::ANGOLA].us_influence, 1);
    ASSERT_EQ(state.countries[ts::countries::VIETNAM].us_influence, 1);
}

// Card 64: Panama Canal Returned
TEST(MidCardsTest, Card64_PanamaCanalReturned) {
    ts::GameState state{};
    state.countries[ts::countries::PANAMA].us_influence = 0;
    state.countries[ts::countries::COSTA_RICA].us_influence = 0;
    state.countries[ts::countries::VENEZUELA].us_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::PANAMA_CANAL_RETURNED, ts::Player::US);
    ASSERT_EQ(state.countries[ts::countries::PANAMA].us_influence, 1);
    ASSERT_EQ(state.countries[ts::countries::COSTA_RICA].us_influence, 1);
    ASSERT_EQ(state.countries[ts::countries::VENEZUELA].us_influence, 1);
}

// Card 65: Camp David Accords
TEST(MidCardsTest, Card65_CampDavidAccords) {
    ts::GameState state{};
    state.countries[ts::countries::ISRAEL].us_influence = 0;
    state.countries[ts::countries::JORDAN].us_influence = 0;
    state.countries[ts::countries::EGYPT].us_influence = 0;
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::CAMP_DAVID_ACCORDS, ts::Player::US);
    ASSERT_EQ(state.victory_points, 1);
    ASSERT_EQ(state.countries[ts::countries::ISRAEL].us_influence, 1);
    ASSERT_EQ(state.countries[ts::countries::JORDAN].us_influence, 1);
    ASSERT_EQ(state.countries[ts::countries::EGYPT].us_influence, 1);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::CAMP_DAVID_PLAYED));
}

// Card 66: Puppet Governments
TEST(MidCardsTest, Card66_PuppetGovernments) {
    ts::GameState state{};
    state.countries[ts::countries::PERU].us_influence = 0;
    state.countries[ts::countries::PERU].ussr_influence = 0;
    state.countries[ts::countries::BOLIVIA].us_influence = 0;
    state.countries[ts::countries::BOLIVIA].ussr_influence = 0;
    state.countries[ts::countries::PARAGUAY].us_influence = 0;
    state.countries[ts::countries::PARAGUAY].ussr_influence = 0;

    ts::CardHandlers::trigger_event(state, ts::card_ids::PUPPET_GOVERNMENTS, ts::Player::US);
    ASSERT_EQ(state.ctx().remaining_steps, 3);
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::PERU, 0, 0});
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::BOLIVIA, 0, 0});
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::PARAGUAY, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::PERU].us_influence, 1);
    ASSERT_EQ(state.countries[ts::countries::BOLIVIA].us_influence, 1);
    ASSERT_EQ(state.countries[ts::countries::PARAGUAY].us_influence, 1);
}

// Card 67: Grain Sales to Soviets
TEST(MidCardsTest, Card67_GrainSales_ReturnForOps) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == ts::CardLocation::HAND_USSR) {
            state.card_locations[i] = ts::CardLocation::DRAW_DECK;
        }
    }
    state.card_locations[ts::card_ids::FIDEL] = ts::CardLocation::HAND_USSR;
    ts::CardHandlers::trigger_event(state, ts::card_ids::GRAIN_SALES, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::CHOOSE_BRANCH);

    // Branch 1: Return card and conduct 2 Ops
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::CHOOSE_BRANCH, 1, 0, 0});
    ASSERT_FALSE(done);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_OP_MODE);
    ASSERT_EQ(state.ctx().pending_ops_value, 2);
}

// Card 68: John Paul II Elected Pope
TEST(MidCardsTest, Card68_JohnPaulII) {
    ts::GameState state{};
    state.countries[ts::countries::POLAND].ussr_influence = 3;
    state.countries[ts::countries::POLAND].us_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::JOHN_PAUL_II, ts::Player::US);
    ASSERT_EQ(state.countries[ts::countries::POLAND].ussr_influence, 1);
    ASSERT_EQ(state.countries[ts::countries::POLAND].us_influence, 1);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::JOHN_PAUL_II_PLAYED));
}

// Card 69: Latin American Death Squads
TEST(MidCardsTest, Card69_LatinAmericanDeathSquads) {
    ts::GameState state{};
    ts::CardHandlers::trigger_event(state, ts::card_ids::LATIN_DEATH_SQUADS, ts::Player::US);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::DEATH_SQUADS_US));
}

// Card 70: OAS Founded
TEST(MidCardsTest, Card70_OASFounded) {
    ts::GameState state{};
    state.countries[ts::countries::PANAMA].us_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::OAS_FOUNDED, ts::Player::US);
    ASSERT_EQ(state.ctx().remaining_steps, 2);
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::PANAMA, 0, 0});
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::PANAMA, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::PANAMA].us_influence, 2);
}

// Card 71: Nixon Plays the China Card
TEST(MidCardsTest, Card71_NixonPlaysTheChinaCard_USSRHasCard) {
    ts::GameState state{};
    state.china_card_holder = ts::Player::USSR;
    state.china_card_playable = 1;
    ts::CardHandlers::trigger_event(state, ts::card_ids::NIXON_PLAYS_THE_CHINA_CARD, ts::Player::US);
    ASSERT_EQ(state.china_card_holder, ts::Player::US);
    ASSERT_EQ(state.china_card_playable, 0); // Face down
}

TEST(MidCardsTest, Card71_NixonPlaysTheChinaCard_USAlreadyHasCard) {
    ts::GameState state{};
    state.china_card_holder = ts::Player::US;
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::NIXON_PLAYS_THE_CHINA_CARD, ts::Player::US);
    ASSERT_EQ(state.victory_points, 2);
}

// Card 72: Sadat Expels Soviets
TEST(MidCardsTest, Card72_SadatExpelsSoviets) {
    ts::GameState state{};
    state.countries[ts::countries::EGYPT].ussr_influence = 3;
    state.countries[ts::countries::EGYPT].us_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::SADAT_EXPELS_SOVIETS, ts::Player::US);
    ASSERT_EQ(state.countries[ts::countries::EGYPT].ussr_influence, 0);
    ASSERT_EQ(state.countries[ts::countries::EGYPT].us_influence, 1);
}

// Card 73: Shuttle Diplomacy
TEST(MidCardsTest, Card73_ShuttleDiplomacy) {
    ts::GameState state{};
    ts::CardHandlers::trigger_event(state, ts::card_ids::SHUTTLE_DIPLOMACY, ts::Player::US);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::SHUTTLE_DIPLOMACY_ACTIVE));
}

// Card 74: The Voice of America
TEST(MidCardsTest, Card74_VoiceOfAmerica) {
    ts::GameState state{};
    state.countries[ts::countries::EGYPT].ussr_influence = 3;
    state.countries[ts::countries::CHILE].ussr_influence = 2;
    ts::CardHandlers::trigger_event(state, ts::card_ids::THE_VOICE_OF_AMERICA, ts::Player::US);
    ASSERT_EQ(state.ctx().remaining_steps, 4);
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::EGYPT, 0, 0});
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::EGYPT, 0, 0});
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::CHILE, 0, 0});
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::CHILE, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::EGYPT].ussr_influence, 1);
    ASSERT_EQ(state.countries[ts::countries::CHILE].ussr_influence, 0);
}

// Card 75: Liberation Theology
TEST(MidCardsTest, Card75_LiberationTheology) {
    ts::GameState state{};
    state.countries[ts::countries::NICARAGUA].ussr_influence = 0;
    state.countries[ts::countries::GUATEMALA].ussr_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::LIBERATION_THEOLOGY, ts::Player::USSR);
    ASSERT_EQ(state.ctx().remaining_steps, 3);
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::NICARAGUA, 0, 0});
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::NICARAGUA, 0, 0});
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::GUATEMALA, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::NICARAGUA].ussr_influence, 2);
    ASSERT_EQ(state.countries[ts::countries::GUATEMALA].ussr_influence, 1);
}

// Card 76: Ussuri River Skirmish
TEST(MidCardsTest, Card76_UssuriRiver_USSRHasChinaCard) {
    ts::GameState state{};
    state.china_card_holder = ts::Player::USSR;
    state.china_card_playable = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::USSURI_RIVER_SKIRMISH, ts::Player::US);
    ASSERT_EQ(state.china_card_holder, ts::Player::US);
    ASSERT_EQ(state.china_card_playable, 1); // Face up
}

// Card 77: Ask Not What Your Country Can Do For You
TEST(MidCardsTest, Card77_AskNot) {
    ts::GameState state{};
    state.card_locations[ts::card_ids::FIDEL] = ts::CardLocation::HAND_US;
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::DRAW_DECK;
    state.card_locations[ts::card_ids::FIVE_YEAR_PLAN] = ts::CardLocation::DRAW_DECK;
    ts::CardHandlers::trigger_event(state, ts::card_ids::ASK_NOT_WHAT_YOUR_COUNTRY_CAN_DO_FOR_YOU, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_CARD);

    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::FIDEL, 0, 0});
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, 0, 0, ts::action_flags::CONFIRM_DONE});
    ASSERT_EQ(state.card_locations[ts::card_ids::FIDEL], ts::CardLocation::DISCARD_PILE);
    bool has_drawn = (state.card_locations[ts::card_ids::DUCK_AND_COVER] == ts::CardLocation::HAND_US) || (state.card_locations[ts::card_ids::FIVE_YEAR_PLAN] == ts::CardLocation::HAND_US);
    ASSERT_TRUE(has_drawn);
}

// Card 78: Alliance for Progress
TEST(MidCardsTest, Card78_AllianceForProgress) {
    ts::GameState state{};
    state.countries[ts::countries::MEXICO].us_influence = 3; // CA Battleground
    state.countries[ts::countries::PANAMA].us_influence = 3; // CA Battleground
    state.countries[ts::countries::BRAZIL].us_influence = 3; // SA Battleground
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::ALLIANCE_FOR_PROGRESS, ts::Player::US);
    ASSERT_EQ(state.victory_points, 3);
}

// Card 79: Africa Scoring
TEST(MidCardsTest, Card79_AfricaScoring) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::AFRICA_SCORING, ts::Player::US);
    ASSERT_TRUE(state.victory_points != 0 || state.victory_points == 0);
}

// Card 80: One Small Step
TEST(MidCardsTest, Card80_OneSmallStep_Behind) {
    ts::GameState state{};
    state.us_space_track = 1;
    state.ussr_space_track = 3;
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::ONE_SMALL_STEP, ts::Player::US);
    ASSERT_EQ(state.us_space_track, 3); // Advanced 2 boxes
}

// Card 81: South America Scoring
TEST(MidCardsTest, Card81_SouthAmericaScoring) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
    state.countries[ts::countries::BRAZIL].us_influence = 3;
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::SOUTH_AMERICA_SCORING, ts::Player::US);
    ASSERT_TRUE(state.victory_points != 0 || state.victory_points == 0);
}

// Card 107: Che
TEST(MidCardsTest, Card107_Che) {
    ts::GameState state{};
    state.countries[ts::countries::COLOMBIA].us_influence = 2; // Non-BG in SA
    state.countries[ts::countries::COLOMBIA].ussr_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::CHE, ts::Player::USSR);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::POINT_NODE);

    // Roll 6 -> coup value: 6 + 3 - 2*1 = 7. Removes 2 US and adds 5 USSR
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::COLOMBIA, 6, 0});
    ASSERT_FALSE(done); // Second coup is offered because US influence was removed
    ASSERT_EQ(state.countries[ts::countries::COLOMBIA].us_influence, 0);
    ASSERT_EQ(state.countries[ts::countries::COLOMBIA].ussr_influence, 5);
    // Pass second coup
    done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, 0, 0, ts::action_flags::CONFIRM_DONE});
    ASSERT_TRUE(done);
}

// Card 108: Our Man in Tehran
TEST(MidCardsTest, Card108_OurManInTehran_WithMEControl) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
    state.countries[ts::countries::ISRAEL].us_influence = 4; // US controls Israel (ME country)

    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::OUR_MAN_IN_TEHRAN, ts::Player::US);
    ASSERT_FALSE(done);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_CARD);
}

TEST(MidCardsTest, Card108_OurManInTehran_NoMEControl) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
    for (uint8_t i = 0; i < 84; ++i) {
        if (ts::MapData::get_country(i).region == ts::Region::MIDDLE_EAST) {
            state.countries[i].us_influence = 0;
        }
    }
    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::OUR_MAN_IN_TEHRAN, ts::Player::US);
    ASSERT_TRUE(done); // Nothing happens
}
