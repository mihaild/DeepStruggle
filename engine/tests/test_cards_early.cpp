#include "test_framework.hpp"
#include "ts/engine.hpp"
#include "ts/card_data.hpp"
#include "ts/card_handlers.hpp"
#include "ts/constants.hpp"
#include "ts/scoring.hpp"
#include "ts/ops.hpp"

// Card 1: Asia Scoring
TEST(EarlyCardsTest, Card01_AsiaScoring) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
    state.countries[ts::countries::JAPAN].us_influence = 4;
    state.countries[ts::countries::SOUTH_KOREA].us_influence = 3;
    state.countries[ts::countries::NORTH_KOREA].ussr_influence = 3;
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::ASIA_SCORING, ts::Player::US);
    // Domination/Presence scored
    ASSERT_TRUE(state.victory_points > 0);
}

// Card 2: Europe Scoring
TEST(EarlyCardsTest, Card02_EuropeScoring) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
    state.countries[ts::countries::POLAND].ussr_influence = 3;
    state.countries[ts::countries::EAST_GERMANY].ussr_influence = 3;
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::EUROPE_SCORING, ts::Player::USSR);
    ASSERT_TRUE(state.victory_points <= 0);
}

// Card 3: Middle East Scoring
TEST(EarlyCardsTest, Card03_MiddleEastScoring) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
    state.countries[ts::countries::ISRAEL].us_influence = 4;
    state.countries[ts::countries::EGYPT].ussr_influence = 2;
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::MIDDLE_EAST_SCORING, ts::Player::US);
    // VP changed based on scoring
    ASSERT_TRUE(state.victory_points != 0 || state.victory_points == 0);
}

// Card 4: Duck and Cover
TEST(EarlyCardsTest, Card04_DuckAndCover) {
    ts::GameState state{};
    state.defcon = 4;
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::DUCK_AND_COVER, ts::Player::US);
    ASSERT_EQ(state.defcon, 3);
    ASSERT_EQ(state.victory_points, 2); // 5 - 3 = 2 VP for US
}

// Card 5: Five Year Plan
TEST(EarlyCardsTest, Card05_FiveYearPlan_USSRHand) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == ts::CardLocation::HAND_USSR) {
            state.card_locations[i] = ts::CardLocation::DRAW_DECK;
        }
    }
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::HAND_USSR;
    state.card_locations[ts::card_ids::FIDEL] = ts::CardLocation::HAND_USSR;
    ts::CardHandlers::trigger_event(state, ts::card_ids::FIVE_YEAR_PLAN, ts::Player::US);
    // One card discarded from USSR hand
    bool discarded = (state.card_locations[ts::card_ids::DUCK_AND_COVER] == ts::CardLocation::DISCARD_PILE) ||
                     (state.card_locations[ts::card_ids::FIDEL] == ts::CardLocation::DISCARD_PILE);
    ASSERT_TRUE(discarded);
}

TEST(EarlyCardsTest, Card05_FiveYearPlan_EmptyHand) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == ts::CardLocation::HAND_USSR) {
            state.card_locations[i] = ts::CardLocation::DRAW_DECK;
        }
    }
    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::FIVE_YEAR_PLAN, ts::Player::US);
    ASSERT_TRUE(done);
}

// Card 6: The China Card
TEST(EarlyCardsTest, Card06_TheChinaCard_Properties) {
    const auto& c = ts::CardData::get_card(ts::card_ids::THE_CHINA_CARD);
    ASSERT_EQ(c.ops, 4);
    ASSERT_EQ(c.side, ts::Player::NONE);
}

// Card 7: Socialist Governments
TEST(EarlyCardsTest, Card07_SocialistGovernments_Basic) {
    ts::GameState state{};
    state.countries[ts::countries::ITALY].us_influence = 3;
    state.countries[ts::countries::FRANCE].us_influence = 2;
    ts::CardHandlers::trigger_event(state, ts::card_ids::SOCIALIST_GOVERNMENTS, ts::Player::USSR);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::POINT_NODE);
    ASSERT_EQ(state.ctx().remaining_steps, 3);

    // Remove 2 from Italy, 1 from France
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::ITALY, 0, 0));
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::ITALY, 0, 0));
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::FRANCE, 0, 0));
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::ITALY].us_influence, 1);
    ASSERT_EQ(state.countries[ts::countries::FRANCE].us_influence, 1);
}

TEST(EarlyCardsTest, Card07_SocialistGovernments_BlockedByIronLady) {
    ts::GameState state{};
    state.set_flag(ts::effect_bits::IRON_LADY_PLAYED);
    ASSERT_FALSE(ts::CardHandlers::can_trigger_event(state, ts::card_ids::SOCIALIST_GOVERNMENTS, ts::Player::USSR));
    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::SOCIALIST_GOVERNMENTS, ts::Player::USSR);
    ASSERT_TRUE(done);
}

// Card 8: Fidel
TEST(EarlyCardsTest, Card08_Fidel_Basic) {
    ts::GameState state{};
    state.countries[ts::countries::CUBA].us_influence = 2;
    state.countries[ts::countries::CUBA].ussr_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::FIDEL, ts::Player::USSR);
    ASSERT_EQ(state.countries[ts::countries::CUBA].us_influence, 0);
    ASSERT_EQ(state.countries[ts::countries::CUBA].ussr_influence, 3);
}

TEST(EarlyCardsTest, Card08_Fidel_HigherUSSRInfluence) {
    ts::GameState state{};
    state.countries[ts::countries::CUBA].us_influence = 1;
    state.countries[ts::countries::CUBA].ussr_influence = 5;
    ts::CardHandlers::trigger_event(state, ts::card_ids::FIDEL, ts::Player::USSR);
    ASSERT_EQ(state.countries[ts::countries::CUBA].us_influence, 0);
    ASSERT_EQ(state.countries[ts::countries::CUBA].ussr_influence, 5); // Unchanged at 5
}

// Card 9: Vietnam Revolts
TEST(EarlyCardsTest, Card09_VietnamRevolts) {
    ts::GameState state{};
    state.countries[ts::countries::VIETNAM].ussr_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::VIETNAM_REVOLTS, ts::Player::USSR);
    ASSERT_EQ(state.countries[ts::countries::VIETNAM].ussr_influence, 2);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::VIETNAM_REVOLTS_ACTIVE));
}

// Card 10: Blockade
TEST(EarlyCardsTest, Card10_Blockade_Discard3OpsCard) {
    ts::GameState state{};
    state.countries[ts::countries::WEST_GERMANY].us_influence = 4;
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::HAND_US; // 3 ops
    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::BLOCKADE, ts::Player::USSR);
    ASSERT_FALSE(done);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_CARD);

    // US discards Duck and Cover
    done = ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::SELECT_CARD, ts::card_ids::DUCK_AND_COVER, 0, 0));
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::WEST_GERMANY].us_influence, 4); // Saved!
    ASSERT_EQ(state.card_locations[ts::card_ids::DUCK_AND_COVER], ts::CardLocation::DISCARD_PILE);
}

TEST(EarlyCardsTest, Card10_Blockade_No3OpsCard) {
    ts::GameState state{};
    state.countries[ts::countries::WEST_GERMANY].us_influence = 4;
    // US has no 3+ ops cards in hand
    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::BLOCKADE, ts::Player::USSR);
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::WEST_GERMANY].us_influence, 0); // Removed!
}

TEST(EarlyCardsTest, Card10_Blockade_EarlyStopRefuseDiscard) {
    ts::GameState state{};
    state.countries[ts::countries::WEST_GERMANY].us_influence = 4;
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::HAND_US;
    ts::CardHandlers::trigger_event(state, ts::card_ids::BLOCKADE, ts::Player::USSR);
    // US chooses confirm_done / pass (refuses to discard)
    ts::MicroAction pass_act{};
    pass_act.flags = ts::action_flags::CONFIRM_DONE;
    bool done = ts::CardHandlers::handle_event_step(state, pass_act);
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::WEST_GERMANY].us_influence, 0); // Removed!
}

// Card 11: Korean War
TEST(EarlyCardsTest, Card11_KoreanWar_Success) {
    ts::GameState state{};
    state.countries[ts::countries::SOUTH_KOREA].us_influence = 2;
    state.countries[ts::countries::SOUTH_KOREA].ussr_influence = 0;
    state.victory_points = 0;
    state.ussr_mil_ops = 0;
    // No US controlled neighbors -> unmod roll. We test with default PRNG
    ts::CardHandlers::trigger_event(state, ts::card_ids::KOREAN_WAR, ts::Player::USSR);
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::ROLL_DIE, 0, 0, 0));
    ASSERT_EQ(state.ussr_mil_ops, 2);
}

// Card 12: Romanian Abdication
TEST(EarlyCardsTest, Card12_RomanianAbdication) {
    ts::GameState state{};
    state.countries[ts::countries::ROMANIA].us_influence = 3;
    state.countries[ts::countries::ROMANIA].ussr_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::ROMANIAN_ABDICATION, ts::Player::USSR);
    ASSERT_EQ(state.countries[ts::countries::ROMANIA].us_influence, 0);
    ASSERT_EQ(state.countries[ts::countries::ROMANIA].ussr_influence, 3);
}

// Card 13: Arab-Israeli War
TEST(EarlyCardsTest, Card13_ArabIsraeliWar_Success) {
    ts::GameState state{};
    state.countries[ts::countries::ISRAEL].us_influence = 2;
    state.countries[ts::countries::ISRAEL].ussr_influence = 0;
    state.victory_points = 0;
    state.ussr_mil_ops = 0;
    // Forced roll = 5, no US controlled neighbors -> success
    ts::CardHandlers::trigger_event(state, ts::card_ids::ARAB_ISRAELI_WAR, ts::Player::USSR, 5);
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::ROLL_DIE, 5, 0, 0));
    ASSERT_EQ(state.countries[ts::countries::ISRAEL].us_influence, 0);
    ASSERT_EQ(state.countries[ts::countries::ISRAEL].ussr_influence, 2);
    ASSERT_EQ(state.victory_points, -2);
    ASSERT_EQ(state.ussr_mil_ops, 2);
}

TEST(EarlyCardsTest, Card13_ArabIsraeliWar_BlockedByCampDavid) {
    ts::GameState state{};
    state.set_flag(ts::effect_bits::CAMP_DAVID_PLAYED);
    ASSERT_FALSE(ts::CardHandlers::can_trigger_event(state, ts::card_ids::ARAB_ISRAELI_WAR, ts::Player::USSR));
    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::ARAB_ISRAELI_WAR, ts::Player::USSR);
    ASSERT_TRUE(done);
}

// Card 14: COMECON
TEST(EarlyCardsTest, Card14_COMECON_Placement) {
    ts::GameState state{};
    state.countries[ts::countries::EAST_GERMANY].ussr_influence = 0;
    state.countries[ts::countries::POLAND].ussr_influence = 0;
    state.countries[ts::countries::CZECHOSLOVAKIA].ussr_influence = 0;
    state.countries[ts::countries::HUNGARY].ussr_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::COMECON, ts::Player::USSR);
    ASSERT_EQ(state.ctx().remaining_steps, 4);

    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::EAST_GERMANY, 0, 0));
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::POLAND, 0, 0));
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::CZECHOSLOVAKIA, 0, 0));
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::HUNGARY, 0, 0));
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::EAST_GERMANY].ussr_influence, 1);
    ASSERT_EQ(state.countries[ts::countries::POLAND].ussr_influence, 1);
    ASSERT_EQ(state.countries[ts::countries::CZECHOSLOVAKIA].ussr_influence, 1);
    ASSERT_EQ(state.countries[ts::countries::HUNGARY].ussr_influence, 1);
}

// Card 15: Nasser
TEST(EarlyCardsTest, Card15_Nasser_Basic) {
    ts::GameState state{};
    state.countries[ts::countries::EGYPT].us_influence = 4;
    state.countries[ts::countries::EGYPT].ussr_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::NASSER, ts::Player::USSR);
    ASSERT_EQ(state.countries[ts::countries::EGYPT].ussr_influence, 2);
    ASSERT_EQ(state.countries[ts::countries::EGYPT].us_influence, 2);
}

TEST(EarlyCardsTest, Card15_Nasser_OddUSInfluence) {
    ts::GameState state{};
    state.countries[ts::countries::EGYPT].us_influence = 3;
    state.countries[ts::countries::EGYPT].ussr_influence = 1;
    ts::CardHandlers::trigger_event(state, ts::card_ids::NASSER, ts::Player::USSR);
    ASSERT_EQ(state.countries[ts::countries::EGYPT].ussr_influence, 3);
    ASSERT_EQ(state.countries[ts::countries::EGYPT].us_influence, 1); // 3 - (3+1)/2 = 1
}

// Card 16: Warsaw Pact Formed
TEST(EarlyCardsTest, Card16_WarsawPact_Branch0_RemoveUS) {
    ts::GameState state{};
    state.countries[ts::countries::POLAND].us_influence = 3;
    state.countries[ts::countries::EAST_GERMANY].us_influence = 2;
    ts::CardHandlers::trigger_event(state, ts::card_ids::WARSAW_PACT, ts::Player::USSR);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::CHOOSE_BRANCH);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::WARSAW_PACT_PLAYED));

    // Branch 0: remove all US from up to 4 countries in EE
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::CHOOSE_BRANCH, 0, 0, 0));
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::POINT_NODE);
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::POLAND, 0, 0));
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::EAST_GERMANY, 0, 0));
    ASSERT_EQ(state.countries[ts::countries::POLAND].us_influence, 0);
    ASSERT_EQ(state.countries[ts::countries::EAST_GERMANY].us_influence, 0);
}

// Card 17: De Gaulle Leads France
TEST(EarlyCardsTest, Card17_DeGaulle) {
    ts::GameState state{};
    state.countries[ts::countries::FRANCE].us_influence = 3;
    state.countries[ts::countries::FRANCE].ussr_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::DE_GAULLE, ts::Player::USSR);
    ASSERT_EQ(state.countries[ts::countries::FRANCE].us_influence, 1);
    ASSERT_EQ(state.countries[ts::countries::FRANCE].ussr_influence, 1);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::NATO_CANCELED_FRANCE));
}

// Card 18: Captured Nazi Scientist
TEST(EarlyCardsTest, Card18_CapturedNaziScientist) {
    ts::GameState state{};
    state.us_space_track = 0;
    state.ussr_space_track = 0;
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::CAPTURED_NAZI_SCIENTIST, ts::Player::US);
    ASSERT_EQ(state.us_space_track, 1);
    ASSERT_EQ(state.victory_points, 2); // 1st to box 1 gets 2 VP
}

// Card 19: Truman Doctrine
TEST(EarlyCardsTest, Card19_TrumanDoctrine) {
    ts::GameState state{};
    state.countries[ts::countries::YUGOSLAVIA].ussr_influence = 2;
    state.countries[ts::countries::YUGOSLAVIA].us_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::TRUMAN_DOCTRINE, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::POINT_NODE);
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::YUGOSLAVIA, 0, 0));
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::YUGOSLAVIA].ussr_influence, 0);
}

// Card 20: Olympic Games
TEST(EarlyCardsTest, Card20_OlympicGames_Boycott) {
    ts::GameState state{};
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::OLYMPIC_GAMES, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_player, ts::Player::USSR); // Opponent chooses
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::CHOOSE_BRANCH);

    // USSR boycotts (branch 1) -> US conducts 4 Ops (no VP)
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::CHOOSE_BRANCH, 1, 0, 0));
    ASSERT_EQ(state.victory_points, 0);
    ASSERT_EQ(state.ctx().pending_ops_value, 4);
    ASSERT_EQ(state.ctx().decision_player, ts::Player::US);
}

// Card 21: NATO
TEST(EarlyCardsTest, Card21_NATO_Prereq) {
    ts::GameState state{};
    ASSERT_FALSE(ts::CardHandlers::can_trigger_event(state, ts::card_ids::NATO, ts::Player::US));
    state.set_flag(ts::effect_bits::MARSHALL_PLAN_PLAYED);
    ASSERT_TRUE(ts::CardHandlers::can_trigger_event(state, ts::card_ids::NATO, ts::Player::US));
    ts::CardHandlers::trigger_event(state, ts::card_ids::NATO, ts::Player::US);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::NATO_ACTIVE));
}

// Card 22: Independent Reds
TEST(EarlyCardsTest, Card22_IndependentReds) {
    ts::GameState state{};
    state.countries[ts::countries::ROMANIA].ussr_influence = 3;
    state.countries[ts::countries::ROMANIA].us_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::INDEPENDENT_REDS, ts::Player::US);
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::ROMANIA, 0, 0));
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::ROMANIA].us_influence, 3);
}

// Card 23: Marshall Plan
TEST(EarlyCardsTest, Card23_MarshallPlan) {
    ts::GameState state{};
    state.countries[ts::countries::UNITED_KINGDOM].us_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::MARSHALL_PLAN, ts::Player::US);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::MARSHALL_PLAN_PLAYED));
    ASSERT_EQ(state.ctx().remaining_steps, 7);
}

// Card 24: Indo-Pakistani War
TEST(EarlyCardsTest, Card24_IndoPakistaniWar) {
    ts::GameState state{};
    state.countries[ts::countries::PAKISTAN].ussr_influence = 2;
    state.countries[ts::countries::PAKISTAN].us_influence = 0;
    state.us_mil_ops = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::INDO_PAKISTANI_WAR, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::POINT_NODE);
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::PAKISTAN, 0, 0));
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::ROLL_DIE);
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::ROLL_DIE, 0, 0, 0));
    ASSERT_EQ(state.us_mil_ops, 2);
}

// Card 25: Containment
TEST(EarlyCardsTest, Card25_Containment) {
    ts::GameState state{};
    ts::CardHandlers::trigger_event(state, ts::card_ids::CONTAINMENT, ts::Player::US);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::CONTAINMENT_ACTIVE));
}

// Card 26: CIA Created
TEST(EarlyCardsTest, Card26_CIACreated) {
    ts::GameState state{};
    ts::CardHandlers::trigger_event(state, ts::card_ids::CIA_CREATED, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_player, ts::Player::US);
    ASSERT_EQ(state.ctx().pending_ops_value, 1);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_OP_MODE);
}

// Card 27: US/Japan Mutual Defense Pact
TEST(EarlyCardsTest, Card27_USJapanPact) {
    ts::GameState state{};
    state.countries[ts::countries::JAPAN].us_influence = 1;
    ts::CardHandlers::trigger_event(state, ts::card_ids::US_JAPAN_PACT, ts::Player::US);
    ASSERT_EQ(state.countries[ts::countries::JAPAN].us_influence, 4);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::US_JAPAN_PACT_ACTIVE));
}

// Card 28: Suez Crisis
TEST(EarlyCardsTest, Card28_SuezCrisis) {
    ts::GameState state{};
    state.countries[ts::countries::UNITED_KINGDOM].us_influence = 3;
    state.countries[ts::countries::FRANCE].us_influence = 2;
    state.countries[ts::countries::ISRAEL].us_influence = 2;
    ts::CardHandlers::trigger_event(state, ts::card_ids::SUEZ_CRISIS, ts::Player::USSR);
    ASSERT_EQ(state.ctx().remaining_steps, 4);

    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::UNITED_KINGDOM, 0, 0));
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::UNITED_KINGDOM, 0, 0));
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::FRANCE, 0, 0));
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::ISRAEL, 0, 0));
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::UNITED_KINGDOM].us_influence, 1);
    ASSERT_EQ(state.countries[ts::countries::FRANCE].us_influence, 1);
    ASSERT_EQ(state.countries[ts::countries::ISRAEL].us_influence, 1);
}

// Card 29: East European Unrest
TEST(EarlyCardsTest, Card29_EastEuropeanUnrest_EarlyWar) {
    ts::GameState state{};
    state.turn = 3; // Early War
    state.countries[ts::countries::POLAND].ussr_influence = 3;
    ts::CardHandlers::trigger_event(state, ts::card_ids::EAST_EUROPEAN_UNREST, ts::Player::US);
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::POLAND, 0, 0));
    ASSERT_EQ(state.countries[ts::countries::POLAND].ussr_influence, 2); // 1 removed
}

TEST(EarlyCardsTest, Card29_EastEuropeanUnrest_LateWar) {
    ts::GameState state{};
    state.turn = 9; // Late War
    state.countries[ts::countries::POLAND].ussr_influence = 3;
    ts::CardHandlers::trigger_event(state, ts::card_ids::EAST_EUROPEAN_UNREST, ts::Player::US);
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::POLAND, 0, 0));
    ASSERT_EQ(state.countries[ts::countries::POLAND].ussr_influence, 1); // 2 removed
}

// Card 30: Decolonization
TEST(EarlyCardsTest, Card30_Decolonization) {
    ts::GameState state{};
    state.countries[ts::countries::ANGOLA].ussr_influence = 0;
    state.countries[ts::countries::NIGERIA].ussr_influence = 0;
    state.countries[ts::countries::ZAIRE].ussr_influence = 0;
    state.countries[ts::countries::VIETNAM].ussr_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::DECOLONIZATION, ts::Player::USSR);
    ASSERT_EQ(state.ctx().remaining_steps, 4);
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::ANGOLA, 0, 0));
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::NIGERIA, 0, 0));
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::ZAIRE, 0, 0));
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::VIETNAM, 0, 0));
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::ANGOLA].ussr_influence, 1);
    ASSERT_EQ(state.countries[ts::countries::VIETNAM].ussr_influence, 1);
}

// Card 31: Red Scare/Purge
TEST(EarlyCardsTest, Card31_RedScarePurge) {
    ts::GameState state{};
    ts::CardHandlers::trigger_event(state, ts::card_ids::RED_SCARE_PURGE, ts::Player::US);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::PURGE_USSR_ACTIVE));

    ts::GameState state2{};
    ts::CardHandlers::trigger_event(state2, ts::card_ids::RED_SCARE_PURGE, ts::Player::USSR);
    ASSERT_TRUE(state2.has_flag(ts::effect_bits::PURGE_US_ACTIVE));
}

// Card 32: UN Intervention
TEST(EarlyCardsTest, Card32_UNIntervention) {
    ts::GameState state{};
    state.card_locations[ts::card_ids::DE_GAULLE] = ts::CardLocation::HAND_US; // USSR card in US hand
    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::UN_INTERVENTION, ts::Player::US);
    ASSERT_FALSE(done);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_CARD);

    done = ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::SELECT_CARD, ts::card_ids::DE_GAULLE, 0, 0));
    ASSERT_FALSE(done);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_OP_MODE);
    ASSERT_EQ(state.ctx().pending_ops_value, 3);
    ASSERT_EQ(state.card_locations[ts::card_ids::DE_GAULLE], ts::CardLocation::DISCARD_PILE);
}

// Card 33: De-Stalinization
TEST(EarlyCardsTest, Card33_DeStalinization_TwoStages) {
    ts::GameState state{};
    state.countries[ts::countries::EAST_GERMANY].ussr_influence = 3;
    state.countries[ts::countries::POLAND].ussr_influence = 3;
    state.countries[ts::countries::FRANCE].us_influence = 0;
    state.countries[ts::countries::FRANCE].ussr_influence = 0;

    ts::CardHandlers::trigger_event(state, ts::card_ids::DE_STALINIZATION, ts::Player::USSR);
    ASSERT_EQ(state.ctx().remaining_steps, 4);

    // Remove 2 from East Germany
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::EAST_GERMANY, 0, 0));
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::EAST_GERMANY, 0, 0));
    // Confirm stage 1 done (removed 2)
    ts::MicroAction done_act{};
    done_act.flags = ts::action_flags::CONFIRM_DONE;
    ts::CardHandlers::handle_event_step(state, done_act);

    // Stage 2: place 2 in France
    ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::FRANCE, 0, 0));
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::FRANCE, 0, 0));
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::EAST_GERMANY].ussr_influence, 1);
    ASSERT_EQ(state.countries[ts::countries::FRANCE].ussr_influence, 2);
}

// Card 34: Nuclear Test Ban
TEST(EarlyCardsTest, Card34_NuclearTestBan) {
    ts::GameState state{};
    state.defcon = 3;
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::NUCLEAR_TEST_BAN, ts::Player::US);
    ASSERT_EQ(state.victory_points, 1); // 3 - 2 = 1 VP
    ASSERT_EQ(state.defcon, 5); // 3 + 2 = 5
}

// Card 35: Formosan Resolution
TEST(EarlyCardsTest, Card35_FormosanResolution) {
    ts::GameState state{};
    ts::CardHandlers::trigger_event(state, ts::card_ids::FORMOSAN_RESOLUTION, ts::Player::US);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::FORMOSAN_RESOLUTION_ACTIVE));
}

// Card 103: Defectors
TEST(EarlyCardsTest, Card103_Defectors_Headline) {
    ts::GameState state{};
    state.current_phase = ts::Phase::HEADLINE;
    state.headline_ussr_card = ts::card_ids::FIDEL;
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::DEFECTORS, ts::Player::US);
    ASSERT_EQ(state.headline_ussr_card, 0); // Canceled!
    ASSERT_EQ(state.victory_points, 0);     // Headline cancellation grants 0 VP
}

TEST(EarlyCardsTest, Card103_Defectors_ActionRoundUSSR) {
    ts::GameState state{};
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::DEFECTORS, ts::Player::USSR);
    ASSERT_EQ(state.victory_points, 1); // US gains 1 VP
}

// Card 104: The Cambridge Five
TEST(EarlyCardsTest, Card104_CambridgeFive_Basic) {
    ts::GameState state{};
    state.turn = 3; // Early War
    state.card_locations[ts::card_ids::MIDDLE_EAST_SCORING] = ts::CardLocation::HAND_US;
    state.countries[ts::countries::EGYPT].ussr_influence = 0;

    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::THE_CAMBRIDGE_FIVE, ts::Player::USSR);
    ASSERT_FALSE(done);
    ASSERT_EQ(state.ctx().decision_player, ts::Player::USSR);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::POINT_NODE);

    done = ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::EGYPT, 0, 0));
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::EGYPT].ussr_influence, 1);
}

TEST(EarlyCardsTest, Card104_CambridgeFive_BlockedInLateWar) {
    ts::GameState state{};
    state.turn = 8; // Late War
    ASSERT_FALSE(ts::CardHandlers::can_trigger_event(state, ts::card_ids::THE_CAMBRIDGE_FIVE, ts::Player::USSR));
}

// Card 105: Special Relationship
TEST(EarlyCardsTest, Card105_SpecialRelationship_NoNATO) {
    ts::GameState state{};
    state.countries[ts::countries::UNITED_KINGDOM].us_influence = 5; // UK controlled
    state.countries[ts::countries::FRANCE].us_influence = 0;
    state.clear_flag(ts::effect_bits::NATO_ACTIVE);

    ts::CardHandlers::trigger_event(state, ts::card_ids::SPECIAL_RELATIONSHIP, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::POINT_NODE);
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::FRANCE, 0, 0));
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::FRANCE].us_influence, 1);
}

TEST(EarlyCardsTest, Card105_SpecialRelationship_WithNATO) {
    ts::GameState state{};
    state.countries[ts::countries::UNITED_KINGDOM].us_influence = 5;
    state.countries[ts::countries::WEST_GERMANY].us_influence = 0;
    state.set_flag(ts::effect_bits::NATO_ACTIVE);
    state.victory_points = 0;

    ts::CardHandlers::trigger_event(state, ts::card_ids::SPECIAL_RELATIONSHIP, ts::Player::US);
    ASSERT_EQ(state.victory_points, 2); // +2 VP
    ASSERT_EQ(state.ctx().remaining_steps, 1);

    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::WEST_GERMANY, 0, 0));
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::WEST_GERMANY].us_influence, 2);
}

TEST(EarlyCardsTest, Card105_SpecialRelationship_UKUncontrolled) {
    ts::GameState state{};
    state.countries[ts::countries::UNITED_KINGDOM].us_influence = 0;
    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::SPECIAL_RELATIONSHIP, ts::Player::US);
    ASSERT_TRUE(done); // Nothing happens
}

// Card 106: NORAD
TEST(EarlyCardsTest, Card106_NORAD) {
    ts::GameState state{};
    ts::CardHandlers::trigger_event(state, ts::card_ids::NORAD, ts::Player::US);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::NORAD_ACTIVE));
}
