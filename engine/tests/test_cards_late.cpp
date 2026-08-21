#include "test_framework.hpp"
#include "ts/engine.hpp"
#include "ts/card_data.hpp"
#include "ts/card_handlers.hpp"
#include "ts/constants.hpp"
#include "ts/scoring.hpp"
#include "ts/map_data.hpp"
#include "ts/ops.hpp"

// Card 82: Iranian Hostage Crisis
TEST(LateCardsTest, Card82_IranianHostageCrisis) {
    ts::GameState state{};
    state.countries[ts::countries::IRAN].us_influence = 2;
    state.countries[ts::countries::IRAN].ussr_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::IRANIAN_HOSTAGE_CRISIS, ts::Player::USSR);
    ASSERT_EQ(state.countries[ts::countries::IRAN].us_influence, 0);
    ASSERT_EQ(state.countries[ts::countries::IRAN].ussr_influence, 2);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::IRANIAN_HOSTAGE_CRISIS_PLAY));
}

// Card 83: The Iron Lady
TEST(LateCardsTest, Card83_TheIronLady) {
    ts::GameState state{};
    state.countries[ts::countries::ARGENTINA].ussr_influence = 0;
    state.countries[ts::countries::UNITED_KINGDOM].ussr_influence = 2;
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::THE_IRON_LADY, ts::Player::US);
    ASSERT_EQ(state.countries[ts::countries::ARGENTINA].ussr_influence, 1);
    ASSERT_EQ(state.countries[ts::countries::UNITED_KINGDOM].ussr_influence, 0);
    ASSERT_EQ(state.victory_points, 1);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::IRON_LADY_PLAYED));
}

// Card 84: Reagan Bombs Libya
TEST(LateCardsTest, Card84_ReaganBombsLibya) {
    ts::GameState state{};
    state.countries[ts::countries::LIBYA].ussr_influence = 4;
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::REAGAN_BOMBS_LIBYA, ts::Player::US);
    ASSERT_EQ(state.victory_points, 2); // 4 / 2 = 2 VP
}

// Card 85: Star Wars
TEST(LateCardsTest, Card85_StarWars_AheadOnSpace) {
    ts::GameState state{};
    state.us_space_track = 3;
    state.ussr_space_track = 1;
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::DISCARD_PILE;
    ASSERT_TRUE(ts::CardHandlers::can_trigger_event(state, ts::card_ids::STAR_WARS, ts::Player::US));

    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::STAR_WARS, ts::Player::US);
    ASSERT_FALSE(done);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_CARD);

    done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::DUCK_AND_COVER, 0, 0});
    ASSERT_TRUE(done);
}

TEST(LateCardsTest, Card85_StarWars_BehindOnSpace) {
    ts::GameState state{};
    state.us_space_track = 1;
    state.ussr_space_track = 3;
    ASSERT_FALSE(ts::CardHandlers::can_trigger_event(state, ts::card_ids::STAR_WARS, ts::Player::US));
}

// Card 86: North Sea Oil
TEST(LateCardsTest, Card86_NorthSeaOil) {
    ts::GameState state{};
    ts::CardHandlers::trigger_event(state, ts::card_ids::NORTH_SEA_OIL, ts::Player::US);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::NORTH_SEA_OIL_PLAYED));
    ASSERT_TRUE(state.has_flag(ts::effect_bits::NORTH_SEA_OIL_ACTIVE));
}

// Card 87: The Reformer
TEST(LateCardsTest, Card87_TheReformer) {
    ts::GameState state{};
    state.countries[ts::countries::FRANCE].ussr_influence = 0;
    state.countries[ts::countries::ITALY].ussr_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::THE_REFORMER, ts::Player::USSR);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::THE_REFORMER_PLAYED));

    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::FRANCE, 0, 0});
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::FRANCE, 0, 0});
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::ITALY, 0, 0});
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::ITALY, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::FRANCE].ussr_influence, 2);
    ASSERT_EQ(state.countries[ts::countries::ITALY].ussr_influence, 2);
}

// Card 88: Marine Barracks Bombing
TEST(LateCardsTest, Card88_MarineBarracksBombing) {
    ts::GameState state{};
    state.countries[ts::countries::LEBANON].us_influence = 3;
    state.countries[ts::countries::ISRAEL].us_influence = 3;
    ts::CardHandlers::trigger_event(state, ts::card_ids::MARINE_BARRACKS_BOMBING, ts::Player::USSR);
    ASSERT_EQ(state.countries[ts::countries::LEBANON].us_influence, 0); // Removed all Lebanon
    ASSERT_EQ(state.ctx().remaining_steps, 2);

    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::ISRAEL, 0, 0});
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::ISRAEL, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::ISRAEL].us_influence, 1);
}

// Card 89: Soviets Shoot Down KAL-007
TEST(LateCardsTest, Card89_SovietsShootDownKAL_SKoreaControlled) {
    ts::GameState state{};
    state.countries[ts::countries::SOUTH_KOREA].us_influence = 3; // Controlled
    state.defcon = 3;
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::SOVIETS_SHOOT_DOWN_KAL_007, ts::Player::US);
    ASSERT_EQ(state.defcon, 2);
    ASSERT_EQ(state.victory_points, 2);
    ASSERT_EQ(state.ctx().pending_ops_value, 4);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_OP_MODE);
}

TEST(LateCardsTest, Card89_SovietsShootDownKAL_SKoreaNotControlled) {
    ts::GameState state{};
    state.countries[ts::countries::SOUTH_KOREA].us_influence = 0;
    state.defcon = 3;
    state.victory_points = 0;
    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::SOVIETS_SHOOT_DOWN_KAL_007, ts::Player::US);
    ASSERT_TRUE(done);
    ASSERT_EQ(state.defcon, 2);
    ASSERT_EQ(state.victory_points, 2);
}

// Card 90: Glasnost
TEST(LateCardsTest, Card90_Glasnost_ReformerPlayed) {
    ts::GameState state{};
    state.set_flag(ts::effect_bits::THE_REFORMER_PLAYED);
    state.defcon = 2;
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::GLASNOST, ts::Player::USSR);
    ASSERT_EQ(state.defcon, 3);
    ASSERT_EQ(state.victory_points, -2);
    ASSERT_EQ(state.ctx().pending_ops_value, 4);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_OP_MODE);
}

// Card 91: Ortega Elected in Nicaragua
TEST(LateCardsTest, Card91_OrtegaElected) {
    ts::GameState state{};
    state.countries[ts::countries::NICARAGUA].us_influence = 3;
    ts::CardHandlers::trigger_event(state, ts::card_ids::ORTEGA_ELECTED_IN_NICARAGUA, ts::Player::USSR);
    ASSERT_EQ(state.countries[ts::countries::NICARAGUA].us_influence, 0);
}

// Card 92: Terrorism
TEST(LateCardsTest, Card92_Terrorism_Standard) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == ts::CardLocation::HAND_US) {
            state.card_locations[i] = ts::CardLocation::DRAW_DECK;
        }
    }
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::HAND_US;
    ts::CardHandlers::trigger_event(state, ts::card_ids::TERRORISM, ts::Player::USSR);
    ASSERT_EQ(state.card_locations[ts::card_ids::DUCK_AND_COVER], ts::CardLocation::DISCARD_PILE);
}

TEST(LateCardsTest, Card92_Terrorism_WithIranianHostageCrisis) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == ts::CardLocation::HAND_US) {
            state.card_locations[i] = ts::CardLocation::DRAW_DECK;
        }
    }
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::HAND_US;
    state.card_locations[ts::card_ids::FIDEL] = ts::CardLocation::HAND_US;
    state.set_flag(ts::effect_bits::IRANIAN_HOSTAGE_CRISIS_PLAY);

    ts::CardHandlers::trigger_event(state, ts::card_ids::TERRORISM, ts::Player::USSR);
    // US must discard 2 cards!
    ASSERT_EQ(state.card_locations[ts::card_ids::DUCK_AND_COVER], ts::CardLocation::DISCARD_PILE);
    ASSERT_EQ(state.card_locations[ts::card_ids::FIDEL], ts::CardLocation::DISCARD_PILE);
}

// Card 93: Iran-Contra Scandal
TEST(LateCardsTest, Card93_IranContraScandal) {
    ts::GameState state{};
    ts::CardHandlers::trigger_event(state, ts::card_ids::IRAN_CONTRA, ts::Player::USSR);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::IRAN_CONTRA_ACTIVE));
}

// Card 94: Chernobyl
TEST(LateCardsTest, Card94_Chernobyl) {
    ts::GameState state{};
    ts::CardHandlers::trigger_event(state, ts::card_ids::CHERNOBYL, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::CHOOSE_BRANCH);
    // Designate Europe (branch 0)
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::CHOOSE_BRANCH, 0, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::CHERNOBYL_ACTIVE));
    ASSERT_EQ(state.ctx().temp_cards[0], 0);
}

// Card 95: Latin American Debt Crisis
TEST(LateCardsTest, Card95_LatinAmericanDebtCrisis_USDiscards) {
    ts::GameState state{};
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::HAND_US; // 3 ops
    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::LATIN_AMERICAN_DEBT_CRISIS, ts::Player::USSR);
    ASSERT_FALSE(done);
    ASSERT_EQ(state.ctx().decision_player, ts::Player::US);

    done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::DUCK_AND_COVER, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.card_locations[ts::card_ids::DUCK_AND_COVER], ts::CardLocation::DISCARD_PILE);
}

TEST(LateCardsTest, Card95_LatinAmericanDebtCrisis_USSRDoubles) {
    ts::GameState state{};
    state.countries[ts::countries::CHILE].ussr_influence = 2;
    state.countries[ts::countries::ARGENTINA].ussr_influence = 1;
    // US has no 3+ ops card in hand
    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::LATIN_AMERICAN_DEBT_CRISIS, ts::Player::USSR);
    ASSERT_FALSE(done);
    ASSERT_EQ(state.ctx().decision_player, ts::Player::USSR);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::POINT_NODE);

    // Double Chile and Argentina
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::CHILE, 0, 0});
    done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::ARGENTINA, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::CHILE].ussr_influence, 4); // 2 -> 4
    ASSERT_EQ(state.countries[ts::countries::ARGENTINA].ussr_influence, 2); // 1 -> 2
}

// Card 96: Tear Down this Wall
TEST(LateCardsTest, Card96_TearDownThisWall) {
    ts::GameState state{};
    state.countries[ts::countries::EAST_GERMANY].us_influence = 0;
    state.set_flag(ts::effect_bits::WILLY_BRANDT_PLAYED);
    state.set_flag(ts::effect_bits::NATO_CANCELED_WEST_GERMANY);

    ts::CardHandlers::trigger_event(state, ts::card_ids::TEAR_DOWN_THIS_WALL, ts::Player::US);
    ASSERT_EQ(state.countries[ts::countries::EAST_GERMANY].us_influence, 3);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::TEAR_DOWN_THIS_WALL_PLAYED));
    ASSERT_FALSE(state.has_flag(ts::effect_bits::WILLY_BRANDT_PLAYED));
    ASSERT_FALSE(state.has_flag(ts::effect_bits::NATO_CANCELED_WEST_GERMANY));
    ASSERT_EQ(state.ctx().pending_ops_value, 3);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_OP_MODE);
}

// Card 97: An Evil Empire
TEST(LateCardsTest, Card97_AnEvilEmpire) {
    ts::GameState state{};
    state.set_flag(ts::effect_bits::FLOWER_POWER_ACTIVE);
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::AN_EVIL_EMPIRE, ts::Player::US);
    ASSERT_EQ(state.victory_points, 1);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::EVIL_EMPIRE_PLAYED));
    ASSERT_FALSE(state.has_flag(ts::effect_bits::FLOWER_POWER_ACTIVE)); // Canceled!
}

// Card 98: Aldrich Ames Remix
TEST(LateCardsTest, Card98_AldrichAmes) {
    ts::GameState state{};
    state.card_locations[ts::card_ids::FIDEL] = ts::CardLocation::HAND_US;
    ts::CardHandlers::trigger_event(state, ts::card_ids::ALDRICH_AMES, ts::Player::USSR);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::ALDRICH_AMES_ACTIVE));
    ASSERT_EQ(state.ctx().decision_player, ts::Player::USSR);

    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::FIDEL, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.card_locations[ts::card_ids::FIDEL], ts::CardLocation::DISCARD_PILE);
}

// Card 99: Pershing II Deployed
TEST(LateCardsTest, Card99_PershingII) {
    ts::GameState state{};
    state.countries[ts::countries::WEST_GERMANY].us_influence = 3;
    state.countries[ts::countries::FRANCE].us_influence = 2;
    state.countries[ts::countries::ITALY].us_influence = 2;
    state.victory_points = 0;

    ts::CardHandlers::trigger_event(state, ts::card_ids::PERSHING_II_DEPLOYED, ts::Player::USSR);
    ASSERT_EQ(state.victory_points, -1);
    ASSERT_EQ(state.ctx().remaining_steps, 3);

    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::WEST_GERMANY, 0, 0});
    ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::FRANCE, 0, 0});
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::ITALY, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::WEST_GERMANY].us_influence, 2);
    ASSERT_EQ(state.countries[ts::countries::FRANCE].us_influence, 1);
    ASSERT_EQ(state.countries[ts::countries::ITALY].us_influence, 1);
}

// Card 100: Wargames
TEST(LateCardsTest, Card100_Wargames_Defcon2Award6VP) {
    ts::GameState state{};
    state.defcon = 2;
    state.victory_points = 10;
    ts::CardHandlers::trigger_event(state, ts::card_ids::WARGAMES, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::CHOOSE_BRANCH);

    // Branch 0: Give 6 VP to opponent and end game
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::CHOOSE_BRANCH, 0, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.victory_points, 4); // 10 - 6 = 4 VP
    ASSERT_EQ(state.current_phase, ts::Phase::GAME_OVER);
}

// Card 101: Solidarity
TEST(LateCardsTest, Card101_Solidarity_WithJohnPaulII) {
    ts::GameState state{};
    state.set_flag(ts::effect_bits::JOHN_PAUL_II_PLAYED);
    state.countries[ts::countries::POLAND].us_influence = 0;
    ASSERT_TRUE(ts::CardHandlers::can_trigger_event(state, ts::card_ids::SOLIDARITY, ts::Player::US));
    ts::CardHandlers::trigger_event(state, ts::card_ids::SOLIDARITY, ts::Player::US);
    ASSERT_EQ(state.countries[ts::countries::POLAND].us_influence, 3);
}

TEST(LateCardsTest, Card101_Solidarity_WithoutJohnPaulII) {
    ts::GameState state{};
    ASSERT_FALSE(ts::CardHandlers::can_trigger_event(state, ts::card_ids::SOLIDARITY, ts::Player::US));
}

// Card 102: Iran-Iraq War
TEST(LateCardsTest, Card102_IranIraqWar) {
    ts::GameState state{};
    state.countries[ts::countries::IRAN].ussr_influence = 2;
    state.countries[ts::countries::IRAN].us_influence = 0;
    state.us_mil_ops = 0;
    state.victory_points = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::IRAN_IRAQ_WAR, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::POINT_NODE);

    // Target Iran with forced roll 5 -> success
    bool done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::IRAN, 5, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[ts::countries::IRAN].us_influence, 2);
    ASSERT_EQ(state.countries[ts::countries::IRAN].ussr_influence, 0);
    ASSERT_EQ(state.victory_points, 2);
    ASSERT_EQ(state.us_mil_ops, 2);
}

// Card 109: Yuri and Samantha
TEST(LateCardsTest, Card109_YuriAndSamantha) {
    ts::GameState state{};
    ts::CardHandlers::trigger_event(state, ts::card_ids::YURI_AND_SAMANTHA, ts::Player::USSR);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::YURI_AND_SAMANTHA_ACTIVE));
}

// Card 110: AWACS Sale to Saudis
TEST(LateCardsTest, Card110_AWACS_Sale) {
    ts::GameState state{};
    state.countries[ts::countries::SAUDI_ARABIA].us_influence = 0;
    ts::CardHandlers::trigger_event(state, ts::card_ids::AWACS_SALE, ts::Player::US);
    ASSERT_EQ(state.countries[ts::countries::SAUDI_ARABIA].us_influence, 2);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::AWACS_PLAYED));
}
