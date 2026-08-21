#include "test_framework.hpp"
#include "ts/engine.hpp"
#include "ts/card_data.hpp"
#include "ts/card_handlers.hpp"
#include "ts/constants.hpp"

// =============================================================================
// CATEGORY A: 3 Individual Cards (FYP, Grain Sales, Star Wars)
// =============================================================================

TEST(ReentrancyTest, 01_Individual_FiveYearPlan) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);

    // (a) USSR hand empty -> graceful no-op
    for (uint8_t i = 1; i <= 110; ++i) state.card_locations[i] = ts::CardLocation::DRAW_DECK;
    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::FIVE_YEAR_PLAN, ts::Player::US);
    ASSERT_TRUE(done);

    // (b) USSR hand has US event (Duck and Cover #4) -> discarded to discard pile, triggers event, DEFCON drops
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::HAND_USSR;
    state.defcon = 5;
    done = ts::CardHandlers::trigger_event(state, ts::card_ids::FIVE_YEAR_PLAN, ts::Player::US);
    ASSERT_TRUE(done);
    ASSERT_EQ(state.defcon, 4);
    ASSERT_EQ(state.card_locations[ts::card_ids::DUCK_AND_COVER], ts::CardLocation::DISCARD_PILE);

    // (c) USSR hand has USSR event (Arab-Israeli War #13) -> discarded without event
    state.card_locations[ts::card_ids::ARAB_ISRAELI_WAR] = ts::CardLocation::HAND_USSR;
    done = ts::CardHandlers::trigger_event(state, ts::card_ids::FIVE_YEAR_PLAN, ts::Player::US);
    ASSERT_TRUE(done);
    ASSERT_EQ(state.card_locations[ts::card_ids::ARAB_ISRAELI_WAR], ts::CardLocation::DISCARD_PILE);

    // (d) USSR hand has Neutral event (Olympic Games #20) -> discarded without event
    state.card_locations[ts::card_ids::OLYMPIC_GAMES] = ts::CardLocation::HAND_USSR;
    done = ts::CardHandlers::trigger_event(state, ts::card_ids::FIVE_YEAR_PLAN, ts::Player::US);
    ASSERT_TRUE(done);
    ASSERT_EQ(state.card_locations[ts::card_ids::OLYMPIC_GAMES], ts::CardLocation::DISCARD_PILE);
}

TEST(ReentrancyTest, 02_Individual_GrainSales) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);

    // (a) USSR hand empty -> US gets 2 Ops directly
    for (uint8_t i = 1; i <= 110; ++i) state.card_locations[i] = ts::CardLocation::DRAW_DECK;
    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::GRAIN_SALES, ts::Player::US);
    ASSERT_FALSE(done); // Prompts for Op mode
    ASSERT_EQ(state.ctx().decision_player, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_OP_MODE);
    ASSERT_EQ(state.ctx().pending_ops_value, 2);

    // (b) USSR holds Duck and Cover (#4) -> US draws it and chooses Branch 0 (play drawn card)
    state.ctx() = ts::DecisionContext{};
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::HAND_USSR;
    done = ts::CardHandlers::trigger_event(state, ts::card_ids::GRAIN_SALES, ts::Player::US);
    ASSERT_FALSE(done);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::CHOOSE_BRANCH);

    // US selects Branch 0: Play drawn card
    ts::MicroAction act_play{ts::DecisionType::CHOOSE_BRANCH, 0, 0, 0};
    ts::CardHandlers::handle_event_step(state, act_play);

    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_PLAY_MODE);
    ASSERT_EQ(state.ctx().pending_op_card, ts::card_ids::DUCK_AND_COVER);

    // (c) USSR holds Soviet card (Arab-Israeli War #13) -> US draws and chooses Branch 1 (return card for 2 Ops)
    state.ctx() = ts::DecisionContext{};
    for (uint8_t i = 1; i <= 110; ++i) state.card_locations[i] = ts::CardLocation::DRAW_DECK;
    state.card_locations[ts::card_ids::ARAB_ISRAELI_WAR] = ts::CardLocation::HAND_USSR;
    ts::CardHandlers::trigger_event(state, ts::card_ids::GRAIN_SALES, ts::Player::US);

    ts::MicroAction act_ops{ts::DecisionType::CHOOSE_BRANCH, 1, 0, 0};
    ts::CardHandlers::handle_event_step(state, act_ops);

    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_OP_MODE);
    ASSERT_EQ(state.ctx().pending_ops_value, 2);
}

TEST(ReentrancyTest, 03_Individual_StarWars) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);

    // (a) Discard pile empty -> returns true immediately
    for (uint8_t i = 1; i <= 110; ++i) state.card_locations[i] = ts::CardLocation::DRAW_DECK;
    state.us_space_track = 3;
    state.ussr_space_track = 1;
    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::STAR_WARS, ts::Player::US);
    ASSERT_TRUE(done);

    // (b) US space <= USSR space -> fails prerequisite
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::DISCARD_PILE;
    state.us_space_track = 1;
    state.ussr_space_track = 2;
    done = ts::CardHandlers::trigger_event(state, ts::card_ids::STAR_WARS, ts::Player::US);
    ASSERT_TRUE(done);

    // (c) US space > USSR space -> US selects Duck and Cover (#4) from discard
    state.us_space_track = 3;
    state.ussr_space_track = 1;
    state.defcon = 5;
    done = ts::CardHandlers::trigger_event(state, ts::card_ids::STAR_WARS, ts::Player::US);
    ASSERT_FALSE(done);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_CARD);
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::STAR_WARS);

    // US selects Duck and Cover -> executes and degrades DEFCON
    ts::MicroAction act{ts::DecisionType::SELECT_CARD, ts::card_ids::DUCK_AND_COVER, 0, 0};
    ts::CardHandlers::handle_event_step(state, act);

    ASSERT_EQ(state.defcon, 4);
}

// =============================================================================
// CATEGORY B: 6 Pairs of 2 Cards
// =============================================================================

TEST(ReentrancyTest, 04_Pair_FYP_then_GrainSales) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);

    for (uint8_t i = 1; i <= 110; ++i) state.card_locations[i] = ts::CardLocation::DRAW_DECK;
    state.card_locations[ts::card_ids::ARAB_ISRAELI_WAR] = ts::CardLocation::HAND_USSR;
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::HAND_USSR;

    // 1. FYP executes and discards Arab-Israeli War
    ts::CardHandlers::trigger_event(state, ts::card_ids::FIVE_YEAR_PLAN, ts::Player::US);
    ASSERT_EQ(state.card_locations[ts::card_ids::ARAB_ISRAELI_WAR], ts::CardLocation::DISCARD_PILE);

    // 2. Grain Sales executes and draws Duck and Cover
    state.ctx() = ts::DecisionContext{};
    ts::CardHandlers::trigger_event(state, ts::card_ids::GRAIN_SALES, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::CHOOSE_BRANCH);
}

TEST(ReentrancyTest, 05_Pair_GrainSales_then_FYP) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);

    for (uint8_t i = 1; i <= 110; ++i) state.card_locations[i] = ts::CardLocation::DRAW_DECK;
    state.card_locations[ts::card_ids::FIVE_YEAR_PLAN] = ts::CardLocation::HAND_USSR;
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::HAND_USSR;
    state.defcon = 5;

    // 1. Grain Sales draws Five Year Plan
    ts::CardHandlers::trigger_event(state, ts::card_ids::GRAIN_SALES, ts::Player::US);
    
    // US plays drawn Five Year Plan
    ts::MicroAction act{ts::DecisionType::CHOOSE_BRANCH, 0, 0, 0};
    ts::CardHandlers::handle_event_step(state, act);

    // Play as Event
    ts::CardHandlers::trigger_event(state, ts::card_ids::FIVE_YEAR_PLAN, ts::Player::US);

    // FYP discards Duck and Cover and triggers it!
    ASSERT_EQ(state.defcon, 4);
    ASSERT_EQ(state.card_locations[ts::card_ids::DUCK_AND_COVER], ts::CardLocation::DISCARD_PILE);
}

TEST(ReentrancyTest, 06_Pair_FYP_then_StarWars) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);

    for (uint8_t i = 1; i <= 110; ++i) state.card_locations[i] = ts::CardLocation::DRAW_DECK;
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::HAND_USSR;
    state.us_space_track = 4;
    state.ussr_space_track = 2;
    state.defcon = 5;

    // 1. FYP discards Duck and Cover into discard pile (and triggers it -> DEFCON drops to 4)
    ts::CardHandlers::trigger_event(state, ts::card_ids::FIVE_YEAR_PLAN, ts::Player::US);
    ASSERT_EQ(state.card_locations[ts::card_ids::DUCK_AND_COVER], ts::CardLocation::DISCARD_PILE);
    ASSERT_EQ(state.defcon, 4);

    // 2. Star Wars retrieves Duck and Cover from discard pile -> triggers again -> DEFCON drops to 3
    state.ctx() = ts::DecisionContext{};
    ts::CardHandlers::trigger_event(state, ts::card_ids::STAR_WARS, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_CARD);

    ts::MicroAction act{ts::DecisionType::SELECT_CARD, ts::card_ids::DUCK_AND_COVER, 0, 0};
    ts::CardHandlers::handle_event_step(state, act);

    ASSERT_EQ(state.defcon, 3);
}

TEST(ReentrancyTest, 07_Pair_StarWars_then_FYP) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);

    for (uint8_t i = 1; i <= 110; ++i) state.card_locations[i] = ts::CardLocation::DRAW_DECK;
    state.card_locations[ts::card_ids::FIVE_YEAR_PLAN] = ts::CardLocation::DISCARD_PILE;
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::HAND_USSR;
    state.us_space_track = 3;
    state.ussr_space_track = 1;
    state.defcon = 5;

    // Star Wars retrieves FYP from discard
    ts::CardHandlers::trigger_event(state, ts::card_ids::STAR_WARS, ts::Player::US);
    ts::MicroAction act{ts::DecisionType::SELECT_CARD, ts::card_ids::FIVE_YEAR_PLAN, 0, 0};
    ts::CardHandlers::handle_event_step(state, act);

    // FYP triggered -> discards Duck and Cover -> DEFCON drops to 4
    ASSERT_EQ(state.defcon, 4);
    ASSERT_EQ(state.card_locations[ts::card_ids::DUCK_AND_COVER], ts::CardLocation::DISCARD_PILE);
}

TEST(ReentrancyTest, 08_Pair_GrainSales_then_StarWars) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);

    for (uint8_t i = 1; i <= 110; ++i) state.card_locations[i] = ts::CardLocation::DRAW_DECK;
    state.card_locations[ts::card_ids::STAR_WARS] = ts::CardLocation::HAND_USSR;
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::DISCARD_PILE;
    state.us_space_track = 3;
    state.ussr_space_track = 1;
    state.defcon = 5;

    // Grain Sales draws Star Wars
    ts::CardHandlers::trigger_event(state, ts::card_ids::GRAIN_SALES, ts::Player::US);
    
    // US selects Branch 0: Play Star Wars as Event
    ts::MicroAction act{ts::DecisionType::CHOOSE_BRANCH, 0, 0, 0};
    ts::CardHandlers::handle_event_step(state, act);

    ts::CardHandlers::trigger_event(state, ts::card_ids::STAR_WARS, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_CARD);
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::STAR_WARS);

    // Star Wars retrieves Duck and Cover
    ts::MicroAction act_sw{ts::DecisionType::SELECT_CARD, ts::card_ids::DUCK_AND_COVER, 0, 0};
    ts::CardHandlers::handle_event_step(state, act_sw);
    ASSERT_EQ(state.defcon, 4);
}

TEST(ReentrancyTest, 09_Pair_StarWars_then_GrainSales) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);

    for (uint8_t i = 1; i <= 110; ++i) state.card_locations[i] = ts::CardLocation::DRAW_DECK;
    state.card_locations[ts::card_ids::GRAIN_SALES] = ts::CardLocation::DISCARD_PILE;
    state.card_locations[ts::card_ids::ARAB_ISRAELI_WAR] = ts::CardLocation::HAND_USSR;
    state.us_space_track = 4;
    state.ussr_space_track = 2;

    // Star Wars retrieves Grain Sales
    ts::CardHandlers::trigger_event(state, ts::card_ids::STAR_WARS, ts::Player::US);
    ts::MicroAction act{ts::DecisionType::SELECT_CARD, ts::card_ids::GRAIN_SALES, 0, 0};
    ts::CardHandlers::handle_event_step(state, act);

    // Grain Sales now active
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::GRAIN_SALES);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::CHOOSE_BRANCH);
}

// =============================================================================
// CATEGORY C: 6 Triples (All 6 Possible 3-Card Permutations)
// =============================================================================

TEST(ReentrancyTest, 10_Triple_Order_StarWars_GrainSales_FYP) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);

    for (uint8_t i = 1; i <= 110; ++i) state.card_locations[i] = ts::CardLocation::DRAW_DECK;
    state.card_locations[ts::card_ids::GRAIN_SALES] = ts::CardLocation::DISCARD_PILE;
    state.card_locations[ts::card_ids::FIVE_YEAR_PLAN] = ts::CardLocation::HAND_USSR;
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::HAND_USSR;
    state.us_space_track = 4;
    state.ussr_space_track = 1;
    state.defcon = 5;

    // 1. Star Wars retrieves Grain Sales
    ts::CardHandlers::trigger_event(state, ts::card_ids::STAR_WARS, ts::Player::US);
    ts::MicroAction act1{ts::DecisionType::SELECT_CARD, ts::card_ids::GRAIN_SALES, 0, 0};
    ts::CardHandlers::handle_event_step(state, act1);

    // 2. Grain Sales draws FYP -> US plays FYP
    ts::MicroAction act2{ts::DecisionType::CHOOSE_BRANCH, 0, 0, 0};
    ts::CardHandlers::handle_event_step(state, act2);

    // 3. FYP triggers and discards Duck and Cover
    ts::CardHandlers::trigger_event(state, ts::card_ids::FIVE_YEAR_PLAN, ts::Player::US);
    ASSERT_EQ(state.defcon, 4);
}

TEST(ReentrancyTest, 11_Triple_Order_StarWars_FYP_GrainSales) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);

    for (uint8_t i = 1; i <= 110; ++i) state.card_locations[i] = ts::CardLocation::DRAW_DECK;
    state.card_locations[ts::card_ids::FIVE_YEAR_PLAN] = ts::CardLocation::DISCARD_PILE;
    state.card_locations[ts::card_ids::GRAIN_SALES] = ts::CardLocation::HAND_USSR;
    state.us_space_track = 3;
    state.ussr_space_track = 0;

    // Star Wars retrieves FYP -> FYP discards Grain Sales
    ts::CardHandlers::trigger_event(state, ts::card_ids::STAR_WARS, ts::Player::US);
    ts::MicroAction act{ts::DecisionType::SELECT_CARD, ts::card_ids::FIVE_YEAR_PLAN, 0, 0};
    ts::CardHandlers::handle_event_step(state, act);

    ASSERT_EQ(state.card_locations[ts::card_ids::GRAIN_SALES], ts::CardLocation::DISCARD_PILE);
}

TEST(ReentrancyTest, 12_Triple_Order_GrainSales_StarWars_FYP) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);

    for (uint8_t i = 1; i <= 110; ++i) state.card_locations[i] = ts::CardLocation::DRAW_DECK;
    state.card_locations[ts::card_ids::STAR_WARS] = ts::CardLocation::HAND_USSR;
    state.card_locations[ts::card_ids::FIVE_YEAR_PLAN] = ts::CardLocation::DISCARD_PILE;
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::HAND_USSR;
    state.us_space_track = 3;
    state.ussr_space_track = 1;
    state.defcon = 5;

    // 1. Grain Sales draws Star Wars
    ts::CardHandlers::trigger_event(state, ts::card_ids::GRAIN_SALES, ts::Player::US);
    ts::MicroAction act1{ts::DecisionType::CHOOSE_BRANCH, 0, 0, 0};
    ts::CardHandlers::handle_event_step(state, act1);

    // 2. Play Star Wars -> retrieves FYP
    ts::CardHandlers::trigger_event(state, ts::card_ids::STAR_WARS, ts::Player::US);
    ts::MicroAction act2{ts::DecisionType::SELECT_CARD, ts::card_ids::FIVE_YEAR_PLAN, 0, 0};
    ts::CardHandlers::handle_event_step(state, act2);

    // 3. FYP discards Duck and Cover -> DEFCON 4
    ASSERT_EQ(state.defcon, 4);
}

TEST(ReentrancyTest, 13_Triple_Order_GrainSales_FYP_StarWars) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);

    for (uint8_t i = 1; i <= 110; ++i) state.card_locations[i] = ts::CardLocation::DRAW_DECK;
    state.card_locations[ts::card_ids::FIVE_YEAR_PLAN] = ts::CardLocation::HAND_USSR;
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::HAND_USSR;
    state.us_space_track = 3;
    state.ussr_space_track = 1;
    state.defcon = 5;

    // Grain Sales draws FYP -> FYP discards Duck and Cover -> DEFCON 4
    ts::CardHandlers::trigger_event(state, ts::card_ids::GRAIN_SALES, ts::Player::US);
    ts::MicroAction act1{ts::DecisionType::CHOOSE_BRANCH, 0, 0, 0};
    ts::CardHandlers::handle_event_step(state, act1);
    ts::CardHandlers::trigger_event(state, ts::card_ids::FIVE_YEAR_PLAN, ts::Player::US);

    ASSERT_EQ(state.card_locations[ts::card_ids::DUCK_AND_COVER], ts::CardLocation::DISCARD_PILE);
    ASSERT_EQ(state.defcon, 4);

    // Star Wars retrieves Duck and Cover -> DEFCON 3
    state.ctx() = ts::DecisionContext{};
    ts::CardHandlers::trigger_event(state, ts::card_ids::STAR_WARS, ts::Player::US);
    ts::MicroAction act2{ts::DecisionType::SELECT_CARD, ts::card_ids::DUCK_AND_COVER, 0, 0};
    ts::CardHandlers::handle_event_step(state, act2);

    ASSERT_EQ(state.defcon, 3);
}

TEST(ReentrancyTest, 14_Triple_Order_FYP_GrainSales_StarWars) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);

    for (uint8_t i = 1; i <= 110; ++i) state.card_locations[i] = ts::CardLocation::DRAW_DECK;
    state.card_locations[ts::card_ids::ARAB_ISRAELI_WAR] = ts::CardLocation::HAND_USSR;
    state.card_locations[ts::card_ids::STAR_WARS] = ts::CardLocation::HAND_USSR;
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::DISCARD_PILE;
    state.us_space_track = 4;
    state.ussr_space_track = 1;
    state.defcon = 5;

    // 1. FYP discards Arab-Israeli War
    ts::CardHandlers::trigger_event(state, ts::card_ids::FIVE_YEAR_PLAN, ts::Player::US);

    // 2. Grain Sales draws Star Wars
    state.ctx() = ts::DecisionContext{};
    ts::CardHandlers::trigger_event(state, ts::card_ids::GRAIN_SALES, ts::Player::US);
    ts::MicroAction act1{ts::DecisionType::CHOOSE_BRANCH, 0, 0, 0};
    ts::CardHandlers::handle_event_step(state, act1);

    // 3. Star Wars retrieves Duck and Cover -> DEFCON 4
    ts::CardHandlers::trigger_event(state, ts::card_ids::STAR_WARS, ts::Player::US);
    ts::MicroAction act2{ts::DecisionType::SELECT_CARD, ts::card_ids::DUCK_AND_COVER, 0, 0};
    ts::CardHandlers::handle_event_step(state, act2);

    ASSERT_EQ(state.defcon, 4);
}

TEST(ReentrancyTest, 15_Triple_Order_FYP_StarWars_GrainSales) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);

    for (uint8_t i = 1; i <= 110; ++i) state.card_locations[i] = ts::CardLocation::DRAW_DECK;
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::HAND_USSR;
    state.us_space_track = 3;
    state.ussr_space_track = 1;
    state.defcon = 5;

    // 1. FYP discards Duck and Cover (only card in USSR hand) -> DEFCON drops to 4
    ts::CardHandlers::trigger_event(state, ts::card_ids::FIVE_YEAR_PLAN, ts::Player::US);
    ASSERT_EQ(state.defcon, 4);
    ASSERT_EQ(state.card_locations[ts::card_ids::DUCK_AND_COVER], ts::CardLocation::DISCARD_PILE);

    // 2. Star Wars retrieves Duck and Cover from discard -> DEFCON drops to 3
    state.ctx() = ts::DecisionContext{};
    ts::CardHandlers::trigger_event(state, ts::card_ids::STAR_WARS, ts::Player::US);
    ts::MicroAction act1{ts::DecisionType::SELECT_CARD, ts::card_ids::DUCK_AND_COVER, 0, 0};
    ts::CardHandlers::handle_event_step(state, act1);
    ASSERT_EQ(state.defcon, 3);

    // 3. Give USSR Arab-Israeli War, then Grain Sales draws it
    state.card_locations[ts::card_ids::ARAB_ISRAELI_WAR] = ts::CardLocation::HAND_USSR;
    state.ctx() = ts::DecisionContext{};
    ts::CardHandlers::trigger_event(state, ts::card_ids::GRAIN_SALES, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::CHOOSE_BRANCH);
}
