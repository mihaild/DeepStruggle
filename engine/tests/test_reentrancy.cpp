#include "test_framework.hpp"
#include "ts/engine.hpp"
#include "ts/card_data.hpp"
#include "ts/card_handlers.hpp"
#include "ts/constants.hpp"

TEST(ReentrancyTest, ContextStackPushAndPop) {
    ts::GameState state{};
    ASSERT_EQ(state.ctx_stack_depth, 0);

    state.ctx().resolving_card = ts::card_ids::STAR_WARS;
    state.ctx().decision_player = ts::Player::US;

    state.push_context();
    ASSERT_EQ(state.ctx_stack_depth, 1);
    state.ctx().resolving_card = ts::card_ids::MARSHALL_PLAN;
    state.ctx().decision_player = ts::Player::US;

    ASSERT_EQ(state.ctx_stack[0].resolving_card, ts::card_ids::STAR_WARS);
    ASSERT_EQ(state.ctx_stack[1].resolving_card, ts::card_ids::MARSHALL_PLAN);

    state.pop_context();
    ASSERT_EQ(state.ctx_stack_depth, 0);
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::STAR_WARS);
}

TEST(ReentrancyTest, FiveYearPlanTriggersSubEvent) {
    ts::GameState state{};
    ts::Engine::init_game(state, 42);

    // Give USSR Duck and Cover (#4)
    for (uint8_t i = 1; i <= 110; ++i) state.card_locations[i] = ts::CardLocation::DRAW_DECK;
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::HAND_USSR;
    state.defcon = 5;

    // US plays Five Year Plan (#5)
    ts::CardHandlers::trigger_event(state, ts::card_ids::FIVE_YEAR_PLAN, ts::Player::US);

    // Five Year Plan discards Duck and Cover and immediately triggers its event!
    ASSERT_EQ(state.defcon, 4); // Degraded by 1 from Duck and Cover
}
