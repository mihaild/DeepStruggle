#include "test_framework.hpp"
#include "ts/engine.hpp"
#include "ts/card_data.hpp"
#include "ts/card_handlers.hpp"
#include "ts/constants.hpp"

TEST(EarlyCardsTest, FidelEvent) {
    ts::GameState state{};
    state.countries[ts::countries::CUBA].us_influence = 2;
    state.countries[ts::countries::CUBA].ussr_influence = 0;

    ts::CardHandlers::trigger_event(state, ts::card_ids::FIDEL, ts::Player::USSR);

    ASSERT_EQ(state.countries[ts::countries::CUBA].us_influence, 0);
    ASSERT_EQ(state.countries[ts::countries::CUBA].ussr_influence, 3);
}

TEST(EarlyCardsTest, NasserEvent) {
    ts::GameState state{};
    state.countries[ts::countries::EGYPT].us_influence = 4;
    state.countries[ts::countries::EGYPT].ussr_influence = 0;

    ts::CardHandlers::trigger_event(state, ts::card_ids::NASSER, ts::Player::USSR);

    ASSERT_EQ(state.countries[ts::countries::EGYPT].ussr_influence, 2);
    ASSERT_EQ(state.countries[ts::countries::EGYPT].us_influence, 2); // Half removed
}

TEST(EarlyCardsTest, TrumanDoctrineEvent) {
    ts::GameState state{};
    state.countries[ts::countries::YUGOSLAVIA].ussr_influence = 2;
    state.countries[ts::countries::YUGOSLAVIA].us_influence = 0;

    // Trigger Truman Doctrine
    ts::CardHandlers::trigger_event(state, ts::card_ids::TRUMAN_DOCTRINE, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::POINT_NODE);

    // US selects Yugoslavia
    ts::MicroAction act{};
    act.decision_type = ts::DecisionType::POINT_NODE;
    act.primary_id = ts::countries::YUGOSLAVIA;
    ts::CardHandlers::handle_event_step(state, act);

    ASSERT_EQ(state.countries[ts::countries::YUGOSLAVIA].ussr_influence, 0);
}
