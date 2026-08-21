#include "test_framework.hpp"
#include "ts/engine.hpp"
#include "ts/card_data.hpp"
#include "ts/card_handlers.hpp"
#include "ts/constants.hpp"

TEST(MidCardsTest, PanamaCanalReturned) {
    ts::GameState state{};
    state.countries[ts::countries::PANAMA].us_influence = 0;
    state.countries[ts::countries::COSTA_RICA].us_influence = 0;
    state.countries[ts::countries::VENEZUELA].us_influence = 0;

    ts::CardHandlers::trigger_event(state, ts::card_ids::PANAMA_CANAL_RETURNED, ts::Player::US);

    ASSERT_EQ(state.countries[ts::countries::PANAMA].us_influence, 1);
    ASSERT_EQ(state.countries[ts::countries::COSTA_RICA].us_influence, 1);
    ASSERT_EQ(state.countries[ts::countries::VENEZUELA].us_influence, 1);
}

TEST(MidCardsTest, SadatExpelsSoviets) {
    ts::GameState state{};
    state.countries[ts::countries::EGYPT].ussr_influence = 3;
    state.countries[ts::countries::EGYPT].us_influence = 0;

    ts::CardHandlers::trigger_event(state, ts::card_ids::SADAT_EXPELS_SOVIETS, ts::Player::US);

    ASSERT_EQ(state.countries[ts::countries::EGYPT].ussr_influence, 0);
    ASSERT_EQ(state.countries[ts::countries::EGYPT].us_influence, 1);
}
