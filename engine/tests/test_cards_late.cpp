#include "test_framework.hpp"
#include "ts/engine.hpp"
#include "ts/card_data.hpp"
#include "ts/card_handlers.hpp"
#include "ts/constants.hpp"

TEST(LateCardsTest, ReaganBombsLibya) {
    ts::GameState state{};
    state.countries[ts::countries::LIBYA].ussr_influence = 5;
    state.victory_points = 0;

    ts::CardHandlers::trigger_event(state, ts::card_ids::REAGAN_BOMBS_LIBYA, ts::Player::US);

    // 5 / 2 = 2 VP for US
    ASSERT_EQ(state.victory_points, 2);
}

TEST(LateCardsTest, SolidarityPrerequisite) {
    ts::GameState state{};
    state.countries[ts::countries::POLAND].us_influence = 0;

    // Without John Paul II played -> event does not add influence
    ts::CardHandlers::trigger_event(state, ts::card_ids::SOLIDARITY, ts::Player::US);
    ASSERT_EQ(state.countries[ts::countries::POLAND].us_influence, 0);

    // With John Paul II played -> adds 3 US influence to Poland
    state.set_flag(ts::effect_bits::JOHN_PAUL_II_PLAYED);
    ts::CardHandlers::trigger_event(state, ts::card_ids::SOLIDARITY, ts::Player::US);
    ASSERT_EQ(state.countries[ts::countries::POLAND].us_influence, 3);
}
