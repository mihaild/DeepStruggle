#include "test_framework.hpp"
#include "ts/engine.hpp"
#include "ts/card_data.hpp"
#include "ts/constants.hpp"
#include "ts/card_handlers.hpp"
#include "ts/ops.hpp"

TEST(DefconSuicideTest, CoupAtDefcon2CausesPhasingPlayerLoss) {
    ts::GameState state{};
    ts::Engine::init_game(state, 12345);

    // Set DEFCON to 2 and active player to USSR
    state.defcon = 2;
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.phasing_player = ts::Player::USSR;
    state.countries[ts::countries::EGYPT].us_influence = 2; // BG country

    // USSR coups Egypt (Battleground) at DEFCON 2
    auto res = ts::Operations::execute_coup(state, ts::Player::USSR, ts::countries::EGYPT, 3, 3);

    ASSERT_TRUE(res.caused_defcon_suicide);
    ASSERT_EQ(state.defcon, 1);
    // Phasing player (USSR) loses immediately -> VP becomes +20 (US win)
    ASSERT_EQ(state.victory_points, 20);
    ASSERT_EQ(state.current_phase, ts::Phase::GAME_OVER);
    ASSERT_TRUE(ts::Engine::is_terminal(state));
    ASSERT_EQ(ts::Engine::get_terminal_utility(state), 1.0f);
}

TEST(DefconSuicideTest, OpponentEventDuckAndCoverAtDefcon2CausesPhasingPlayerLoss) {
    ts::GameState state{};
    ts::Engine::init_game(state, 12345);

    // Set DEFCON to 2, USSR is phasing player, playing Duck and Cover (#4, US event) for Ops
    state.defcon = 2;
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.phasing_player = ts::Player::USSR;

    // Trigger Duck and Cover event on USSR's turn
    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::DUCK_AND_COVER, ts::Player::US);
    ASSERT_TRUE(done);

    // Duck and Cover degraded DEFCON to 1 -> Phasing player (USSR) loses immediately!
    ASSERT_EQ(state.defcon, 1);
    ASSERT_EQ(state.victory_points, 20);
    ASSERT_EQ(state.current_phase, ts::Phase::GAME_OVER);
}

// The coup that takes DEFCON to 1 still resolves first. The rules carry it out -- die rolled,
// influence moved, military ops credited -- and only then does the game end with the couping
// player losing. The engine used to return the moment DEFCON hit 1, before the roll, freezing
// the board exactly where the decisive coup was made: at turn 5's headline of ts-replayer
// game 102 the USSR's Panama coup removed two US influence in the human log and none here.
TEST(DefconSuicideTest, CoupThatDropsDefconToOneStillMovesInfluence) {
    ts::GameState state{};
    ts::Engine::init_game(state, 12345);

    state.defcon = 2;
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.phasing_player = ts::Player::USSR;
    state.countries[ts::countries::EGYPT].us_influence = 2;
    state.countries[ts::countries::EGYPT].ussr_influence = 0;
    const uint8_t milops_before = state.ussr_mil_ops;

    // 4 Ops against Egypt's stability 2 clears 2*2 on any roll, so the coup cannot fail.
    auto res = ts::Operations::execute_coup(state, ts::Player::USSR, ts::countries::EGYPT, 4, 3);

    ASSERT_TRUE(res.success);
    ASSERT_EQ(state.countries[ts::countries::EGYPT].us_influence, 0);
    ASSERT_GT(state.ussr_mil_ops, milops_before);
    ASSERT_TRUE(res.caused_defcon_suicide);
    ASSERT_EQ(state.defcon, 1);
    ASSERT_EQ(state.current_phase, ts::Phase::GAME_OVER);
}
