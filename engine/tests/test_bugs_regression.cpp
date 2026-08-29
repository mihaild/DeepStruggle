#include "ts/observation.hpp"
#include "test_framework.hpp"
#include "ts/state_machine.hpp"
#include "ts/engine.hpp"
#include "ts/map_data.hpp"
#include "ts/card_data.hpp"
#include "ts/action_mask.hpp"
#include "ts/card_handlers.hpp"

using namespace ts;

TEST(RegressionTest, USHeadlinesDeStalinizationTriggersUSSRChoice) {
    GameState state{};
    Engine::init_game(state, 42);

    // Setup USSR influence
    state.countries[countries::EAST_GERMANY].ussr_influence = 3;
    state.countries[countries::POLAND].ussr_influence = 3;

    // Transition to Headline
    state.current_phase = Phase::HEADLINE;
    state.ctx().decision_player = Player::US;
    state.ctx().decision_type = DecisionType::SELECT_CARD;

    // US headlines De-Stalinization (#33)
    MicroAction us_act(DecisionType::SELECT_CARD, card_ids::DE_STALINIZATION, 0, 0);
    bool ok1 = Engine::step(state, us_act);
    ASSERT_TRUE(ok1);

    // USSR headlines Asia Scoring (#1)
    MicroAction ussr_act(DecisionType::SELECT_CARD, card_ids::ASIA_SCORING, 0, 0);
    bool ok2 = Engine::step(state, ussr_act);
    ASSERT_TRUE(ok2);

    // Headline resolution: De-Stalinization (3 ops) vs Asia Scoring (0 ops) -> De-Stalinization goes first!
    // De-Stalinization gives USSR a choice to remove influence!
    ASSERT_EQ(state.current_phase, Phase::HEADLINE);
    ASSERT_EQ(state.ctx().decision_player, Player::USSR);
    ASSERT_EQ(state.ctx().decision_type, DecisionType::POINT_NODE);
    ASSERT_EQ(state.ctx().resolving_card, card_ids::DE_STALINIZATION);
}

TEST(RegressionTest, DeGaulleOpsFirstTriggersOpponentEventAfterOps) {
    GameState state{};
    Engine::init_game(state, 42);

    // Give US De Gaulle (#17)
    state.card_locations[card_ids::DE_GAULLE] = CardLocation::HAND_US;
    state.current_phase = Phase::ACTION_ROUND;
    state.action_round = 1;
    state.phasing_player = Player::US;
    state.ctx().decision_player = Player::US;
    state.ctx().decision_type = DecisionType::SELECT_CARD;

    // Put 3 US influence in France
    state.countries[countries::FRANCE].us_influence = 3;
    state.countries[countries::FRANCE].ussr_influence = 0;

    // 1. Select Card De Gaulle (#17, 3 Ops, USSR event)
    Engine::step(state, MicroAction(DecisionType::SELECT_CARD, card_ids::DE_GAULLE, 0, 0));
    ASSERT_EQ(state.ctx().decision_type, DecisionType::SELECT_PLAY_MODE);

    // 2. Select OPS
    Engine::step(state, MicroAction(DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(PlayMode::OPS), 0, 0));
    ASSERT_EQ(state.ctx().decision_type, DecisionType::CHOOSE_TIMING_BRANCH);

    // 3. Choose OPS_FIRST
    Engine::step(state, MicroAction(DecisionType::CHOOSE_TIMING_BRANCH, static_cast<uint8_t>(TimingBranch::OPS_FIRST), 0, 0));
    ASSERT_EQ(state.ctx().decision_type, DecisionType::SELECT_OP_MODE);

    // 4. Select INFLUENCE mode
    Engine::step(state, MicroAction(DecisionType::SELECT_OP_MODE, static_cast<uint8_t>(OpMode::INFLUENCE), 0, 0));
    ASSERT_EQ(state.ctx().decision_type, DecisionType::POINT_NODE);

    // 5. Place 3 ops in UK (ID 1)
    Engine::step(state, MicroAction(DecisionType::POINT_NODE, countries::UNITED_KINGDOM, 0, 0));
    Engine::step(state, MicroAction(DecisionType::POINT_NODE, countries::UNITED_KINGDOM, 0, 0));
    Engine::step(state, MicroAction(DecisionType::POINT_NODE, countries::UNITED_KINGDOM, 0, 0));

    // After ops finished, De Gaulle event MUST have resolved automatically!
    // France should have: US influence 3 - 2 = 1, USSR influence 0 + 1 = 1, NATO canceled
    ASSERT_EQ(state.countries[countries::FRANCE].us_influence, 1);
    ASSERT_EQ(state.countries[countries::FRANCE].ussr_influence, 1);
    ASSERT_TRUE(state.has_flag(effect_bits::NATO_CANCELED_FRANCE));
    ASSERT_EQ(state.card_locations[card_ids::DE_GAULLE], CardLocation::REMOVED_FROM_GAME);
}

TEST(RegressionTest, RealignmentOnlyTargetsCountriesWithOpponentInfluenceAndRolls) {
    GameState state{};
    Engine::init_game(state, 42);

    state.countries[countries::FRANCE].us_influence = 2;
    state.countries[countries::FRANCE].ussr_influence = 0;
    state.countries[countries::ITALY].us_influence = 0;
    state.countries[countries::ITALY].ussr_influence = 2;

    // US playing card for realignment
    state.current_phase = Phase::ACTION_ROUND;
    state.action_round = 1;
    state.phasing_player = Player::US;
    state.ctx().decision_player = Player::US;
    state.ctx().decision_type = DecisionType::SELECT_CARD;
    state.card_locations[card_ids::DUCK_AND_COVER] = CardLocation::HAND_US; // 3 Ops US card

    Engine::step(state, MicroAction(DecisionType::SELECT_CARD, card_ids::DUCK_AND_COVER, 0, 0));
    Engine::step(state, MicroAction(DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(PlayMode::OPS), 0, 0));
    Engine::step(state, MicroAction(DecisionType::SELECT_OP_MODE, static_cast<uint8_t>(OpMode::REALIGN), 0, 0));

    ASSERT_EQ(state.ctx().decision_type, DecisionType::POINT_NODE);
    ASSERT_EQ(state.ctx().op_mode, OpMode::REALIGN);

    // Verify ActionMask for Realignment: France has 0 USSR influence, so France CANNOT be targeted!
    // Italy has 2 USSR influence, so Italy CAN be targeted!
    uint8_t mask[84];
    size_t out_size = 0;
    ActionMask::generate_mask(state, mask, &out_size);
    ASSERT_EQ(out_size, 84);
    ASSERT_EQ(mask[countries::FRANCE], 0);
    ASSERT_EQ(mask[countries::ITALY], 1);

    // Execute realignment on Italy with forced rolls (US roll 6, USSR roll 1)
    // US should win by 5, removing all 2 USSR influence from Italy
    MicroAction realign_act(DecisionType::POINT_NODE, countries::ITALY, 6, 1);
    Engine::step(state, realign_act);
    ASSERT_EQ(state.ctx().decision_type, DecisionType::ROLL_DIE);
    Engine::step(state, MicroAction(DecisionType::ROLL_DIE, 6, 1, 0));
    ASSERT_EQ(state.countries[countries::ITALY].ussr_influence, 0);
}

TEST(RegressionTest, ArabIsraeliWarWithRoll2Fails) {
    GameState state{};
    Engine::init_game(state, 42);

    state.countries[countries::ISRAEL].us_influence = 2;
    state.countries[countries::ISRAEL].ussr_influence = 0;
    state.countries[countries::JORDAN].us_influence = 2; // US controls Jordan (-1 mod)

    state.card_locations[card_ids::ARAB_ISRAELI_WAR] = CardLocation::HAND_USSR;
    state.current_phase = Phase::ACTION_ROUND;
    state.action_round = 1;
    state.phasing_player = Player::USSR;
    state.ctx().decision_player = Player::USSR;
    state.ctx().decision_type = DecisionType::SELECT_CARD;

    // 1. Select Card
    Engine::step(state, MicroAction(DecisionType::SELECT_CARD, card_ids::ARAB_ISRAELI_WAR, 0, 0));
    // 2. Select Play Mode EVENT with forced roll = 2
    Engine::step(state, MicroAction(DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(PlayMode::EVENT), 2, 0));
    ASSERT_EQ(state.ctx().decision_type, DecisionType::ROLL_DIE);
    Engine::step(state, MicroAction(DecisionType::ROLL_DIE, 2, 0, 0));

    // Roll 2 - 1 = 1 (< 4), so Arab-Israeli War fails!
    // Israel must still have 2 US influence and 0 USSR influence
    ASSERT_EQ(state.countries[countries::ISRAEL].us_influence, 2);
    ASSERT_EQ(state.countries[countries::ISRAEL].ussr_influence, 0);
    // USSR mil ops gained 2 regardless
    ASSERT_EQ(state.ussr_mil_ops, 2);
}

TEST(RegressionTest, NoChainInfluencePlacementDuringSameOp) {
    GameState state{};
    Engine::init_game(state, 42);

    // Give US influence ONLY in France (adjacent to Algeria), but 0 in Morocco and 0 in West Africa
    state.countries[countries::FRANCE].us_influence = 3;
    state.countries[countries::ALGERIA].us_influence = 2; // Adjacent to Morocco
    state.countries[countries::MOROCCO].us_influence = 0;
    state.countries[countries::WEST_AFRICA].us_influence = 0;

    state.card_locations[card_ids::DUCK_AND_COVER] = CardLocation::HAND_US; // 3 Ops
    state.current_phase = Phase::ACTION_ROUND;
    state.action_round = 1;
    state.phasing_player = Player::US;
    state.ctx().decision_player = Player::US;
    state.ctx().decision_type = DecisionType::SELECT_CARD;

    Engine::step(state, MicroAction(DecisionType::SELECT_CARD, card_ids::DUCK_AND_COVER, 0, 0));
    Engine::step(state, MicroAction(DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(PlayMode::OPS), 0, 0));
    Engine::step(state, MicroAction(DecisionType::SELECT_OP_MODE, static_cast<uint8_t>(OpMode::INFLUENCE), 0, 0));

    // 1st point placed in Morocco (legal because adjacent to Algeria)
    uint8_t mask[84];
    size_t out_size = 0;
    ActionMask::generate_mask(state, mask, &out_size);
    ASSERT_EQ(mask[countries::MOROCCO], 1);
    ASSERT_EQ(mask[countries::WEST_AFRICA], 0);

    Engine::step(state, MicroAction(DecisionType::POINT_NODE, countries::MOROCCO, 0, 0));
    ASSERT_EQ(state.countries[countries::MOROCCO].us_influence, 1);

    // 2nd point: West Africa must STILL be 0 (no chain placement allowed in same AR)
    ActionMask::generate_mask(state, mask, &out_size);
    ASSERT_EQ(mask[countries::WEST_AFRICA], 0);
    // Morocco can receive a 2nd point because it now has influence
    ASSERT_EQ(mask[countries::MOROCCO], 1);
}

TEST(RegressionTest, WarsawPactEasternEuropeOnlyAndMax2) {
    GameState state{};
    Engine::init_game(state, 42);

    state.card_locations[card_ids::WARSAW_PACT] = CardLocation::HAND_USSR;
    state.current_phase = Phase::ACTION_ROUND;
    state.action_round = 1;
    state.phasing_player = Player::USSR;
    state.ctx().decision_player = Player::USSR;
    state.ctx().decision_type = DecisionType::SELECT_CARD;

    Engine::step(state, MicroAction(DecisionType::SELECT_CARD, card_ids::WARSAW_PACT, 0, 0));
    Engine::step(state, MicroAction(DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(PlayMode::EVENT), 0, 0));

    // Choose Branch 1: Add 5 USSR influence in Eastern Europe (max 2 per country)
    Engine::step(state, MicroAction(DecisionType::CHOOSE_BRANCH, 1, 0, 0));
    ASSERT_EQ(state.ctx().decision_type, DecisionType::POINT_NODE);

    // Verify ActionMask: Only Eastern Europe countries are legal!
    uint8_t mask[84];
    size_t out_size = 0;
    ActionMask::generate_mask(state, mask, &out_size);
    for (uint8_t i = 0; i < 84; ++i) {
        if (MapData::get_country(i).in_eastern_europe) {
            ASSERT_EQ(mask[i], 1);
        } else {
            ASSERT_EQ(mask[i], 0);
        }
    }

    // Place 2 in Poland (ID 15)
    Engine::step(state, MicroAction(DecisionType::POINT_NODE, countries::POLAND, 0, 0));
    Engine::step(state, MicroAction(DecisionType::POINT_NODE, countries::POLAND, 0, 0));

    // Poland reached 2 placements -> Poland mask must now be 0!
    ActionMask::generate_mask(state, mask, &out_size);
    ASSERT_EQ(mask[countries::POLAND], 0);
}

TEST(RegressionTest, ObservationHiddenOpponentHand) {
    GameState state{};
    Engine::init_game(state, 42);

    state.card_locations[5] = CardLocation::HAND_USSR;
    state.card_locations[10] = CardLocation::HAND_US;
    state.card_locations[12] = CardLocation::DISCARD_PILE;

    ObservationBuffer obs_us{};
    Observation::extract(state, Player::US, &obs_us);

    // Card 10 (US hand) from US perspective is MY_HAND (canon_loc = 1)
    size_t off_10 = (10 - 1) * 12;
    ASSERT_TRUE(obs_us.card_features[off_10 + 1] == 1.0f);
    ASSERT_TRUE(obs_us.card_features[off_10 + 0] == 0.0f);
    ASSERT_TRUE(obs_us.card_features[off_10 + 2] == 0.0f);

    // Card 5 (USSR hand) from US perspective must be HIDDEN (canon_loc = 0)
    size_t off_5 = (5 - 1) * 12;
    ASSERT_TRUE(obs_us.card_features[off_5 + 0] == 1.0f);
    ASSERT_TRUE(obs_us.card_features[off_5 + 1] == 0.0f);
    ASSERT_TRUE(obs_us.card_features[off_5 + 2] == 0.0f);

    // Global feature 70 is opp_hand_cnt / 10.0f (public count is preserved)
    ASSERT_TRUE(obs_us.global_features[70] > 0.0f);

    ObservationBuffer obs_ussr{};
    Observation::extract(state, Player::USSR, &obs_ussr);

    // Card 5 (USSR hand) from USSR perspective is MY_HAND (canon_loc = 1)
    ASSERT_TRUE(obs_ussr.card_features[off_5 + 1] == 1.0f);
    ASSERT_TRUE(obs_ussr.card_features[off_5 + 0] == 0.0f);
    ASSERT_TRUE(obs_ussr.card_features[off_5 + 2] == 0.0f);

    // Card 10 (US hand) from USSR perspective must be HIDDEN (canon_loc = 0)
    ASSERT_TRUE(obs_ussr.card_features[off_10 + 0] == 1.0f);
    ASSERT_TRUE(obs_ussr.card_features[off_10 + 1] == 0.0f);
    ASSERT_TRUE(obs_ussr.card_features[off_10 + 2] == 0.0f);
}

TEST(RegressionTest, ObservationActionHistoryDerotation) {
    GameState state{};
    Engine::init_game(state, 42);

    // Record 20 action tokens with sequential card_ids 1..20
    for (uint8_t i = 1; i <= 20; ++i) {
        ActionToken tok{};
        tok.acting_player = (i % 2 == 1) ? Player::US : Player::USSR;
        tok.action_type = ActionType::PLAY_EVENT;
        tok.card_id = i;
        tok.target_id = i;
        tok.ops_value = 2;
        tok.die_roll = 3;
        tok.us_inf_delta = 1;
        tok.ussr_inf_delta = 0;
        tok.defcon_after = 4;
        state.action_history.record(tok);
    }

    ObservationBuffer obs{};
    Observation::extract(state, Player::US, &obs);

    // History sequence has 16 items.
    // Index 15 must be the most recent action (token with card_id = 20)
    // Index 0 must be the oldest action in the 16-token window (token with card_id = 5)
    for (size_t h = 0; h < 16; ++h) {
        uint8_t expected_card_id = 5 + h; // 5..20
        float expected_card_norm = static_cast<float>(expected_card_id) / 110.0f;
        size_t h_off = h * 32;
        float diff = std::abs(obs.history_sequence[h_off + 2] - expected_card_norm);
        ASSERT_TRUE(diff < 1e-5f);
    }
}

// DEFCON-1 losses are classified by *agency*, not by side. A player who chooses the
// action that drops DEFCON to 1 committed an avoidable blunder (flag clear); a player
// forced to fire an opponent-associated event while playing that card for Operations was
// squeezed and had no safe alternative (flag set). Before this was unified, the Duck and
// Cover and KAL-007 sites keyed the flag off `loser == USSR`, so the label depended on
// which side lost rather than on who chose the action.
TEST(RegressionTest, DefconOneProvokedFlagTracksAgencyNotSide) {
    // Case 1: US is phasing and fires its OWN event (Duck and Cover) at DEFCON 2.
    // Entirely avoidable -> unprovoked.
    {
        GameState state{};
        Engine::init_game(state, 7);
        state.current_phase = Phase::ACTION_ROUND;
        state.phasing_player = Player::US;
        state.defcon = 2;
        state.clear_flag(effect_bits::DEFCON_SUICIDE_PROVOKED);

        CardHandlers::trigger_event(state, card_ids::DUCK_AND_COVER, Player::US);

        ASSERT_EQ(static_cast<int>(state.current_phase), static_cast<int>(Phase::GAME_OVER));
        ASSERT_EQ(static_cast<int>(state.victory_points), -20); // phasing US loses
        ASSERT_TRUE(!state.has_flag(effect_bits::DEFCON_SUICIDE_PROVOKED));
    }

    // Case 2: USSR is phasing and plays the same US-associated card for Operations, so
    // the US event fires against them. Forced -> provoked.
    {
        GameState state{};
        Engine::init_game(state, 7);
        state.current_phase = Phase::ACTION_ROUND;
        state.phasing_player = Player::USSR;
        state.defcon = 2;
        state.clear_flag(effect_bits::DEFCON_SUICIDE_PROVOKED);

        CardHandlers::trigger_event(state, card_ids::DUCK_AND_COVER, Player::US);

        ASSERT_EQ(static_cast<int>(state.current_phase), static_cast<int>(Phase::GAME_OVER));
        ASSERT_EQ(static_cast<int>(state.victory_points), 20); // phasing USSR loses
        ASSERT_TRUE(state.has_flag(effect_bits::DEFCON_SUICIDE_PROVOKED));
    }

    // Case 3: the mirror of case 1 on the other side. USSR fires KAL-007 (a US event)
    // while US is phasing -> US loses and it is provoked, i.e. the flag is driven by
    // agency and NOT by "USSR is the loser" as the old side-based check assumed.
    {
        GameState state{};
        Engine::init_game(state, 7);
        state.current_phase = Phase::ACTION_ROUND;
        state.phasing_player = Player::US;
        state.defcon = 2;
        state.clear_flag(effect_bits::DEFCON_SUICIDE_PROVOKED);

        CardHandlers::trigger_event(state, card_ids::SOVIETS_SHOOT_DOWN_KAL_007, Player::USSR);

        ASSERT_EQ(static_cast<int>(state.current_phase), static_cast<int>(Phase::GAME_OVER));
        ASSERT_EQ(static_cast<int>(state.victory_points), -20); // phasing US loses
        ASSERT_TRUE(state.has_flag(effect_bits::DEFCON_SUICIDE_PROVOKED));
    }
}
