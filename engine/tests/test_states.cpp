#include <algorithm>
#include "test_framework.hpp"
#include "ts/engine.hpp"
#include "ts/state_machine.hpp"
#include "ts/card_data.hpp"
#include "ts/card_handlers.hpp"
#include "ts/constants.hpp"
#include "ts/scoring.hpp"
#include "ts/map_data.hpp"
#include "ts/ops.hpp"
#include "ts/space_race.hpp"
#include "ts/action_mask.hpp"
#include "ts/serialization.hpp"

// =============================================================================
// 1. Phase State Machine Tests
// =============================================================================

TEST(StatesTest, Phase_SETUP_Flow) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 123);
    ASSERT_EQ(state.current_phase, ts::Phase::SETUP);
    ASSERT_EQ(state.ctx().decision_player, ts::Player::USSR);
    ASSERT_EQ(state.ctx().remaining_steps, 6);

    // USSR 6 placements in Eastern Europe
    for (int i = 0; i < 6; ++i) {
        ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::POLAND, 0, 0}));
    }
    ASSERT_EQ(state.ctx().decision_player, ts::Player::US);
    ASSERT_EQ(state.ctx().remaining_steps, 7);

    // US 7 placements in Western Europe (Stage 0)
    for (int i = 0; i < 7; ++i) {
        ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::WEST_GERMANY, 0, 0}));
    }
    ASSERT_EQ(state.current_phase, ts::Phase::SETUP);
    ASSERT_EQ(state.ctx().decision_player, ts::Player::US);
    ASSERT_EQ(state.ctx().remaining_steps, 2);

    // US 2 bonus placements in countries with US influence (Stage 1)
    for (int i = 0; i < 2; ++i) {
        ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::WEST_GERMANY, 0, 0}));
    }
    ASSERT_EQ(state.current_phase, ts::Phase::HEADLINE);
}

TEST(StatesTest, Phase_HEADLINE_Flow) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 123);
    state.current_phase = ts::Phase::HEADLINE;
    state.ctx().decision_player = ts::Player::US;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::hand_of(ts::Player::US);
    state.card_locations[ts::card_ids::SOCIALIST_GOVERNMENTS] = ts::hand_of(ts::Player::USSR);

    // US selects Duck and Cover (3 Ops)
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::DUCK_AND_COVER, 0, 0}));
    ASSERT_EQ(state.ctx().decision_player, ts::Player::USSR);

    // USSR selects Socialist Governments (3 Ops)
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::SOCIALIST_GOVERNMENTS, 0, 0}));
}

TEST(StatesTest, Phase_ACTION_ROUND_PlayerAlternation) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 123);
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.action_round = 1;
    state.phasing_player = ts::Player::USSR;
    state.ctx().decision_player = ts::Player::USSR;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::hand_of(ts::Player::USSR);

    // USSR plays Duck & Cover for Ops
    ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::DUCK_AND_COVER, 0, 0});
    ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::PlayMode::OPS), 0, 0});
    ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::CHOOSE_TIMING_BRANCH, static_cast<uint8_t>(ts::TimingBranch::OPS_FIRST), 0, 0});
    ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_OP_MODE, static_cast<uint8_t>(ts::OpMode::INFLUENCE), 0, 0});
    for (int i = 0; i < 3; ++i) {
        ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::NORTH_KOREA, 0, 0});
    }

    // Alternates to US for AR 1
    ASSERT_EQ(state.phasing_player, ts::Player::US);
    ASSERT_EQ(state.action_round, 1);
}

TEST(StatesTest, Phase_FINAL_SCORING_And_GAME_OVER) {
    ts::GameState state{};
    state.turn = 10;
    state.victory_points = 5;

    ts::Scoring::execute_final_scoring(state);
    ASSERT_EQ(state.current_phase, ts::Phase::GAME_OVER);
}

// =============================================================================
// 2. DecisionType Tests
// =============================================================================

TEST(StatesTest, DecisionType_AllTypes_ActionMaskGeneration) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 123);
    uint8_t mask[112]{};
    size_t out_size = 0;

    // 1. SELECT_CARD
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.phasing_player = ts::Player::US;
    state.ctx().decision_player = ts::Player::US;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;
    state.card_locations[ts::card_ids::DEFECTORS] = ts::hand_of(ts::Player::US);
    ts::ActionMask::generate_mask(state, mask, &out_size);
    ASSERT_EQ(out_size, 112);
    ASSERT_EQ(mask[ts::card_ids::DEFECTORS], 1);

    // 2. SELECT_PLAY_MODE
    // Defectors CANNOT be played as event by US during Action Round
    state.ctx().decision_type = ts::DecisionType::SELECT_PLAY_MODE;
    state.ctx().pending_op_card = ts::card_ids::DEFECTORS;
    ts::ActionMask::generate_mask(state, mask, &out_size);
    ASSERT_EQ(out_size, 4);
    ASSERT_EQ(mask[static_cast<uint8_t>(ts::PlayMode::EVENT)], 0);
    ASSERT_EQ(mask[static_cast<uint8_t>(ts::PlayMode::OPS)], 1);

    // Friendly card with valid event (Duck and Cover) CAN be played as Event or Ops
    state.ctx().pending_op_card = ts::card_ids::DUCK_AND_COVER;
    ts::ActionMask::generate_mask(state, mask, &out_size);
    ASSERT_EQ(mask[static_cast<uint8_t>(ts::PlayMode::EVENT)], 1);
    ASSERT_EQ(mask[static_cast<uint8_t>(ts::PlayMode::OPS)], 1);

    // 3. CHOOSE_TIMING_BRANCH
    state.ctx().decision_type = ts::DecisionType::CHOOSE_TIMING_BRANCH;
    ts::ActionMask::generate_mask(state, mask, &out_size);
    ASSERT_EQ(out_size, 2);
    ASSERT_EQ(mask[static_cast<uint8_t>(ts::TimingBranch::EVENT_FIRST)], 1);
    ASSERT_EQ(mask[static_cast<uint8_t>(ts::TimingBranch::OPS_FIRST)], 1);

    // 4. SELECT_OP_MODE
    state.ctx().decision_type = ts::DecisionType::SELECT_OP_MODE;
    state.ctx().pending_ops_value = 3;
    state.countries[ts::countries::CANADA].us_influence = 2;
    state.countries[ts::countries::ARGENTINA].ussr_influence = 2;
    state.defcon = 5;
    ts::ActionMask::generate_mask(state, mask, &out_size);
    ASSERT_EQ(out_size, 3);
    ASSERT_EQ(mask[static_cast<uint8_t>(ts::OpMode::INFLUENCE)], 1);
    ASSERT_EQ(mask[static_cast<uint8_t>(ts::OpMode::COUP)], 1);
    ASSERT_EQ(mask[static_cast<uint8_t>(ts::OpMode::REALIGN)], 1);

    // 5. POINT_NODE
    state.ctx().decision_type = ts::DecisionType::POINT_NODE;
    state.ctx().op_mode = ts::OpMode::INFLUENCE;
    state.ctx().remaining_steps = 2;
    state.countries[ts::countries::CANADA].us_influence = 2;
    ts::ActionMask::generate_mask(state, mask, &out_size);
    ASSERT_EQ(out_size, 84);
    ASSERT_EQ(mask[ts::countries::CANADA], 1);

    // 6. CHOOSE_BRANCH
    state.ctx().decision_type = ts::DecisionType::CHOOSE_BRANCH;
    state.ctx().resolving_card = ts::card_ids::WARSAW_PACT;
    ts::ActionMask::generate_mask(state, mask, &out_size);
    ASSERT_EQ(out_size, 8);
    ASSERT_EQ(mask[0], 1);
    ASSERT_EQ(mask[1], 1);
}

// =============================================================================
// 3. Persistent Game State Mechanics & Invariants
// =============================================================================

TEST(StatesTest, GameState_Chernobyl_BlocksUSSRInfluencePlacement_AndResetsAtTurnEnd) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 123);
    for (uint8_t i = 1; i <= 110; ++i) {
        if (ts::CardData::is_scoring_card(i)) {
            state.card_locations[i] = ts::CardLocation::DISCARD_PILE;
        }
    }
    state.countries[ts::countries::EAST_GERMANY].ussr_influence = 3; // In Europe
    state.countries[ts::countries::NORTH_KOREA].ussr_influence = 3;  // In Asia

    // 1. US designates Europe (region 0) for Chernobyl
    uint64_t ch_flags = ts::effect_bits::CHERNOBYL_ACTIVE | (static_cast<uint64_t>(ts::Region::EUROPE) << ts::effect_bits::CHERNOBYL_REGION_SHIFT);
    state.set_flag(ch_flags);

    // USSR cannot place influence in Europe
    ASSERT_FALSE(ts::Operations::can_place_influence(state, ts::Player::USSR, ts::countries::EAST_GERMANY));
    ASSERT_FALSE(ts::Operations::can_place_influence(state, ts::Player::USSR, ts::countries::POLAND));

    // But USSR CAN place influence in Asia (North Korea)
    ASSERT_TRUE(ts::Operations::can_place_influence(state, ts::Player::USSR, ts::countries::NORTH_KOREA));

    // 2. End turn -> Chernobyl is cleared by TURN_CLEANUP_MASK
    ts::StateMachine::end_turn(state);
    ASSERT_FALSE(state.has_flag(ts::effect_bits::CHERNOBYL_ACTIVE));
    ASSERT_TRUE(ts::Operations::can_place_influence(state, ts::Player::USSR, ts::countries::EAST_GERMANY));
}

TEST(StatesTest, GameState_TheReformer_BlocksUSSRCoupsInEurope) {
    ts::GameState state{};
    state.defcon = 5; // Defcon allows European coups normally
    state.countries[ts::countries::WEST_GERMANY].us_influence = 2;
    state.countries[ts::countries::EGYPT].us_influence = 2;

    // 1. Without Reformer: USSR can coup in Europe at DEFCON 5
    ASSERT_TRUE(ts::Operations::can_coup(state, ts::Player::USSR, ts::countries::WEST_GERMANY));

    // 2. Set THE_REFORMER_PLAYED
    state.set_flag(ts::effect_bits::THE_REFORMER_PLAYED);

    // USSR cannot coup anywhere in Europe
    ASSERT_FALSE(ts::Operations::can_coup(state, ts::Player::USSR, ts::countries::WEST_GERMANY));
    ASSERT_FALSE(ts::Operations::can_coup(state, ts::Player::USSR, ts::countries::FRANCE));

    // But USSR CAN coup in Middle East (Egypt)
    ASSERT_TRUE(ts::Operations::can_coup(state, ts::Player::USSR, ts::countries::EGYPT));
}

TEST(StatesTest, GameState_SALT_Negotiations_AppliesMinus1ToCoups_AndResetsAtTurnEnd) {
    ts::GameState state{};
    for (uint8_t i = 1; i <= 110; ++i) {
        if (ts::CardData::is_scoring_card(i)) {
            state.card_locations[i] = ts::CardLocation::DISCARD_PILE;
        }
    }
    state.countries[ts::countries::EGYPT].ussr_influence = 2; // Stability 2

    // 1. Without SALT: 2 Ops, roll 3 -> Total = 3 + 2 = 5. Margin = 5 - 2*2 = 1.
    auto res1 = ts::Operations::execute_coup(state, ts::Player::US, ts::countries::EGYPT, 2, 3);
    ASSERT_EQ(res1.margin, 1);
    ASSERT_EQ(state.countries[ts::countries::EGYPT].ussr_influence, 1);

    // 2. With SALT: 2 Ops, roll 3 -> Total = 3 - 1 + 2 = 4. Margin = 4 - 2*2 = 0.
    state.set_flag(ts::effect_bits::SALT_ACTIVE);
    auto res2 = ts::Operations::execute_coup(state, ts::Player::US, ts::countries::EGYPT, 2, 3);
    ASSERT_EQ(res2.margin, 0);

    // 3. End of turn clears SALT
    ts::StateMachine::end_turn(state);
    ASSERT_FALSE(state.has_flag(ts::effect_bits::SALT_ACTIVE));
}

TEST(StatesTest, GameState_NATO_ProtectionAndCancellations) {
    ts::GameState state{};
    state.defcon = 5;
    state.countries[ts::countries::FRANCE].us_influence = 3;       // US-controlled (stability 3)
    state.countries[ts::countries::WEST_GERMANY].us_influence = 4; // US-controlled (stability 4)
    state.countries[ts::countries::ITALY].us_influence = 2;        // US-controlled (stability 2)

    // 1. Without NATO: USSR can coup/realign US-controlled European countries at DEFCON 5
    ASSERT_TRUE(ts::Operations::can_coup(state, ts::Player::USSR, ts::countries::FRANCE));
    ASSERT_TRUE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::FRANCE));

    // 2. Set NATO_ACTIVE: USSR blocked from couping/realigning all US-controlled European countries
    state.set_flag(ts::effect_bits::NATO_ACTIVE);
    ASSERT_FALSE(ts::Operations::can_coup(state, ts::Player::USSR, ts::countries::FRANCE));
    ASSERT_FALSE(ts::Operations::can_coup(state, ts::Player::USSR, ts::countries::WEST_GERMANY));
    ASSERT_FALSE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::ITALY));

    // 3. De Gaulle cancels NATO for France only
    state.set_flag(ts::effect_bits::NATO_CANCELED_FRANCE);
    ASSERT_TRUE(ts::Operations::can_coup(state, ts::Player::USSR, ts::countries::FRANCE));
    ASSERT_FALSE(ts::Operations::can_coup(state, ts::Player::USSR, ts::countries::WEST_GERMANY));

    // 4. Willy Brandt cancels NATO for West Germany only
    state.set_flag(ts::effect_bits::NATO_CANCELED_WEST_GERMANY);
    ASSERT_TRUE(ts::Operations::can_coup(state, ts::Player::USSR, ts::countries::WEST_GERMANY));

    // 5. Tear Down This Wall restores NATO for West Germany
    state.clear_flag(ts::effect_bits::NATO_CANCELED_WEST_GERMANY);
    ASSERT_FALSE(ts::Operations::can_coup(state, ts::Player::USSR, ts::countries::WEST_GERMANY));
}

TEST(StatesTest, GameState_US_Japan_Pact_BlocksUSSRCoupsAndRealignments) {
    ts::GameState state{};
    state.defcon = 5;
    state.countries[ts::countries::JAPAN].us_influence = 3;

    // 1. Without Pact: USSR can coup Japan at DEFCON 5
    ASSERT_TRUE(ts::Operations::can_coup(state, ts::Player::USSR, ts::countries::JAPAN));
    ASSERT_TRUE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::JAPAN));

    // 2. Set US_JAPAN_PACT_ACTIVE: USSR blocked
    state.set_flag(ts::effect_bits::US_JAPAN_PACT_ACTIVE);
    ASSERT_FALSE(ts::Operations::can_coup(state, ts::Player::USSR, ts::countries::JAPAN));
    ASSERT_FALSE(ts::Operations::can_realign(state, ts::Player::USSR, ts::countries::JAPAN));
}

TEST(StatesTest, GameState_IranContra_AppliesMinus1ToUSRealignment_AndResetsAtTurnEnd) {
    ts::GameState state{};
    for (uint8_t i = 1; i <= 110; ++i) {
        if (ts::CardData::is_scoring_card(i)) {
            state.card_locations[i] = ts::CardLocation::DISCARD_PILE;
        }
    }
    state.countries[ts::countries::ARGENTINA].ussr_influence = 2;

    // 1. Without Iran-Contra: US roll 4, USSR roll 4 -> Tie (0 removed)
    auto res1 = ts::Operations::execute_realign(state, ts::Player::US, ts::countries::ARGENTINA, 4, 4);
    ASSERT_EQ(res1.us_mod, 0);

    // 2. With Iran-Contra: US suffered -1 mod -> US total = 3 vs USSR total = 4
    state.set_flag(ts::effect_bits::IRAN_CONTRA_ACTIVE);
    auto res2 = ts::Operations::execute_realign(state, ts::Player::US, ts::countries::ARGENTINA, 4, 4);
    ASSERT_EQ(res2.us_mod, -1);

    // 3. End of turn clears Iran-Contra
    ts::StateMachine::end_turn(state);
    ASSERT_FALSE(state.has_flag(ts::effect_bits::IRAN_CONTRA_ACTIVE));
}

TEST(StatesTest, GameState_YuriAndSamantha_AwardsUSSR1VPPerUSCoup_AndResetsAtTurnEnd) {
    ts::GameState state{};
    for (uint8_t i = 1; i <= 110; ++i) {
        if (ts::CardData::is_scoring_card(i)) {
            state.card_locations[i] = ts::CardLocation::DISCARD_PILE;
        }
    }
    state.defcon = 5;
    state.countries[ts::countries::EGYPT].ussr_influence = 2;
    state.victory_points = 0;
    state.set_flag(ts::effect_bits::YURI_AND_SAMANTHA_ACTIVE);

    // US executes coup -> Yuri and Samantha gives +1 VP to USSR (-1 VP net)
    ts::Operations::execute_coup(state, ts::Player::US, ts::countries::EGYPT, 2, 3);
    ASSERT_EQ(state.victory_points, -1);

    // End of turn clears Yuri and Samantha
    ts::StateMachine::end_turn(state);
    ASSERT_FALSE(state.has_flag(ts::effect_bits::YURI_AND_SAMANTHA_ACTIVE));
}

TEST(StatesTest, GameState_VietnamRevolts_GrantsUSSRBonusOpsInSEAsia_AndResetsAtTurnEnd) {
    ts::GameState state{};
    for (uint8_t i = 1; i <= 110; ++i) {
        if (ts::CardData::is_scoring_card(i)) {
            state.card_locations[i] = ts::CardLocation::DISCARD_PILE;
        }
    }

    // 1. Normal Ops: 2 Ops card in Asia = 2 Ops
    uint8_t ops1 = ts::Operations::get_effective_ops(state, ts::card_ids::FIDEL, ts::Player::USSR, ts::Region::ASIA);
    ASSERT_EQ(ops1, 2);

    // 2. With Vietnam Revolts: 2 Ops card in Asia = 3 Ops
    state.set_flag(ts::effect_bits::VIETNAM_REVOLTS_ACTIVE);
    uint8_t ops2 = ts::Operations::get_effective_ops(state, ts::card_ids::FIDEL, ts::Player::USSR, ts::Region::ASIA);
    ASSERT_EQ(ops2, 3);

    // In Europe, remains 2 Ops
    uint8_t ops_europe = ts::Operations::get_effective_ops(state, ts::card_ids::FIDEL, ts::Player::USSR, ts::Region::EUROPE);
    ASSERT_EQ(ops_europe, 2);

    // 3. End of turn clears Vietnam Revolts
    ts::StateMachine::end_turn(state);
    ASSERT_FALSE(state.has_flag(ts::effect_bits::VIETNAM_REVOLTS_ACTIVE));
}

TEST(StatesTest, GameState_FormosanResolution_TaiwanBecomesBattleground_UntilChinaCardPlayed) {
    ts::GameState state{};
    state.countries[ts::countries::TAIWAN].us_influence = 3; // US controls Taiwan
    state.set_flag(ts::effect_bits::FORMOSAN_RESOLUTION_ACTIVE);

    // When Formosan Resolution is active and US controls Taiwan, Taiwan counts as BG in Asia
    auto res = ts::Scoring::evaluate_region(state, ts::Region::ASIA);
    ASSERT_EQ(res.us_battlegrounds, 1);

    // When US plays China Card, Formosan Resolution is cancelled
    state.clear_flag(ts::effect_bits::FORMOSAN_RESOLUTION_ACTIVE);
    auto res2 = ts::Scoring::evaluate_region(state, ts::Region::ASIA);
    ASSERT_EQ(res2.us_battlegrounds, 0); // Taiwan is non-BG again
}

TEST(StatesTest, GameState_ShuttleDiplomacy_SubtractsUSSRBattleground_AndClearsOnScoring) {
    ts::GameState state{};
    state.countries[ts::countries::EGYPT].ussr_influence = 2; // BG
    state.countries[ts::countries::ISRAEL].ussr_influence = 4; // BG
    state.set_flag(ts::effect_bits::SHUTTLE_DIPLOMACY_ACTIVE);

    // Shuttle diplomacy subtracts 1 USSR battleground on Middle East scoring
    auto summary = ts::Scoring::evaluate_region(state, ts::Region::MIDDLE_EAST);
    ASSERT_EQ(summary.ussr_battlegrounds, 2); // Base is 2

    // score_region clears the flag after scoring
    ts::Scoring::score_region(state, ts::Region::MIDDLE_EAST);
    ASSERT_FALSE(state.has_flag(ts::effect_bits::SHUTTLE_DIPLOMACY_ACTIVE));
}

TEST(StatesTest, SetupPhase_USBonusInfluencePlacement) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 12345);

    ASSERT_EQ(state.current_phase, ts::Phase::SETUP);
    ASSERT_EQ(state.ctx().decision_player, ts::Player::USSR);
    ASSERT_EQ(state.ctx().remaining_steps, 6);

    // USSR attempts illegal placement outside Eastern Europe (e.g. Canada)
    ts::MicroAction act_illegal{};
    act_illegal.decision_type = ts::DecisionType::POINT_NODE;
    act_illegal.primary_id = ts::countries::CANADA;
    ASSERT_FALSE(ts::StateMachine::step(state, act_illegal));

    // USSR legally places 3 in East Germany and 3 in Poland
    for (int i = 0; i < 3; ++i) {
        ts::MicroAction act{};
        act.decision_type = ts::DecisionType::POINT_NODE;
        act.primary_id = ts::countries::EAST_GERMANY;
        ASSERT_TRUE(ts::StateMachine::step(state, act));
    }
    for (int i = 0; i < 3; ++i) {
        ts::MicroAction act{};
        act.decision_type = ts::DecisionType::POINT_NODE;
        act.primary_id = ts::countries::POLAND;
        ASSERT_TRUE(ts::StateMachine::step(state, act));
    }

    // Now transitioned to US Stage 0 (7 in Western Europe)
    ASSERT_EQ(state.current_phase, ts::Phase::SETUP);
    ASSERT_EQ(state.ctx().decision_player, ts::Player::US);
    ASSERT_EQ(state.ctx().remaining_steps, 7);
    ASSERT_EQ(state.ctx().pending_ops_value, 0);

    // US attempts placement outside Western Europe (e.g. Iran) during Stage 0 -> rejected
    ts::MicroAction act_iran{};
    act_iran.decision_type = ts::DecisionType::POINT_NODE;
    act_iran.primary_id = ts::countries::IRAN;
    ASSERT_FALSE(ts::StateMachine::step(state, act_iran));

    // Verify ActionMask during Stage 0 allows Western Europe and forbids Iran
    uint8_t mask[84] = {0};
    size_t out_sz = 0; ts::ActionMask::generate_mask(state, mask, &out_sz);
    ASSERT_EQ(mask[ts::countries::WEST_GERMANY], 1);
    ASSERT_EQ(mask[ts::countries::ITALY], 1);
    ASSERT_EQ(mask[ts::countries::IRAN], 0);

    // US places 4 in West Germany and 3 in Italy
    for (int i = 0; i < 4; ++i) {
        ts::MicroAction act{};
        act.decision_type = ts::DecisionType::POINT_NODE;
        act.primary_id = ts::countries::WEST_GERMANY;
        ASSERT_TRUE(ts::StateMachine::step(state, act));
    }
    for (int i = 0; i < 3; ++i) {
        ts::MicroAction act{};
        act.decision_type = ts::DecisionType::POINT_NODE;
        act.primary_id = ts::countries::ITALY;
        ASSERT_TRUE(ts::StateMachine::step(state, act));
    }

    // Now transitioned to US Stage 1 (2 Bonus influence in countries with US presence)
    ASSERT_EQ(state.current_phase, ts::Phase::SETUP);
    ASSERT_EQ(state.ctx().decision_player, ts::Player::US);
    ASSERT_EQ(state.ctx().remaining_steps, 2);
    ASSERT_EQ(state.ctx().pending_ops_value, 1);

    // Verify ActionMask during Stage 1:
    // Countries with US influence (Iran = 1, West Germany = 4, Canada = 2) must be 1.
    // Countries without US influence (Iraq = 0, Chile = 0, Poland = 0) must be 0.
    std::fill(std::begin(mask), std::end(mask), 0);
    out_sz = 0; ts::ActionMask::generate_mask(state, mask, &out_sz);
    ASSERT_EQ(mask[ts::countries::IRAN], 1);
    ASSERT_EQ(mask[ts::countries::WEST_GERMANY], 1);
    ASSERT_EQ(mask[ts::countries::CANADA], 1);
    ASSERT_EQ(mask[ts::countries::IRAQ], 0);
    ASSERT_EQ(mask[ts::countries::CHILE], 0);
    ASSERT_EQ(mask[ts::countries::POLAND], 0);

    // US attempts illegal bonus placement in Iraq (0 US influence) -> rejected
    ts::MicroAction act_iraq{};
    act_iraq.decision_type = ts::DecisionType::POINT_NODE;
    act_iraq.primary_id = ts::countries::IRAQ;
    ASSERT_FALSE(ts::StateMachine::step(state, act_iraq));

    // US bonus point 1: Iran (boosts from 1 to 2)
    uint8_t iran_before = state.countries[ts::countries::IRAN].us_influence;
    ASSERT_EQ(iran_before, 1);
    ASSERT_TRUE(ts::StateMachine::step(state, act_iran));
    ASSERT_EQ(state.countries[ts::countries::IRAN].us_influence, 2);
    ASSERT_EQ(state.ctx().remaining_steps, 1);

    // US bonus point 2: West Germany (boosts from 4 to 5)
    uint8_t wg_before = state.countries[ts::countries::WEST_GERMANY].us_influence;
    ASSERT_EQ(wg_before, 4);
    ts::MicroAction act_wg{};
    act_wg.decision_type = ts::DecisionType::POINT_NODE;
    act_wg.primary_id = ts::countries::WEST_GERMANY;
    ASSERT_TRUE(ts::StateMachine::step(state, act_wg));
    ASSERT_EQ(state.countries[ts::countries::WEST_GERMANY].us_influence, 5);

    // Setup is completely finished -> Turn 1 begins in HEADLINE phase!
    ASSERT_EQ(state.current_phase, ts::Phase::HEADLINE);
    ASSERT_EQ(state.turn, 1);
}
