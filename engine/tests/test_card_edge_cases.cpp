#include "ts/action_mask.hpp"
#include "test_framework.hpp"
#include "ts/card_handlers.hpp"
#include "ts/card_data.hpp"
#include "ts/map_data.hpp"
#include "ts/ops.hpp"
#include "ts/scoring.hpp"
#include "ts/state_machine.hpp"
#include "ts/space_race.hpp"

namespace ts {

// 1. User Notices & Specific Requirements
TEST(CardEdgeCasesTest, Blockade_USCanDiscard_EvenIfWestGermanyEmpty) {
    GameState state{};
    // West Germany has 0 US influence
    state.countries[countries::WEST_GERMANY].us_influence = 0;
    state.card_locations[card_ids::DUCK_AND_COVER] = CardLocation::HAND_US; // 3 Ops card
    
    // USSR plays Blockade -> prompts US to discard
    bool done = CardHandlers::trigger_event(state, card_ids::BLOCKADE, Player::USSR);
    ASSERT_FALSE(done);
    ASSERT_EQ(state.ctx().decision_player, Player::US);
    ASSERT_EQ(state.ctx().decision_type, DecisionType::SELECT_CARD);

    // US discards Duck and Cover
    done = CardHandlers::handle_event_step(state, MicroAction{DecisionType::SELECT_CARD, card_ids::DUCK_AND_COVER, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.card_locations[card_ids::DUCK_AND_COVER], CardLocation::DISCARD_PILE);
    ASSERT_EQ(state.countries[countries::WEST_GERMANY].us_influence, 0);
}

TEST(CardEdgeCasesTest, OpsModifiers_RedScare_And_Containment_Or_Brezhnev_NetZero) {
    GameState state{};
    state.set_flag(effect_bits::CONTAINMENT_ACTIVE);
    state.set_flag(effect_bits::PURGE_US_ACTIVE);
    state.set_flag(effect_bits::BREZHNEV_DOCTRINE_ACTIVE);
    state.set_flag(effect_bits::PURGE_USSR_ACTIVE);

    // 1-op, 2-op, 3-op, 4-op cards should have exact base ops
    ASSERT_EQ(Operations::get_effective_ops(state, card_ids::BLOCKADE, Player::US), 1);
    ASSERT_EQ(Operations::get_effective_ops(state, card_ids::KOREAN_WAR, Player::US), 2);
    ASSERT_EQ(Operations::get_effective_ops(state, card_ids::DUCK_AND_COVER, Player::US), 3);
    ASSERT_EQ(Operations::get_effective_ops(state, card_ids::MARSHALL_PLAN, Player::US), 4);

    ASSERT_EQ(Operations::get_effective_ops(state, card_ids::BLOCKADE, Player::USSR), 1);
    ASSERT_EQ(Operations::get_effective_ops(state, card_ids::KOREAN_WAR, Player::USSR), 2);
    ASSERT_EQ(Operations::get_effective_ops(state, card_ids::DUCK_AND_COVER, Player::USSR), 3);
    ASSERT_EQ(Operations::get_effective_ops(state, card_ids::MARSHALL_PLAN, Player::USSR), 4);
}

TEST(CardEdgeCasesTest, OpsModifiers_Containment_And_Brezhnev_CapAt4Ops_ExceptChinaCardInAsia) {
    GameState state{};
    state.set_flag(effect_bits::CONTAINMENT_ACTIVE);
    state.set_flag(effect_bits::BREZHNEV_DOCTRINE_ACTIVE);

    // 1-op -> 2, 2-op -> 3, 3-op -> 4, 4-op -> 4 (capped)
    ASSERT_EQ(Operations::get_effective_ops(state, card_ids::BLOCKADE, Player::US), 2);
    ASSERT_EQ(Operations::get_effective_ops(state, card_ids::KOREAN_WAR, Player::US), 3);
    ASSERT_EQ(Operations::get_effective_ops(state, card_ids::DUCK_AND_COVER, Player::US), 4);
    ASSERT_EQ(Operations::get_effective_ops(state, card_ids::MARSHALL_PLAN, Player::US), 4); // Capped at 4

    ASSERT_EQ(Operations::get_effective_ops(state, card_ids::BLOCKADE, Player::USSR), 2);
    ASSERT_EQ(Operations::get_effective_ops(state, card_ids::KOREAN_WAR, Player::USSR), 3);
    ASSERT_EQ(Operations::get_effective_ops(state, card_ids::DUCK_AND_COVER, Player::USSR), 4);
    ASSERT_EQ(Operations::get_effective_ops(state, card_ids::MARSHALL_PLAN, Player::USSR), 4); // Capped at 4

    // China Card: 4 base, but 5 in Asia!
    ASSERT_EQ(Operations::get_effective_ops(state, card_ids::THE_CHINA_CARD, Player::US, Region::EUROPE), 4);
    ASSERT_EQ(Operations::get_effective_ops(state, card_ids::THE_CHINA_CARD, Player::US, Region::ASIA), 5);
}

TEST(CardEdgeCasesTest, ChinaCard_OpsModifiers_WithPurge_Containment_Brezhnev_InAsia) {
    // 1. Base China Card
    GameState s1{};
    ASSERT_EQ(Operations::get_effective_ops(s1, card_ids::THE_CHINA_CARD, Player::USSR, Region::EUROPE), 4);
    ASSERT_EQ(Operations::get_effective_ops(s1, card_ids::THE_CHINA_CARD, Player::USSR, Region::ASIA), 5);

    // 2. With Purge
    GameState s2{};
    s2.set_flag(effect_bits::PURGE_USSR_ACTIVE);
    ASSERT_EQ(Operations::get_effective_ops(s2, card_ids::THE_CHINA_CARD, Player::USSR, Region::EUROPE), 3);
    ASSERT_EQ(Operations::get_effective_ops(s2, card_ids::THE_CHINA_CARD, Player::USSR, Region::ASIA), 4);

    // 3. With Purge + Brezhnev
    GameState s3{};
    s3.set_flag(effect_bits::PURGE_USSR_ACTIVE);
    s3.set_flag(effect_bits::BREZHNEV_DOCTRINE_ACTIVE);
    ASSERT_EQ(Operations::get_effective_ops(s3, card_ids::THE_CHINA_CARD, Player::USSR, Region::EUROPE), 4);
    ASSERT_EQ(Operations::get_effective_ops(s3, card_ids::THE_CHINA_CARD, Player::USSR, Region::ASIA), 5);

    // 4. US with Purge + Containment
    GameState s4{};
    s4.set_flag(effect_bits::PURGE_US_ACTIVE);
    s4.set_flag(effect_bits::CONTAINMENT_ACTIVE);
    ASSERT_EQ(Operations::get_effective_ops(s4, card_ids::THE_CHINA_CARD, Player::US, Region::EUROPE), 4);
    ASSERT_EQ(Operations::get_effective_ops(s4, card_ids::THE_CHINA_CARD, Player::US, Region::ASIA), 5);
}

TEST(CardEdgeCasesTest, KitchenDebates_RemovedIfLeadInBGs_DiscardedIfNot) {
    // 1. US leads in battlegrounds -> +2 VP, REMOVED_FROM_GAME
    GameState s1{};
    s1.countries[countries::WEST_GERMANY].us_influence = 4; // US BG
    s1.countries[countries::PANAMA].us_influence = 3;       // US BG
    s1.victory_points = 0;
    CardHandlers::trigger_event(s1, card_ids::KITCHEN_DEBATES, Player::US);
    ASSERT_EQ(s1.victory_points, 2);
    ASSERT_EQ(s1.card_locations[card_ids::KITCHEN_DEBATES], CardLocation::REMOVED_FROM_GAME);

    // 2. US does not lead in battlegrounds -> 0 VP, DISCARD_PILE
    GameState s2{};
    s2.countries[countries::EAST_GERMANY].ussr_influence = 4; // USSR BG
    s2.countries[countries::POLAND].ussr_influence = 4;       // USSR BG
    s2.victory_points = 0;
    CardHandlers::trigger_event(s2, card_ids::KITCHEN_DEBATES, Player::US);
    ASSERT_EQ(s2.victory_points, 0);
    ASSERT_EQ(s2.card_locations[card_ids::KITCHEN_DEBATES], CardLocation::DISCARD_PILE);
}

TEST(CardEdgeCasesTest, MissileEnvy_OpponentHoldsOnlyScoringCards_NoTransfer) {
    GameState state{};
    // USSR plays Missile Envy, US holds only scoring cards
    state.card_locations[card_ids::ASIA_SCORING] = CardLocation::HAND_US;
    state.card_locations[card_ids::EUROPE_SCORING] = CardLocation::HAND_US;

    bool done = CardHandlers::trigger_event(state, card_ids::MISSILE_ENVY, Player::USSR);
    ASSERT_TRUE(done); // Clean no-op, no cards transferred
    ASSERT_EQ(state.card_locations[card_ids::ASIA_SCORING], CardLocation::HAND_US);
    ASSERT_EQ(state.card_locations[card_ids::EUROPE_SCORING], CardLocation::HAND_US);
}

TEST(CardEdgeCasesTest, WillyBrandt_CancelledByTearDownThisWall_AndBlocksSubsequentPlay) {
    GameState state{};
    ASSERT_TRUE(CardHandlers::can_trigger_event(state, card_ids::WILLY_BRANDT, Player::USSR));

    // Play Tear Down This Wall
    CardHandlers::trigger_event(state, card_ids::TEAR_DOWN_THIS_WALL, Player::US);
    ASSERT_TRUE(state.has_flag(effect_bits::TEAR_DOWN_THIS_WALL_PLAYED));

    // Willy Brandt cannot be triggered as event after Tear Down This Wall
    ASSERT_FALSE(CardHandlers::can_trigger_event(state, card_ids::WILLY_BRANDT, Player::USSR));
}

TEST(CardEdgeCasesTest, OrtegaElected_CanCoupCuba_AndAdjacentCountries) {
    GameState state{};
    state.countries[countries::NICARAGUA].us_influence = 3;
    state.countries[countries::CUBA].us_influence = 3; // Cuba is adjacent to Nicaragua
    state.countries[countries::CUBA].ussr_influence = 0;

    bool done = CardHandlers::trigger_event(state, card_ids::ORTEGA_ELECTED_IN_NICARAGUA, Player::USSR);
    ASSERT_FALSE(done);
    ASSERT_EQ(state.countries[countries::NICARAGUA].us_influence, 0);
    ASSERT_EQ(state.ctx().decision_player, Player::USSR);
    ASSERT_EQ(state.ctx().decision_type, DecisionType::POINT_NODE);

    // Action mask allows Cuba (71), Costa Rica (68), Honduras (67)
    uint8_t mask[84];
    size_t out_size = 0;
    CardHandlers::get_event_action_mask(state, mask, &out_size);
    ASSERT_EQ(mask[countries::CUBA], 1);
    ASSERT_EQ(mask[countries::COSTA_RICA], 1);
    ASSERT_EQ(mask[countries::HONDURAS], 1);
    ASSERT_EQ(mask[countries::FRANCE], 0);

    // Coup Cuba with roll 6: roll 6 + 2 ops - 2 * 3 (stability) = 2 coup value
    done = CardHandlers::handle_event_step(state, MicroAction{DecisionType::POINT_NODE, countries::CUBA, 6, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[countries::CUBA].us_influence, 1); // 3 - 2 = 1
    ASSERT_EQ(state.ussr_mil_ops, 2);
}

TEST(CardEdgeCasesTest, AldrichAmes_WithSingleCardInUSHand) {
    GameState state{};
    state.card_locations[card_ids::DUCK_AND_COVER] = CardLocation::HAND_US;

    bool done = CardHandlers::trigger_event(state, card_ids::ALDRICH_AMES, Player::USSR);
    ASSERT_FALSE(done);
    ASSERT_EQ(state.ctx().decision_player, Player::USSR);
    ASSERT_EQ(state.ctx().decision_type, DecisionType::SELECT_CARD);

    // Action mask allows selecting the single card
    uint8_t mask[112];
    size_t out_size = 0;
    CardHandlers::get_event_action_mask(state, mask, &out_size);
    ASSERT_EQ(mask[card_ids::DUCK_AND_COVER], 1);

    // USSR discards Duck and Cover
    done = CardHandlers::handle_event_step(state, MicroAction{DecisionType::SELECT_CARD, card_ids::DUCK_AND_COVER, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.card_locations[card_ids::DUCK_AND_COVER], CardLocation::DISCARD_PILE);
}

TEST(CardEdgeCasesTest, Reshuffle_ImmediateWhenDrawDeckEmpty) {
    GameState state{};
    StateMachine::init_new_game(state, 42);

    // Clear US hand to 0 cards
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == CardLocation::HAND_US) {
            state.card_locations[i] = CardLocation::DISCARD_PILE;
        }
    }

    // Drain draw deck so exactly 1 card remains
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == CardLocation::DRAW_DECK) {
            state.card_locations[i] = CardLocation::DISCARD_PILE;
        }
    }
    state.card_locations[card_ids::FIDEL] = CardLocation::DRAW_DECK;
    state.card_locations[card_ids::DUCK_AND_COVER] = CardLocation::DISCARD_PILE;
    state.card_locations[card_ids::FIVE_YEAR_PLAN] = CardLocation::DISCARD_PILE;
    
    // Dealing cards to US draws Fidel (last card), triggering immediate reshuffle of discard pile into draw deck
    StateMachine::deal_cards_to_hands(state);

    // Discard pile cards should now be in draw deck / hands, not stuck in discard pile
    ASSERT_EQ(state.card_locations[card_ids::FIDEL], CardLocation::HAND_US);
    ASSERT_NE(state.card_locations[card_ids::DUCK_AND_COVER], CardLocation::DISCARD_PILE);
    ASSERT_EQ(state.card_locations[card_ids::DUCK_AND_COVER], CardLocation::DRAW_DECK);
}

// 2. War Cards Roll-Failure Tests
TEST(CardEdgeCasesTest, KoreanWar_FailedRoll_AwardsMilOps_NoVPOrInfluence) {
    GameState state{};
    state.countries[countries::SOUTH_KOREA].us_influence = 2;
    state.countries[countries::SOUTH_KOREA].ussr_influence = 0;
    state.victory_points = 0;
    state.ussr_mil_ops = 0;

    // Trigger with forced roll 2 (failure: 2 <= 3)
    CardHandlers::trigger_event(state, card_ids::KOREAN_WAR, Player::USSR, 2);
    ASSERT_EQ(state.ussr_mil_ops, 2); // Mil Ops awarded
    ASSERT_EQ(state.victory_points, 0); // 0 VP
    ASSERT_EQ(state.countries[countries::SOUTH_KOREA].us_influence, 2); // Influence untouched
    ASSERT_EQ(state.countries[countries::SOUTH_KOREA].ussr_influence, 0);
}

TEST(CardEdgeCasesTest, ArabIsraeliWar_FailedRoll_WithAdjacentModifiers) {
    GameState state{};
    state.countries[countries::ISRAEL].us_influence = 2; // -1 modifier (controlled)
    state.countries[countries::EGYPT].us_influence = 2;  // -1 modifier (controlled)
    state.countries[countries::JORDAN].us_influence = 2; // -1 modifier (controlled)
    state.victory_points = 0;
    state.ussr_mil_ops = 0;

    // Total modifier is -3. Forced roll 5: 5 - 3 = 2 <= 3 (failure)
    CardHandlers::trigger_event(state, card_ids::ARAB_ISRAELI_WAR, Player::USSR, 5);
    ASSERT_EQ(state.ussr_mil_ops, 2);
    ASSERT_EQ(state.victory_points, 0);
    ASSERT_EQ(state.countries[countries::ISRAEL].us_influence, 2);
}

TEST(CardEdgeCasesTest, IndoPakistaniWar_FailedRoll_And_MilOps) {
    GameState state{};
    state.countries[countries::INDIA].us_influence = 3;
    state.countries[countries::INDIA].ussr_influence = 0;
    state.victory_points = 0;
    state.ussr_mil_ops = 0;

    bool done = CardHandlers::trigger_event(state, card_ids::INDO_PAKISTANI_WAR, Player::USSR);
    ASSERT_FALSE(done);
    ASSERT_EQ(state.ctx().decision_type, DecisionType::POINT_NODE);

    // Choose to invade India with forced roll 2
    done = CardHandlers::handle_event_step(state, MicroAction{DecisionType::POINT_NODE, countries::INDIA, 2, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.ussr_mil_ops, 2);
    ASSERT_EQ(state.victory_points, 0);
    ASSERT_EQ(state.countries[countries::INDIA].us_influence, 3);
}

TEST(CardEdgeCasesTest, BrushWar_FailedRoll_And_StabilityRestriction) {
    GameState state{};
    state.countries[countries::ZAIRE].us_influence = 2; // Stability 1
    state.victory_points = 0;
    state.us_mil_ops = 0;

    bool done = CardHandlers::trigger_event(state, card_ids::BRUSH_WAR, Player::US);
    ASSERT_FALSE(done);
    
    // Action mask should disallow stability 3+ countries (e.g. West Germany)
    uint8_t mask[84];
    size_t out_size = 0;
    CardHandlers::get_event_action_mask(state, mask, &out_size);
    ASSERT_EQ(mask[countries::ZAIRE], 1);
    ASSERT_EQ(mask[countries::WEST_GERMANY], 0);

    // Invade Zaire with forced roll 2 (failure)
    done = CardHandlers::handle_event_step(state, MicroAction{DecisionType::POINT_NODE, countries::ZAIRE, 2, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.us_mil_ops, 3); // +3 Mil Ops awarded
    ASSERT_EQ(state.victory_points, 0);
    ASSERT_EQ(state.countries[countries::ZAIRE].us_influence, 2);
}

TEST(CardEdgeCasesTest, IranIraqWar_FailedRoll_And_MilOps) {
    GameState state{};
    state.countries[countries::IRAQ].us_influence = 2;
    state.victory_points = 0;
    state.ussr_mil_ops = 0;

    bool done = CardHandlers::trigger_event(state, card_ids::IRAN_IRAQ_WAR, Player::USSR);
    ASSERT_FALSE(done);

    // Invade Iraq with forced roll 1
    done = CardHandlers::handle_event_step(state, MicroAction{DecisionType::POINT_NODE, countries::IRAQ, 1, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.ussr_mil_ops, 2);
    ASSERT_EQ(state.victory_points, 0);
    ASSERT_EQ(state.countries[countries::IRAQ].us_influence, 2);
}

// 3. DEFCON 2 Boundaries & Suicide Invariants
TEST(CardEdgeCasesTest, DuckAndCover_AtDefcon2_PlayedByUS_USLoses) {
    GameState state{};
    state.defcon = 2;
    state.phasing_player = Player::US;
    CardHandlers::trigger_event(state, card_ids::DUCK_AND_COVER, Player::US);
    ASSERT_EQ(state.defcon, 1);
    ASSERT_EQ(state.current_phase, Phase::GAME_OVER);
    ASSERT_EQ(state.victory_points, -20); // USSR wins!
}

TEST(CardEdgeCasesTest, DuckAndCover_AtDefcon2_PlayedByUSSRForOps_USSRLoses) {
    GameState state{};
    state.defcon = 2;
    state.phasing_player = Player::USSR; // USSR is phasing player playing opponent card
    CardHandlers::trigger_event(state, card_ids::DUCK_AND_COVER, Player::US);
    ASSERT_EQ(state.defcon, 1);
    ASSERT_EQ(state.current_phase, Phase::GAME_OVER);
    ASSERT_EQ(state.victory_points, 20); // US wins!
}

TEST(CardEdgeCasesTest, OlympicGames_BoycottAtDefcon2_CausesDefconSuicide) {
    GameState state{};
    state.defcon = 2;
    state.phasing_player = Player::US; // US sponsored Olympics

    // USSR boycotts -> degrades DEFCON to 1 -> phasing player (US) loses!
    bool done = CardHandlers::trigger_event(state, card_ids::OLYMPIC_GAMES, Player::US);
    ASSERT_FALSE(done);

    done = CardHandlers::handle_event_step(state, MicroAction{DecisionType::CHOOSE_BRANCH, 1, 0, 0}); // Branch 1: Boycott
    ASSERT_TRUE(done);
    ASSERT_EQ(state.defcon, 1);
    ASSERT_EQ(state.current_phase, Phase::GAME_OVER);
    ASSERT_EQ(state.victory_points, -20); // USSR wins!
}

TEST(CardEdgeCasesTest, HowILearnedToStopWorrying_SetTo1_CausesDefconSuicide) {
    GameState state{};
    state.defcon = 3;
    state.phasing_player = Player::US;

    bool done = CardHandlers::trigger_event(state, card_ids::HOW_I_LEARNED_TO_STOP_WORRYING, Player::US);
    ASSERT_FALSE(done);

    // Set DEFCON to 1
    done = CardHandlers::handle_event_step(state, MicroAction{DecisionType::CHOOSE_BRANCH, 1, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.defcon, 1);
    ASSERT_EQ(state.current_phase, Phase::GAME_OVER);
    ASSERT_EQ(state.victory_points, -20); // USSR wins!
}

TEST(CardEdgeCasesTest, SovietsShootDownKAL007_AtDefcon2_CausesDefconSuicide) {
    GameState state{};
    state.defcon = 2;
    state.phasing_player = Player::US;

    CardHandlers::trigger_event(state, card_ids::SOVIETS_SHOOT_DOWN_KAL_007, Player::US);
    ASSERT_EQ(state.defcon, 1);
    ASSERT_EQ(state.current_phase, Phase::GAME_OVER);
    ASSERT_EQ(state.victory_points, -20); // USSR wins!
}

TEST(CardEdgeCasesTest, Summit_DefconSuicide_WhenDegradedTo1) {
    GameState state{};
    state.defcon = 2;
    state.phasing_player = Player::US;

    // US wins summit with forced high roll
    bool done = CardHandlers::trigger_event(state, card_ids::SUMMIT, Player::US);
    ASSERT_FALSE(done);

    // US chooses to degrade DEFCON by 1 -> DEFCON 1 -> Game Over
    done = CardHandlers::handle_event_step(state, MicroAction{DecisionType::CHOOSE_BRANCH, 1, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.defcon, 1);
    ASSERT_EQ(state.current_phase, Phase::GAME_OVER);
    ASSERT_EQ(state.victory_points, -20); // USSR wins!
}

// 4. Multi-Step Placement & Removal Boundary Constraints
TEST(CardEdgeCasesTest, SocialistGovernments_WesternEurope_FewerThan3Influence) {
    GameState state{};
    // Only 1 US influence exists in Western Europe (France)
    state.countries[countries::FRANCE].us_influence = 1;

    bool done = CardHandlers::trigger_event(state, card_ids::SOCIALIST_GOVERNMENTS, Player::USSR);
    ASSERT_FALSE(done);

    done = CardHandlers::handle_event_step(state, MicroAction{DecisionType::POINT_NODE, countries::FRANCE, 0, 0});
    ASSERT_FALSE(done);
    ASSERT_EQ(state.countries[countries::FRANCE].us_influence, 0);

    // Confirm done / early stop since no more US influence in Western Europe
    MicroAction done_act{};
    done_act.flags = action_flags::CONFIRM_DONE;
    done = CardHandlers::handle_event_step(state, done_act);
    ASSERT_TRUE(done);
}

TEST(CardEdgeCasesTest, SuezCrisis_FewerThan4InfluenceTotal) {
    GameState state{};
    // UK has 1, France has 1, Israel has 0 -> total 2 US influence
    state.countries[countries::UNITED_KINGDOM].us_influence = 1;
    state.countries[countries::FRANCE].us_influence = 1;
    state.countries[countries::ISRAEL].us_influence = 0;

    bool done = CardHandlers::trigger_event(state, card_ids::SUEZ_CRISIS, Player::USSR);
    ASSERT_FALSE(done);

    CardHandlers::handle_event_step(state, MicroAction{DecisionType::POINT_NODE, countries::UNITED_KINGDOM, 0, 0});
    CardHandlers::handle_event_step(state, MicroAction{DecisionType::POINT_NODE, countries::FRANCE, 0, 0});
    
    // Confirm done / early stop since no more US influence in valid countries
    MicroAction done_act{};
    done_act.flags = action_flags::CONFIRM_DONE;
    done = CardHandlers::handle_event_step(state, done_act);
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[countries::UNITED_KINGDOM].us_influence, 0);
    ASSERT_EQ(state.countries[countries::FRANCE].us_influence, 0);
}

TEST(CardEdgeCasesTest, TrumanDoctrine_RestrictsToUncontrolledEuropeanCountries) {
    GameState state{};
    state.countries[countries::POLAND].ussr_influence = 3; // Controlled by USSR
    state.countries[countries::FINLAND].ussr_influence = 1; // Uncontrolled
    state.countries[countries::FRANCE].us_influence = 3;   // Controlled by US

    bool done = CardHandlers::trigger_event(state, card_ids::TRUMAN_DOCTRINE, Player::US);
    ASSERT_FALSE(done);

    uint8_t mask[84];
    size_t out_size = 0;
    CardHandlers::get_event_action_mask(state, mask, &out_size);
    ASSERT_EQ(mask[countries::FINLAND], 1);
    ASSERT_EQ(mask[countries::POLAND], 0);
    ASSERT_EQ(mask[countries::FRANCE], 0);

    done = CardHandlers::handle_event_step(state, MicroAction{DecisionType::POINT_NODE, countries::FINLAND, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[countries::FINLAND].ussr_influence, 0);
}

TEST(CardEdgeCasesTest, VoiceOfAmerica_RestrictsToNonEuropeanCountries) {
    GameState state{};
    state.countries[countries::EAST_GERMANY].ussr_influence = 3; // Europe
    state.countries[countries::CUBA].ussr_influence = 3;         // Non-Europe

    bool done = CardHandlers::trigger_event(state, card_ids::THE_VOICE_OF_AMERICA, Player::US);
    ASSERT_FALSE(done);

    uint8_t mask[84];
    size_t out_size = 0;
    CardHandlers::get_event_action_mask(state, mask, &out_size);
    ASSERT_EQ(mask[countries::CUBA], 1);
    ASSERT_EQ(mask[countries::EAST_GERMANY], 0);
}

TEST(CardEdgeCasesTest, Che_SecondCoupGrantedOnlyOnSuccess) {
    // 1. First coup succeeds in removing US influence -> second coup offered
    GameState s1{};
    s1.countries[countries::COLOMBIA].us_influence = 2;
    s1.countries[countries::COLOMBIA].ussr_influence = 0;
    s1.countries[countries::PERU].us_influence = 2;

    bool done = CardHandlers::trigger_event(s1, card_ids::CHE, Player::USSR);
    ASSERT_FALSE(done);

    // First coup in Colombia with forced roll 6: 6 + 3 - 2 * 1 = 7 (removes 2 US, adds 5 USSR)
    done = CardHandlers::handle_event_step(s1, MicroAction{DecisionType::POINT_NODE, countries::COLOMBIA, 6, 0});
    ASSERT_TRUE(done); // First step completes
    ASSERT_EQ(s1.countries[countries::COLOMBIA].us_influence, 0);
    ASSERT_EQ(s1.countries[countries::COLOMBIA].ussr_influence, 5);
}

TEST(CardEdgeCasesTest, SpecialRelationship_UKUncontrolled_NoEffect) {
    GameState state{};
    state.countries[countries::UNITED_KINGDOM].us_influence = 0; // UK uncontrolled
    state.victory_points = 0;

    bool done = CardHandlers::trigger_event(state, card_ids::SPECIAL_RELATIONSHIP, Player::US);
    ASSERT_TRUE(done); // Immediate no-op
    ASSERT_EQ(state.victory_points, 0);
}


TEST(CardEdgeCasesTest, OlympicGames_Boycott_OpsModifiers_Suite) {
    // 1. Default Boycott (4 Ops)
    {
        GameState s{};
        s.defcon = 4;
        CardHandlers::trigger_event(s, card_ids::OLYMPIC_GAMES, Player::US);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::CHOOSE_BRANCH, 1, 0, 0}); // USSR boycotts
        ASSERT_EQ(s.victory_points, 2);
        ASSERT_EQ(s.defcon, 3);
        ASSERT_EQ(s.ctx().pending_ops_value, 4);
    }
    {
        GameState s{};
        s.defcon = 4;
        CardHandlers::trigger_event(s, card_ids::OLYMPIC_GAMES, Player::USSR);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::CHOOSE_BRANCH, 1, 0, 0}); // US boycotts
        ASSERT_EQ(s.victory_points, -2);
        ASSERT_EQ(s.defcon, 3);
        ASSERT_EQ(s.ctx().pending_ops_value, 4);
    }

    // 2. Boycott under Red Scare / Purge (3 Ops)
    {
        GameState s{};
        s.defcon = 4;
        s.set_flag(effect_bits::PURGE_US_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::OLYMPIC_GAMES, Player::US);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::CHOOSE_BRANCH, 1, 0, 0});
        ASSERT_EQ(s.ctx().pending_ops_value, 3);
    }
    {
        GameState s{};
        s.defcon = 4;
        s.set_flag(effect_bits::PURGE_USSR_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::OLYMPIC_GAMES, Player::USSR);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::CHOOSE_BRANCH, 1, 0, 0});
        ASSERT_EQ(s.ctx().pending_ops_value, 3);
    }

    // 3. Boycott under Containment / Brezhnev Doctrine (4 Ops capped)
    {
        GameState s{};
        s.defcon = 4;
        s.set_flag(effect_bits::CONTAINMENT_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::OLYMPIC_GAMES, Player::US);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::CHOOSE_BRANCH, 1, 0, 0});
        ASSERT_EQ(s.ctx().pending_ops_value, 4);
    }
    {
        GameState s{};
        s.defcon = 4;
        s.set_flag(effect_bits::BREZHNEV_DOCTRINE_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::OLYMPIC_GAMES, Player::USSR);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::CHOOSE_BRANCH, 1, 0, 0});
        ASSERT_EQ(s.ctx().pending_ops_value, 4);
    }

    // 4. Boycott under Red Scare + Containment / Brezhnev Doctrine (4 Ops net zero)
    {
        GameState s{};
        s.defcon = 4;
        s.set_flag(effect_bits::PURGE_US_ACTIVE);
        s.set_flag(effect_bits::CONTAINMENT_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::OLYMPIC_GAMES, Player::US);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::CHOOSE_BRANCH, 1, 0, 0});
        ASSERT_EQ(s.ctx().pending_ops_value, 4);
    }
    {
        GameState s{};
        s.defcon = 4;
        s.set_flag(effect_bits::PURGE_USSR_ACTIVE);
        s.set_flag(effect_bits::BREZHNEV_DOCTRINE_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::OLYMPIC_GAMES, Player::USSR);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::CHOOSE_BRANCH, 1, 0, 0});
        ASSERT_EQ(s.ctx().pending_ops_value, 4);
    }
}


// Tests for all cards allowing operations under Red Scare / Containment / Brezhnev Doctrine
TEST(CardEdgeCasesTest, CIACreated_OpsModifiers_Suite) {
    // 1. Default (1 Op)
    {
        GameState s{};
        CardHandlers::trigger_event(s, card_ids::CIA_CREATED, Player::US);
        ASSERT_EQ(s.ctx().pending_ops_value, 1);
    }
    // 2. Containment (2 Ops)
    {
        GameState s{};
        s.set_flag(effect_bits::CONTAINMENT_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::CIA_CREATED, Player::US);
        ASSERT_EQ(s.ctx().pending_ops_value, 2);
    }
    // 3. Purge (1 Op - min 1)
    {
        GameState s{};
        s.set_flag(effect_bits::PURGE_US_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::CIA_CREATED, Player::US);
        ASSERT_EQ(s.ctx().pending_ops_value, 1);
    }
    // 4. Containment + Purge (1 Op - net zero)
    {
        GameState s{};
        s.set_flag(effect_bits::CONTAINMENT_ACTIVE);
        s.set_flag(effect_bits::PURGE_US_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::CIA_CREATED, Player::US);
        ASSERT_EQ(s.ctx().pending_ops_value, 1);
    }
}

TEST(CardEdgeCasesTest, LoneGunman_OpsModifiers_Suite) {
    // 1. Default (1 Op)
    {
        GameState s{};
        CardHandlers::trigger_event(s, card_ids::LONE_GUNMAN, Player::USSR);
        ASSERT_EQ(s.ctx().pending_ops_value, 1);
    }
    // 2. Brezhnev (2 Ops)
    {
        GameState s{};
        s.set_flag(effect_bits::BREZHNEV_DOCTRINE_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::LONE_GUNMAN, Player::USSR);
        ASSERT_EQ(s.ctx().pending_ops_value, 2);
    }
    // 3. Purge (1 Op - min 1)
    {
        GameState s{};
        s.set_flag(effect_bits::PURGE_USSR_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::LONE_GUNMAN, Player::USSR);
        ASSERT_EQ(s.ctx().pending_ops_value, 1);
    }
    // 4. Brezhnev + Purge (1 Op - net zero)
    {
        GameState s{};
        s.set_flag(effect_bits::BREZHNEV_DOCTRINE_ACTIVE);
        s.set_flag(effect_bits::PURGE_USSR_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::LONE_GUNMAN, Player::USSR);
        ASSERT_EQ(s.ctx().pending_ops_value, 1);
    }
}

TEST(CardEdgeCasesTest, Junta_OpsModifiers_Suite) {
    // US play: 2 base ops
    {
        GameState s{};
        CardHandlers::trigger_event(s, card_ids::JUNTA, Player::US);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::POINT_NODE, countries::CHILE, 0, 0});
        ASSERT_EQ(s.ctx().pending_ops_value, 2);
    }
    {
        GameState s{};
        s.set_flag(effect_bits::CONTAINMENT_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::JUNTA, Player::US);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::POINT_NODE, countries::CHILE, 0, 0});
        ASSERT_EQ(s.ctx().pending_ops_value, 3);
    }
    {
        GameState s{};
        s.set_flag(effect_bits::PURGE_US_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::JUNTA, Player::US);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::POINT_NODE, countries::CHILE, 0, 0});
        ASSERT_EQ(s.ctx().pending_ops_value, 1);
    }
    {
        GameState s{};
        s.set_flag(effect_bits::CONTAINMENT_ACTIVE);
        s.set_flag(effect_bits::PURGE_US_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::JUNTA, Player::US);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::POINT_NODE, countries::CHILE, 0, 0});
        ASSERT_EQ(s.ctx().pending_ops_value, 2);
    }

    // USSR play: 2 base ops
    {
        GameState s{};
        CardHandlers::trigger_event(s, card_ids::JUNTA, Player::USSR);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::POINT_NODE, countries::CHILE, 0, 0});
        ASSERT_EQ(s.ctx().pending_ops_value, 2);
    }
    {
        GameState s{};
        s.set_flag(effect_bits::BREZHNEV_DOCTRINE_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::JUNTA, Player::USSR);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::POINT_NODE, countries::CHILE, 0, 0});
        ASSERT_EQ(s.ctx().pending_ops_value, 3);
    }
    {
        GameState s{};
        s.set_flag(effect_bits::PURGE_USSR_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::JUNTA, Player::USSR);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::POINT_NODE, countries::CHILE, 0, 0});
        ASSERT_EQ(s.ctx().pending_ops_value, 1);
    }
    {
        GameState s{};
        s.set_flag(effect_bits::BREZHNEV_DOCTRINE_ACTIVE);
        s.set_flag(effect_bits::PURGE_USSR_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::JUNTA, Player::USSR);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::POINT_NODE, countries::CHILE, 0, 0});
        ASSERT_EQ(s.ctx().pending_ops_value, 2);
    }
}

TEST(CardEdgeCasesTest, ABMTreaty_OpsModifiers_Suite) {
    // US play
    {
        GameState s{};
        CardHandlers::trigger_event(s, card_ids::ABM_TREATY, Player::US);
        ASSERT_EQ(s.ctx().pending_ops_value, 4);
    }
    {
        GameState s{};
        s.set_flag(effect_bits::CONTAINMENT_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::ABM_TREATY, Player::US);
        ASSERT_EQ(s.ctx().pending_ops_value, 4); // Capped at 4
    }
    {
        GameState s{};
        s.set_flag(effect_bits::PURGE_US_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::ABM_TREATY, Player::US);
        ASSERT_EQ(s.ctx().pending_ops_value, 3);
    }
    {
        GameState s{};
        s.set_flag(effect_bits::CONTAINMENT_ACTIVE);
        s.set_flag(effect_bits::PURGE_US_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::ABM_TREATY, Player::US);
        ASSERT_EQ(s.ctx().pending_ops_value, 4);
    }

    // USSR play
    {
        GameState s{};
        CardHandlers::trigger_event(s, card_ids::ABM_TREATY, Player::USSR);
        ASSERT_EQ(s.ctx().pending_ops_value, 4);
    }
    {
        GameState s{};
        s.set_flag(effect_bits::BREZHNEV_DOCTRINE_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::ABM_TREATY, Player::USSR);
        ASSERT_EQ(s.ctx().pending_ops_value, 4); // Capped at 4
    }
    {
        GameState s{};
        s.set_flag(effect_bits::PURGE_USSR_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::ABM_TREATY, Player::USSR);
        ASSERT_EQ(s.ctx().pending_ops_value, 3);
    }
    {
        GameState s{};
        s.set_flag(effect_bits::BREZHNEV_DOCTRINE_ACTIVE);
        s.set_flag(effect_bits::PURGE_USSR_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::ABM_TREATY, Player::USSR);
        ASSERT_EQ(s.ctx().pending_ops_value, 4);
    }
}

TEST(CardEdgeCasesTest, Glasnost_OpsModifiers_Suite) {
    // Glasnost grants 4 Ops to USSR if The Reformer is played
    {
        GameState s{};
        s.set_flag(effect_bits::THE_REFORMER_PLAYED);
        CardHandlers::trigger_event(s, card_ids::GLASNOST, Player::USSR);
        ASSERT_EQ(s.ctx().pending_ops_value, 4);
    }
    {
        GameState s{};
        s.set_flag(effect_bits::THE_REFORMER_PLAYED);
        s.set_flag(effect_bits::BREZHNEV_DOCTRINE_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::GLASNOST, Player::USSR);
        ASSERT_EQ(s.ctx().pending_ops_value, 4); // Capped at 4
    }
    {
        GameState s{};
        s.set_flag(effect_bits::THE_REFORMER_PLAYED);
        s.set_flag(effect_bits::PURGE_USSR_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::GLASNOST, Player::USSR);
        ASSERT_EQ(s.ctx().pending_ops_value, 3);
    }
    {
        GameState s{};
        s.set_flag(effect_bits::THE_REFORMER_PLAYED);
        s.set_flag(effect_bits::BREZHNEV_DOCTRINE_ACTIVE);
        s.set_flag(effect_bits::PURGE_USSR_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::GLASNOST, Player::USSR);
        ASSERT_EQ(s.ctx().pending_ops_value, 4);
    }
}

TEST(CardEdgeCasesTest, SovietsShootDownKAL007_OpsModifiers_Suite) {
    // KAL-007 grants 4 Ops to US if South Korea is US controlled
    {
        GameState s{};
        s.defcon = 4;
        s.countries[countries::SOUTH_KOREA].us_influence = 3; // Controlled
        CardHandlers::trigger_event(s, card_ids::SOVIETS_SHOOT_DOWN_KAL_007, Player::US);
        ASSERT_EQ(s.ctx().pending_ops_value, 4);
    }
    {
        GameState s{};
        s.defcon = 4;
        s.countries[countries::SOUTH_KOREA].us_influence = 3;
        s.set_flag(effect_bits::CONTAINMENT_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::SOVIETS_SHOOT_DOWN_KAL_007, Player::US);
        ASSERT_EQ(s.ctx().pending_ops_value, 4); // Capped at 4
    }
    {
        GameState s{};
        s.defcon = 4;
        s.countries[countries::SOUTH_KOREA].us_influence = 3;
        s.set_flag(effect_bits::PURGE_US_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::SOVIETS_SHOOT_DOWN_KAL_007, Player::US);
        ASSERT_EQ(s.ctx().pending_ops_value, 3);
    }
    {
        GameState s{};
        s.defcon = 4;
        s.countries[countries::SOUTH_KOREA].us_influence = 3;
        s.set_flag(effect_bits::CONTAINMENT_ACTIVE);
        s.set_flag(effect_bits::PURGE_US_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::SOVIETS_SHOOT_DOWN_KAL_007, Player::US);
        ASSERT_EQ(s.ctx().pending_ops_value, 4);
    }
}

TEST(CardEdgeCasesTest, Che_OpsModifiers_Suite) {
    // Che performs coup using modified 3 base ops
    // Country: Colombia (stability 1). Roll 1:
    // Default (3 ops): 1 + 3 - 2 = 2 coup val (removes 2 US)
    {
        GameState s{};
        s.countries[countries::COLOMBIA].us_influence = 3;
        CardHandlers::trigger_event(s, card_ids::CHE, Player::USSR);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::POINT_NODE, countries::COLOMBIA, 1, 0});
        ASSERT_EQ(s.countries[countries::COLOMBIA].us_influence, 1); // 3 - 2 = 1
    }
    // Brezhnev (4 ops): 1 + 4 - 2 = 3 coup val (removes 3 US)
    {
        GameState s{};
        s.set_flag(effect_bits::BREZHNEV_DOCTRINE_ACTIVE);
        s.countries[countries::COLOMBIA].us_influence = 3;
        CardHandlers::trigger_event(s, card_ids::CHE, Player::USSR);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::POINT_NODE, countries::COLOMBIA, 1, 0});
        ASSERT_EQ(s.countries[countries::COLOMBIA].us_influence, 0); // 3 - 3 = 0
    }
    // Purge (2 ops): 1 + 2 - 2 = 1 coup val (removes 1 US)
    {
        GameState s{};
        s.set_flag(effect_bits::PURGE_USSR_ACTIVE);
        s.countries[countries::COLOMBIA].us_influence = 3;
        CardHandlers::trigger_event(s, card_ids::CHE, Player::USSR);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::POINT_NODE, countries::COLOMBIA, 1, 0});
        ASSERT_EQ(s.countries[countries::COLOMBIA].us_influence, 2); // 3 - 1 = 2
    }
    // Brezhnev + Purge (3 ops): 1 + 3 - 2 = 2 coup val (removes 2 US)
    {
        GameState s{};
        s.set_flag(effect_bits::BREZHNEV_DOCTRINE_ACTIVE);
        s.set_flag(effect_bits::PURGE_USSR_ACTIVE);
        s.countries[countries::COLOMBIA].us_influence = 3;
        CardHandlers::trigger_event(s, card_ids::CHE, Player::USSR);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::POINT_NODE, countries::COLOMBIA, 1, 0});
        ASSERT_EQ(s.countries[countries::COLOMBIA].us_influence, 1); // 3 - 2 = 1
    }
}

TEST(CardEdgeCasesTest, OrtegaElected_OpsModifiers_Suite) {
    // Ortega performs coup using modified 2 base ops
    // Country: Honduras (stability 2). Roll 4:
    // Default (2 ops): 4 + 2 - 4 = 2 coup val (removes 2 US)
    {
        GameState s{};
        s.countries[countries::HONDURAS].us_influence = 3;
        CardHandlers::trigger_event(s, card_ids::ORTEGA_ELECTED_IN_NICARAGUA, Player::USSR);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::POINT_NODE, countries::HONDURAS, 4, 0});
        ASSERT_EQ(s.countries[countries::HONDURAS].us_influence, 1); // 3 - 2 = 1
    }
    // Brezhnev (3 ops): 4 + 3 - 4 = 3 coup val (removes 3 US)
    {
        GameState s{};
        s.set_flag(effect_bits::BREZHNEV_DOCTRINE_ACTIVE);
        s.countries[countries::HONDURAS].us_influence = 3;
        CardHandlers::trigger_event(s, card_ids::ORTEGA_ELECTED_IN_NICARAGUA, Player::USSR);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::POINT_NODE, countries::HONDURAS, 4, 0});
        ASSERT_EQ(s.countries[countries::HONDURAS].us_influence, 0); // 3 - 3 = 0
    }
    // Purge (1 op): 4 + 1 - 4 = 1 coup val (removes 1 US)
    {
        GameState s{};
        s.set_flag(effect_bits::PURGE_USSR_ACTIVE);
        s.countries[countries::HONDURAS].us_influence = 3;
        CardHandlers::trigger_event(s, card_ids::ORTEGA_ELECTED_IN_NICARAGUA, Player::USSR);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::POINT_NODE, countries::HONDURAS, 4, 0});
        ASSERT_EQ(s.countries[countries::HONDURAS].us_influence, 2); // 3 - 1 = 2
    }
    // Brezhnev + Purge (2 ops): 4 + 2 - 4 = 2 coup val (removes 2 US)
    {
        GameState s{};
        s.set_flag(effect_bits::BREZHNEV_DOCTRINE_ACTIVE);
        s.set_flag(effect_bits::PURGE_USSR_ACTIVE);
        s.countries[countries::HONDURAS].us_influence = 3;
        CardHandlers::trigger_event(s, card_ids::ORTEGA_ELECTED_IN_NICARAGUA, Player::USSR);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::POINT_NODE, countries::HONDURAS, 4, 0});
        ASSERT_EQ(s.countries[countries::HONDURAS].us_influence, 1); // 3 - 2 = 1
    }
}

TEST(CardEdgeCasesTest, TearDownThisWall_OpsModifiers_Suite) {
    // US performs 3 Ops in Europe
    {
        GameState s{};
        CardHandlers::trigger_event(s, card_ids::TEAR_DOWN_THIS_WALL, Player::US);
        ASSERT_EQ(s.ctx().pending_ops_value, 3);
    }
    {
        GameState s{};
        s.set_flag(effect_bits::CONTAINMENT_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::TEAR_DOWN_THIS_WALL, Player::US);
        ASSERT_EQ(s.ctx().pending_ops_value, 4);
    }
    {
        GameState s{};
        s.set_flag(effect_bits::PURGE_US_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::TEAR_DOWN_THIS_WALL, Player::US);
        ASSERT_EQ(s.ctx().pending_ops_value, 2);
    }
    {
        GameState s{};
        s.set_flag(effect_bits::CONTAINMENT_ACTIVE);
        s.set_flag(effect_bits::PURGE_US_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::TEAR_DOWN_THIS_WALL, Player::US);
        ASSERT_EQ(s.ctx().pending_ops_value, 3);
    }
}

TEST(CardEdgeCasesTest, GrainSales_OpsModifiers_Suite) {
    // US conducts 2 Ops if USSR has no cards or card returned
    {
        GameState s{};
        CardHandlers::trigger_event(s, card_ids::GRAIN_SALES, Player::US);
        ASSERT_EQ(s.ctx().pending_ops_value, 2);
    }
    {
        GameState s{};
        s.set_flag(effect_bits::CONTAINMENT_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::GRAIN_SALES, Player::US);
        ASSERT_EQ(s.ctx().pending_ops_value, 3);
    }
    {
        GameState s{};
        s.set_flag(effect_bits::PURGE_US_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::GRAIN_SALES, Player::US);
        ASSERT_EQ(s.ctx().pending_ops_value, 1);
    }
    {
        GameState s{};
        s.set_flag(effect_bits::CONTAINMENT_ACTIVE);
        s.set_flag(effect_bits::PURGE_US_ACTIVE);
        CardHandlers::trigger_event(s, card_ids::GRAIN_SALES, Player::US);
        ASSERT_EQ(s.ctx().pending_ops_value, 2);
    }
}


TEST(CardEdgeCasesTest, WarsawPact_Branch0_NoUSInfluence_ResolvesImmediately_NeverHighlightsCanada) {
    GameState state{};
    StateMachine::init_new_game(state, 42);

    // Setup: No US influence anywhere in Eastern Europe
    for (uint8_t i = 0; i < 84; ++i) {
        if (MapData::get_country(i).in_eastern_europe) {
            state.countries[i].us_influence = 0;
            state.countries[i].ussr_influence = 0;
        }
    }
    state.countries[countries::CANADA].us_influence = 4; // Canada has US influence

    // Trigger Warsaw Pact event
    bool done = CardHandlers::trigger_event(state, card_ids::WARSAW_PACT, Player::USSR);
    ASSERT_FALSE(done);
    ASSERT_EQ(state.ctx().decision_type, DecisionType::CHOOSE_BRANCH);

    // USSR chooses branch 0 (remove US influence from 4 EE countries)
    done = CardHandlers::handle_event_step(state, MicroAction{DecisionType::CHOOSE_BRANCH, 0, 0, 0});
    
    // Since there is NO US influence in Eastern Europe, it must resolve immediately!
    ASSERT_TRUE(done);
    ASSERT_EQ(state.ctx().resolving_card, 0);

    // Now test when Poland has US influence: mask should ONLY contain Poland, NEVER Canada (id 0)
    state.countries[countries::POLAND].us_influence = 2;
    done = CardHandlers::trigger_event(state, card_ids::WARSAW_PACT, Player::USSR);
    ASSERT_FALSE(done);
    
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().max_per_country = 1;
    state.ctx().remaining_steps = 1;
    
    uint8_t mask[84];
    size_t out_size = 0;
    ActionMask::generate_mask(state, mask, &out_size);
    ASSERT_EQ(out_size, 84);
    ASSERT_EQ(mask[countries::CANADA], 0); // Canada must NOT be legal!
    ASSERT_EQ(mask[countries::POLAND], 1); // Poland IS legal!
}

} // namespace ts

// =============================================================================
// Space Race Edge Cases and Ability Tests
// =============================================================================

TEST(CardEdgeCasesTest, SpaceRace_AllBoxAbilities_Animal_ManInSpace_SpaceWalk_MoonLanding) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);

    // Box 0: Initial state - no abilities
    ASSERT_FALSE(ts::SpaceRace::has_animal_in_space(state, ts::Player::US));
    ASSERT_FALSE(ts::SpaceRace::has_animal_in_space(state, ts::Player::USSR));
    ASSERT_FALSE(ts::SpaceRace::has_man_in_space(state, ts::Player::US));
    ASSERT_FALSE(ts::SpaceRace::has_space_walk(state, ts::Player::US));
    ASSERT_FALSE(ts::SpaceRace::has_space_station_ar8(state, ts::Player::US));

    // Box 2: Animal in Space (2 attempts per turn)
    state.us_space_track = 2;
    state.ussr_space_track = 1;
    ASSERT_TRUE(ts::SpaceRace::has_animal_in_space(state, ts::Player::US));
    ASSERT_FALSE(ts::SpaceRace::has_animal_in_space(state, ts::Player::USSR));

    // When opponent catches up to Box 2, leader privilege is canceled
    state.ussr_space_track = 2;
    ASSERT_FALSE(ts::SpaceRace::has_animal_in_space(state, ts::Player::US));
    ASSERT_FALSE(ts::SpaceRace::has_animal_in_space(state, ts::Player::USSR));

    // Box 4: Lunar Orbit (Opponent reveals headline first)
    state.us_space_track = 4;
    state.ussr_space_track = 3;
    ASSERT_TRUE(ts::SpaceRace::has_man_in_space(state, ts::Player::US));
    ASSERT_FALSE(ts::SpaceRace::has_man_in_space(state, ts::Player::USSR));
    state.ussr_space_track = 4;
    ASSERT_FALSE(ts::SpaceRace::has_man_in_space(state, ts::Player::US));

    // Box 6: Space Walk (Discard 1 held card at end of turn)
    state.us_space_track = 6;
    state.ussr_space_track = 5;
    ASSERT_TRUE(ts::SpaceRace::has_space_walk(state, ts::Player::US));
    ASSERT_FALSE(ts::SpaceRace::has_space_walk(state, ts::Player::USSR));
    state.ussr_space_track = 6;
    ASSERT_FALSE(ts::SpaceRace::has_space_walk(state, ts::Player::US));

    // Box 8: Moon Landing (8th Action Round)
    state.us_space_track = 8;
    state.ussr_space_track = 7;
    ASSERT_TRUE(ts::SpaceRace::has_space_station_ar8(state, ts::Player::US));
    ASSERT_FALSE(ts::SpaceRace::has_space_station_ar8(state, ts::Player::USSR));
    state.ussr_space_track = 8;
    ASSERT_FALSE(ts::SpaceRace::has_space_station_ar8(state, ts::Player::US));
}

TEST(CardEdgeCasesTest, CapturedNaziScientist_AdvancesTrack_AwardsVPsAndAbilities_AndRespectsMax) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.victory_points = 0;

    // 1. US advances from Box 0 -> Box 1 (1st to reach: +2 VP)
    state.us_space_track = 0;
    state.ussr_space_track = 0;
    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::CAPTURED_NAZI_SCIENTIST, ts::Player::US);
    ASSERT_TRUE(done);
    ASSERT_EQ(state.us_space_track, 1);
    ASSERT_EQ(state.victory_points, 2); // 1st player VP award

    // 2. USSR advances from Box 0 -> Box 1 (2nd to reach: +1 VP to USSR -> net +1 VP to US)
    done = ts::CardHandlers::trigger_event(state, ts::card_ids::CAPTURED_NAZI_SCIENTIST, ts::Player::USSR);
    ASSERT_TRUE(done);
    ASSERT_EQ(state.ussr_space_track, 1);
    ASSERT_EQ(state.victory_points, 1); // 2 - 1 = +1 VP

    // 3. US advances from Box 1 -> Box 2 (Animal in Space: 0 VP, but activates 2 attempts/turn)
    done = ts::CardHandlers::trigger_event(state, ts::card_ids::CAPTURED_NAZI_SCIENTIST, ts::Player::US);
    ASSERT_TRUE(done);
    ASSERT_EQ(state.us_space_track, 2);
    ASSERT_EQ(state.victory_points, 1); // No VP change on Box 2
    ASSERT_TRUE(ts::SpaceRace::has_animal_in_space(state, ts::Player::US));

    // 4. USSR advances to Box 8 (Moon Landing): awards 2 VP to USSR (net 1 - 2 = -1 VP)
    state.ussr_space_track = 7;
    state.us_space_track = 6;
    done = ts::CardHandlers::trigger_event(state, ts::card_ids::CAPTURED_NAZI_SCIENTIST, ts::Player::USSR);
    ASSERT_TRUE(done);
    ASSERT_EQ(state.ussr_space_track, 8);
    ASSERT_EQ(state.victory_points, -1); // 1 - 2 = -1 VP
    ASSERT_TRUE(ts::SpaceRace::has_space_station_ar8(state, ts::Player::USSR));

    // 5. When already at Box 8 (max), cannot advance further
    int8_t vp_before = state.victory_points;
    done = ts::CardHandlers::trigger_event(state, ts::card_ids::CAPTURED_NAZI_SCIENTIST, ts::Player::USSR);
    ASSERT_TRUE(done);
    ASSERT_EQ(state.ussr_space_track, 8);
    ASSERT_EQ(state.victory_points, vp_before); // No change
}

TEST(CardEdgeCasesTest, OneSmallStep_Advances2SpacesWhenBehind_AwardsVPWhenLandingOnVPBox_NoVPWhenJumpingOver) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);

    // Case 1: Jumping over a VP box does NOT award VP
    // USSR is at Box 3, US is at Box 0.
    // US plays One Small Step: advances +2 to Box 2 (Animal in Space, 0 VP).
    // Box 1 had 2 VP, but US jumps over it without landing -> 0 VP awarded.
    state.victory_points = 0;
    state.us_space_track = 0;
    state.ussr_space_track = 3;
    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::ONE_SMALL_STEP, ts::Player::US);
    ASSERT_TRUE(done);
    ASSERT_EQ(state.us_space_track, 2);
    ASSERT_EQ(state.victory_points, 0); // Jumped over Box 1 -> No VP!

    // Case 2: Landing on a VP box as 1st player awards full 1st VP
    // USSR is at Box 4, US is at Box 3.
    // US plays One Small Step: advances +2 to Box 5 (Lunar Probe).
    // Box 5 awards 3 VP (1st) / 1 VP (2nd). US reaches Box 5 before USSR (USSR is at 4).
    state.victory_points = 0;
    state.us_space_track = 3;
    state.ussr_space_track = 4;
    done = ts::CardHandlers::trigger_event(state, ts::card_ids::ONE_SMALL_STEP, ts::Player::US);
    ASSERT_TRUE(done);
    ASSERT_EQ(state.us_space_track, 5);
    ASSERT_EQ(state.victory_points, 3); // Landed on Box 5 as 1st -> +3 VP!

    // Case 3: Landing on a VP box as 2nd player awards 2nd VP
    // USSR is at Box 6, US is at Box 3.
    // US plays One Small Step: advances +2 to Box 5.
    // USSR is at 6 (already passed 5), so US lands on Box 5 as 2nd player -> receives 1 VP.
    state.victory_points = 0;
    state.us_space_track = 3;
    state.ussr_space_track = 6;
    done = ts::CardHandlers::trigger_event(state, ts::card_ids::ONE_SMALL_STEP, ts::Player::US);
    ASSERT_TRUE(done);
    ASSERT_EQ(state.us_space_track, 5);
    ASSERT_EQ(state.victory_points, 1); // Landed on Box 5 as 2nd -> +1 VP!

    // Case 4: Not behind opponent -> Does NOT advance
    state.victory_points = 0;
    state.us_space_track = 4;
    state.ussr_space_track = 4; // Tied
    done = ts::CardHandlers::trigger_event(state, ts::card_ids::ONE_SMALL_STEP, ts::Player::US);
    ASSERT_TRUE(done);
    ASSERT_EQ(state.us_space_track, 4); // No advance
    ASSERT_EQ(state.victory_points, 0);

    state.us_space_track = 5;
    state.ussr_space_track = 3; // Ahead
    done = ts::CardHandlers::trigger_event(state, ts::card_ids::ONE_SMALL_STEP, ts::Player::US);
    ASSERT_TRUE(done);
    ASSERT_EQ(state.us_space_track, 5); // No advance
    ASSERT_EQ(state.victory_points, 0);
}

TEST(CardEdgeCasesTest, Defectors_PlayedByUSSRDuringActionRound_Awards1VPToUS) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.action_round = 1;
    state.phasing_player = ts::Player::USSR;
    state.victory_points = 0;

    // When USSR plays Defectors for Ops or event during AR, event triggers with executor US
    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::DEFECTORS, ts::Player::US);
    ASSERT_TRUE(done);
    ASSERT_EQ(state.victory_points, 1); // US gains 1 VP!
}

TEST(CardEdgeCasesTest, Defectors_HeadlinePhase_CancelsUSSRHeadlineWithoutVP) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.current_phase = ts::Phase::HEADLINE;
    state.headline_ussr_card = ts::card_ids::SOCIALIST_GOVERNMENTS;
    state.victory_points = 0;

    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::DEFECTORS, ts::Player::US);
    ASSERT_TRUE(done);
    ASSERT_EQ(state.headline_ussr_card, 0); // USSR headline cancelled
    ASSERT_EQ(state.victory_points, 0);     // No VP in headline phase
}

TEST(CardEdgeCasesTest, GrainSales_HeadlinedByUS_DrawsAndExecutesCard_CleanlyAdvancesToAR1) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.current_phase = ts::Phase::HEADLINE;
    state.ctx().decision_player = ts::Player::US;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == ts::CardLocation::HAND_USSR) {
            state.card_locations[i] = ts::CardLocation::DRAW_DECK;
        }
    }
    state.card_locations[ts::card_ids::GRAIN_SALES] = ts::CardLocation::HAND_US;
    state.card_locations[ts::card_ids::WE_WILL_BURY_YOU] = ts::CardLocation::HAND_USSR;
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::CardLocation::HAND_USSR;
    state.defcon = 4;
    state.victory_points = 0;

    // US headlines Grain Sales (2 Ops)
    ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::GRAIN_SALES, 0, 0});
    // USSR headlines We Will Bury You (4 Ops)
    ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::WE_WILL_BURY_YOU, 0, 0});

    // Stage 1: We Will Bury You (4 Ops) resolves first -> DEFCON drops to 3
    ASSERT_EQ(state.defcon, 3);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::WE_WILL_BURY_YOU_PENDING));

    // Stage 2: Grain Sales (2 Ops) resolves -> US is prompted to choose branch for drawn Duck and Cover
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::CHOOSE_BRANCH);
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::GRAIN_SALES);

    // US selects Branch 0: Play drawn card
    ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::CHOOSE_BRANCH, 0, 0, 0});
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_PLAY_MODE);
    ASSERT_EQ(state.ctx().pending_op_card, ts::card_ids::DUCK_AND_COVER);

    // US plays Duck and Cover for PlayMode::EVENT (0)
    ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_PLAY_MODE, 0, 0, 0});

    // Duck and Cover resolves: DEFCON drops to 2, US gets 3 VP
    ASSERT_EQ(state.defcon, 2);
    ASSERT_EQ(state.victory_points, 3);

    // Headline phase MUST be completely finished and now at AR 1 with USSR choosing card!
    ASSERT_EQ(state.current_phase, ts::Phase::ACTION_ROUND);
    ASSERT_EQ(state.action_round, 1);
    ASSERT_EQ(state.phasing_player, ts::Player::USSR);
    ASSERT_EQ(state.ctx().decision_player, ts::Player::USSR);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_CARD);
}
