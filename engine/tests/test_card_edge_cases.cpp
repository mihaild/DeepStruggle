#include "ts/engine.hpp"
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
    state.card_locations[card_ids::DUCK_AND_COVER] = ts::hand_of(ts::Player::US); // 3 Ops card
    
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
    state.card_locations[card_ids::ASIA_SCORING] = ts::hand_of(ts::Player::US);
    state.card_locations[card_ids::EUROPE_SCORING] = ts::hand_of(ts::Player::US);

    bool done = CardHandlers::trigger_event(state, card_ids::MISSILE_ENVY, Player::USSR);
    ASSERT_TRUE(done); // Clean no-op, no cards transferred
    ASSERT_TRUE(in_hand_of(state.card_locations[card_ids::ASIA_SCORING], Player::US));
    ASSERT_TRUE(in_hand_of(state.card_locations[card_ids::EUROPE_SCORING], Player::US));
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
    // A free coup is still a coup and needs US influence to remove. This test used to leave
    // Costa Rica and Honduras empty and assert they were offered anyway, which is the defect
    // ts-replayer game 139 turn 9 AR 2 ran into: Ortega offered Cuba at US 0 / USSR 3, and
    // taking that battleground dropped DEFCON to 1 and ended the game.
    state.countries[countries::COSTA_RICA].us_influence = 1;
    state.countries[countries::HONDURAS].us_influence = 1;

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
    ASSERT_EQ(mask[countries::FRANCE], 0);   // not adjacent to Nicaragua

    // Adjacent, but with no US influence there is nothing to coup.
    state.countries[countries::HONDURAS].us_influence = 0;
    CardHandlers::get_event_action_mask(state, mask, &out_size);
    ASSERT_EQ(mask[countries::HONDURAS], 0);
    state.countries[countries::HONDURAS].us_influence = 1;
    CardHandlers::get_event_action_mask(state, mask, &out_size);

    // Choosing the target opens a chance node; the coup resolves when that is rolled.
    // Roll 6: 6 + 2 ops - 2 * 3 (stability) = 2 coup value
    done = CardHandlers::handle_event_step(state, MicroAction{DecisionType::POINT_NODE, countries::CUBA, 0, 0});
    ASSERT_FALSE(done);
    ASSERT_EQ(state.ctx().decision_type, DecisionType::ROLL_DIE);
    done = CardHandlers::handle_event_step(state, MicroAction{DecisionType::ROLL_DIE, 6, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(state.countries[countries::CUBA].us_influence, 1); // 3 - 2 = 1
    // The event grants the attempt, not the Operations to buy it with, so nothing is spent and
    // no military operations are earned. The logs agree: an Ortega coup prints no "Military Ops
    // to N" line, while a coup the same player makes with a card's own Ops does.
    ASSERT_EQ(state.ussr_mil_ops, 0);
}

TEST(CardEdgeCasesTest, AldrichAmes_WithSingleCardInUSHand) {
    GameState state{};
    state.card_locations[card_ids::DUCK_AND_COVER] = ts::hand_of(ts::Player::US);

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
        if (state.card_locations[i] == ts::hand_of(ts::Player::US)) {
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
    ASSERT_TRUE(in_hand_of(state.card_locations[card_ids::FIDEL], Player::US));
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
    CardHandlers::handle_event_step(state, MicroAction(DecisionType::ROLL_DIE, 2, 0, 0));
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
    CardHandlers::handle_event_step(state, MicroAction(DecisionType::ROLL_DIE, 5, 0, 0));
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
    ASSERT_FALSE(done);
    ASSERT_EQ(state.ctx().decision_type, DecisionType::ROLL_DIE);
    done = CardHandlers::handle_event_step(state, MicroAction(DecisionType::ROLL_DIE, 2, 0, 0));
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
    ASSERT_FALSE(done);
    ASSERT_EQ(state.ctx().decision_type, DecisionType::ROLL_DIE);
    done = CardHandlers::handle_event_step(state, MicroAction(DecisionType::ROLL_DIE, 2, 0, 0));
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
    ASSERT_FALSE(done);
    ASSERT_EQ(state.ctx().decision_type, DecisionType::ROLL_DIE);
    done = CardHandlers::handle_event_step(state, MicroAction(DecisionType::ROLL_DIE, 1, 0, 0));
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

    // US triggers Summit -> chance node for die rolls
    bool done = CardHandlers::trigger_event(state, card_ids::SUMMIT, Player::US);
    ASSERT_FALSE(done);
    ASSERT_EQ(state.ctx().decision_type, DecisionType::ROLL_DIE);

    // Roll die: US rolls 6, USSR rolls 1 -> US wins Summit by 5
    done = CardHandlers::handle_event_step(state, MicroAction{DecisionType::ROLL_DIE, 6, 1, 0});
    ASSERT_FALSE(done);
    ASSERT_EQ(state.ctx().decision_type, DecisionType::CHOOSE_BRANCH);

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

    // First coup in Colombia with forced roll 6: 6 + 3 - 2 * 1 = 7 (removes 2 US, adds 5 USSR).
    // Each coup takes two steps now: the target choice, then its own chance node.
    CardHandlers::handle_event_step(s1, MicroAction{DecisionType::POINT_NODE, countries::COLOMBIA, 0, 0});
    done = CardHandlers::handle_event_step(s1, MicroAction{DecisionType::ROLL_DIE, 6, 0, 0});
    ASSERT_FALSE(done); // Second coup is offered because US influence was removed
    ASSERT_EQ(s1.countries[countries::COLOMBIA].us_influence, 0);
    ASSERT_EQ(s1.countries[countries::COLOMBIA].ussr_influence, 5);

    // Second coup in Peru with forced roll 6
    CardHandlers::handle_event_step(s1, MicroAction{DecisionType::POINT_NODE, countries::PERU, 0, 0});
    done = CardHandlers::handle_event_step(s1, MicroAction{DecisionType::ROLL_DIE, 6, 0, 0});
    ASSERT_TRUE(done);
    ASSERT_EQ(s1.countries[countries::PERU].us_influence, 0);
    ASSERT_EQ(s1.countries[countries::PERU].ussr_influence, 3);
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
    // 1. Default Boycott (4 Ops, 0 VP)
    {
        GameState s{};
        s.defcon = 4;
        CardHandlers::trigger_event(s, card_ids::OLYMPIC_GAMES, Player::US);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::CHOOSE_BRANCH, 1, 0, 0}); // USSR boycotts
        ASSERT_EQ(s.victory_points, 0);
        ASSERT_EQ(s.defcon, 3);
        ASSERT_EQ(s.ctx().pending_ops_value, 4);
    }
    {
        GameState s{};
        s.defcon = 4;
        CardHandlers::trigger_event(s, card_ids::OLYMPIC_GAMES, Player::USSR);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::CHOOSE_BRANCH, 1, 0, 0}); // US boycotts
        ASSERT_EQ(s.victory_points, 0);
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
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::POINT_NODE, countries::COLOMBIA, 0, 0});
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::ROLL_DIE, 1, 0, 0});
        ASSERT_EQ(s.countries[countries::COLOMBIA].us_influence, 1); // 3 - 2 = 1
    }
    // Brezhnev (4 ops): 1 + 4 - 2 = 3 coup val (removes 3 US)
    {
        GameState s{};
        s.set_flag(effect_bits::BREZHNEV_DOCTRINE_ACTIVE);
        s.countries[countries::COLOMBIA].us_influence = 3;
        CardHandlers::trigger_event(s, card_ids::CHE, Player::USSR);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::POINT_NODE, countries::COLOMBIA, 0, 0});
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::ROLL_DIE, 1, 0, 0});
        ASSERT_EQ(s.countries[countries::COLOMBIA].us_influence, 0); // 3 - 3 = 0
    }
    // Purge (2 ops): 1 + 2 - 2 = 1 coup val (removes 1 US)
    {
        GameState s{};
        s.set_flag(effect_bits::PURGE_USSR_ACTIVE);
        s.countries[countries::COLOMBIA].us_influence = 3;
        CardHandlers::trigger_event(s, card_ids::CHE, Player::USSR);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::POINT_NODE, countries::COLOMBIA, 0, 0});
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::ROLL_DIE, 1, 0, 0});
        ASSERT_EQ(s.countries[countries::COLOMBIA].us_influence, 2); // 3 - 1 = 2
    }
    // Brezhnev + Purge (3 ops): 1 + 3 - 2 = 2 coup val (removes 2 US)
    {
        GameState s{};
        s.set_flag(effect_bits::BREZHNEV_DOCTRINE_ACTIVE);
        s.set_flag(effect_bits::PURGE_USSR_ACTIVE);
        s.countries[countries::COLOMBIA].us_influence = 3;
        CardHandlers::trigger_event(s, card_ids::CHE, Player::USSR);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::POINT_NODE, countries::COLOMBIA, 0, 0});
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::ROLL_DIE, 1, 0, 0});
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
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::POINT_NODE, countries::HONDURAS, 0, 0});
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::ROLL_DIE, 4, 0, 0});
        ASSERT_EQ(s.countries[countries::HONDURAS].us_influence, 1); // 3 - 2 = 1
    }
    // Brezhnev (3 ops): 4 + 3 - 4 = 3 coup val (removes 3 US)
    {
        GameState s{};
        s.set_flag(effect_bits::BREZHNEV_DOCTRINE_ACTIVE);
        s.countries[countries::HONDURAS].us_influence = 3;
        CardHandlers::trigger_event(s, card_ids::ORTEGA_ELECTED_IN_NICARAGUA, Player::USSR);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::POINT_NODE, countries::HONDURAS, 0, 0});
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::ROLL_DIE, 4, 0, 0});
        ASSERT_EQ(s.countries[countries::HONDURAS].us_influence, 0); // 3 - 3 = 0
    }
    // Purge (1 op): 4 + 1 - 4 = 1 coup val (removes 1 US)
    {
        GameState s{};
        s.set_flag(effect_bits::PURGE_USSR_ACTIVE);
        s.countries[countries::HONDURAS].us_influence = 3;
        CardHandlers::trigger_event(s, card_ids::ORTEGA_ELECTED_IN_NICARAGUA, Player::USSR);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::POINT_NODE, countries::HONDURAS, 0, 0});
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::ROLL_DIE, 4, 0, 0});
        ASSERT_EQ(s.countries[countries::HONDURAS].us_influence, 2); // 3 - 1 = 2
    }
    // Brezhnev + Purge (2 ops): 4 + 2 - 4 = 2 coup val (removes 2 US)
    {
        GameState s{};
        s.set_flag(effect_bits::BREZHNEV_DOCTRINE_ACTIVE);
        s.set_flag(effect_bits::PURGE_USSR_ACTIVE);
        s.countries[countries::HONDURAS].us_influence = 3;
        CardHandlers::trigger_event(s, card_ids::ORTEGA_ELECTED_IN_NICARAGUA, Player::USSR);
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::POINT_NODE, countries::HONDURAS, 0, 0});
        CardHandlers::handle_event_step(s, MicroAction{DecisionType::ROLL_DIE, 4, 0, 0});
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

TEST(CardEdgeCasesTest, Defectors_USActionRound_CannotBePlayedAsEvent) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.action_round = 1;
    state.phasing_player = ts::Player::US;
    state.ctx().decision_player = ts::Player::US;

    // 1. can_trigger_event returns false for US during Action Round
    ASSERT_FALSE(ts::CardHandlers::can_trigger_event(state, ts::card_ids::DEFECTORS, ts::Player::US));

    // 2. Action mask for SELECT_PLAY_MODE must NOT have EVENT bit set
    state.ctx().decision_type = ts::DecisionType::SELECT_PLAY_MODE;
    state.ctx().pending_op_card = ts::card_ids::DEFECTORS;
    uint8_t mask[128]{};
    size_t out_size = 0;
    ts::ActionMask::generate_mask(state, mask, &out_size);
    ASSERT_EQ(out_size, static_cast<size_t>(ts::Resolution::COUNT));
    ASSERT_EQ(mask[static_cast<uint8_t>(ts::Resolution::EVENT)], 0); // Event is illegal!
    ASSERT_TRUE((mask[static_cast<size_t>(ts::Resolution::OPS_INFLUENCE)] || mask[static_cast<size_t>(ts::Resolution::OPS_COUP)] || mask[static_cast<size_t>(ts::Resolution::OPS_REALIGN)]));   // Ops is legal

    // 3. Attempting to step with PlayMode::EVENT must be rejected by StateMachine
    bool step_result = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::Resolution::EVENT), 0, 0});
    ASSERT_FALSE(step_result); // Rejected as illegal move
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
        if (state.card_locations[i] == ts::hand_of(ts::Player::USSR)) {
            state.card_locations[i] = ts::CardLocation::DRAW_DECK;
        }
    }
    state.card_locations[ts::card_ids::GRAIN_SALES] = ts::hand_of(ts::Player::US);
    state.card_locations[ts::card_ids::WE_WILL_BURY_YOU] = ts::hand_of(ts::Player::USSR);
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::hand_of(ts::Player::USSR);
    state.defcon = 4;
    state.victory_points = 0;

    // US headlines Grain Sales (2 Ops)
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::GRAIN_SALES, 0, 0}));
    // USSR headlines We Will Bury You (4 Ops)
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::WE_WILL_BURY_YOU, 0, 0}));

    // Stage 1: We Will Bury You (4 Ops) resolves first -> DEFCON drops to 3
    ASSERT_EQ(state.defcon, 3);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::WE_WILL_BURY_YOU_PENDING));

    // Stage 2: Grain Sales (2 Ops) resolves -> US is prompted to choose branch for drawn Duck and Cover
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::CHOOSE_BRANCH);
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::GRAIN_SALES);

    // US selects Branch 0: Play drawn card
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::CHOOSE_BRANCH, 0, 0, 0}));
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_PLAY_MODE);
    ASSERT_EQ(state.ctx().pending_op_card, ts::card_ids::DUCK_AND_COVER);

    // US plays Duck and Cover for PlayMode::EVENT (0)
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_PLAY_MODE, 0, 0, 0}));

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

TEST(CardEdgeCasesTest, GrainSales_Headline_OpponentCardOpsFirst_StillFiresItsEvent) {
    // Grain Sales works the same in a headline as in an action round: the US takes a card out
    // of the USSR hand, and if it is the USSR's own the US chooses whether its Event or its
    // Operations resolve first. OPS_FIRST leaves the Event owed until the Ops are spent, and
    // the headline path used to unwind and advance without ever firing it.
    //
    // At turn 4's headline of ts-replayer game 137 the US headlines Grain Sales, takes Willy
    // Brandt, chooses its Operations first and realigns Cuba twice; Willy Brandt's Event --
    // 1 VP to the USSR, 1 Influence into West Germany, and the card into play -- never
    // happened, and the engine went straight to AR1.
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.current_phase = ts::Phase::HEADLINE;
    state.ctx().decision_player = ts::Player::US;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == ts::hand_of(ts::Player::USSR) ||
            state.card_locations[i] == ts::hand_of(ts::Player::US)) {
            state.card_locations[i] = ts::CardLocation::DRAW_DECK;
        }
    }
    state.card_locations[ts::card_ids::GRAIN_SALES] = ts::hand_of(ts::Player::US);
    state.card_locations[ts::card_ids::WE_WILL_BURY_YOU] = ts::hand_of(ts::Player::USSR);
    state.card_locations[ts::card_ids::WILLY_BRANDT] = ts::hand_of(ts::Player::USSR);
    state.defcon = 4;
    state.victory_points = 0;
    const int8_t west_germany_before = state.countries[ts::countries::WEST_GERMANY].ussr_influence;

    // US headlines Grain Sales (2 Ops); the USSR headlines a 4 Ops card, which resolves first
    // and asks nothing.
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::GRAIN_SALES, 0, 0}));
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::WE_WILL_BURY_YOU, 0, 0}));
    const int8_t vp_before = state.victory_points;

    // Grain Sales resolves and offers the US the card it took.
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::CHOOSE_BRANCH);
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::GRAIN_SALES);
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::CHOOSE_BRANCH, 0, 0, 0}));
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_PLAY_MODE);
    ASSERT_EQ(state.ctx().pending_op_card, ts::card_ids::WILLY_BRANDT);

    // It is the USSR's own card, so Operations is the only play mode, and the ordering is a
    // choice of its own.
    ASSERT_TRUE(ts::Engine::step(state, ts::MicroAction{ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::Resolution::OPS_INFLUENCE), 0, 0}));
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::POINT_NODE);
    while (state.ctx().decision_type == ts::DecisionType::POINT_NODE &&
           state.ctx().resolving_card == 0) {
        ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::CANADA, 0, 0}));
    }

    // ...and only then does Willy Brandt's Event happen.
    ASSERT_EQ(state.victory_points, vp_before - 1);          // 1 VP to the USSR
    ASSERT_EQ(state.countries[ts::countries::WEST_GERMANY].ussr_influence,
              west_germany_before + 1);
    ASSERT_TRUE(state.has_flag(ts::effect_bits::WILLY_BRANDT_PLAYED));
    ASSERT_FALSE(ts::in_hand_of(state.card_locations[ts::card_ids::WILLY_BRANDT],
                                ts::Player::US));

    // And the headline is over, with nothing left on the stack.
    ASSERT_EQ(state.current_phase, ts::Phase::ACTION_ROUND);
    ASSERT_EQ(state.ctx_stack_depth, 0);
}

// Defectors cancels the USSR headline however it reaches the table, not only when the US
// headlines it. Three cards can put it there mid-headline: Five Year Plan discards it out of
// the USSR hand, Grain Sales To Soviets hands it to the US to play, and Star Wars takes it out
// of the discard pile. At turn 2's headline of ts-replayer game 313 the USSR headlines Vietnam
// Revolts against Five Year Plan -- the higher Ops, so it resolves first -- and the Defectors
// it discards leaves Vietnam untouched in the log.
namespace {

// The USSR headlines Vietnam Revolts (2 Ops) against a US headline of `us_headline`, with the
// cards named in `ussr_hand` and `us_hand` held. Both headlines are selected, so the higher
// Ops card resolves first.
ts::GameState defectors_headline(uint8_t us_headline,
                                 std::initializer_list<uint8_t> us_hand,
                                 std::initializer_list<uint8_t> ussr_hand) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.current_phase = ts::Phase::HEADLINE;
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == ts::hand_of(ts::Player::US) ||
            state.card_locations[i] == ts::hand_of(ts::Player::USSR)) {
            state.card_locations[i] = ts::CardLocation::DRAW_DECK;
        }
    }
    state.card_locations[us_headline] = ts::hand_of(ts::Player::US);
    for (uint8_t c : us_hand) state.card_locations[c] = ts::hand_of(ts::Player::US);
    state.card_locations[ts::card_ids::VIETNAM_REVOLTS] = ts::hand_of(ts::Player::USSR);
    for (uint8_t c : ussr_hand) state.card_locations[c] = ts::hand_of(ts::Player::USSR);
    state.ctx().decision_player = ts::Player::US;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, us_headline, 0, 0}));
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::VIETNAM_REVOLTS, 0, 0}));
    return state;
}

}  // namespace

TEST(CardEdgeCasesTest, Defectors_DiscardedByFiveYearPlanInHeadline_CancelsUSSRHeadline) {
    ts::GameState state = defectors_headline(ts::card_ids::FIVE_YEAR_PLAN,
                                             {}, {ts::card_ids::DEFECTORS});
    // Five Year Plan is 3 Ops against Vietnam Revolts' 2, so it resolves first and discards
    // the only card the USSR holds.
    ASSERT_EQ(state.card_locations[ts::card_ids::DEFECTORS], ts::CardLocation::DISCARD_PILE);
    ASSERT_EQ(state.countries[ts::countries::VIETNAM].ussr_influence, 0);
    ASSERT_FALSE(state.has_flag(ts::effect_bits::VIETNAM_REVOLTS_ACTIVE));
    ASSERT_EQ(state.card_locations[ts::card_ids::VIETNAM_REVOLTS], ts::CardLocation::DISCARD_PILE);
}

TEST(CardEdgeCasesTest, Defectors_HandedOverByGrainSalesInHeadline_CancelsUSSRHeadline) {
    ts::GameState state = defectors_headline(ts::card_ids::GRAIN_SALES,
                                             {}, {ts::card_ids::DEFECTORS});
    // Grain Sales is 2 Ops and Vietnam Revolts 2, and the US wins ties, so Grain Sales
    // resolves first and offers the US the card it drew.
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::CHOOSE_BRANCH);
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::GRAIN_SALES);
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::CHOOSE_BRANCH, 0, 0, 0}));
    ASSERT_EQ(state.ctx().pending_op_card, ts::card_ids::DEFECTORS);
    // Defectors is a US card, so the US may play it as its Event -- and in a headline it is
    // legal to do so, unlike in an action round.
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::Resolution::EVENT), 0, 0}));
    ASSERT_EQ(state.countries[ts::countries::VIETNAM].ussr_influence, 0);
    ASSERT_FALSE(state.has_flag(ts::effect_bits::VIETNAM_REVOLTS_ACTIVE));
}

TEST(CardEdgeCasesTest, Defectors_TakenByStarWarsInHeadline_CancelsUSSRHeadline) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.current_phase = ts::Phase::HEADLINE;
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == ts::hand_of(ts::Player::US) ||
            state.card_locations[i] == ts::hand_of(ts::Player::USSR) ||
            state.card_locations[i] == ts::CardLocation::DISCARD_PILE) {
            state.card_locations[i] = ts::CardLocation::DRAW_DECK;
        }
    }
    state.us_space_track = 4;      // Star Wars needs the US ahead on the space track
    state.ussr_space_track = 0;
    state.card_locations[ts::card_ids::STAR_WARS] = ts::hand_of(ts::Player::US);
    state.card_locations[ts::card_ids::VIETNAM_REVOLTS] = ts::hand_of(ts::Player::USSR);
    state.card_locations[ts::card_ids::DEFECTORS] = ts::CardLocation::DISCARD_PILE;
    state.ctx().decision_player = ts::Player::US;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::STAR_WARS, 0, 0}));
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::VIETNAM_REVOLTS, 0, 0}));

    // Star Wars is 2 Ops against Vietnam Revolts' 2 and the US wins ties, so it resolves first
    // and asks which card to take out of the discard pile.
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_CARD);
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::STAR_WARS);
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::DEFECTORS, 0, 0}));

    ASSERT_EQ(state.countries[ts::countries::VIETNAM].ussr_influence, 0);
    ASSERT_FALSE(state.has_flag(ts::effect_bits::VIETNAM_REVOLTS_ACTIVE));
    ASSERT_EQ(state.card_locations[ts::card_ids::VIETNAM_REVOLTS], ts::CardLocation::DISCARD_PILE);
}

TEST(CardEdgeCasesTest, Defectors_AfterTheUSSRHeadlineHasResolved_CancelsNothing) {
    // The USSR headlines Five Year Plan, whose event is the US's to execute, and the discard
    // it takes is Defectors. The USSR's headline is that very card and it has already
    // resolved, so there is nothing left to cancel -- and the US's own headline stands.
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.current_phase = ts::Phase::HEADLINE;
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == ts::hand_of(ts::Player::US) ||
            state.card_locations[i] == ts::hand_of(ts::Player::USSR)) {
            state.card_locations[i] = ts::CardLocation::DRAW_DECK;
        }
    }
    state.card_locations[ts::card_ids::FIVE_YEAR_PLAN] = ts::hand_of(ts::Player::USSR);
    state.card_locations[ts::card_ids::DEFECTORS] = ts::hand_of(ts::Player::USSR);
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::hand_of(ts::Player::US);
    state.ctx().decision_player = ts::Player::USSR;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::FIVE_YEAR_PLAN, 0, 0}));
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::DUCK_AND_COVER, 0, 0}));
    ASSERT_EQ(state.card_locations[ts::card_ids::DEFECTORS], ts::CardLocation::DISCARD_PILE);
    ASSERT_EQ(state.defcon, 4);   // Duck and Cover resolved: DEFCON degrades by one
}

// Each headline card resolves in a decision frame of its own. What lasts belongs to the state,
// not to the frame -- and the countries a card has already placed in are kept in the frame, as
// the visited bitmap that enforces "no more than one per country".
TEST(CardEdgeCasesTest, Headline_SecondCardStartsFromAFreshContext) {
    // Turn 7's headline of ts-replayer game 92: the US headlines Colonial Rear Guards and the
    // USSR headlines Decolonization. Both place one Influence per country in Africa or
    // Southeast Asia, and the log has them overlapping in Zaire, Angola and Nigeria.
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.current_phase = ts::Phase::HEADLINE;
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == ts::hand_of(ts::Player::US) ||
            state.card_locations[i] == ts::hand_of(ts::Player::USSR)) {
            state.card_locations[i] = ts::CardLocation::DRAW_DECK;
        }
    }
    state.card_locations[ts::card_ids::COLONIAL_REAR_GUARDS] = ts::hand_of(ts::Player::US);
    state.card_locations[ts::card_ids::DECOLONIZATION] = ts::hand_of(ts::Player::USSR);
    state.ctx().decision_player = ts::Player::US;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::COLONIAL_REAR_GUARDS, 0, 0}));
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::DECOLONIZATION, 0, 0}));

    // Colonial Rear Guards is 2 Ops against Decolonization's 2 and the US wins ties, so it
    // resolves first: four US Influence, one per country.
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::COLONIAL_REAR_GUARDS);
    const uint8_t shared[] = {ts::countries::ZAIRE, ts::countries::ANGOLA,
                              ts::countries::NIGERIA};
    for (uint8_t cid : shared) {
        ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, cid, 0, 0}));
    }
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::ZIMBABWE, 0, 0}));

    // ...and Decolonization may place in the same countries, because its frame is its own.
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::DECOLONIZATION);
    ASSERT_EQ(state.ctx().decision_player, ts::Player::USSR);
    uint8_t mask[128] = {0};
    size_t mask_size = 0;
    ts::ActionMask::generate_mask(state, mask, &mask_size);
    for (uint8_t cid : shared) {
        ASSERT_EQ(mask[cid], 1);
    }
    for (uint8_t cid : shared) {
        ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, cid, 0, 0}));
    }
    ASSERT_EQ(state.countries[ts::countries::ZAIRE].ussr_influence, 1);
    ASSERT_EQ(state.countries[ts::countries::ANGOLA].ussr_influence, 1);
    ASSERT_EQ(state.countries[ts::countries::NIGERIA].ussr_influence, 1);
}

// A trap takes a card of 2 Ops or more every action round, and a scoring card is not one. It
// may still be played on either of two counts: nothing else in hand is eligible, or the player
// holds as many scoring cards as they have rounds left to play them in -- otherwise the trap
// would carry them past the turn's end, which loses the game outright.
namespace {

ts::GameState trapped_ussr(uint8_t turn, uint8_t action_round,
                           std::initializer_list<uint8_t> hand) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == ts::hand_of(ts::Player::US) ||
            state.card_locations[i] == ts::hand_of(ts::Player::USSR)) {
            state.card_locations[i] = ts::CardLocation::DRAW_DECK;
        }
    }
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.turn = turn;
    state.action_round = action_round;
    state.phasing_player = ts::Player::USSR;
    state.set_flag(ts::effect_bits::BEAR_TRAP_ACTIVE);
    for (uint8_t c : hand) state.card_locations[c] = ts::hand_of(ts::Player::USSR);
    state.ctx().decision_player = ts::Player::USSR;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;
    return state;
}

}  // namespace

TEST(CardEdgeCasesTest, Trap_ScoringCardIsPlayableOnTheLastRoundDespiteAnEligibleDiscard) {
    // Turn 4 AR7 of ts-replayer game 63: the USSR has spent two rounds discarding to Bear Trap
    // and plays Central America Scoring on the last one. One scoring card, one round left.
    ts::GameState state = trapped_ussr(4, 7, {ts::card_ids::CENTRAL_AMERICA_SCORING,
                                              ts::card_ids::SOCIALIST_GOVERNMENTS});
    uint8_t mask[128] = {0};
    size_t mask_size = 0;
    ts::ActionMask::generate_mask(state, mask, &mask_size);
    ASSERT_EQ(mask[ts::card_ids::CENTRAL_AMERICA_SCORING], 1);
    ASSERT_EQ(mask[ts::card_ids::SOCIALIST_GOVERNMENTS], 1);   // the discard is still on offer
}

TEST(CardEdgeCasesTest, Trap_ScoringCardWaitsWhileThereAreRoundsToSpare) {
    // The same hand three rounds earlier: one scoring card against three rounds left, so the
    // trap takes its discard first.
    ts::GameState state = trapped_ussr(4, 5, {ts::card_ids::CENTRAL_AMERICA_SCORING,
                                              ts::card_ids::SOCIALIST_GOVERNMENTS});
    uint8_t mask[128] = {0};
    size_t mask_size = 0;
    ts::ActionMask::generate_mask(state, mask, &mask_size);
    ASSERT_EQ(mask[ts::card_ids::CENTRAL_AMERICA_SCORING], 0);
    ASSERT_EQ(mask[ts::card_ids::SOCIALIST_GOVERNMENTS], 1);
}

TEST(CardEdgeCasesTest, Trap_TwoScoringCardsAndTwoRoundsLeftPlayThroughIt) {
    ts::GameState state = trapped_ussr(4, 6, {ts::card_ids::CENTRAL_AMERICA_SCORING,
                                              ts::card_ids::AFRICA_SCORING,
                                              ts::card_ids::SOCIALIST_GOVERNMENTS});
    uint8_t mask[128] = {0};
    size_t mask_size = 0;
    ts::ActionMask::generate_mask(state, mask, &mask_size);
    ASSERT_EQ(mask[ts::card_ids::CENTRAL_AMERICA_SCORING], 1);
    ASSERT_EQ(mask[ts::card_ids::AFRICA_SCORING], 1);
}

TEST(CardEdgeCasesTest, Trap_ScoringCardIsPlayableWithNothingTheTrapWillTake) {
    ts::GameState state = trapped_ussr(4, 5, {ts::card_ids::CENTRAL_AMERICA_SCORING});
    uint8_t mask[128] = {0};
    size_t mask_size = 0;
    ts::ActionMask::generate_mask(state, mask, &mask_size);
    ASSERT_EQ(mask[ts::card_ids::CENTRAL_AMERICA_SCORING], 1);
}

TEST(CardEdgeCasesTest, SpaceRace_Box1_EarthSatellite_VPAwards_FirstAndSecond) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.victory_points = 0;
    state.us_space_track = 0;
    state.ussr_space_track = 0;

    // US attempts Box 1 with forced roll 2 (success, 1-3)
    bool ok1 = ts::SpaceRace::attempt_space(state, ts::Player::US, ts::card_ids::DUCK_AND_COVER, 2);
    ASSERT_TRUE(ok1);
    ASSERT_EQ(state.us_space_track, 1);
    ASSERT_EQ(state.victory_points, 2); // 1st player gets 2 VP

    // USSR attempts Box 1 with forced roll 3 (success)
    bool ok2 = ts::SpaceRace::attempt_space(state, ts::Player::USSR, ts::card_ids::SOCIALIST_GOVERNMENTS, 3);
    ASSERT_TRUE(ok2);
    ASSERT_EQ(state.ussr_space_track, 1);
    ASSERT_EQ(state.victory_points, 1); // 2nd player gets 1 VP (2 - 1 = +1 VP)
}

TEST(CardEdgeCasesTest, SpaceRace_Box2_AnimalInSpace_Grants2AttemptsPerTurn_AndCancelsWhenOpponentReaches) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.us_space_track = 1;
    state.ussr_space_track = 0;

    // US reaches Box 2
    bool ok = ts::SpaceRace::attempt_space(state, ts::Player::US, ts::card_ids::DUCK_AND_COVER, 2);
    ASSERT_TRUE(ok);
    ASSERT_EQ(state.us_space_track, 2);
    ASSERT_TRUE(ts::SpaceRace::has_animal_in_space(state, ts::Player::US));
    ASSERT_FALSE(ts::SpaceRace::has_animal_in_space(state, ts::Player::USSR));

    // Reset turn attempts
    state.set_space_turns_used(ts::Player::US, 0);
    state.set_space_turns_used(ts::Player::USSR, 0);

    // US can attempt space race 1st time
    ASSERT_TRUE(ts::SpaceRace::can_attempt_space(state, ts::Player::US, ts::card_ids::DUCK_AND_COVER));
    state.record_space_attempt(ts::Player::US);

    // US can attempt space race 2nd time in same turn!
    ASSERT_TRUE(ts::SpaceRace::can_attempt_space(state, ts::Player::US, ts::card_ids::DUCK_AND_COVER));
    state.record_space_attempt(ts::Player::US);

    // US cannot attempt 3rd time
    ASSERT_FALSE(ts::SpaceRace::can_attempt_space(state, ts::Player::US, ts::card_ids::DUCK_AND_COVER));

    // USSR can only attempt 1 time
    ASSERT_TRUE(ts::SpaceRace::can_attempt_space(state, ts::Player::USSR, ts::card_ids::SOCIALIST_GOVERNMENTS));
    state.record_space_attempt(ts::Player::USSR);
    ASSERT_FALSE(ts::SpaceRace::can_attempt_space(state, ts::Player::USSR, ts::card_ids::SOCIALIST_GOVERNMENTS));

    // USSR catches up to Box 2 -> privilege is cancelled for US!
    state.ussr_space_track = 2;
    ASSERT_FALSE(ts::SpaceRace::has_animal_in_space(state, ts::Player::US));
    ASSERT_FALSE(ts::SpaceRace::has_animal_in_space(state, ts::Player::USSR));
}

TEST(CardEdgeCasesTest, SpaceRace_Box3_ManInOrbit_VPAwards_First2VP_Second0VP) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.victory_points = 0;
    state.us_space_track = 2;
    state.ussr_space_track = 2;

    // US advances to Box 3 (Man in Orbit)
    bool ok1 = ts::SpaceRace::attempt_space(state, ts::Player::US, ts::card_ids::DUCK_AND_COVER, 1);
    ASSERT_TRUE(ok1);
    ASSERT_EQ(state.us_space_track, 3);
    ASSERT_EQ(state.victory_points, 2); // 1st player gets 2 VP

    // USSR advances to Box 3
    bool ok2 = ts::SpaceRace::attempt_space(state, ts::Player::USSR, ts::card_ids::SOCIALIST_GOVERNMENTS, 1);
    ASSERT_TRUE(ok2);
    ASSERT_EQ(state.ussr_space_track, 3);
    ASSERT_EQ(state.victory_points, 2); // 2nd player gets 0 VP (remains +2 VP)
}

TEST(CardEdgeCasesTest, SpaceRace_Box4_ManInSpace_OpponentRevealsHeadlineFirst_AndCancelsWhenOpponentReaches) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.us_space_track = 4;
    state.ussr_space_track = 3;

    ASSERT_TRUE(ts::SpaceRace::has_man_in_space(state, ts::Player::US));
    ASSERT_FALSE(ts::SpaceRace::has_man_in_space(state, ts::Player::USSR));

    // Start turn -> USSR must be prompted for Headline first!
    ts::StateMachine::start_turn(state);
    ASSERT_EQ(state.current_phase, ts::Phase::HEADLINE);
    ASSERT_EQ(state.ctx().decision_player, ts::Player::USSR);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_CARD);

    // USSR selects headline #31
    state.card_locations[ts::card_ids::RED_SCARE_PURGE] = ts::hand_of(ts::Player::USSR);
    state.card_locations[ts::card_ids::DEFECTORS] = ts::hand_of(ts::Player::US);
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::RED_SCARE_PURGE, 0, 0}));

    // Now US is prompted for Headline second, and USSR's headline is already known in state.headline_ussr_card!
    ASSERT_EQ(state.ctx().decision_player, ts::Player::US);
    ASSERT_EQ(state.headline_ussr_card, ts::card_ids::RED_SCARE_PURGE);

    // US selects headline #103 Defectors -> cancels USSR headline!
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::DEFECTORS, 0, 0}));

    // USSR catches up to Box 4 -> privilege cancelled
    state.ussr_space_track = 4;
    ASSERT_FALSE(ts::SpaceRace::has_man_in_space(state, ts::Player::US));
}

TEST(CardEdgeCasesTest, SpaceRace_Box5_LunarProbe_VPAwards_First3VP_Second1VP) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.victory_points = 0;
    state.us_space_track = 4;
    state.ussr_space_track = 4;

    // US advances to Box 5 (requires 3 Ops)
    bool ok1 = ts::SpaceRace::attempt_space(state, ts::Player::US, ts::card_ids::CONTAINMENT, 1);
    ASSERT_TRUE(ok1);
    ASSERT_EQ(state.us_space_track, 5);
    ASSERT_EQ(state.victory_points, 3); // 1st player gets 3 VP

    // USSR advances to Box 5
    bool ok2 = ts::SpaceRace::attempt_space(state, ts::Player::USSR, ts::card_ids::RED_SCARE_PURGE, 1);
    ASSERT_TRUE(ok2);
    ASSERT_EQ(state.ussr_space_track, 5);
    ASSERT_EQ(state.victory_points, 2); // 2nd player gets 1 VP (3 - 1 = +2 VP)
}

TEST(CardEdgeCasesTest, SpaceRace_Box6_SpaceWalk_AllowsDiscardAtTurnEnd_AndCancelsWhenOpponentReaches) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.turn = 2;
    state.action_round = 6;
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.phasing_player = ts::Player::US;
    state.us_space_track = 6;
    state.ussr_space_track = 5;

    ASSERT_TRUE(ts::SpaceRace::has_space_walk(state, ts::Player::US));

    // Populate draw deck with plenty of dummy cards so reshuffle does not occur
    for (uint8_t i = 1; i <= 35; ++i) {
        state.card_locations[i] = ts::CardLocation::DRAW_DECK;
    }
    // Put a toxic card in US hand
    state.card_locations[ts::card_ids::RED_SCARE_PURGE] = ts::hand_of(ts::Player::US);

    // Step to finish AR 6 of Turn 2 (last AR in early war)
    ts::StateMachine::advance_after_action_round(state);

    // The turn's cleanup is a step of its own now (RollType::TURN_CLEANUP), so the last action
    // round leaves a chance node rather than running the turn end inline. Drain it exactly as
    // every other chance node is drained.
    ASSERT_EQ(state.ctx().decision_player, ts::Player::NONE);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::ROLL_DIE);
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::ROLL_DIE, 0, 0, 0}));

    // End of turn reached -> US with Space Walk is prompted to discard a card!
    ASSERT_EQ(state.ctx().decision_player, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_CARD);
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::SPACE_WALK_DISCARD);
    ASSERT_EQ(state.ctx().allow_early_stop, 1);

    // US discards Red Scare/Purge (#31)
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::RED_SCARE_PURGE, 0, 0}));

    // Card should now be in discard pile without event having triggered (Red Scare flag never set)
    ASSERT_FALSE(state.has_flag(ts::effect_bits::PURGE_US_ACTIVE));
    ASSERT_FALSE(state.has_flag(ts::effect_bits::PURGE_USSR_ACTIVE));
    
    ASSERT_EQ(state.card_locations[ts::card_ids::RED_SCARE_PURGE], ts::CardLocation::DISCARD_PILE);

    // Turn 2 is completed and advanced to Turn 3!
    ASSERT_EQ(state.turn, 3);
    ASSERT_EQ(state.current_phase, ts::Phase::HEADLINE);

    // If USSR catches up to Box 6, privilege is cancelled
    state.ussr_space_track = 6;
    ASSERT_FALSE(ts::SpaceRace::has_space_walk(state, ts::Player::US));
}

TEST(CardEdgeCasesTest, SpaceRace_Box7_SpaceStation_VPAwards_First4VP_Second2VP) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.victory_points = 0;
    state.us_space_track = 6;
    state.ussr_space_track = 6;

    // USSR advances to Box 7 (Space Station, 3 Ops)
    bool ok1 = ts::SpaceRace::attempt_space(state, ts::Player::USSR, ts::card_ids::RED_SCARE_PURGE, 1);
    ASSERT_TRUE(ok1);
    ASSERT_EQ(state.ussr_space_track, 7);
    ASSERT_EQ(state.victory_points, -4); // USSR 1st gets 4 VP

    // US advances to Box 7
    bool ok2 = ts::SpaceRace::attempt_space(state, ts::Player::US, ts::card_ids::CONTAINMENT, 1);
    ASSERT_TRUE(ok2);
    ASSERT_EQ(state.us_space_track, 7);
    ASSERT_EQ(state.victory_points, -2); // US 2nd gets 2 VP (-4 + 2 = -2 VP)
}

TEST(CardEdgeCasesTest, SpaceRace_Box8_EagleBearLanded_Grants8thActionRound_And2VPFirst0VPSecond) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.turn = 5;
    state.victory_points = 0;
    state.us_space_track = 7;
    state.ussr_space_track = 7;

    // US advances to Box 8 (requires 4 Ops, roll 1-2)
    bool ok = ts::SpaceRace::attempt_space(state, ts::Player::US, ts::card_ids::MARSHALL_PLAN, 1);
    ASSERT_TRUE(ok);
    ASSERT_EQ(state.us_space_track, 8);
    ASSERT_EQ(state.victory_points, 2); // 2 VP first
    ASSERT_TRUE(ts::SpaceRace::has_space_station_ar8(state, ts::Player::US));

    // Turn 5 AR 7: US plays in AR 7
    state.turn = 5;
    state.action_round = 7;
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.phasing_player = ts::Player::US;
    state.china_card_playable = 0; // China card face down
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == ts::hand_of(ts::Player::USSR)) {
            state.card_locations[i] = ts::CardLocation::DISCARD_PILE;
        }
    }
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::hand_of(ts::Player::US);

    // Advance after US AR 7 -> Because US has Space Box 8, 8th AR begins (USSR has 0 cards so auto-passes to US)
    ts::StateMachine::advance_after_action_round(state);
    ASSERT_EQ(state.action_round, 8);
    ASSERT_EQ(state.phasing_player, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_player, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_CARD);

    // USSR catches up to Box 8 -> cancels 8th AR for US
    state.ussr_space_track = 8;
    ASSERT_FALSE(ts::SpaceRace::has_space_station_ar8(state, ts::Player::US));
}

TEST(CardEdgeCasesTest, IndependentReds_NoTargets_FinishesImmediately) {
    ts::GameState state{};
    state.rng_state = 42; state.turn = 1;

    // Clear USSR influence in all 5 allowed countries
    state.countries[ts::countries::YUGOSLAVIA].ussr_influence = 0;
    state.countries[ts::countries::ROMANIA].ussr_influence = 0;
    state.countries[ts::countries::BULGARIA].ussr_influence = 0;
    state.countries[ts::countries::HUNGARY].ussr_influence = 0;
    state.countries[ts::countries::CZECHOSLOVAKIA].ussr_influence = 0;

    // Trigger Independent Reds
    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::INDEPENDENT_REDS, ts::Player::US);
    // When no targets exist, it must finish immediately with done=true and resolving_card=0
    ASSERT_TRUE(done);
    ASSERT_EQ(state.ctx().resolving_card, 0);
}

TEST(CardEdgeCasesTest, IndependentReds_RestrictedTo5AllowedCountries_RejectsCanada) {
    ts::GameState state{};
    state.rng_state = 42; state.turn = 1;

    // Set USSR influence in Romania and Bulgaria only
    state.countries[ts::countries::YUGOSLAVIA].ussr_influence = 0;
    state.countries[ts::countries::ROMANIA].ussr_influence = 2;
    state.countries[ts::countries::BULGARIA].ussr_influence = 1;
    state.countries[ts::countries::HUNGARY].ussr_influence = 0;
    state.countries[ts::countries::CZECHOSLOVAKIA].ussr_influence = 0;
    state.countries[ts::countries::CANADA].ussr_influence = 0;
    state.countries[ts::countries::CANADA].us_influence = 0;

    bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::INDEPENDENT_REDS, ts::Player::US);
    ASSERT_FALSE(done);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::POINT_NODE);
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::INDEPENDENT_REDS);

    // Verify ActionMask only has ROMANIA and BULGARIA marked as legal
    uint8_t mask[128] = {0};
    size_t mask_size = 0;
    ts::ActionMask::generate_mask(state, mask, &mask_size);
    ASSERT_EQ(mask[ts::countries::CANADA], 0); // Canada must be ILLEGAL
    ASSERT_EQ(mask[ts::countries::ROMANIA], 1); // Romania must be legal
    ASSERT_EQ(mask[ts::countries::BULGARIA], 1); // Bulgaria must be legal
    ASSERT_EQ(mask[ts::countries::HUNGARY], 0); // Hungary has 0 USSR influence -> illegal

    // Attempt to target Canada (country 0) -> must be rejected
    bool ok_canada = ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::CANADA, 0, 0));
    ASSERT_FALSE(ok_canada);
    ASSERT_EQ(state.countries[ts::countries::CANADA].us_influence, 0);

    // Target Romania (country 17) -> must succeed and add 2 US influence
    bool ok_romania = ts::CardHandlers::handle_event_step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::ROMANIA, 0, 0));
    ASSERT_TRUE(ok_romania);
    ASSERT_EQ(state.countries[ts::countries::ROMANIA].us_influence, 2);
    ASSERT_EQ(state.ctx().resolving_card, 0);
}

TEST(CardEdgeCasesTest, ChinaCard_CannotBePlayedAsEvent_OpsAndSpaceLegal) {
    ts::GameState state{};
    state.rng_state = 42;
    state.turn = 1;
    state.action_round = 1;
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.phasing_player = ts::Player::USSR;
    state.china_card_holder = ts::Player::USSR;
    state.china_card_playable = 1;
    state.ctx().decision_player = ts::Player::USSR;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;

    // Step 1: USSR selects China Card
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction(ts::DecisionType::SELECT_CARD, ts::card_ids::THE_CHINA_CARD, 0, 0)));
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_PLAY_MODE);
    ASSERT_EQ(state.ctx().pending_op_card, ts::card_ids::THE_CHINA_CARD);

    // The China Card carries no Event of its own, so Event is illegal. Operations and the
    // Space Race are both legal: racing with the best Ops card in the game is a poor play and
    // not an illegal one, and at turn 10 AR4 of ts-replayer game 247 the US races to box 5
    // with it.
    uint8_t mask[128] = {0};
    size_t mask_size = 0;
    ts::ActionMask::generate_mask(state, mask, &mask_size);
    ASSERT_EQ(mask[static_cast<size_t>(ts::Resolution::EVENT)], 0); // Event ILLEGAL
    ASSERT_TRUE((mask[static_cast<size_t>(ts::Resolution::OPS_INFLUENCE)] || mask[static_cast<size_t>(ts::Resolution::OPS_COUP)] || mask[static_cast<size_t>(ts::Resolution::OPS_REALIGN)]));   // Ops LEGAL
    ASSERT_EQ(mask[static_cast<size_t>(ts::Resolution::SPACE)], 1); // Space LEGAL

    // Attempting to step with EVENT mode must be rejected
    bool ok_event = ts::StateMachine::step(state, ts::MicroAction(ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::Resolution::EVENT), 0, 0));
    ASSERT_FALSE(ok_event);
}

TEST(CardEdgeCasesTest, ChinaCard_RacedForSpace_PassesToOpponentAndIsNeverDiscarded) {
    ts::GameState state{};
    state.rng_state = 42;
    state.turn = 10;
    state.action_round = 4;
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.phasing_player = ts::Player::US;
    state.china_card_holder = ts::Player::US;
    state.china_card_playable = 1;
    state.us_space_track = 4;   // box 5 wants 3 Ops; the China Card has 4
    // A card in each hand, so the turn does not end under the attempt and flip the China Card
    // face up again before the assertions below.
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::hand_of(ts::Player::US);
    state.card_locations[ts::card_ids::FIVE_YEAR_PLAN] = ts::hand_of(ts::Player::USSR);
    state.ctx().decision_player = ts::Player::US;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;

    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction(ts::DecisionType::SELECT_CARD, ts::card_ids::THE_CHINA_CARD, 0, 0)));
    bool ok_space = ts::StateMachine::step(state, ts::MicroAction(ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::Resolution::SPACE), 0, 0));
    ASSERT_TRUE(ok_space);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::ROLL_DIE);

    // A roll of 3 makes box 5 (needs 3 or less).
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction(ts::DecisionType::ROLL_DIE, 3, 0, 0)));
    ASSERT_EQ(state.us_space_track, 5);

    // It passes to the opponent face down, and is not in the discard pile.
    ASSERT_EQ(state.china_card_holder, ts::Player::USSR);
    ASSERT_EQ(state.china_card_playable, 0);
    ASSERT_NE(state.card_locations[ts::card_ids::THE_CHINA_CARD], ts::CardLocation::DISCARD_PILE);

    // And it cost an attempt, as any other card would. (The US is past Animal In Space here,
    // so they have a second one this turn.)
    ASSERT_EQ(state.get_space_turns_used(ts::Player::US), 1);
}

TEST(CardEdgeCasesTest, OpponentCard_CannotBePlayedAsEvent_OnlyOpsAndSpaceLegal) {
    ts::GameState state{};
    state.rng_state = 42;
    state.turn = 1;
    state.action_round = 1;
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.phasing_player = ts::Player::USSR;
    state.card_locations[ts::card_ids::DUCK_AND_COVER] = ts::hand_of(ts::Player::USSR); // US Event Card
    state.ctx().decision_player = ts::Player::USSR;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;

    // Step 1: USSR selects US card (Duck and Cover)
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction(ts::DecisionType::SELECT_CARD, ts::card_ids::DUCK_AND_COVER, 0, 0)));
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_PLAY_MODE);

    // Verify ActionMask: EVENT (0) must be 0 for opponent card!
    uint8_t mask[128] = {0};
    size_t mask_size = 0;
    ts::ActionMask::generate_mask(state, mask, &mask_size);
    // P17: EVENT on an OPPONENT's card is legal, and it means EVENT-FIRST -- the event resolves
    // and the Ops choice is deferred. Under the old representation this was reached by choosing
    // OPS and then CHOOSE_TIMING_BRANCH(EVENT_FIRST); the merged node carries the timing, so the
    // option appears here instead of being illegal. What is still impossible is playing an
    // opponent's card for its event INSTEAD of the Ops, and that is checked below by the Ops
    // choice surviving the event.
    ASSERT_EQ(mask[static_cast<size_t>(ts::Resolution::EVENT)], 1); // event-first LEGAL
    ASSERT_TRUE((mask[static_cast<size_t>(ts::Resolution::OPS_INFLUENCE)] || mask[static_cast<size_t>(ts::Resolution::OPS_COUP)] || mask[static_cast<size_t>(ts::Resolution::OPS_REALIGN)]));   // Ops LEGAL
    ASSERT_EQ(mask[static_cast<size_t>(ts::Resolution::SPACE)], 1); // Space LEGAL (3 Ops vs Box 1 min 2)

    // Choosing it resolves the opponent's event and leaves the Ops still to be spent.
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction(ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::Resolution::EVENT), 0, 0)));
    ASSERT_EQ(state.ctx().timing_branch, static_cast<uint8_t>(ts::TimingBranch::EVENT_FIRST));
}

TEST(CardEdgeCasesTest, FormosanResolution_CancelledWhenUSPlaysChinaCard) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.action_round = 1;
    state.phasing_player = ts::Player::US;
    state.china_card_holder = ts::Player::US;
    state.china_card_playable = 1;
    state.set_flag(ts::effect_bits::FORMOSAN_RESOLUTION_ACTIVE);
    state.ctx().decision_player = ts::Player::US;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;

    ASSERT_TRUE(state.has_flag(ts::effect_bits::FORMOSAN_RESOLUTION_ACTIVE));

    // Step 1: US selects China Card
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction(ts::DecisionType::SELECT_CARD, ts::card_ids::THE_CHINA_CARD, 0, 0)));
    // Step 2: US selects OPS mode
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction(ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::Resolution::OPS_INFLUENCE), 0, 0)));

    // Formosan resolution flag is cancelled upon US playing China Card for Ops!
    ASSERT_FALSE(state.has_flag(ts::effect_bits::FORMOSAN_RESOLUTION_ACTIVE));
}

TEST(CardEdgeCasesTest, FormosanResolution_NotCancelledWhenUSSRPlaysChinaCard) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.action_round = 1;
    state.phasing_player = ts::Player::USSR;
    state.china_card_holder = ts::Player::USSR;
    state.china_card_playable = 1;
    state.set_flag(ts::effect_bits::FORMOSAN_RESOLUTION_ACTIVE);
    state.ctx().decision_player = ts::Player::USSR;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;

    ASSERT_TRUE(state.has_flag(ts::effect_bits::FORMOSAN_RESOLUTION_ACTIVE));

    // Step 1: USSR selects China Card
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction(ts::DecisionType::SELECT_CARD, ts::card_ids::THE_CHINA_CARD, 0, 0)));
    // Step 2: USSR selects OPS mode
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction(ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::Resolution::OPS_INFLUENCE), 0, 0)));

    // Formosan resolution flag remains ACTIVE when USSR plays China Card
    ASSERT_TRUE(state.has_flag(ts::effect_bits::FORMOSAN_RESOLUTION_ACTIVE));
}

TEST(CardEdgeCasesTest, DieRollRecord_ResetToNoneOnNextStep) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.action_round = 1;
    state.defcon = 5;
    state.phasing_player = ts::Player::USSR;
    state.ctx().decision_player = ts::Player::USSR;
    state.ctx().decision_type = ts::DecisionType::SELECT_OP_MODE;
    state.ctx().pending_ops_value = 3;

    // Step 1: USSR chooses COUP
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction(ts::DecisionType::SELECT_OP_MODE, 1, 0, 0)));
    ASSERT_EQ(state.last_roll.type, ts::RollType::NONE);

    // Step 2: USSR points node (Iran) -> transitions to ROLL_DIE
    state.countries[ts::countries::IRAN].us_influence = 2;
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::IRAN, 0, 0)));
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::ROLL_DIE);
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction(ts::DecisionType::ROLL_DIE, 5, 0, 0)));

    ASSERT_EQ(state.last_roll.type, ts::RollType::COUP);
    ASSERT_EQ(state.last_roll.roller, ts::Player::USSR);
    ASSERT_EQ(state.last_roll.country_id, ts::countries::IRAN);
    ASSERT_EQ(state.last_roll.roll1, 5);
    ASSERT_EQ(state.last_roll.mod1, 3); // Ops
    ASSERT_TRUE(state.last_roll.success);

    // Step 3: Next step (US turn, SELECT_CARD), last_roll MUST be reset to NONE
    ASSERT_EQ(state.phasing_player, ts::Player::US);
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction(ts::DecisionType::SELECT_CARD, ts::card_ids::DUCK_AND_COVER, 0, 0)));
    ASSERT_EQ(state.last_roll.type, ts::RollType::NONE);
    ASSERT_EQ(state.last_roll.roll1, 0);
}

TEST(CardEdgeCasesTest, DieRollRecord_BrushWar_PopulatedOnTargetResolution) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.action_round = 1;
    state.phasing_player = ts::Player::USSR;
    state.ctx().decision_player = ts::Player::USSR;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;
    state.card_locations[ts::card_ids::BRUSH_WAR] = ts::hand_of(ts::Player::USSR);

    // Step 1: USSR selects Brush War (#36)
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction(ts::DecisionType::SELECT_CARD, ts::card_ids::BRUSH_WAR, 0, 0)));
    ASSERT_EQ(state.last_roll.type, ts::RollType::NONE);

    // Step 2: USSR selects EVENT mode -> transitions to POINT_NODE. NO ROLL YET!
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction(ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::Resolution::EVENT), 0, 0)));
    ASSERT_EQ(state.last_roll.type, ts::RollType::NONE);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::POINT_NODE);
    ASSERT_EQ(state.ctx().resolving_card, ts::card_ids::BRUSH_WAR);

    // Step 3: USSR points to Brazil (#78, stability 2) -> transitions to ROLL_DIE
    state.countries[ts::countries::BRAZIL].us_influence = 2;
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction(ts::DecisionType::POINT_NODE, ts::countries::BRAZIL, 0, 0)));
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::ROLL_DIE);
    ASSERT_TRUE(ts::StateMachine::step(state, ts::MicroAction(ts::DecisionType::ROLL_DIE, 4, 0, 0)));

    ASSERT_EQ(state.last_roll.type, ts::RollType::WAR_EVENT);
    ASSERT_EQ(state.last_roll.roller, ts::Player::USSR);
    ASSERT_EQ(state.last_roll.card_id, ts::card_ids::BRUSH_WAR);
    ASSERT_EQ(state.last_roll.country_id, ts::countries::BRAZIL);
    ASSERT_EQ(state.last_roll.roll1, 4);
    ASSERT_TRUE(state.last_roll.success);
}


// =============================================================================
// COMPLEX EVENT CHAINS & FULL ACTION ROUND RE-ENTRANCY TESTS
// =============================================================================

// Scenario 1:
// US plays Five Year Plan (#5) -> discards Grain Sales (#67) from USSR hand.
// Grain Sales triggers as US event -> US draws Star Wars (#85) from USSR hand and plays it.
// Star Wars triggers -> US retrieves ABM Treaty (#57) from discard pile.
// ABM Treaty triggers -> DEFCON +1, US gets 4 Ops to coup / realign.
// Complete Coup and Realignment -> entire Action Round finishes and advances to USSR AR.
TEST(CardEdgeCasesTest, Chain_FYP_GrainSales_StarWars_ABMTreaty_Full_AR) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);

    // Setup: Turn 7 Mid/Late War, AR 1, US phasing
    state.turn = 7;
    state.action_round = 1;
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.phasing_player = ts::Player::US;
    state.defcon = 4;
    state.us_space_track = 4;
    state.ussr_space_track = 1; // Star Wars prerequisite met (US > USSR space track)

    // Put all cards in draw deck initially
    for (uint8_t i = 1; i <= 110; ++i) state.card_locations[i] = ts::CardLocation::DRAW_DECK;

    // US hand has Five Year Plan (#5)
    state.card_locations[ts::card_ids::FIVE_YEAR_PLAN] = ts::hand_of(ts::Player::US);
    // USSR hand has Grain Sales (#67) and Star Wars (#85)
    state.card_locations[ts::card_ids::GRAIN_SALES] = ts::hand_of(ts::Player::USSR);
    state.card_locations[ts::card_ids::STAR_WARS] = ts::hand_of(ts::Player::USSR);
    // Discard pile has ABM Treaty (#57)
    state.card_locations[ts::card_ids::ABM_TREATY] = ts::CardLocation::DISCARD_PILE;

    // Target battleground: Iran (ID 25) with USSR influence 2, US 0
    state.countries[ts::countries::IRAN].ussr_influence = 2;
    state.countries[ts::countries::IRAN].us_influence = 0;

    // Initial decision context for US Action Round start
    state.ctx().decision_player = ts::Player::US;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;

    // Step 1: US selects Five Year Plan (#5)
    bool ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::FIVE_YEAR_PLAN, 0, 0});
    ASSERT_TRUE(ok);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_PLAY_MODE);

    // Step 2: US selects Play Mode: EVENT
    ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::Resolution::EVENT), 0, 0});
    ASSERT_TRUE(ok);

    // If Grain Sales triggers -> US draws Star Wars and chooses Branch 0 (play drawn card)
    if (state.ctx().decision_type == ts::DecisionType::CHOOSE_BRANCH) {
        ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::CHOOSE_BRANCH, 0, 0, 0});
        ASSERT_TRUE(ok);
    }

    // Now Star Wars prompts US to select card from discard pile
    ASSERT_EQ(state.ctx().decision_player, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_CARD);

    // Step 3: US selects ABM Treaty (#57) from discard pile
    ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::ABM_TREATY, 0, 0});
    ASSERT_TRUE(ok);

    // ABM Treaty triggers: DEFCON improved to 5, US gets 4 Ops
    ASSERT_EQ(state.defcon, 5);
    ASSERT_EQ(state.ctx().decision_player, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_OP_MODE);
    ASSERT_EQ(state.ctx().pending_ops_value, 4);

    // Step 4: US chooses Op Mode: COUP (1)
    ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_OP_MODE, static_cast<uint8_t>(ts::OpMode::COUP), 0, 0});
    ASSERT_TRUE(ok);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::POINT_NODE);

    // Step 5: US targets Iran (ID 25) -> transitions to ROLL_DIE
    ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::IRAN, 0, 0});
    ASSERT_TRUE(ok);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::ROLL_DIE);
    ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::ROLL_DIE, 5, 0, 0});
    ASSERT_TRUE(ok);

    // Iran coup resolved: DEFCON degraded from 5 to 4, US military ops updated
    ASSERT_EQ(state.defcon, 4);
    ASSERT_EQ(state.us_mil_ops, 4);

    // Verification: The entire Action Round has completed cleanly!
    // Next state must be USSR Action Round (AR 1)
    ASSERT_EQ(state.current_phase, ts::Phase::ACTION_ROUND);
    ASSERT_EQ(state.phasing_player, ts::Player::USSR);
    ASSERT_EQ(state.action_round, 2);
    ASSERT_EQ(state.ctx().decision_player, ts::Player::USSR);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_CARD);
}

// Scenario 2:
// US plays Star Wars (#85) -> retrieves Five Year Plan (#5) from discard.
// Five Year Plan discards Grain Sales (#67) from USSR hand.
// Grain Sales draws Glasnost (#90) from USSR hand.
// Scenario 2.1: US plays Glasnost for Coup, then USSR executes Glasnost event / Realignment.
// Full AR completes cleanly.
TEST(CardEdgeCasesTest, Chain_StarWars_FYP_GrainSales_Glasnost_Full_AR) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);

    state.turn = 9;
    state.action_round = 2;
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.phasing_player = ts::Player::US;
    state.defcon = 4;
    state.us_space_track = 5;
    state.ussr_space_track = 2;

    for (uint8_t i = 1; i <= 110; ++i) state.card_locations[i] = ts::CardLocation::DRAW_DECK;

    state.card_locations[ts::card_ids::STAR_WARS] = ts::hand_of(ts::Player::US);
    state.card_locations[ts::card_ids::FIVE_YEAR_PLAN] = ts::CardLocation::DISCARD_PILE;
    state.card_locations[ts::card_ids::GRAIN_SALES] = ts::hand_of(ts::Player::USSR);
    state.card_locations[ts::card_ids::GLASNOST] = ts::hand_of(ts::Player::USSR);

    // Influence in Cuba (ID 67) for Coup target
    state.countries[ts::countries::CUBA].ussr_influence = 3;
    state.countries[ts::countries::CUBA].us_influence = 0;

    // Influence in Poland (ID 3) for realignment target
    state.countries[ts::countries::POLAND].ussr_influence = 2;
    state.countries[ts::countries::POLAND].us_influence = 1;

    state.ctx().decision_player = ts::Player::US;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;

    // 1. US plays Star Wars (#85)
    bool ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::STAR_WARS, 0, 0});
    ASSERT_TRUE(ok);
    ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::Resolution::EVENT), 0, 0});
    ASSERT_TRUE(ok);

    // 2. Star Wars selects Five Year Plan (#5) from discard
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_CARD);
    ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::FIVE_YEAR_PLAN, 0, 0});
    ASSERT_TRUE(ok);

    // 3. FYP discards Grain Sales (#67) -> Grain Sales draws Glasnost (#90) -> Prompt branch 0
    if (state.ctx().decision_type == ts::DecisionType::CHOOSE_BRANCH) {
        ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::CHOOSE_BRANCH, 0, 0, 0});
        ASSERT_TRUE(ok);
    }

    // 4. US now plays Glasnost (#90) for OPS
    if (state.ctx().decision_type == ts::DecisionType::SELECT_PLAY_MODE) {
        ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::Resolution::OPS_INFLUENCE), 0, 0});
        ASSERT_TRUE(ok);
    }

    // If timing branch prompt appears: US chooses OPS_FIRST (0)
    // P17: choosing an OPS_* resolution on an opponent's card IS ops-first, so the separate
    // timing step this replaced no longer exists.
    ASSERT_EQ(state.ctx().timing_branch, static_cast<uint8_t>(ts::TimingBranch::OPS_FIRST));

    // 5. US selects Op Mode: COUP (1)
    if (state.ctx().decision_type == ts::DecisionType::SELECT_OP_MODE) {
        ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_OP_MODE, static_cast<uint8_t>(ts::OpMode::COUP), 0, 0});
        ASSERT_TRUE(ok);
    }

    // 6. US targets Cuba (ID 67) with roll 4
    if (state.ctx().decision_type == ts::DecisionType::POINT_NODE) {
        ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::CUBA, 0, 0});
        ASSERT_TRUE(ok);
        if (state.ctx().decision_type == ts::DecisionType::ROLL_DIE) {
            ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::ROLL_DIE, 4, 0, 0});
            ASSERT_TRUE(ok);
        }
    }

    // 7. USSR Glasnost event triggers (USSR gets Realignment or Ops if prompted)
    while (state.ctx().decision_player == ts::Player::USSR && state.ctx().decision_type != ts::DecisionType::SELECT_CARD) {
        if (state.ctx().decision_type == ts::DecisionType::POINT_NODE) {
            if (state.ctx().allow_early_stop) {
                ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, 255, 0, ts::action_flags::CONFIRM_DONE});
            } else {
                ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::POLAND, 0, 0});
            }
            ASSERT_TRUE(ok);
        } else if (state.ctx().decision_type == ts::DecisionType::CHOOSE_BRANCH) {
            ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::CHOOSE_BRANCH, 0, 0, 0});
            ASSERT_TRUE(ok);
        } else if (state.ctx().decision_type == ts::DecisionType::SELECT_OP_MODE) {
            ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_OP_MODE, static_cast<uint8_t>(ts::OpMode::REALIGN), 0, 0});
            ASSERT_TRUE(ok);
        } else if (state.ctx().decision_type == ts::DecisionType::ROLL_DIE || state.ctx().decision_player == ts::Player::NONE) {
            ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::ROLL_DIE, 0, 0, 0});
            ASSERT_TRUE(ok);
        } else {
            break;
        }
    }

    // Verification: AR finishes and advances to USSR turn
    ASSERT_EQ(state.current_phase, ts::Phase::ACTION_ROUND);
    ASSERT_EQ(state.phasing_player, ts::Player::USSR);
    ASSERT_EQ(state.action_round, 3);
    ASSERT_EQ(state.ctx().decision_player, ts::Player::USSR);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_CARD);
}

// Scenario 3:
// USSR plays Grain Sales (#67) as Ops.
// US event triggers on Grain Sales -> draws Star Wars (#85) from USSR hand.
// US uses Star Wars to retrieve Five Year Plan (#5) from discard.
// Five Year Plan discards Soviets Shoot Down KAL-007 (#89) from USSR hand.
// KAL-007 triggers as US event -> US conducts Realignment in South Korea/Asia.
// Once complete, USSR conducts its Grain Sales Ops and the entire AR completes cleanly.
TEST(CardEdgeCasesTest, Chain_USSR_GrainSales_StarWars_FYP_KAL007_Full_AR) {
    ts::GameState state{};
    ts::StateMachine::init_new_game(state, 42);

    state.turn = 9;
    state.action_round = 3;
    state.current_phase = ts::Phase::ACTION_ROUND;
    state.phasing_player = ts::Player::USSR;
    state.defcon = 4;
    state.us_space_track = 6;
    state.ussr_space_track = 3;

    for (uint8_t i = 1; i <= 110; ++i) state.card_locations[i] = ts::CardLocation::DRAW_DECK;

    state.card_locations[ts::card_ids::GRAIN_SALES] = ts::hand_of(ts::Player::USSR);
    state.card_locations[ts::card_ids::FIVE_YEAR_PLAN] = ts::CardLocation::DISCARD_PILE;
    state.card_locations[ts::card_ids::STAR_WARS] = ts::hand_of(ts::Player::USSR);

    // Influence in South Korea and Japan
    state.countries[ts::countries::SOUTH_KOREA].us_influence = 4;
    state.countries[ts::countries::SOUTH_KOREA].ussr_influence = 1;
    state.countries[ts::countries::JAPAN].us_influence = 3;

    // Target for USSR ops later: Afghanistan
    state.countries[ts::countries::AFGHANISTAN].ussr_influence = 0;

    state.ctx().decision_player = ts::Player::USSR;
    state.ctx().decision_type = ts::DecisionType::SELECT_CARD;

    // 1. USSR plays Grain Sales (#67) for Ops
    bool ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::GRAIN_SALES, 0, 0});
    ASSERT_TRUE(ok);
    // 2. P17: Grain Sales is an opponent card for the USSR, and this test wants EVENT-FIRST --
    //    the event resolves and the Ops are spent afterwards. That is now the EVENT option at
    //    the merged node; the separate CHOOSE_TIMING_BRANCH step it replaced is gone.
    ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::Resolution::EVENT), 0, 0});
    ASSERT_TRUE(ok);
    // No timing_branch assertion here: Grain Sales' event opens a sub-decision, so ctx() is the
    // pushed event frame and the staged Ops frame carrying EVENT_FIRST is below it on the stack.
    // The De Gaulle regression above checks that field, on an event that completes immediately.

    // 3. Grain Sales US prompt: Branch 0 (play drawn Star Wars #85)
    if (state.ctx().decision_type == ts::DecisionType::CHOOSE_BRANCH) {
        ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::CHOOSE_BRANCH, 0, 0, 0});
        ASSERT_TRUE(ok);
    }

    // Give USSR KAL-007 (#89) before Five Year Plan executes
    state.card_locations[ts::card_ids::SOVIETS_SHOOT_DOWN_KAL_007] = ts::hand_of(ts::Player::USSR);

    // 3b. US selects Play Mode for Star Wars (#85): EVENT (0)
    if (state.ctx().decision_type == ts::DecisionType::SELECT_PLAY_MODE) {
        ASSERT_EQ(state.ctx().decision_player, ts::Player::US);
        ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(ts::Resolution::EVENT), 0, 0});
        ASSERT_TRUE(ok);
    }

    // 4. Star Wars prompts US to select card from discard -> Selects Five Year Plan (#5)
    ASSERT_EQ(state.ctx().decision_player, ts::Player::US);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_CARD);
    ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_CARD, ts::card_ids::FIVE_YEAR_PLAN, 0, 0});
    ASSERT_TRUE(ok);

    // 5. FYP discards KAL-007 (#89) -> KAL-007 triggers as US event (US gets Ops for realignment)
    while (state.ctx().decision_player == ts::Player::US && state.ctx().decision_type != ts::DecisionType::SELECT_CARD) {
        if (state.ctx().decision_type == ts::DecisionType::SELECT_OP_MODE) {
            ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_OP_MODE, static_cast<uint8_t>(ts::OpMode::REALIGN), 0, 0});
            ASSERT_TRUE(ok);
        } else if (state.ctx().decision_type == ts::DecisionType::POINT_NODE) {
            if (state.ctx().allow_early_stop) {
                ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, 255, 0, ts::action_flags::CONFIRM_DONE});
            } else {
                ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::SOUTH_KOREA, 0, 0});
            }
            ASSERT_TRUE(ok);
        } else if (state.ctx().decision_type == ts::DecisionType::CHOOSE_BRANCH) {
            ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::CHOOSE_BRANCH, 0, 0, 0});
            ASSERT_TRUE(ok);
        } else {
            break;
        }
    }

    // 6. After US event chain completes, USSR gets to conduct its Grain Sales Ops (2 Ops)
    while (state.ctx().decision_player == ts::Player::USSR && state.ctx().decision_type != ts::DecisionType::SELECT_CARD) {
        if (state.ctx().decision_type == ts::DecisionType::SELECT_OP_MODE) {
            ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::SELECT_OP_MODE, static_cast<uint8_t>(ts::OpMode::INFLUENCE), 0, 0});
            ASSERT_TRUE(ok);
        } else if (state.ctx().decision_type == ts::DecisionType::POINT_NODE) {
            ok = ts::StateMachine::step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::AFGHANISTAN, 0, 0});
            ASSERT_TRUE(ok);
        } else {
            break;
        }
    }

    // Verification: AR finishes cleanly and advances to next turn
    ASSERT_EQ(state.current_phase, ts::Phase::ACTION_ROUND);
    ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::SELECT_CARD);
}

TEST(CardEdgeCasesTest, WarEvents_UnifiedWrapper_Suite) {
    // 1. Arab-Israeli War disabled by Camp David
    {
        ts::GameState state;
        ts::Engine::init_game(state, 42);
        state.set_flag(ts::effect_bits::CAMP_DAVID_PLAYED);
        bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::ARAB_ISRAELI_WAR, ts::Player::USSR);
        ASSERT_TRUE(done); // Immediately completed with no effect
    }

    // 2. Arab-Israeli War checks Israel control & modifiers
    {
        ts::GameState state;
        ts::Engine::init_game(state, 42);
        state.countries[ts::countries::ISRAEL].us_influence = 4; // US controls Israel (stab 4) -> -1 modifier
        state.countries[ts::countries::EGYPT].us_influence = 2;  // US controls Egypt (stab 2) -> -1 modifier
        state.ussr_mil_ops = 0;
        state.victory_points = 0;

        // Roll 5 + (-2 mod) = 3 < 4 -> Failure
        bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::ARAB_ISRAELI_WAR, ts::Player::USSR);
        ASSERT_FALSE(done);
        ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::ROLL_DIE);
        done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::ROLL_DIE, 5, 0, 0});
        ASSERT_TRUE(done);
        ASSERT_EQ(state.ussr_mil_ops, 2); // MilOps awarded regardless of outcome
        ASSERT_EQ(state.victory_points, 0); // No VP on failure
        ASSERT_EQ(state.countries[ts::countries::ISRAEL].us_influence, 4);

        // Roll 6 + (-2 mod) = 4 >= 4 -> Success
        done = ts::CardHandlers::trigger_event(state, ts::card_ids::ARAB_ISRAELI_WAR, ts::Player::USSR);
        ASSERT_FALSE(done);
        done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::ROLL_DIE, 6, 0, 0});
        ASSERT_TRUE(done);
        ASSERT_EQ(state.victory_points, -2);
        ASSERT_EQ(state.countries[ts::countries::ISRAEL].us_influence, 0);
        ASSERT_EQ(state.countries[ts::countries::ISRAEL].ussr_influence, 4);
    }

    // 3. Indo-Pakistani War Target Selection & Flower Power
    {
        ts::GameState state;
        ts::Engine::init_game(state, 42);
        state.set_flag(ts::effect_bits::FLOWER_POWER_ACTIVE);
        state.countries[ts::countries::INDIA].ussr_influence = 3;
        state.us_mil_ops = 0;
        state.victory_points = 0;

        bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::INDO_PAKISTANI_WAR, ts::Player::US);
        ASSERT_FALSE(done);
        ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::POINT_NODE);

        // Invalid target (Canada) rejected
        bool step_ok = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::CANADA, 0, 0});
        ASSERT_FALSE(step_ok);
        ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::POINT_NODE);

        // Target India with roll 5 -> Success -> +2 VP for war, -2 VP for Flower Power -> net 0 VP
        step_ok = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::INDIA, 0, 0});
        ASSERT_FALSE(step_ok);
        ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::ROLL_DIE);

        done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::ROLL_DIE, 5, 0, 0});
        ASSERT_TRUE(done);
        ASSERT_EQ(state.us_mil_ops, 2);
        ASSERT_EQ(state.victory_points, 0); // +2 war - 2 Flower Power = 0
        ASSERT_EQ(state.countries[ts::countries::INDIA].ussr_influence, 0);
        ASSERT_EQ(state.countries[ts::countries::INDIA].us_influence, 3);
    }

    // 4. Brush War NATO Protection, Stability Validation & MilOps +3
    {
        ts::GameState state;
        ts::Engine::init_game(state, 42);
        state.set_flag(ts::effect_bits::NATO_ACTIVE);
        state.countries[ts::countries::GREECE].us_influence = 2; // US controls Greece (Europe)
        state.countries[ts::countries::MEXICO].us_influence = 2; // CA, stab 2
        state.countries[ts::countries::ITALY].us_influence = 3;  // stab 3 (illegal for Brush War)
        state.ussr_mil_ops = 0;
        state.victory_points = 0;

        bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::BRUSH_WAR, ts::Player::USSR);
        ASSERT_FALSE(done);
        ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::POINT_NODE);

        // Target Italy (stab 3) rejected
        bool step_ok = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::ITALY, 0, 0});
        ASSERT_FALSE(step_ok);

        // Target Greece (protected by NATO) rejected for USSR
        step_ok = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::GREECE, 0, 0});
        ASSERT_FALSE(step_ok);

        // Target Mexico (stab 2) accepted -> roll 3 is success -> +3 MilOps, 1 VP
        step_ok = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::MEXICO, 0, 0});
        ASSERT_FALSE(step_ok);
        ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::ROLL_DIE);

        done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::ROLL_DIE, 3, 0, 0});
        ASSERT_TRUE(done);
        ASSERT_EQ(state.ussr_mil_ops, 3); // Brush war gives 3 MilOps
        ASSERT_EQ(state.victory_points, -1); // USSR gets 1 VP
        ASSERT_EQ(state.countries[ts::countries::MEXICO].us_influence, 0);
        ASSERT_EQ(state.countries[ts::countries::MEXICO].ussr_influence, 2);
    }

    // 5. Iran-Iraq War Target Selection
    {
        ts::GameState state;
        ts::Engine::init_game(state, 42);
        state.countries[ts::countries::IRAN].us_influence = 0;
        state.countries[ts::countries::IRAN].ussr_influence = 2;
        state.us_mil_ops = 0;
        state.victory_points = 0;

        bool done = ts::CardHandlers::trigger_event(state, ts::card_ids::IRAN_IRAQ_WAR, ts::Player::US);
        ASSERT_FALSE(done);
        ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::POINT_NODE);

        // Invalid target (Israel) rejected
        bool step_ok = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::ISRAEL, 0, 0});
        ASSERT_FALSE(step_ok);

        // Target Iran accepted -> roll 4 -> Success -> 2 VP, 2 MilOps
        step_ok = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::POINT_NODE, ts::countries::IRAN, 0, 0});
        ASSERT_FALSE(step_ok);
        ASSERT_EQ(state.ctx().decision_type, ts::DecisionType::ROLL_DIE);

        done = ts::CardHandlers::handle_event_step(state, ts::MicroAction{ts::DecisionType::ROLL_DIE, 4, 0, 0});
        ASSERT_TRUE(done);
        ASSERT_EQ(state.us_mil_ops, 2);
        ASSERT_EQ(state.victory_points, 2);
        ASSERT_EQ(state.countries[ts::countries::IRAN].ussr_influence, 0);
        ASSERT_EQ(state.countries[ts::countries::IRAN].us_influence, 2);
    }
}
