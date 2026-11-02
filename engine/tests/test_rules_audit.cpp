// Regression tests for the rules audit against the card text, the Deluxe rulebook, the FAQ and
// the struggler engine's rulings table (docs/RULES_SOURCES.md there). One test per defect, each
// written to fail on the engine as it stood before the fix.
#include "test_framework.hpp"
#include "ts/engine.hpp"
#include "ts/state_machine.hpp"
#include "ts/action_mask.hpp"
#include "ts/card_handlers.hpp"
#include "ts/card_data.hpp"
#include "ts/map_data.hpp"
#include "ts/scoring.hpp"
#include "ts/ops.hpp"

using namespace ts;

namespace {

// A US or USSR action round, nothing in either hand. The caller deals what it needs.
GameState action_round_for(Player p) {
    GameState state{};
    Engine::init_game(state, 42);
    for (uint8_t c = 1; c <= 110; ++c) {
        if (is_in_any_hand(state.card_locations[c])) state.card_locations[c] = CardLocation::DRAW_DECK;
    }
    state.current_phase = Phase::ACTION_ROUND;
    state.action_round = 2;
    state.turn = 4;
    state.phasing_player = p;
    state.ctx() = DecisionContext{};
    state.ctx().decision_player = p;
    state.ctx().decision_type = DecisionType::SELECT_CARD;
    return state;
}

// Resolve a war event against `target` with the die forced to `roll`.
void fight(GameState& state, uint8_t card, Player attacker, uint8_t target, uint8_t roll) {
    ASSERT_FALSE(CardHandlers::trigger_event(state, card, attacker));
    ASSERT_FALSE(CardHandlers::handle_event_step(
        state, MicroAction{DecisionType::POINT_NODE, target, 0, 0}));
    ASSERT_TRUE(CardHandlers::handle_event_step(
        state, MicroAction{DecisionType::ROLL_DIE, roll, 0, 0}));
}

} // namespace

// --- War rolls: the defender's superpower is an adjacent controlled country -------------------

TEST(RulesAudit, BrushWarOnMexico_UnitedStatesCountsAgainstTheUSSR) {
    // Mexico's only neighbours are Guatemala and the United States. With Guatemala uncontrolled
    // the one -1 is the superpower's: a 3 fails where it used to succeed, and a 4 succeeds.
    GameState s{};
    s.countries[countries::MEXICO].us_influence = 1;
    fight(s, card_ids::BRUSH_WAR, Player::USSR, countries::MEXICO, 3);
    ASSERT_EQ(s.victory_points, 0);
    ASSERT_EQ(s.countries[countries::MEXICO].us_influence, 1);
    ASSERT_EQ(s.last_roll.mod1, -1);

    GameState t{};
    t.countries[countries::MEXICO].us_influence = 1;
    fight(t, card_ids::BRUSH_WAR, Player::USSR, countries::MEXICO, 4);
    ASSERT_EQ(t.victory_points, -1);
    ASSERT_EQ(t.countries[countries::MEXICO].ussr_influence, 1);
}

TEST(RulesAudit, BrushWarOnAfghanistan_SovietUnionCountsAgainstTheUS) {
    GameState s{};
    s.countries[countries::AFGHANISTAN].ussr_influence = 1;
    fight(s, card_ids::BRUSH_WAR, Player::US, countries::AFGHANISTAN, 3);
    ASSERT_EQ(s.victory_points, 0);
    ASSERT_EQ(s.last_roll.mod1, -1);
}

TEST(RulesAudit, WarRoll_OwnSuperpowerGivesTheAttackerNothing) {
    // Only the defender's superpower counts: the USSR attacking Afghanistan has no penalty.
    GameState s{};
    s.countries[countries::AFGHANISTAN].us_influence = 1;
    fight(s, card_ids::BRUSH_WAR, Player::USSR, countries::AFGHANISTAN, 3);
    ASSERT_EQ(s.last_roll.mod1, 0);
    ASSERT_EQ(s.victory_points, -1);
}

// --- UN Intervention needs an opponent-associated companion to be played as an Event ----------

TEST(RulesAudit, UNIntervention_NotAnEventWithoutACompanion) {
    GameState s = action_round_for(Player::US);
    s.card_locations[card_ids::UN_INTERVENTION] = hand_of(Player::US);
    s.card_locations[card_ids::DUCK_AND_COVER] = hand_of(Player::US);   // a US card: no companion
    ASSERT_TRUE(Engine::step(s, MicroAction{DecisionType::SELECT_CARD, card_ids::UN_INTERVENTION, 0, 0}));
    ASSERT_EQ(s.ctx().decision_type, DecisionType::SELECT_PLAY_MODE);

    uint8_t mask[FLAT_ACTION_SPACE_SIZE];
    ActionMask::generate_flat_mask_212(s, mask);
    ASSERT_EQ(mask[flat_slots::RESOLUTION + static_cast<size_t>(Resolution::EVENT)], 0);
    ASSERT_EQ(mask[flat_slots::OPS_INFLUENCE], 1);

    // With a USSR card in hand it is an Event again.
    GameState t = action_round_for(Player::US);
    t.card_locations[card_ids::UN_INTERVENTION] = hand_of(Player::US);
    t.card_locations[card_ids::DE_GAULLE] = hand_of(Player::US);
    ASSERT_TRUE(Engine::step(t, MicroAction{DecisionType::SELECT_CARD, card_ids::UN_INTERVENTION, 0, 0}));
    ActionMask::generate_flat_mask_212(t, mask);
    ASSERT_EQ(mask[flat_slots::RESOLUTION + static_cast<size_t>(Resolution::EVENT)], 1);
}

TEST(RulesAudit, UNIntervention_WithoutACompanionCannotDodgeWeWillBuryYou) {
    // The exploit the fizzle allowed: spend the named round on an empty UN Intervention and the
    // 3 VP never came. Playing it for Operations is all that is left, and that pays.
    GameState s = action_round_for(Player::US);
    s.set_flag(effect_bits::WE_WILL_BURY_YOU_PENDING);
    s.card_locations[card_ids::UN_INTERVENTION] = hand_of(Player::US);
    s.card_locations[card_ids::DUCK_AND_COVER] = hand_of(Player::US);
    ASSERT_TRUE(Engine::step(s, MicroAction{DecisionType::SELECT_CARD, card_ids::UN_INTERVENTION, 0, 0}));
    ASSERT_FALSE(Engine::step(s, MicroAction{DecisionType::SELECT_PLAY_MODE,
                                             static_cast<uint8_t>(Resolution::EVENT), 0, 0}));
    ASSERT_TRUE(Engine::step(s, MicroAction{DecisionType::SELECT_PLAY_MODE,
                                            static_cast<uint8_t>(Resolution::OPS_INFLUENCE), 0, 0}));
    ASSERT_EQ(s.victory_points, -3);
    ASSERT_FALSE(s.has_flag(effect_bits::WE_WILL_BURY_YOU_PENDING));
}

// --- We Will Bury You on a trapped round -------------------------------------------------------

TEST(RulesAudit, WeWillBuryYou_SettlesOnAQuagmireDiscardRound) {
    GameState s = action_round_for(Player::US);
    s.set_flag(effect_bits::QUAGMIRE_ACTIVE);
    s.set_flag(effect_bits::WE_WILL_BURY_YOU_PENDING);
    s.card_locations[card_ids::DUCK_AND_COVER] = hand_of(Player::US);   // 3 Ops, a legal discard
    ASSERT_TRUE(Engine::step(s, MicroAction{DecisionType::SELECT_CARD, card_ids::DUCK_AND_COVER, 0, 0}));
    // The round is spent on the discard, so UN Intervention was not played in it: 3 VP now,
    // not in whatever later round the US next plays a card.
    ASSERT_EQ(s.victory_points, -3);
    ASSERT_FALSE(s.has_flag(effect_bits::WE_WILL_BURY_YOU_PENDING));
    ASSERT_EQ(s.ctx().decision_type, DecisionType::ROLL_DIE);
}

TEST(RulesAudit, WeWillBuryYou_SettlesOnAQuagmireRoundWithNoLegalDiscard) {
    // Nothing in the US hand is worth 2 Ops, so the trap takes nothing and the round is passed.
    // It is still the US's "next action round" and UN Intervention was not played in it: the
    // 3 VP are owed now, not deferred to the next round the US plays a card in.
    GameState s = action_round_for(Player::US);
    s.set_flag(effect_bits::QUAGMIRE_ACTIVE);
    s.set_flag(effect_bits::WE_WILL_BURY_YOU_PENDING);
    s.card_locations[card_ids::TRUMAN_DOCTRINE] = hand_of(Player::US);   // 1 Op: not a legal discard

    uint8_t mask[256];
    size_t mask_size = 0;
    ActionMask::generate_mask(s, mask, &mask_size);
    ASSERT_EQ(mask[0], 1);                              // only the pass is offered
    ASSERT_EQ(mask[card_ids::TRUMAN_DOCTRINE], 0);

    ASSERT_TRUE(Engine::step(s, MicroAction{DecisionType::SELECT_CARD, 0, 0, 0}));
    ASSERT_EQ(s.victory_points, -3);
    ASSERT_FALSE(s.has_flag(effect_bits::WE_WILL_BURY_YOU_PENDING));
    ASSERT_EQ(s.card_locations[card_ids::TRUMAN_DOCTRINE], hand_of(Player::US));   // nothing was taken
    ASSERT_EQ(s.phasing_player, Player::USSR);           // the round is over
}

// --- NORAD arms on a move to DEFCON 2, not on a setting of 2 at 2 -----------------------------

TEST(RulesAudit, CubanMissileCrisisAtDefcon2_DoesNotArmNorad) {
    GameState s{};
    s.defcon = 2;
    CardHandlers::trigger_event(s, card_ids::CUBAN_MISSILE_CRISIS, Player::USSR);
    ASSERT_EQ(s.defcon, 2);
    ASSERT_EQ(s.defcon_dropped_to_2, 0);

    GameState t{};
    t.defcon = 4;
    CardHandlers::trigger_event(t, card_ids::CUBAN_MISSILE_CRISIS, Player::USSR);
    ASSERT_EQ(t.defcon_dropped_to_2, 1);
}

TEST(RulesAudit, HowILearnedSettingTwoAtTwo_DoesNotArmNorad) {
    GameState s{};
    s.defcon = 2;
    ASSERT_FALSE(CardHandlers::trigger_event(s, card_ids::HOW_I_LEARNED_TO_STOP_WORRYING, Player::US));
    ASSERT_TRUE(CardHandlers::handle_event_step(
        s, MicroAction{DecisionType::CHOOSE_BRANCH, 2, 0, action_flags::DEFCON_VALUE}));
    ASSERT_EQ(s.defcon_dropped_to_2, 0);

    GameState t{};
    t.defcon = 4;
    ASSERT_FALSE(CardHandlers::trigger_event(t, card_ids::HOW_I_LEARNED_TO_STOP_WORRYING, Player::US));
    ASSERT_TRUE(CardHandlers::handle_event_step(
        t, MicroAction{DecisionType::CHOOSE_BRANCH, 2, 0, action_flags::DEFCON_VALUE}));
    ASSERT_EQ(t.defcon_dropped_to_2, 1);
}

// --- U-2 Incident's rider on the Grain Sales route ---------------------------------------------

TEST(RulesAudit, GrainSalesUNIntervention_PaysTheU2Rider) {
    GameState s = action_round_for(Player::US);
    s.set_flag(effect_bits::U2_INCIDENT_ACTIVE);
    s.card_locations[card_ids::DE_GAULLE] = hand_of(Player::USSR);   // the only card Grain Sales can draw
    s.card_locations[card_ids::UN_INTERVENTION] = hand_of(Player::US);
    s.ctx().decision_player = Player::US;
    ASSERT_FALSE(CardHandlers::trigger_event(s, card_ids::GRAIN_SALES, Player::US));
    ASSERT_EQ(s.ctx().pending_op_card, card_ids::DE_GAULLE);
    ASSERT_FALSE(CardHandlers::handle_event_step(
        s, MicroAction{DecisionType::SELECT_CARD, card_ids::UN_INTERVENTION, 0, 0}));
    ASSERT_EQ(s.victory_points, -1);
    ASSERT_EQ(s.ctx().decision_type, DecisionType::SELECT_OP_MODE);
}

TEST(RulesAudit, GrainSalesUNIntervention_OnAWarCardChargesNoFlowerPower) {
    GameState s = action_round_for(Player::US);
    s.set_flag(effect_bits::FLOWER_POWER_ACTIVE);
    s.card_locations[card_ids::KOREAN_WAR] = hand_of(Player::USSR);
    s.card_locations[card_ids::UN_INTERVENTION] = hand_of(Player::US);
    ASSERT_FALSE(CardHandlers::trigger_event(s, card_ids::GRAIN_SALES, Player::US));
    ASSERT_FALSE(CardHandlers::handle_event_step(
        s, MicroAction{DecisionType::SELECT_CARD, card_ids::UN_INTERVENTION, 0, 0}));
    // A lent war card is not played for its own Ops or Event (ts-replayer game 304, T7 AR7).
    ASSERT_EQ(s.victory_points, 0);
}

// --- Summit counts regions without the scoring-only card effects -------------------------------

namespace {

// USSR controls three of the six Middle East battlegrounds (Iraq, Libya, Egypt) and Syria; the
// US controls two battlegrounds (Israel, Iran) and Jordan. USSR Domination, 4 countries to 3 --
// unless Shuttle Diplomacy takes a battleground and a country off its count, which leaves 3 to 3
// and no one dominating.
GameState middle_east_on_a_knife_edge() {
    GameState s{};
    auto ussr = [&](uint8_t c) { s.countries[c].ussr_influence = MapData::get_country(c).stability; };
    auto us = [&](uint8_t c) { s.countries[c].us_influence = MapData::get_country(c).stability; };
    ussr(countries::SYRIA); ussr(countries::IRAQ); ussr(countries::LIBYA); ussr(countries::EGYPT);
    us(countries::ISRAEL); us(countries::IRAN); us(countries::JORDAN);
    return s;
}

} // namespace

TEST(RulesAudit, Summit_IgnoresShuttleDiplomacy) {
    GameState s = middle_east_on_a_knife_edge();
    ASSERT_TRUE(Scoring::dominates_or_controls(s, Region::MIDDLE_EAST, Player::USSR));
    s.set_flag(effect_bits::SHUTTLE_DIPLOMACY_ACTIVE);
    // A scoring would drop the USSR out of Domination...
    ASSERT_EQ(Scoring::evaluate_region(s, Region::MIDDLE_EAST).ussr_status, RegionalStatus::PRESENCE);
    // ...Summit does not.
    ASSERT_TRUE(Scoring::dominates_or_controls(s, Region::MIDDLE_EAST, Player::USSR));

    // And through the event: dice 3 each, the USSR's one region decides it.
    ASSERT_FALSE(CardHandlers::trigger_event(s, card_ids::SUMMIT, Player::US));
    ASSERT_FALSE(CardHandlers::handle_event_step(s, MicroAction{DecisionType::ROLL_DIE, 3, 3, 0}));
    ASSERT_EQ(s.victory_points, -2);
    ASSERT_EQ(s.ctx().decision_player, Player::USSR);
}

TEST(RulesAudit, Summit_IgnoresFormosanResolution) {
    // Asia with Taiwan US-controlled: under Formosan Resolution a scoring counts Taiwan as a
    // battleground. The US holds three countries and two battlegrounds to the USSR's two and
    // one; with Taiwan a battleground that is 3 to 1 either way, so pick a split Taiwan tips.
    GameState s{};
    auto us = [&](uint8_t c) { s.countries[c].us_influence = MapData::get_country(c).stability; };
    auto ussr = [&](uint8_t c) { s.countries[c].ussr_influence = MapData::get_country(c).stability; };
    // US: Taiwan (non-BG), Japan (BG). USSR: North Korea (BG), Afghanistan (non-BG).
    us(countries::TAIWAN); us(countries::JAPAN);
    ussr(countries::NORTH_KOREA); ussr(countries::AFGHANISTAN);
    // US adds a third country so it has more countries: Australia (non-BG).
    us(countries::AUSTRALIA);
    // Without Formosan: US 3 countries, 1 BG; USSR 2 countries, 1 BG -> no US Domination.
    ASSERT_FALSE(Scoring::dominates_or_controls(s, Region::ASIA, Player::US));
    s.set_flag(effect_bits::FORMOSAN_RESOLUTION_ACTIVE);
    // A scoring counts Taiwan: 2 BGs to 1 -> Domination.
    ASSERT_EQ(Scoring::evaluate_region(s, Region::ASIA).us_status, RegionalStatus::DOMINATION);
    // Summit does not.
    ASSERT_FALSE(Scoring::dominates_or_controls(s, Region::ASIA, Player::US));
}

// --- Event Ops and the Vietnam Revolts ladder --------------------------------------------------

TEST(RulesAudit, OlympicBoycottUnderVietnamRevolts_KeepsFourOutsideSoutheastAsia) {
    // The USSR sponsors, the US boycotts: 4 Ops, +1 while they stay in Southeast Asia. The
    // ladder fell back to the Olympic Games' printed 2 on the first placement outside, so the
    // USSR lost two of the four Ops the event grants.
    GameState s = action_round_for(Player::USSR);
    s.defcon = 4;
    s.set_flag(effect_bits::VIETNAM_REVOLTS_ACTIVE);
    s.ctx().decision_player = Player::USSR;
    ASSERT_FALSE(CardHandlers::trigger_event(s, card_ids::OLYMPIC_GAMES, Player::USSR));
    ASSERT_FALSE(CardHandlers::handle_event_step(s, MicroAction{DecisionType::CHOOSE_BRANCH, 1, 0, 0}));
    ASSERT_EQ(s.ctx().pending_ops_value, 5);
    ASSERT_EQ(s.ctx().decision_type, DecisionType::SELECT_OP_MODE);
    ASSERT_TRUE(Engine::step(s, MicroAction{DecisionType::SELECT_OP_MODE,
                                            static_cast<uint8_t>(OpMode::INFLUENCE), 0, 0}));
    // East Germany has USSR Influence from setup.
    ASSERT_TRUE(Engine::step(s, MicroAction{DecisionType::POINT_NODE, countries::EAST_GERMANY, 0, 0}));
    ASSERT_EQ(s.ctx().pending_ops_value, 4);
    ASSERT_EQ(s.ctx().remaining_steps, 3);
}

TEST(RulesAudit, OlympicBoycottUnderVietnamRevoltsAndPurge_FallsBackToTheModifiedGrant) {
    // The fallback is the grant as modified (rule 7.4.3): 4 less Purge's 1 is 3, plus Vietnam
    // Revolts while in Southeast Asia is 4. Leaving the subregion falls back to 3 -- not to the
    // card's printed 2, and not to an unmodified 4.
    GameState s = action_round_for(Player::USSR);
    s.defcon = 4;
    s.set_flag(effect_bits::VIETNAM_REVOLTS_ACTIVE);
    s.set_flag(effect_bits::PURGE_USSR_ACTIVE);
    ASSERT_FALSE(CardHandlers::trigger_event(s, card_ids::OLYMPIC_GAMES, Player::USSR));
    ASSERT_FALSE(CardHandlers::handle_event_step(s, MicroAction{DecisionType::CHOOSE_BRANCH, 1, 0, 0}));
    ASSERT_EQ(s.ctx().pending_ops_value, 4);
    ASSERT_TRUE(Engine::step(s, MicroAction{DecisionType::SELECT_OP_MODE,
                                            static_cast<uint8_t>(OpMode::INFLUENCE), 0, 0}));
    ASSERT_TRUE(Engine::step(s, MicroAction{DecisionType::POINT_NODE, countries::EAST_GERMANY, 0, 0}));
    ASSERT_EQ(s.ctx().pending_ops_value, 3);
    ASSERT_EQ(s.ctx().remaining_steps, 2);
}

// --- A card played on a pushed frame (ENG-6) ---------------------------------------------------
//
// Grain Sales fired through another card's Event plays the card it draws on the frame pushed for it.
// Ops spent on a pushed frame used to be taken for Ops an Event granted: the card stayed in the hand
// it was drawn into, playable again, and an opponent's card owed no Event.

namespace {

void take(GameState& s, DecisionType t, uint8_t id) {
    ASSERT_TRUE(StateMachine::step(s, MicroAction{t, id, 0, 0}));
}

// The US plays Star Wars as its Event and takes Grain Sales, which draws `drawn`, the only card the
// USSR holds. Filler keeps the US in the turn.
GameState star_wars_into_grain_sales(uint8_t drawn) {
    GameState s = action_round_for(Player::US);
    s.turn = 9;
    s.us_space_track = 4;
    s.ussr_space_track = 1;
    s.card_locations[card_ids::STAR_WARS] = hand_of(Player::US);
    s.card_locations[card_ids::GRAIN_SALES] = CardLocation::DISCARD_PILE;
    s.card_locations[card_ids::DUCK_AND_COVER] = hand_of(Player::US);
    s.card_locations[drawn] = hand_of(Player::USSR);
    take(s, DecisionType::SELECT_CARD, card_ids::STAR_WARS);
    take(s, DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(Resolution::EVENT));
    take(s, DecisionType::SELECT_CARD, card_ids::GRAIN_SALES);
    ASSERT_EQ(s.ctx_stack_depth, 1);
    ASSERT_EQ(s.ctx().pending_op_card, drawn);
    return s;
}

void spend_on_canada(GameState& s) {
    for (int i = 0; i < 8 && s.ctx().decision_player == Player::US &&
                    s.ctx().decision_type == DecisionType::POINT_NODE; ++i) {
        take(s, DecisionType::POINT_NODE, countries::CANADA);
    }
}

} // namespace

TEST(PushedFrame, StarWarsGrainSales_ANeutralCardSpentForOpsIsDiscarded) {
    GameState s = star_wars_into_grain_sales(card_ids::NUCLEAR_TEST_BAN);
    take(s, DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(Resolution::OPS_INFLUENCE));
    spend_on_canada(s);
    ASSERT_EQ(s.card_locations[card_ids::NUCLEAR_TEST_BAN], CardLocation::DISCARD_PILE);
    ASSERT_EQ(s.card_locations[card_ids::GRAIN_SALES], CardLocation::DISCARD_PILE);
    ASSERT_EQ(s.ctx_stack_depth, 0);
}

TEST(PushedFrame, StarWarsGrainSales_AUSSRCardFiresItsEventOnceAndIsRemoved) {
    GameState s = star_wars_into_grain_sales(card_ids::NASSER);
    const int before = s.countries[countries::EGYPT].ussr_influence;
    take(s, DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(Resolution::OPS_INFLUENCE));
    spend_on_canada(s);
    ASSERT_EQ(s.card_locations[card_ids::NASSER], CardLocation::REMOVED_FROM_GAME);
    ASSERT_EQ(s.countries[countries::EGYPT].ussr_influence, before + 2);
    ASSERT_EQ(s.ctx_stack_depth, 0);
    ASSERT_EQ(s.ctx().resolving_card, 0);
}

TEST(PushedFrame, StarWarsGrainSales_AStarredUSCardSpentForOpsIsOnlyDiscarded) {
    // The US's own card for Ops: its Event never happened, so a starred card is not removed.
    GameState s = star_wars_into_grain_sales(card_ids::MARSHALL_PLAN);
    ASSERT_TRUE(CardData::get_card(card_ids::MARSHALL_PLAN).one_time);
    take(s, DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(Resolution::OPS_INFLUENCE));
    spend_on_canada(s);
    ASSERT_EQ(s.card_locations[card_ids::MARSHALL_PLAN], CardLocation::DISCARD_PILE);
    ASSERT_EQ(s.ctx_stack_depth, 0);
}

TEST(PushedFrame, StarWarsGrainSales_AFriendlyCardsEventLeavesNoFrameBehind) {
    GameState s = star_wars_into_grain_sales(card_ids::NUCLEAR_TEST_BAN);
    take(s, DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(Resolution::EVENT));
    ASSERT_EQ(s.card_locations[card_ids::NUCLEAR_TEST_BAN], CardLocation::DISCARD_PILE);
    ASSERT_EQ(s.ctx_stack_depth, 0);
    ASSERT_EQ(s.ctx().decision_type, DecisionType::SELECT_CARD);
}

TEST(PushedFrame, StarWarsGrainSales_ASpaceAttemptLeavesNoFrameBehind) {
    GameState s = star_wars_into_grain_sales(card_ids::NUCLEAR_TEST_BAN);
    take(s, DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(Resolution::SPACE));
    ASSERT_TRUE(StateMachine::step(s, MicroAction{DecisionType::ROLL_DIE, 1, 0, 0}));
    ASSERT_NE(s.card_locations[card_ids::NUCLEAR_TEST_BAN], hand_of(Player::US));
    ASSERT_EQ(s.ctx_stack_depth, 0);
}

TEST(PushedFrame, GrainSalesOwnTwoOpsAtDepthDoNotFireAnything) {
    // The decline: Ops really granted by an Event, on a pushed frame. Nothing changes here.
    GameState s = star_wars_into_grain_sales(card_ids::NASSER);
    const int before = s.countries[countries::EGYPT].ussr_influence;
    ASSERT_TRUE(StateMachine::step(s, MicroAction{DecisionType::SELECT_PLAY_MODE, 0, 0, action_flags::CONFIRM_DONE}));
    ASSERT_EQ(s.card_locations[card_ids::NASSER], hand_of(Player::USSR, /*known=*/true));
    take(s, DecisionType::SELECT_OP_MODE, static_cast<uint8_t>(OpMode::INFLUENCE));
    spend_on_canada(s);
    ASSERT_EQ(s.card_locations[card_ids::NASSER], hand_of(Player::USSR, /*known=*/true));
    ASSERT_EQ(s.countries[countries::EGYPT].ussr_influence, before);
    ASSERT_EQ(s.ctx_stack_depth, 0);
}
