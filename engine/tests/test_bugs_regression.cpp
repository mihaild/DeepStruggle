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

    // Put the two cards in the hands that play them. This used to be left to chance: the test
    // named cards without placing them and `Engine::step` did not check, so it passed on whatever
    // seed 42 happened to deal. `step` now validates against the legal mask, and a card that is
    // not in your hand is not a legal play.
    state.card_locations[card_ids::DE_STALINIZATION] = CardLocation::HAND_US_KNOWN;
    state.card_locations[card_ids::ASIA_SCORING] = CardLocation::HAND_USSR_KNOWN;

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
    state.card_locations[card_ids::DE_GAULLE] = ts::hand_of(ts::Player::US);
    state.current_phase = Phase::ACTION_ROUND;
    state.action_round = 1;
    state.phasing_player = Player::US;
    state.ctx().decision_player = Player::US;
    state.ctx().decision_type = DecisionType::SELECT_CARD;

    // Put 3 US influence in France
    state.countries[countries::FRANCE].us_influence = 3;
    state.countries[countries::FRANCE].ussr_influence = 0;

    // 1. Select Card De Gaulle (#17, 3 Ops, USSR event)
    ASSERT_TRUE(Engine::step(state, MicroAction(DecisionType::SELECT_CARD, card_ids::DE_GAULLE, 0, 0)));
    ASSERT_EQ(state.ctx().decision_type, DecisionType::SELECT_PLAY_MODE);

    // 2. P17: one resolution. "Ops for placement" on an opponent's card IS ops-first -- the
    //    three steps this replaced were SELECT_PLAY_MODE(OPS), CHOOSE_TIMING_BRANCH(OPS_FIRST)
    //    and SELECT_OP_MODE(INFLUENCE). timing_branch must still record it, or the event would
    //    not know to fire afterwards, which is what the rest of this test checks.
    ASSERT_TRUE(Engine::step(state, MicroAction(DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(Resolution::OPS_INFLUENCE), 0, 0)));
    ASSERT_EQ(state.ctx().decision_type, DecisionType::POINT_NODE);
    ASSERT_EQ(state.ctx().timing_branch, static_cast<uint8_t>(TimingBranch::OPS_FIRST));

    // 5. Place 3 ops in UK (ID 1)
    ASSERT_TRUE(Engine::step(state, MicroAction(DecisionType::POINT_NODE, countries::UNITED_KINGDOM, 0, 0)));
    ASSERT_TRUE(Engine::step(state, MicroAction(DecisionType::POINT_NODE, countries::UNITED_KINGDOM, 0, 0)));
    ASSERT_TRUE(Engine::step(state, MicroAction(DecisionType::POINT_NODE, countries::UNITED_KINGDOM, 0, 0)));

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
    state.card_locations[card_ids::DUCK_AND_COVER] = ts::hand_of(ts::Player::US); // 3 Ops US card

    ASSERT_TRUE(Engine::step(state, MicroAction(DecisionType::SELECT_CARD, card_ids::DUCK_AND_COVER, 0, 0)));
    ASSERT_TRUE(Engine::step(state, MicroAction(DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(Resolution::OPS_REALIGN), 0, 0)));

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
    ASSERT_TRUE(Engine::step(state, realign_act));
    ASSERT_EQ(state.ctx().decision_type, DecisionType::ROLL_DIE);
    ASSERT_TRUE(Engine::step(state, MicroAction(DecisionType::ROLL_DIE, 6, 1, 0)));
    ASSERT_EQ(state.countries[countries::ITALY].ussr_influence, 0);
}

TEST(RegressionTest, ArabIsraeliWarWithRoll2Fails) {
    GameState state{};
    Engine::init_game(state, 42);

    state.countries[countries::ISRAEL].us_influence = 2;
    state.countries[countries::ISRAEL].ussr_influence = 0;
    state.countries[countries::JORDAN].us_influence = 2; // US controls Jordan (-1 mod)

    state.card_locations[card_ids::ARAB_ISRAELI_WAR] = ts::hand_of(ts::Player::USSR);
    state.current_phase = Phase::ACTION_ROUND;
    state.action_round = 1;
    state.phasing_player = Player::USSR;
    state.ctx().decision_player = Player::USSR;
    state.ctx().decision_type = DecisionType::SELECT_CARD;

    // 1. Select Card
    ASSERT_TRUE(Engine::step(state, MicroAction(DecisionType::SELECT_CARD, card_ids::ARAB_ISRAELI_WAR, 0, 0)));
    // 2. Select Play Mode EVENT with forced roll = 2
    ASSERT_TRUE(Engine::step(state, MicroAction(DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(Resolution::EVENT), 2, 0)));
    ASSERT_EQ(state.ctx().decision_type, DecisionType::ROLL_DIE);
    ASSERT_TRUE(Engine::step(state, MicroAction(DecisionType::ROLL_DIE, 2, 0, 0)));

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

    state.card_locations[card_ids::DUCK_AND_COVER] = ts::hand_of(ts::Player::US); // 3 Ops
    state.current_phase = Phase::ACTION_ROUND;
    state.action_round = 1;
    state.phasing_player = Player::US;
    state.ctx().decision_player = Player::US;
    state.ctx().decision_type = DecisionType::SELECT_CARD;

    ASSERT_TRUE(Engine::step(state, MicroAction(DecisionType::SELECT_CARD, card_ids::DUCK_AND_COVER, 0, 0)));
    ASSERT_TRUE(Engine::step(state, MicroAction(DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(Resolution::OPS_INFLUENCE), 0, 0)));

    // 1st point placed in Morocco (legal because adjacent to Algeria)
    uint8_t mask[84];
    size_t out_size = 0;
    ActionMask::generate_mask(state, mask, &out_size);
    ASSERT_EQ(mask[countries::MOROCCO], 1);
    ASSERT_EQ(mask[countries::WEST_AFRICA], 0);

    ASSERT_TRUE(Engine::step(state, MicroAction(DecisionType::POINT_NODE, countries::MOROCCO, 0, 0)));
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

    state.card_locations[card_ids::WARSAW_PACT] = ts::hand_of(ts::Player::USSR);
    state.current_phase = Phase::ACTION_ROUND;
    state.action_round = 1;
    state.phasing_player = Player::USSR;
    state.ctx().decision_player = Player::USSR;
    state.ctx().decision_type = DecisionType::SELECT_CARD;

    ASSERT_TRUE(Engine::step(state, MicroAction(DecisionType::SELECT_CARD, card_ids::WARSAW_PACT, 0, 0)));
    ASSERT_TRUE(Engine::step(state, MicroAction(DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(Resolution::EVENT), 0, 0)));

    // Choose Branch 1: Add 5 USSR influence in Eastern Europe (max 2 per country)
    ASSERT_TRUE(Engine::step(state, MicroAction(DecisionType::CHOOSE_BRANCH, 1, 0, 0)));
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
    ASSERT_TRUE(Engine::step(state, MicroAction(DecisionType::POINT_NODE, countries::POLAND, 0, 0)));
    ASSERT_TRUE(Engine::step(state, MicroAction(DecisionType::POINT_NODE, countries::POLAND, 0, 0)));

    // Poland reached 2 placements -> Poland mask must now be 0!
    ActionMask::generate_mask(state, mask, &out_size);
    ASSERT_EQ(mask[countries::POLAND], 0);
}

TEST(RegressionTest, ObservationHiddenOpponentHand) {
    GameState state{};
    Engine::init_game(state, 42);

    state.card_locations[5] = ts::hand_of(ts::Player::USSR);
    state.card_locations[10] = ts::hand_of(ts::Player::US);
    state.card_locations[12] = CardLocation::DISCARD_PILE;

    ObservationBufferV23 obs_us{};
    Observation::extract(state, Player::US, &obs_us);

    // Card 10 (US hand) from US perspective is MY_HAND
    size_t off_10 = (10 - 1) * card_slots::V23_FEATURES;
    ASSERT_TRUE(obs_us.card_features[off_10 + 1] == 1.0f);
    ASSERT_TRUE(obs_us.card_features[off_10 + 0] == 0.0f);
    ASSERT_TRUE(obs_us.card_features[off_10 + 2] == 0.0f);

    // Card 5 (USSR hand) from US perspective must be hidden: it folds in with the draw deck,
    // because an unseen opponent card and a card still in the deck are exactly what the observer
    // cannot tell apart. KNOWN_OPPONENT_HAND stays clear -- nobody has shown it.
    size_t off_5 = (5 - 1) * card_slots::V23_FEATURES;
    ASSERT_TRUE(obs_us.card_features[off_5 + 0] == 1.0f);
    ASSERT_TRUE(obs_us.card_features[off_5 + 1] == 0.0f);
    ASSERT_TRUE(obs_us.card_features[off_5 + 2] == 0.0f);

    // Global feature 70 is opp_hand_cnt / 10.0f (public count is preserved)
    ASSERT_TRUE(obs_us.global_features[70] > 0.0f);

    ObservationBufferV23 obs_ussr{};
    Observation::extract(state, Player::USSR, &obs_ussr);

    // Card 5 (USSR hand) from USSR perspective is MY_HAND
    ASSERT_TRUE(obs_ussr.card_features[off_5 + 1] == 1.0f);
    ASSERT_TRUE(obs_ussr.card_features[off_5 + 0] == 0.0f);
    ASSERT_TRUE(obs_ussr.card_features[off_5 + 2] == 0.0f);

    // Card 10 (US hand) from USSR perspective must be hidden
    ASSERT_TRUE(obs_ussr.card_features[off_10 + 0] == 1.0f);
    ASSERT_TRUE(obs_ussr.card_features[off_10 + 1] == 0.0f);
    ASSERT_TRUE(obs_ussr.card_features[off_10 + 2] == 0.0f);
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

// Summit (#45): the roll winner may raise or lower DEFCON. Lowering it to 1 ends the game
// with the PHASING player losing, as at every other DEFCON-1 site -- so a non-phasing
// Summit winner at DEFCON 2 wins outright by degrading, rather than losing. This site
// previously assigned the loss to the roll winner (decision_player), inverting that.
TEST(RegressionTest, SummitDefconOneLossFallsOnPhasingPlayer) {
    // USSR is phasing; US wins the Summit roll and chooses to degrade DEFCON from 2 to 1.
    // The phasing USSR takes the loss (+20 VP = US win), and it is provoked for USSR.
    {
        GameState state{};
        Engine::init_game(state, 11);
        state.current_phase = Phase::ACTION_ROUND;
        state.phasing_player = Player::USSR;
        state.defcon = 2;
        state.clear_flag(effect_bits::DEFCON_SUICIDE_PROVOKED);
        state.ctx().resolving_card = card_ids::SUMMIT;
        state.ctx().decision_player = Player::US;
        state.ctx().decision_type = DecisionType::CHOOSE_BRANCH;

        MicroAction degrade(DecisionType::CHOOSE_BRANCH, 1, 0, 0);
        CardHandlers::handle_event_step(state, degrade);

        ASSERT_EQ(static_cast<int>(state.current_phase), static_cast<int>(Phase::GAME_OVER));
        ASSERT_EQ(static_cast<int>(state.victory_points), 20); // phasing USSR loses
        ASSERT_TRUE(state.has_flag(effect_bits::DEFCON_SUICIDE_PROVOKED));
    }

    // Mirror: US is phasing and also wins the roll, so degrading is self-inflicted.
    {
        GameState state{};
        Engine::init_game(state, 11);
        state.current_phase = Phase::ACTION_ROUND;
        state.phasing_player = Player::US;
        state.defcon = 2;
        state.clear_flag(effect_bits::DEFCON_SUICIDE_PROVOKED);
        state.ctx().resolving_card = card_ids::SUMMIT;
        state.ctx().decision_player = Player::US;
        state.ctx().decision_type = DecisionType::CHOOSE_BRANCH;

        MicroAction degrade(DecisionType::CHOOSE_BRANCH, 1, 0, 0);
        CardHandlers::handle_event_step(state, degrade);

        ASSERT_EQ(static_cast<int>(state.current_phase), static_cast<int>(Phase::GAME_OVER));
        ASSERT_EQ(static_cast<int>(state.victory_points), -20); // phasing US loses
        ASSERT_TRUE(!state.has_flag(effect_bits::DEFCON_SUICIDE_PROVOKED));
    }
}

// Advances a fresh game to the first ROLL_DIE node, playing the lowest legal flat action.
// Returns false if none is reached, so a test fails loudly rather than passing vacuously.
static bool advance_to_roll_node(GameState& state, uint64_t seed) {
    Engine::init_game(state, seed);
    uint8_t mask[FLAT_ACTION_SPACE_SIZE];
    for (int step = 0; step < 4000; ++step) {
        if (Engine::is_terminal(state)) return false;
        if (state.ctx().decision_type == DecisionType::ROLL_DIE) return true;
        ActionMask::generate_flat_mask_212(state, mask);
        int chosen = -1;
        for (int i = 0; i < FLAT_ACTION_SPACE_SIZE; ++i) {
            if (mask[i]) { chosen = i; break; }
        }
        if (chosen < 0) return false;
        ASSERT_TRUE(Engine::step(state, ActionMask::decode_flat_action_212(state, static_cast<uint16_t>(chosen))));
    }
    return false;
}

// A ROLL_DIE node had no case in the flat mask switch at all. Nothing set a bit, so the
// "ensure at least one action is legal" fallback at the bottom supplied 211 -- the generic
// confirm/done -- which decodes to primary_id 255. For every other decision type 255 is the
// harmless "no selection" sentinel; for ROLL_DIE primary_id IS the forced die value, and
// Operations reads `forced_roll > 0` as "a die was forced". So the one legal action at a
// chance node rolled a 255, and space race attempts and coups through the flat path succeeded
// automatically.
TEST(RegressionTest, FlatRollActionRollsTheDieInsteadOfForcing255) {
    GameState state{};
    ASSERT_TRUE(advance_to_roll_node(state, 99));

    uint8_t mask[FLAT_ACTION_SPACE_SIZE];
    ActionMask::generate_flat_mask_212(state, mask);
    int legal_count = 0;
    for (int i = 0; i < FLAT_ACTION_SPACE_SIZE; ++i) legal_count += mask[i] ? 1 : 0;
    ASSERT_EQ(legal_count, 1);
    // P17: slot 115, not the shared 211. A chance node with one legal action is not a decline,
    // and sharing the index with confirm/done is the aliasing that caused this bug in the first
    // place -- the decode needed a special case to tell them apart. Now 211 means exactly one
    // thing and this node has its own index.
    ASSERT_TRUE(mask[115] != 0);
    ASSERT_TRUE(mask[ts::flat_slots::CONFIRM_DONE] == 0);

    const MicroAction decoded = ActionMask::decode_flat_action_212(state, 115);
    ASSERT_EQ(static_cast<int>(decoded.decision_type), static_cast<int>(DecisionType::ROLL_DIE));
    // 0 means "roll normally". Anything else here is a forced die.
    ASSERT_EQ(static_cast<int>(decoded.primary_id), 0);
    ASSERT_EQ(static_cast<int>(decoded.secondary_id), 0);

    // Stepping the flat action must be indistinguishable from an explicit unforced roll.
    GameState via_flat = state;
    GameState via_explicit = state;
    ASSERT_TRUE(Engine::step(via_flat, ActionMask::decode_flat_action_212(via_flat, 115)));
    ASSERT_TRUE(Engine::step(via_explicit, MicroAction(DecisionType::ROLL_DIE, 0, 0, 0)));
    ASSERT_EQ(static_cast<int>(via_flat.last_die_roll),
              static_cast<int>(via_explicit.last_die_roll));
    ASSERT_EQ(static_cast<int>(via_flat.victory_points),
              static_cast<int>(via_explicit.victory_points));
    ASSERT_TRUE(via_flat.last_die_roll <= 6);
}

// ROLL_DIE is the one decision the 212-wide mask cannot constrain: its primary_id and
// secondary_id are the dice themselves, not indices, because forcing a roll is a deliberate
// replayer and test affordance. So the range is checked in StateMachine::step or nowhere, and
// without it a hand-built action carrying 26 was applied as though it were a die.
TEST(RegressionTest, ForcedDieOutsideOneToSixIsRejected) {
    GameState state{};
    ASSERT_TRUE(advance_to_roll_node(state, 99));

    for (int bad : {7, 26, 200, 255}) {
        GameState probe = state;
        const int vp_before = probe.victory_points;
        ASSERT_TRUE(!Engine::step(probe, MicroAction(DecisionType::ROLL_DIE,
                                                     static_cast<uint8_t>(bad), 0, 0)));
        // A rejected action must leave the state untouched, not half-applied.
        ASSERT_EQ(static_cast<int>(probe.victory_points), vp_before);
        ASSERT_EQ(static_cast<int>(probe.ctx().decision_type),
                  static_cast<int>(DecisionType::ROLL_DIE));

        // The opponent's die, used by realignment, is guarded the same way.
        GameState probe2 = state;
        ASSERT_TRUE(!Engine::step(probe2, MicroAction(DecisionType::ROLL_DIE, 3,
                                                      static_cast<uint8_t>(bad), 0)));
    }

    // Every value the affordance exists for is still accepted.
    for (int good = 1; good <= 6; ++good) {
        GameState probe = state;
        ASSERT_TRUE(Engine::step(probe, MicroAction(DecisionType::ROLL_DIE,
                                                    static_cast<uint8_t>(good), 0, 0)));
    }
}

// Advance to a SELECT_CARD node inside an action round, which is where the forced-play rule binds.
static bool advance_to_action_round_card(GameState& state, uint64_t seed) {
    Engine::init_game(state, seed);
    uint8_t mask[FLAT_ACTION_SPACE_SIZE];
    for (int step = 0; step < 4000; ++step) {
        if (Engine::is_terminal(state)) return false;
        if (state.ctx().decision_type == DecisionType::SELECT_CARD &&
            state.current_phase == Phase::ACTION_ROUND) {
            return true;
        }
        ActionMask::generate_flat_mask_212(state, mask);
        int chosen = -1;
        for (int i = 0; i < FLAT_ACTION_SPACE_SIZE; ++i) {
            if (mask[i]) { chosen = i; break; }
        }
        if (chosen < 0) return false;
        ASSERT_TRUE(Engine::step(state, ActionMask::decode_flat_action_212(state, static_cast<uint16_t>(chosen))));
    }
    return false;
}

static void give(GameState& s, uint8_t card, Player p) {
    s.card_locations[card] = (p == Player::US) ? CardLocation::HAND_US_KNOWN
                                               : CardLocation::HAND_USSR_KNOWN;
}

static int count_selectable_cards(GameState& s) {
    uint8_t mask[FLAT_ACTION_SPACE_SIZE];
    ActionMask::generate_flat_mask_212(s, mask);
    int n = 0;
    for (int i = 0; i < 110; ++i) if (mask[i]) n++;
    return n;
}

// Missile Envy obliges its recipient to play THAT CARD for Operations on their next action round.
// `forced_card_id` only ever holds MISSILE_ENVY -- there is no other forced card -- so the old
// condition `forced_card_id == card || forced_card_id == MISSILE_ENVY` was always true while a
// force was live, and the rule became "every card must go to Ops".
TEST(RegressionTest, ForcedPlayBindsOnlyMissileEnvyAndOnlyInAnActionRound) {
    GameState base{};
    ASSERT_TRUE(advance_to_action_round_card(base, 4242));
    const Player p = base.ctx().decision_player;

    // Live force: only Missile Envy is selectable, and the China Card is suppressed with the rest.
    {
        GameState s = base;
        give(s, card_ids::MISSILE_ENVY, p);
        s.forced_card_player = p;
        s.forced_card_id = card_ids::MISSILE_ENVY;
        uint8_t mask[FLAT_ACTION_SPACE_SIZE];
        ActionMask::generate_flat_mask_212(s, mask);
        ASSERT_EQ(count_selectable_cards(s), 1);
        ASSERT_TRUE(mask[card_ids::MISSILE_ENVY - 1] != 0);
        ASSERT_TRUE(mask[card_ids::THE_CHINA_CARD - 1] == 0);
    }

    // A headline is unconstrained: the card says "on their next action round".
    {
        GameState s = base;
        give(s, card_ids::MISSILE_ENVY, p);
        s.forced_card_player = p;
        s.forced_card_id = card_ids::MISSILE_ENVY;
        s.current_phase = Phase::HEADLINE;
        ASSERT_TRUE(count_selectable_cards(s) > 1);
    }

    // A stale force -- the flag set, the card gone -- must restrict nothing. 21 positions in
    // 1,280,000 of sampled self-play carried one.
    {
        GameState s = base;
        s.card_locations[card_ids::MISSILE_ENVY] = CardLocation::DISCARD_PILE;
        s.forced_card_player = p;
        s.forced_card_id = card_ids::MISSILE_ENVY;
        ASSERT_TRUE(count_selectable_cards(s) > 1);
    }
}

// Every play mode the mask offers must be accepted. The state machine's copy of the forced-play
// rule had none of the three guards, so it refused modes the mask had offered: EVENT in a
// headline, SPACE on the China Card, and -- worst -- EVENT on a scoring card, where the mask's
// only legal action was refused and the position could not advance at all.
TEST(RegressionTest, MaskAndStepAgreeOnPlayModesUnderAForce) {
    GameState base{};
    ASSERT_TRUE(advance_to_action_round_card(base, 4242));
    const Player p = base.ctx().decision_player;

    const uint8_t cards_to_try[] = {card_ids::THE_CHINA_CARD, card_ids::MISSILE_ENVY, 13};
    const Phase phases[] = {Phase::ACTION_ROUND, Phase::HEADLINE};

    for (Phase ph : phases) {
        for (uint8_t card : cards_to_try) {
            GameState s = base;
            give(s, card_ids::MISSILE_ENVY, p);
            s.forced_card_player = p;
            s.forced_card_id = card_ids::MISSILE_ENVY;
            s.current_phase = ph;
            s.ctx().decision_type = DecisionType::SELECT_PLAY_MODE;
            s.ctx().pending_op_card = card;

            uint8_t mask[FLAT_ACTION_SPACE_SIZE];
            ActionMask::generate_flat_mask_212(s, mask);
            int offered = 0;
            for (int i = 110; i <= 113; ++i) {
                if (!mask[i]) continue;
                offered++;
                GameState probe = s;
                ASSERT_TRUE(Engine::step(
                    probe, ActionMask::decode_flat_action_212(probe, static_cast<uint16_t>(i))));
            }
            ASSERT_TRUE(offered > 0);
        }
    }
}

// Held scoring outranks the force and DEFERS it: holding scoring cards past the end of the turn
// loses the game outright, so the obligation that can lose it wins, and the forced flags stay set
// to bind on the next action round.
TEST(RegressionTest, HeldScoringOutranksTheForceAndDefersIt) {
    GameState base{};
    ASSERT_TRUE(advance_to_action_round_card(base, 4242));
    const Player p = base.ctx().decision_player;

    GameState s = base;
    give(s, card_ids::MISSILE_ENVY, p);
    s.forced_card_player = p;
    s.forced_card_id = card_ids::MISSILE_ENVY;
    s.turn = 10;
    s.action_round = 7;          // one round left
    give(s, 1, p);               // Asia Scoring
    give(s, 2, p);               // Europe Scoring

    uint8_t mask[FLAT_ACTION_SPACE_SIZE];
    ActionMask::generate_flat_mask_212(s, mask);
    for (int i = 0; i < 110; ++i) {
        if (mask[i]) {
            ASSERT_TRUE(CardData::is_scoring_card(static_cast<uint8_t>(i + 1)));
        }
    }
    ASSERT_TRUE(mask[card_ids::MISSILE_ENVY - 1] == 0);
    // Deferred, not discharged.
    ASSERT_EQ(static_cast<int>(s.forced_card_id), static_cast<int>(card_ids::MISSILE_ENVY));
}
