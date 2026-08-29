#include "ts/state_machine.hpp"
#include "ts/map_data.hpp"
#include "ts/card_data.hpp"
#include "ts/card_handlers.hpp"
#include "ts/scoring.hpp"
#include "ts/space_race.hpp"
#include "ts/ops.hpp"
#include "ts/prng.hpp"
#include "ts/action_mask.hpp"
#include <algorithm>
#include <cstring>

namespace ts {

void StateMachine::add_era_cards_to_deck(GameState& state, WarEra era) noexcept {
    for (uint8_t i = 1; i <= 110; ++i) {
        if (i == card_ids::THE_CHINA_CARD) continue; // The China Card is not in deck
        if (CardData::get_card(i).era == era) {
            state.card_locations[i] = CardLocation::DRAW_DECK;
        }
    }
}

void StateMachine::reshuffle_discard_into_draw(GameState& state) noexcept {
    for (uint8_t i = 1; i <= 110; ++i) {
        if (i == card_ids::THE_CHINA_CARD) continue;
        if (state.card_locations[i] == CardLocation::DISCARD_PILE) {
            state.card_locations[i] = CardLocation::DRAW_DECK;
        }
    }
}

void StateMachine::deal_cards_to_hands(GameState& state) noexcept {
    uint8_t target_hand = (state.turn <= 3) ? 8 : 9;

    auto deal_to_player = [&](Player p) {
        CardLocation hand_loc = (p == Player::US) ? CardLocation::HAND_US : CardLocation::HAND_USSR;
        uint8_t current_count = 0;
        for (uint8_t i = 1; i <= 110; ++i) {
            if (state.card_locations[i] == hand_loc) current_count++;
        }

        while (current_count < target_hand) {
            // Count cards in draw deck
            uint8_t draw_cards[111];
            uint8_t draw_count = 0;
            for (uint8_t i = 1; i <= 110; ++i) {
                if (state.card_locations[i] == CardLocation::DRAW_DECK) {
                    draw_cards[draw_count++] = i;
                }
            }

            if (draw_count == 0) {
                reshuffle_discard_into_draw(state);
                for (uint8_t i = 1; i <= 110; ++i) {
                    if (state.card_locations[i] == CardLocation::DRAW_DECK) {
                        draw_cards[draw_count++] = i;
                    }
                }
                if (draw_count == 0) break; // Entire deck exhausted
            }

            uint32_t chosen_idx = Prng::random_index(state.rng_state, draw_count);
            uint8_t chosen_card = draw_cards[chosen_idx];
            state.card_locations[chosen_card] = hand_loc;
            current_count++;
            if (draw_count == 1) {
                reshuffle_discard_into_draw(state);
            }
        }
    };

    deal_to_player(Player::USSR);
    deal_to_player(Player::US);
}

void StateMachine::init_new_game(GameState& state, uint64_t seed) noexcept {
    state = GameState{};
    state.rng_state = (seed != 0) ? seed : 0x123456789ABCDEF0ULL;

    // 1. Tracks initialization
    state.victory_points = 0;
    state.defcon = 5;
    state.us_mil_ops = 0;
    state.ussr_mil_ops = 0;
    state.us_space_track = 0;
    state.ussr_space_track = 0;
    state.turn = 1;
    state.action_round = 0;

    // 2. Fixed Setup Influence Placements
    // USSR: Syria(22)=1, Iraq(24)=1, North Korea(43)=3, East Germany(14)=3, Finland(5)=1
    state.countries[countries::SYRIA].ussr_influence = 1;
    state.countries[countries::IRAQ].ussr_influence = 1;
    state.countries[countries::NORTH_KOREA].ussr_influence = 3;
    state.countries[countries::EAST_GERMANY].ussr_influence = 3;
    state.countries[countries::FINLAND].ussr_influence = 1;

    // US: Iran(25)=1, Israel(23)=1, Japan(45)=1, Australia(41)=1, Philippines(40)=1, South Korea(44)=1, Panama(70)=1, South Africa(63)=1, United Kingdom(1)=5
    state.countries[countries::IRAN].us_influence = 1;
    state.countries[countries::ISRAEL].us_influence = 1;
    state.countries[countries::JAPAN].us_influence = 1;
    state.countries[countries::AUSTRALIA].us_influence = 4;
    state.countries[countries::PHILIPPINES].us_influence = 1;
    state.countries[countries::SOUTH_KOREA].us_influence = 1;
    state.countries[countries::PANAMA].us_influence = 1;
    state.countries[countries::SOUTH_AFRICA].us_influence = 1;
    state.countries[countries::UNITED_KINGDOM].us_influence = 5;
    state.countries[countries::CANADA].us_influence = 2;

    // 3. China Card to USSR
    state.china_card_holder = Player::USSR;
    state.china_card_playable = 1;
    state.forced_card_player = Player::NONE;
    state.forced_card_id = 0;
    state.card_locations[card_ids::THE_CHINA_CARD] = CardLocation::ONGOING_EVENT;

    // 4. Early War Cards & Deal Hands
    add_era_cards_to_deck(state, WarEra::EARLY);
    deal_cards_to_hands(state);

    // 5. Setup Phase
    state.current_phase = Phase::SETUP;
    state.phasing_player = Player::USSR;
    state.ctx().decision_player = Player::USSR;
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().remaining_steps = 6; // USSR places 6 in Eastern Europe
}

void StateMachine::start_turn(GameState& state) noexcept {
    if (state.current_phase == Phase::GAME_OVER) return;

    // Reset decision context cleanly for the new turn
    state.ctx() = DecisionContext{};

    // Phase A: Improve DEFCON
    state.defcon = static_cast<uint8_t>(std::min(5, static_cast<int>(state.defcon) + 1));

    // Phase B: Deal Cards
    deal_cards_to_hands(state);

    // Phase C: Headline Phase
    state.current_phase = Phase::HEADLINE;
    state.action_round = 0;
    state.headline_us_card = 0;
    state.headline_ussr_card = 0;
    state.headline_first_card = 0;
    state.headline_second_card = 0;
    state.headline_stage = 0;
    state.headline_first_owner = Player::NONE;
    state.headline_second_owner = Player::NONE;

    // Space 4 Priority check
    bool us_space4 = SpaceRace::has_man_in_space(state, Player::US);
    bool ussr_space4 = SpaceRace::has_man_in_space(state, Player::USSR);

    if (us_space4 && !ussr_space4) {
        state.ctx().decision_player = Player::USSR; // USSR reveals first
    } else if (ussr_space4 && !us_space4) {
        state.ctx().decision_player = Player::US; // US reveals first
    } else {
        state.ctx().decision_player = Player::US;
    }

    state.ctx().decision_type = DecisionType::SELECT_CARD;
}

void StateMachine::advance_headline_step(GameState& state) noexcept {
    // Check if we need to resolve headline 2 or enter AR 1
    if (state.headline_stage == 1) {
        // Move to Stage 2: Second headline
        state.headline_stage = 2;
        uint8_t h2_card = state.headline_second_card;
        Player h2_owner = state.headline_second_owner;

        if (h2_card != 0 && state.current_phase != Phase::GAME_OVER) {
            bool ussr_cancelled = (state.headline_us_card == card_ids::DEFECTORS && h2_owner == Player::USSR);
            if (ussr_cancelled) {
                state.card_locations[h2_card] = CardLocation::DISCARD_PILE;
                advance_headline_step(state);
                return;
            }
            const auto& c_info = CardData::get_card(h2_card);
            Player exec_player = (c_info.side == get_opponent(h2_owner)) ? get_opponent(h2_owner) : h2_owner;
            state.phasing_player = h2_owner;
            state.ctx().decision_player = exec_player;
            state.ctx().resolving_card = h2_card;

            bool done = CardHandlers::trigger_event(state, h2_card, exec_player);
            if (h2_card != card_ids::KITCHEN_DEBATES) {
                if (h2_card == card_ids::SHUTTLE_DIPLOMACY && state.has_flag(effect_bits::SHUTTLE_DIPLOMACY_ACTIVE)) {
                    state.card_locations[h2_card] = CardLocation::ONGOING_EVENT;
                } else {
                    state.card_locations[h2_card] = c_info.one_time ? CardLocation::REMOVED_FROM_GAME : CardLocation::DISCARD_PILE;
                }
            }
            if (done) {
                advance_headline_step(state);
            }
            return;
        }
    }

    // Both headlines resolved -> Start Action Round 1
    state.headline_stage = 3;
    if (state.current_phase != Phase::GAME_OVER) {
        state.current_phase = Phase::ACTION_ROUND;
        state.action_round = 1;
        state.phasing_player = Player::USSR;
        state.ctx() = DecisionContext{};
        state.ctx().decision_player = Player::USSR;
        state.ctx().decision_type = DecisionType::SELECT_CARD;
    }
}

void StateMachine::advance_after_ops(GameState& state) noexcept {
    if (state.current_phase == Phase::GAME_OVER) return;
    if (state.current_phase == Phase::HEADLINE) {
        advance_headline_step(state);
        return;
    }
    uint8_t card = state.ctx().pending_op_card;
    Player p = state.phasing_player;
    uint8_t timing = state.ctx().timing_branch;

    if (timing == static_cast<uint8_t>(TimingBranch::OPS_FIRST) && CardData::is_opponent_card(card, p)) {
        // Trigger opponent event!
        Player opp = get_opponent(p);
        const auto& c_info = CardData::get_card(card);
        state.ctx().decision_player = opp;
        state.ctx().resolving_card = card;
        state.ctx().timing_branch = 255; // cleared

        bool done = CardHandlers::trigger_event(state, card, opp);
        if (card != card_ids::KITCHEN_DEBATES) {
            if (card == card_ids::SHUTTLE_DIPLOMACY && state.has_flag(effect_bits::SHUTTLE_DIPLOMACY_ACTIVE)) {
                state.card_locations[card] = CardLocation::ONGOING_EVENT;
            } else {
                state.card_locations[card] = c_info.one_time ? CardLocation::REMOVED_FROM_GAME : CardLocation::DISCARD_PILE;
            }
        }
        if (done) {
            advance_after_action_round(state);
        }
        return;
    }

    // Friendly / Neutral / Already triggered event
    if (card == card_ids::THE_CHINA_CARD) {
        state.china_card_holder = get_opponent(p);
        state.china_card_playable = 0; // Passes to opponent face down
        if (p == Player::US) {
            state.clear_flag(effect_bits::FORMOSAN_RESOLUTION_ACTIVE);
        }
    } else if (card != 0) {
        const auto& c_info = CardData::get_card(card);
        if (card == card_ids::SHUTTLE_DIPLOMACY && state.has_flag(effect_bits::SHUTTLE_DIPLOMACY_ACTIVE)) {
            state.card_locations[card] = CardLocation::ONGOING_EVENT;
        } else {
            state.card_locations[card] = c_info.one_time ? CardLocation::REMOVED_FROM_GAME : CardLocation::DISCARD_PILE;
        }
    }
    advance_after_action_round(state);
}

void StateMachine::advance_after_action_round(GameState& state) noexcept {
    if (state.current_phase == Phase::GAME_OVER) return;
    // Check NORAD
    if (state.defcon_dropped_to_2_in_ar && state.has_flag(effect_bits::NORAD_ACTIVE) &&
        Scoring::is_controlled_by(state, countries::CANADA, Player::US)) {
        // US gets 1 free influence placement
        state.defcon_dropped_to_2_in_ar = 0;
        state.ctx().decision_player = Player::US;
        state.ctx().decision_type = DecisionType::POINT_NODE;
        state.ctx().remaining_steps = 1;
        state.ctx().resolving_card = card_ids::NORAD;
        return;
    }
    state.defcon_dropped_to_2_in_ar = 0;

    // Determine max ARs for this turn
    uint8_t max_ar = (state.turn <= 3) ? 6 : 7;
    bool us_has_ar8 = state.has_flag(effect_bits::NORTH_SEA_OIL_ACTIVE) || SpaceRace::has_space_station_ar8(state, Player::US);
    bool ussr_has_ar8 = SpaceRace::has_space_station_ar8(state, Player::USSR);
    if ((us_has_ar8 || ussr_has_ar8) && state.turn >= 4) {
        max_ar = 8;
    }

    // Alternate phasing player
    if (state.phasing_player == Player::USSR) {
        state.phasing_player = Player::US;
    } else {
        state.phasing_player = Player::USSR;
        state.action_round++;
    }

    if (state.action_round > max_ar) {
        // All Action Rounds completed for this turn -> proceed to turn end
        end_turn(state);
        return;
    }

    // Start next Action Round
    state.ctx() = DecisionContext{};
    state.ctx().decision_player = state.phasing_player;

    // Check if player has cards
    CardLocation hand = (state.phasing_player == Player::US) ? CardLocation::HAND_US : CardLocation::HAND_USSR;
    uint8_t count = 0;
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == hand) count++;
    }

    if (count == 0 && !(state.china_card_holder == state.phasing_player && state.china_card_playable)) {
        // Player has no cards -> automatically pass to next player
        advance_after_action_round(state);
        return;
    }

    state.ctx().decision_type = DecisionType::SELECT_CARD;
}

void StateMachine::end_turn(GameState& state) noexcept {
    // Phase E: Military Operations Status Check
    Scoring::evaluate_military_ops(state);
    if (state.current_phase == Phase::GAME_OVER) return;

    // Check Space Walk (Box 6) Privilege
    bool us_sw = SpaceRace::has_space_walk(state, Player::US);
    bool ussr_sw = SpaceRace::has_space_walk(state, Player::USSR);
    Player sw_player = us_sw ? Player::US : (ussr_sw ? Player::USSR : Player::NONE);

    if (sw_player != Player::NONE) {
        CardLocation loc = (sw_player == Player::US) ? CardLocation::HAND_US : CardLocation::HAND_USSR;
        bool has_cards = false;
        for (uint8_t i = 1; i <= 110; ++i) {
            if (state.card_locations[i] == loc) {
                has_cards = true;
                break;
            }
        }
        if (has_cards) {
            state.ctx() = DecisionContext{};
            state.ctx().decision_player = sw_player;
            state.ctx().decision_type = DecisionType::SELECT_CARD;
            state.ctx().resolving_card = card_ids::SPACE_WALK_DISCARD;
            state.ctx().allow_early_stop = 1;
            return;
        }
    }

    finish_end_turn(state);
}

void StateMachine::finish_end_turn(GameState& state) noexcept {
    // Phase F: Check held cards (Scoring cards cannot be held!)
    bool us_holds = false;
    bool ussr_holds = false;
    for (uint8_t i = 1; i <= 110; ++i) {
        if (CardData::is_scoring_card(i)) {
            if (state.card_locations[i] == CardLocation::HAND_US) us_holds = true;
            if (state.card_locations[i] == CardLocation::HAND_USSR) ussr_holds = true;
        }
    }
    if (us_holds || ussr_holds) {
        state.current_phase = Phase::GAME_OVER;
        if (us_holds && ussr_holds) {
            state.victory_points = 0;
        } else if (us_holds) {
            state.victory_points = -20;
        } else {
            state.victory_points = 20;
        }
        return;
    }

    // Phase G: Flip China Card face up
    state.china_card_playable = 1;

    // Phase H: Advance Turn
    state.persistent_effects &= ~effect_bits::TURN_CLEANUP_MASK;
    state.turn_aggregates.clear();

    state.turn++;
    state.action_round = 0;

    if (state.turn == 4) {
        add_era_cards_to_deck(state, WarEra::MID);
        reshuffle_discard_into_draw(state);
    } else if (state.turn == 8) {
        add_era_cards_to_deck(state, WarEra::LATE);
        reshuffle_discard_into_draw(state);
    }

    if (state.turn <= 10) {
        start_turn(state);
    } else {
        // Phase I: Final Scoring
        Scoring::execute_final_scoring(state);
    }
}

static void snapshot_op_influence(GameState& state, Player p) noexcept {
    state.ctx().start_influence_nodes = {};
    for (uint8_t i = 0; i < 84; ++i) {
        if (state.countries[i].get_influence(p) > 0) {
            state.ctx().set_start_influence(i);
        }
    }
}

bool StateMachine::step(GameState& state, const MicroAction& action) noexcept {
    // Reset ephemeral die roll record for the current step
    state.last_roll = DieRollRecord{};
    state.last_die_roll = 0;
    state.last_opp_die_roll = 0;
    if (state.current_phase == Phase::GAME_OVER) return false;

    // 1. SETUP PHASE
    if (state.current_phase == Phase::SETUP) {
        if (state.ctx().decision_player == Player::USSR) {
            uint8_t cid = action.primary_id;
            if (cid < 84 && MapData::get_country(cid).in_eastern_europe) {
                state.countries[cid].add_influence(Player::USSR, 1);
                state.ctx().remaining_steps--;
                if (state.ctx().remaining_steps == 0) {
                    // Transition to US placing 7 in Western Europe
                    state.ctx().decision_player = Player::US;
                    state.ctx().remaining_steps = 7;
                }
                return true;
            }
            return false;
        } else if (state.ctx().decision_player == Player::US) {
            uint8_t cid = action.primary_id;
            if (state.ctx().pending_ops_value == 0) {
                // Stage 0: 7 points in Western Europe
                if (cid < 84 && MapData::get_country(cid).in_western_europe) {
                    state.countries[cid].add_influence(Player::US, 1);
                    state.ctx().remaining_steps--;
                    if (state.ctx().remaining_steps == 0) {
                        // Transition to US placing 2 bonus influence in countries with existing US presence
                        state.ctx().pending_ops_value = 1; // Stage 1: Bonus placement
                        state.ctx().remaining_steps = 2;
                    }
                    return true;
                }
                return false;
            } else {
                // Stage 1: 2 bonus influence in any country with existing US influence
                if (cid < 84 && state.countries[cid].us_influence > 0) {
                    state.countries[cid].add_influence(Player::US, 1);
                    state.ctx().remaining_steps--;
                    if (state.ctx().remaining_steps == 0) {
                        // Setup complete -> start Turn 1
                        start_turn(state);
                    }
                    return true;
                }
                return false;
            }
        }
    }

    // 2. HEADLINE PHASE CARD SELECTION
    if (state.current_phase == Phase::HEADLINE && state.ctx().temp_cards[4] == 0 && state.ctx().decision_type == DecisionType::SELECT_CARD && state.ctx().resolving_card == 0) {
        Player p = state.ctx().decision_player;
        uint8_t card = action.primary_id;
        if (p == Player::US && state.headline_us_card == 0) {
            state.headline_us_card = card;
            if (state.headline_ussr_card == 0) {
                state.ctx().decision_player = Player::USSR;
                state.ctx().decision_type = DecisionType::SELECT_CARD;
                return true;
            }
        } else if (p == Player::USSR && state.headline_ussr_card == 0) {
            state.headline_ussr_card = card;
            if (state.headline_us_card == 0) {
                state.ctx().decision_player = Player::US;
                state.ctx().decision_type = DecisionType::SELECT_CARD;
                return true;
            }
        }

        // Both headlines selected -> resolve headlines in ops priority order
        uint8_t us_h = state.headline_us_card;
        uint8_t ussr_h = state.headline_ussr_card;
        bool ussr_cancelled = (us_h == card_ids::DEFECTORS);

        uint8_t us_hv = CardData::get_card(us_h).ops;
        uint8_t ussr_hv = CardData::get_card(ussr_h).ops;
        bool us_first = (us_hv >= ussr_hv); // US wins ties

        uint8_t first_card = us_first ? us_h : ussr_h;
        Player first_owner = us_first ? Player::US : Player::USSR;
        uint8_t second_card = us_first ? ussr_h : us_h;
        Player second_owner = us_first ? Player::USSR : Player::US;

        state.headline_first_card = first_card;
        state.headline_second_card = second_card;
        state.headline_first_owner = first_owner;
        state.headline_second_owner = second_owner;
        state.headline_stage = 1; // Stage 1: Resolving 1st headline

        if (first_owner == Player::USSR && ussr_cancelled) {
            state.card_locations[first_card] = CardLocation::DISCARD_PILE;
            advance_headline_step(state);
            return true;
        }

        const auto& c_info = CardData::get_card(first_card);
        Player exec_player = (c_info.side == get_opponent(first_owner)) ? get_opponent(first_owner) : first_owner;
        state.phasing_player = first_owner;
        state.ctx().decision_player = exec_player;
        state.ctx().resolving_card = first_card;

        bool done = CardHandlers::trigger_event(state, first_card, exec_player);
        if (first_card != card_ids::KITCHEN_DEBATES) {
            if (first_card == card_ids::SHUTTLE_DIPLOMACY && state.has_flag(effect_bits::SHUTTLE_DIPLOMACY_ACTIVE)) {
                state.card_locations[first_card] = CardLocation::ONGOING_EVENT;
            } else {
                state.card_locations[first_card] = c_info.one_time ? CardLocation::REMOVED_FROM_GAME : CardLocation::DISCARD_PILE;
            }
        }
        if (done) {
            advance_headline_step(state);
        }
        return true;
    }

    // 3. INTERACTIVE DECISIONS (HEADLINE OR ACTION ROUNDS)
    if (state.current_phase == Phase::ACTION_ROUND || state.current_phase == Phase::HEADLINE) {
        DecisionType dt = state.ctx().decision_type;
        Player p = state.ctx().decision_player;

        // If resolving active event sub-decision
        if (state.ctx().resolving_card != 0) {
            if (state.ctx().resolving_card == card_ids::SPACE_WALK_DISCARD) {
                uint8_t card = action.primary_id;
                if (card >= 1 && card <= 110 && !action.is_confirm_done()) {
                    state.card_locations[card] = CardLocation::DISCARD_PILE;
                }
                state.ctx() = DecisionContext{};
                finish_end_turn(state);
                return true;
            }

            bool finished = CardHandlers::handle_event_step(state, action);
            if (finished || action.is_confirm_done()) {
                state.ctx().resolving_card = 0;
                if (state.ctx_stack_depth > 0) {
                    state.pop_context();
                    if (state.ctx().decision_type == DecisionType::SELECT_OP_MODE) {
                        // Resumed from EVENT_FIRST
                    } else {
                        // Resumed from nested card execution (Missile Envy, Five Year Plan, Star Wars)
                        state.ctx().resolving_card = 0;
                        if (state.current_phase == Phase::HEADLINE) {
                            advance_headline_step(state);
                        } else {
                            advance_after_action_round(state);
                        }
                    }
                } else if (state.current_phase == Phase::HEADLINE) {
                    advance_headline_step(state);
                } else {
                    advance_after_action_round(state);
                }
            }
            return true;
        }

        switch (dt) {
            case DecisionType::SELECT_CARD: {
                uint8_t card = action.primary_id;
                if (card == 0 || action.is_confirm_done()) {
                    if (state.current_phase == Phase::HEADLINE) advance_headline_step(state);
                    else advance_after_action_round(state);
                    return true;
                }
                state.ctx().pending_op_card = card;

                if (state.forced_card_player == p && (state.forced_card_id == card || state.forced_card_id == 0)) {
                    state.forced_card_player = Player::NONE;
                    state.forced_card_id = 0;
                }

                // Quagmire / Bear Trap handling
                bool trapped = (p == Player::US && state.has_flag(effect_bits::QUAGMIRE_ACTIVE)) ||
                               (p == Player::USSR && state.has_flag(effect_bits::BEAR_TRAP_ACTIVE));

                if (trapped) {
                    const auto& c_info = CardData::get_card(card);
                    if (c_info.ops >= 2) {
                        state.card_locations[card] = CardLocation::DISCARD_PILE;
                        state.ctx().temp_cards[0] = card;
                        state.ctx().temp_cards[1] = static_cast<uint8_t>(RollType::TRAP_ESCAPE);
                        state.ctx().temp_cards[2] = action.secondary_id;
                        state.ctx().decision_player = Player::NONE;
                        state.ctx().decision_type = DecisionType::ROLL_DIE;
                        return true;
                    }
                }

                // We Will Bury You check on US Action Round
                if (p == Player::US && state.current_phase == Phase::ACTION_ROUND && state.has_flag(effect_bits::WE_WILL_BURY_YOU_PENDING)) {
                    if (card != card_ids::UN_INTERVENTION) {
                        state.clear_flag(effect_bits::WE_WILL_BURY_YOU_PENDING);
                        state.victory_points = static_cast<int8_t>(std::max(-20, state.victory_points - 3));
                        if (state.victory_points <= -20) {
                            state.current_phase = Phase::GAME_OVER;
                            return true;
                        }
                    }
                }

                // Scoring card auto-resolve
                if (CardData::is_scoring_card(card)) {
                    state.card_locations[card] = CardData::get_card(card).one_time ? CardLocation::REMOVED_FROM_GAME : CardLocation::DISCARD_PILE;
                    CardHandlers::trigger_event(state, card, p);
                    if (state.current_phase != Phase::GAME_OVER) {
                        if (state.current_phase == Phase::HEADLINE) advance_headline_step(state);
                        else advance_after_action_round(state);
                    }
                    return true;
                }

                // Transition to SELECT_PLAY_MODE
                state.ctx().decision_type = DecisionType::SELECT_PLAY_MODE;
                return true;
            }

            case DecisionType::SELECT_PLAY_MODE: {
                uint8_t card = state.ctx().pending_op_card;
                PlayMode mode = static_cast<PlayMode>(action.primary_id);

                // We Will Bury You check on US Action Round when playing UN Intervention
                if (p == Player::US && state.current_phase == Phase::ACTION_ROUND && state.has_flag(effect_bits::WE_WILL_BURY_YOU_PENDING)) {
                    state.clear_flag(effect_bits::WE_WILL_BURY_YOU_PENDING);
                    if (mode != PlayMode::EVENT) {
                        state.victory_points = static_cast<int8_t>(std::max(-20, state.victory_points - 3));
                        if (state.victory_points <= -20) {
                            state.current_phase = Phase::GAME_OVER;
                            return true;
                        }
                    }
                }

                if (mode == PlayMode::SPACE) {
                    state.ctx().temp_cards[0] = card;
                    state.ctx().temp_cards[1] = static_cast<uint8_t>(RollType::SPACE_RACE);
                    state.ctx().temp_cards[2] = action.secondary_id;
                    state.ctx().decision_player = Player::NONE;
                    state.ctx().decision_type = DecisionType::ROLL_DIE;
                    return true;
                }

                if (mode == PlayMode::EVENT) {
                    if (card == card_ids::THE_CHINA_CARD ||
                        (card == card_ids::DEFECTORS && p == Player::US && state.current_phase == Phase::ACTION_ROUND) ||
                        CardData::is_opponent_card(card, p) ||
                        !CardHandlers::can_trigger_event(state, card, p)) {
                        return false; // Illegal event play
                    }
                    const auto& c_info = CardData::get_card(card);
                    bool done = CardHandlers::trigger_event(state, card, p, action.secondary_id);
                    if (card != card_ids::KITCHEN_DEBATES) {
                        if (card == card_ids::SHUTTLE_DIPLOMACY && state.has_flag(effect_bits::SHUTTLE_DIPLOMACY_ACTIVE)) {
                            state.card_locations[card] = CardLocation::ONGOING_EVENT;
                        } else {
                            state.card_locations[card] = c_info.one_time ? CardLocation::REMOVED_FROM_GAME : CardLocation::DISCARD_PILE;
                        }
                    }
                    if (done && state.current_phase != Phase::GAME_OVER) {
                        if (state.current_phase == Phase::HEADLINE) advance_headline_step(state);
                        else advance_after_action_round(state);
                    }
                    return true;
                }

                if (mode == PlayMode::OPS) {
                    if (p == Player::US && CardData::is_war_card(card) && state.has_flag(effect_bits::FLOWER_POWER_ACTIVE)) {
                        state.victory_points = static_cast<int8_t>(std::max(-20, state.victory_points - 2));
                        if (state.victory_points <= -20) {
                            state.current_phase = Phase::GAME_OVER;
                            return true;
                        }
                    }
                    if (card == card_ids::THE_CHINA_CARD && p == Player::US) {
                        state.clear_flag(effect_bits::FORMOSAN_RESOLUTION_ACTIVE);
                    }
                    if (CardData::is_opponent_card(card, p)) {
                        state.ctx().decision_type = DecisionType::CHOOSE_TIMING_BRANCH;
                        return true;
                    }
                    state.ctx().decision_type = DecisionType::SELECT_OP_MODE;
                    state.ctx().pending_ops_value = Operations::get_effective_ops(state, card, p, Region::ASIA);
                    return true;
                }
                return false;
            }

            case DecisionType::CHOOSE_TIMING_BRANCH: {
                uint8_t card = state.ctx().pending_op_card;
                TimingBranch branch = static_cast<TimingBranch>(action.primary_id);
                state.ctx().timing_branch = static_cast<uint8_t>(branch);

                if (branch == TimingBranch::OPS_FIRST) {
                    state.ctx().decision_type = DecisionType::SELECT_OP_MODE;
                    state.ctx().pending_ops_value = Operations::get_effective_ops(state, card, p, Region::ASIA);
                    return true;
                }

                if (branch == TimingBranch::EVENT_FIRST) {
                    state.ctx().timing_branch = static_cast<uint8_t>(TimingBranch::EVENT_FIRST);
                    state.ctx().pending_ops_value = Operations::get_effective_ops(state, card, p, Region::ASIA);
                    state.ctx().decision_type = DecisionType::SELECT_OP_MODE;
                    state.ctx().decision_player = p;
                    state.push_context();

                    Player opp = get_opponent(p);
                    const auto& c_info = CardData::get_card(card);
                    state.ctx().decision_player = opp;
                    state.ctx().resolving_card = card;

                    bool done = CardHandlers::trigger_event(state, card, opp);
                    if (card != card_ids::KITCHEN_DEBATES) {
                        state.card_locations[card] = c_info.one_time ? CardLocation::REMOVED_FROM_GAME : CardLocation::DISCARD_PILE;
                    }
                    if (done) {
                        state.pop_context();
                    }
                    return true;
                }
                return false;
            }

            case DecisionType::SELECT_OP_MODE: {
                if (action.is_confirm_done() || action.primary_id == 255 ||
                    (action.primary_id == 0 && (state.ctx().pending_op_card == card_ids::JUNTA || state.ctx().pending_op_card == card_ids::TEAR_DOWN_THIS_WALL))) {
                    if (state.current_phase == Phase::HEADLINE) advance_headline_step(state);
                    else advance_after_ops(state);
                    return true;
                }
                OpMode op_mode = static_cast<OpMode>(action.primary_id);
                uint8_t ops = state.ctx().pending_ops_value;
                state.ctx().op_mode = op_mode;

                if (op_mode == OpMode::INFLUENCE) {
                    if (state.ctx().pending_op_card == card_ids::JUNTA || state.ctx().pending_op_card == card_ids::TEAR_DOWN_THIS_WALL) {
                        advance_after_ops(state);
                        return true;
                    }
                    snapshot_op_influence(state, p);
                    state.ctx().decision_type = DecisionType::POINT_NODE;
                    state.ctx().remaining_steps = ops;
                    state.ctx().allow_early_stop = 1;
                    return true;
                }

                if (op_mode == OpMode::COUP) {
                    state.ctx().decision_type = DecisionType::POINT_NODE;
                    state.ctx().remaining_steps = 1;
                    state.ctx().max_per_country = 1;
                    state.ctx().allow_early_stop = 0;
                    return true;
                }

                if (op_mode == OpMode::REALIGN) {
                    state.ctx().decision_type = DecisionType::POINT_NODE;
                    state.ctx().remaining_steps = ops;
                    state.ctx().allow_early_stop = 1;
                    return true;
                }
                return false;
            }

            case DecisionType::POINT_NODE: {
                if (action.is_confirm_done() || action.primary_id == 84) {
                    // Early stop Ops
                    advance_after_ops(state);
                    return true;
                }

                uint8_t cid = action.primary_id;
                uint8_t forced_roll = action.secondary_id;
                uint8_t forced_opp_roll = action.flags;

                if (state.ctx().op_mode == OpMode::COUP) {
                    uint8_t coup_ops = state.ctx().pending_ops_value;
                    uint8_t op_card = state.ctx().pending_op_card;
                    const auto& c_info = MapData::get_country(cid);
                    if (op_card == card_ids::THE_CHINA_CARD) {
                        coup_ops = Operations::get_effective_ops(state, op_card, p, c_info.region);
                    } else if (p == Player::USSR && state.has_flag(effect_bits::VIETNAM_REVOLTS_ACTIVE)) {
                        coup_ops = Operations::get_effective_ops(state, op_card, p, c_info.in_southeast_asia ? Region::ASIA : Region::NONE_REGION);
                    }
                    state.ctx().temp_cards[0] = cid;
                    state.ctx().temp_cards[1] = static_cast<uint8_t>(RollType::COUP);
                    state.ctx().temp_cards[2] = forced_roll;
                    state.ctx().temp_cards[3] = (p == Player::US) ? 1 : (p == Player::USSR ? 2 : 0);
                    state.ctx().pending_ops_value = coup_ops;
                    state.ctx().decision_player = Player::NONE;
                    state.ctx().decision_type = DecisionType::ROLL_DIE;
                    return true;
                }

                if (state.ctx().op_mode == OpMode::REALIGN) {
                    uint8_t forced_us = (p == Player::US) ? forced_roll : forced_opp_roll;
                    uint8_t forced_ussr = (p == Player::USSR) ? forced_roll : forced_opp_roll;
                    state.ctx().temp_cards[0] = cid;
                    state.ctx().temp_cards[1] = static_cast<uint8_t>(RollType::REALIGNMENT);
                    state.ctx().temp_cards[2] = forced_us;
                    state.ctx().temp_cards[3] = forced_ussr;
                    state.ctx().temp_cards[4] = (p == Player::US) ? 1 : (p == Player::USSR ? 2 : 0);
                    state.ctx().decision_player = Player::NONE;
                    state.ctx().decision_type = DecisionType::ROLL_DIE;
                    return true;
                }

                // Influence placement
                if (cid < 84) {
                    uint8_t cost = Operations::get_influence_cost(state, p, cid);
                    if (state.ctx().remaining_steps >= cost && Operations::can_place_influence(state, p, cid)) {
                        Operations::place_influence(state, p, cid);
                        state.ctx().remaining_steps -= cost;

                        uint8_t op_card = state.ctx().pending_op_card;
                        const auto& c_info = MapData::get_country(cid);
                        if (op_card == card_ids::THE_CHINA_CARD && c_info.region != Region::ASIA) {
                            uint8_t non_asia_base = Operations::get_effective_ops(state, op_card, p, Region::NONE_REGION);
                            uint8_t total_spent = state.ctx().pending_ops_value - state.ctx().remaining_steps;
                            if (total_spent >= non_asia_base) {
                                state.ctx().remaining_steps = 0;
                            } else {
                                state.ctx().remaining_steps = non_asia_base - total_spent;
                            }
                        } else if (p == Player::USSR && state.has_flag(effect_bits::VIETNAM_REVOLTS_ACTIVE) && !c_info.in_southeast_asia) {
                            uint8_t non_se_base = Operations::get_effective_ops(state, op_card, p, Region::NONE_REGION);
                            uint8_t total_spent = state.ctx().pending_ops_value - state.ctx().remaining_steps;
                            if (total_spent >= non_se_base) {
                                state.ctx().remaining_steps = 0;
                            } else {
                                state.ctx().remaining_steps = non_se_base - total_spent;
                            }
                        }

                        if (state.ctx().remaining_steps == 0) {
                            advance_after_ops(state);
                        }
                        return true;
                    }
                }

                if (state.ctx().allow_early_stop || action.primary_id == 0 || action.primary_id >= 84) {
                    advance_after_ops(state);
                    return true;
                }
                return false;
            }

                        case DecisionType::ROLL_DIE: {
                RollType rt = static_cast<RollType>(state.ctx().temp_cards[1]);
                uint8_t forced_r1 = action.primary_id != 0 ? action.primary_id : state.ctx().temp_cards[2];
                uint8_t forced_r2 = action.secondary_id != 0 ? action.secondary_id : state.ctx().temp_cards[3];

                if (rt == RollType::COUP) {
                    uint8_t cid = state.ctx().temp_cards[0];
                    uint8_t coup_ops = state.ctx().pending_ops_value;
                    Player coup_player = (state.ctx().temp_cards[3] == 1) ? Player::US : ((state.ctx().temp_cards[3] == 2) ? Player::USSR : state.phasing_player);
                    Operations::execute_coup(state, coup_player, cid, coup_ops, forced_r1);
                    if (state.current_phase != Phase::GAME_OVER) {
                        advance_after_ops(state);
                    }
                    return true;
                }

                if (rt == RollType::REALIGNMENT) {
                    uint8_t cid = state.ctx().temp_cards[0];
                    Player realign_player = (state.ctx().temp_cards[4] == 1) ? Player::US : ((state.ctx().temp_cards[4] == 2) ? Player::USSR : state.phasing_player);
                    uint8_t forced_us = (realign_player == Player::US) ? forced_r1 : forced_r2;
                    uint8_t forced_ussr = (realign_player == Player::USSR) ? forced_r1 : forced_r2;
                    Operations::execute_realign(state, realign_player, cid, forced_us, forced_ussr);
                    state.ctx().remaining_steps -= 1;

                    uint8_t op_card = state.ctx().pending_op_card;
                    const auto& c_info = MapData::get_country(cid);
                    if (op_card == card_ids::THE_CHINA_CARD && c_info.region != Region::ASIA) {
                        uint8_t non_asia_base = Operations::get_effective_ops(state, op_card, realign_player, Region::NONE_REGION);
                        uint8_t total_spent = state.ctx().pending_ops_value - state.ctx().remaining_steps;
                        if (total_spent >= non_asia_base) {
                            state.ctx().remaining_steps = 0;
                        } else {
                            state.ctx().remaining_steps = non_asia_base - total_spent;
                        }
                    } else if (realign_player == Player::USSR && state.has_flag(effect_bits::VIETNAM_REVOLTS_ACTIVE) && !c_info.in_southeast_asia) {
                        uint8_t non_se_base = Operations::get_effective_ops(state, op_card, realign_player, Region::NONE_REGION);
                        uint8_t total_spent = state.ctx().pending_ops_value - state.ctx().remaining_steps;
                        if (total_spent >= non_se_base) {
                            state.ctx().remaining_steps = 0;
                        } else {
                            state.ctx().remaining_steps = non_se_base - total_spent;
                        }
                    }

                    if (state.ctx().remaining_steps == 0) {
                        advance_after_ops(state);
                    } else {
                        state.ctx().decision_player = realign_player;
                        state.ctx().decision_type = DecisionType::POINT_NODE;
                        state.ctx().allow_early_stop = 1;
                    }
                    return true;
                }

                if (rt == RollType::SPACE_RACE) {
                    uint8_t card = state.ctx().temp_cards[0];
                    Player space_player = state.phasing_player;
                    SpaceRace::attempt_space(state, space_player, card, forced_r1);
                    if (state.current_phase != Phase::GAME_OVER) {
                        if (state.current_phase == Phase::HEADLINE) advance_headline_step(state);
                        else advance_after_action_round(state);
                    }
                    return true;
                }

                if (rt == RollType::TRAP_ESCAPE) {
                    uint8_t card = state.ctx().temp_cards[0];
                    Player trapped_p = state.phasing_player;
                    uint8_t roll = (forced_r1 >= 1 && forced_r1 <= 6) ? forced_r1 : Prng::roll_d6(state.rng_state);
                    state.last_die_roll = roll;
                    bool escaped = (roll <= 4);
                    if (escaped) {
                        if (trapped_p == Player::US) state.clear_flag(effect_bits::QUAGMIRE_ACTIVE);
                        else state.clear_flag(effect_bits::BEAR_TRAP_ACTIVE);
                    }
                    state.last_roll = DieRollRecord{
                        .type = RollType::TRAP_ESCAPE,
                        .roller = trapped_p,
                        .card_id = card,
                        .country_id = 255,
                        .roll1 = roll,
                        .mod1 = 0,
                        .roll2 = 0,
                        .mod2 = 4, // escape threshold
                        .success = escaped,
                        .net_delta = 0
                    };
                    if (state.current_phase == Phase::HEADLINE) advance_headline_step(state);
                    else advance_after_action_round(state);
                    return true;
                }

                return false;
            }

            case DecisionType::CHOOSE_BRANCH: {
                if (state.ctx().resolving_card != 0) {
                    bool finished = CardHandlers::handle_event_step(state, action);
                    if (finished || action.is_confirm_done()) {
                        state.ctx().resolving_card = 0;
                        if (state.ctx_stack_depth > 0) {
                            state.pop_context();
                        } else if (state.current_phase == Phase::HEADLINE) {
                            advance_headline_step(state);
                        } else {
                            advance_after_action_round(state);
                        }
                    }
                    return true;
                }
                return false;
            }

            default:
                break;
        }
    }

    return false;
}

} // namespace ts
