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
    state.action_round = 1;

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
    state.countries[countries::AUSTRALIA].us_influence = 1;
    state.countries[countries::PHILIPPINES].us_influence = 1;
    state.countries[countries::SOUTH_KOREA].us_influence = 1;
    state.countries[countries::PANAMA].us_influence = 1;
    state.countries[countries::SOUTH_AFRICA].us_influence = 1;
    state.countries[countries::UNITED_KINGDOM].us_influence = 5;
    state.countries[countries::CANADA].us_influence = 2;

    // 3. China Card to USSR
    state.china_card_holder = Player::USSR;
    state.china_card_playable = 1;
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

    // Phase A: Improve DEFCON
    state.defcon = static_cast<uint8_t>(std::min(5, static_cast<int>(state.defcon) + 1));

    // Phase B: Deal Cards
    deal_cards_to_hands(state);

    // Phase C: Headline Phase
    state.current_phase = Phase::HEADLINE;
    state.headline_us_card = 0;
    state.headline_ussr_card = 0;

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
    uint8_t stage = state.ctx().temp_cards[4];
    if (stage == 1) {
        // Move to Stage 2: Second headline
        uint8_t h2_card = state.ctx().temp_cards[1];
        Player h2_owner = static_cast<Player>(state.ctx().temp_cards[3]);
        state.ctx().temp_cards[4] = 2;

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
            state.card_locations[h2_card] = c_info.one_time ? CardLocation::REMOVED_FROM_GAME : CardLocation::DISCARD_PILE;
            if (done) {
                advance_headline_step(state);
            }
            return;
        }
    }

    // Both headlines resolved -> Start Action Round 1
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
        state.card_locations[card] = c_info.one_time ? CardLocation::REMOVED_FROM_GAME : CardLocation::DISCARD_PILE;
        if (done) {
            advance_after_action_round(state);
        }
        return;
    }

    // Friendly / Neutral / Already triggered event
    if (card == card_ids::THE_CHINA_CARD) {
        state.china_card_holder = get_opponent(p);
        state.china_card_playable = 0; // Passes to opponent face down
    } else if (card != 0) {
        const auto& c_info = CardData::get_card(card);
        state.card_locations[card] = c_info.one_time ? CardLocation::REMOVED_FROM_GAME : CardLocation::DISCARD_PILE;
    }
    advance_after_action_round(state);
}

void StateMachine::advance_after_action_round(GameState& state) noexcept {
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
    if (us_has_ar8 && state.phasing_player == Player::US && state.action_round == 7) {
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

    // Phase F: Check held cards (Scoring cards cannot be held!)
    for (uint8_t i = 1; i <= 110; ++i) {
        if (CardData::is_scoring_card(i)) {
            if (state.card_locations[i] == CardLocation::HAND_US) {
                state.victory_points = -20;
                state.current_phase = Phase::GAME_OVER;
                return;
            }
            if (state.card_locations[i] == CardLocation::HAND_USSR) {
                state.victory_points = 20;
                state.current_phase = Phase::GAME_OVER;
                return;
            }
        }
    }

    // Phase G: Flip China Card face up
    state.china_card_playable = 1;

    // Phase H: Advance Turn
    state.persistent_effects &= ~effect_bits::TURN_CLEANUP_MASK;
    state.turn_aggregates.clear();
    state.us_space_turns_used = 0;
    state.ussr_space_turns_used = 0;

    state.turn++;
    state.action_round = 1;

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
            if (cid < 84 && MapData::get_country(cid).in_western_europe) {
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

    // 2. HEADLINE PHASE
    if (state.current_phase == Phase::HEADLINE) {
        Player p = state.ctx().decision_player;

        // If resolving active event sub-decision during Headline
        if (state.ctx().resolving_card != 0) {
            bool finished = CardHandlers::handle_event_step(state, action);
            if (finished) {
                if (state.ctx_stack_depth > 0) {
                    state.pop_context();
                } else {
                    advance_headline_step(state);
                }
            }
            return true;
        }

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

        state.ctx().temp_cards[0] = first_card;
        state.ctx().temp_cards[1] = second_card;
        state.ctx().temp_cards[2] = static_cast<uint8_t>(first_owner);
        state.ctx().temp_cards[3] = static_cast<uint8_t>(second_owner);
        state.ctx().temp_cards[4] = 1; // Stage 1

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
        state.card_locations[first_card] = c_info.one_time ? CardLocation::REMOVED_FROM_GAME : CardLocation::DISCARD_PILE;
        if (done) {
            advance_headline_step(state);
        }
        return true;
    }

    // 3. ACTION ROUNDS & INTERACTIVE DECISIONS
    if (state.current_phase == Phase::ACTION_ROUND) {
        DecisionType dt = state.ctx().decision_type;
        Player p = state.ctx().decision_player;

        // If resolving active event sub-decision
        if (state.ctx().resolving_card != 0) {
            bool finished = CardHandlers::handle_event_step(state, action);
            if (finished) {
                if (state.ctx_stack_depth > 0) {
                    state.pop_context();
                    if (state.ctx().decision_type == DecisionType::SELECT_OP_MODE) {
                        // Resumed from EVENT_FIRST
                    }
                } else {
                    advance_after_action_round(state);
                }
            }
            return true;
        }

        switch (dt) {
            case DecisionType::SELECT_CARD: {
                uint8_t card = action.primary_id;
                state.ctx().pending_op_card = card;

                // Quagmire / Bear Trap handling
                bool trapped = (p == Player::US && state.has_flag(effect_bits::QUAGMIRE_ACTIVE)) ||
                               (p == Player::USSR && state.has_flag(effect_bits::BEAR_TRAP_ACTIVE));

                if (trapped) {
                    const auto& c_info = CardData::get_card(card);
                    if (c_info.ops >= 2) {
                        state.card_locations[card] = CardLocation::DISCARD_PILE;
                        uint8_t roll = (action.secondary_id >= 1 && action.secondary_id <= 6) ? action.secondary_id : Prng::roll_d6(state.rng_state);
                        if (roll <= 4) {
                            if (p == Player::US) state.clear_flag(effect_bits::QUAGMIRE_ACTIVE);
                            else state.clear_flag(effect_bits::BEAR_TRAP_ACTIVE);
                        }
                        advance_after_action_round(state);
                        return true;
                    }
                }

                // Scoring card auto-resolve
                if (CardData::is_scoring_card(card)) {
                    state.card_locations[card] = CardData::get_card(card).one_time ? CardLocation::REMOVED_FROM_GAME : CardLocation::DISCARD_PILE;
                    CardHandlers::trigger_event(state, card, p);
                    if (state.current_phase != Phase::GAME_OVER) {
                        advance_after_action_round(state);
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

                if (mode == PlayMode::SPACE) {
                    SpaceRace::attempt_space(state, p, card, action.secondary_id);
                    if (state.current_phase != Phase::GAME_OVER) {
                        advance_after_action_round(state);
                    }
                    return true;
                }

                if (mode == PlayMode::EVENT) {
                    const auto& c_info = CardData::get_card(card);
                    state.card_locations[card] = c_info.one_time ? CardLocation::REMOVED_FROM_GAME : CardLocation::DISCARD_PILE;
                    bool done = CardHandlers::trigger_event(state, card, p, action.secondary_id);
                    if (done && state.current_phase != Phase::GAME_OVER) {
                        advance_after_action_round(state);
                    }
                    return true;
                }

                if (mode == PlayMode::OPS) {
                    // Check if opponent card
                    if (CardData::is_opponent_card(card, p)) {
                        state.ctx().decision_type = DecisionType::CHOOSE_TIMING_BRANCH;
                        return true;
                    } else {
                        // Friendly / neutral
                        snapshot_op_influence(state, p);
                        state.ctx().pending_ops_value = Operations::get_effective_ops(state, card, p);
                        state.ctx().decision_type = DecisionType::SELECT_OP_MODE;
                        return true;
                    }
                }
                return false;
            }

            case DecisionType::CHOOSE_TIMING_BRANCH: {
                uint8_t card = state.ctx().pending_op_card;
                TimingBranch branch = static_cast<TimingBranch>(action.primary_id);

                if (branch == TimingBranch::OPS_FIRST) {
                    snapshot_op_influence(state, p);
                    state.ctx().timing_branch = static_cast<uint8_t>(TimingBranch::OPS_FIRST);
                    state.ctx().pending_ops_value = Operations::get_effective_ops(state, card, p);
                    state.ctx().decision_type = DecisionType::SELECT_OP_MODE;
                    return true;
                } else {
                    // EVENT_FIRST
                    Player opp = get_opponent(p);
                    state.ctx().timing_branch = static_cast<uint8_t>(TimingBranch::EVENT_FIRST);
                    state.push_context();
                    state.ctx().decision_player = opp;
                    state.ctx().resolving_card = card;
                    bool done = CardHandlers::trigger_event(state, card, opp);
                    if (done) {
                        state.pop_context();
                        if (state.current_phase != Phase::GAME_OVER) {
                            state.ctx().pending_ops_value = Operations::get_effective_ops(state, card, p);
                            state.ctx().decision_type = DecisionType::SELECT_OP_MODE;
                        }
                    }
                    return true;
                }
            }

            case DecisionType::SELECT_OP_MODE: {
                OpMode op_mode = static_cast<OpMode>(action.primary_id);
                uint8_t ops = state.ctx().pending_ops_value;
                state.ctx().op_mode = op_mode;

                if (op_mode == OpMode::INFLUENCE) {
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
                if (action.is_confirm_done()) {
                    // Early stop Ops
                    advance_after_ops(state);
                    return true;
                }

                uint8_t cid = action.primary_id;
                uint8_t forced_roll = action.secondary_id;
                uint8_t forced_opp_roll = action.flags;

                if (state.ctx().op_mode == OpMode::COUP) {
                    Operations::execute_coup(state, p, cid, state.ctx().pending_ops_value, forced_roll);
                    if (state.current_phase != Phase::GAME_OVER) {
                        advance_after_ops(state);
                    }
                    return true;
                }

                if (state.ctx().op_mode == OpMode::REALIGN) {
                    uint8_t forced_us = (p == Player::US) ? forced_roll : forced_opp_roll;
                    uint8_t forced_ussr = (p == Player::USSR) ? forced_roll : forced_opp_roll;
                    Operations::execute_realign(state, p, cid, forced_us, forced_ussr);
                    state.ctx().remaining_steps -= 1;
                    if (state.ctx().remaining_steps == 0) {
                        advance_after_ops(state);
                    }
                    return true;
                }

                // Influence placement
                uint8_t cost = Operations::get_influence_cost(state, p, cid);
                if (state.ctx().remaining_steps >= cost) {
                    Operations::place_influence(state, p, cid);
                    state.ctx().remaining_steps -= cost;
                    if (state.ctx().remaining_steps == 0) {
                        advance_after_ops(state);
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
