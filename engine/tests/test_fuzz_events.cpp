#include "ts/engine.hpp"
#include "test_framework.hpp"
#include "ts/prng.hpp"
#include "ts/constants.hpp"
#include "ts/card_data.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <string>
#include <algorithm>

int main(int argc, char** argv) {
    uint64_t max_steps = 100000;
    uint64_t max_games = 0;
    uint64_t seed = 42;
    uint32_t event_bias_pct = 95; // 95% probability of playing events

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--steps" && i + 1 < argc) {
            max_steps = std::stoull(argv[++i]);
        } else if (arg == "--games" && i + 1 < argc) {
            max_games = std::stoull(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoull(argv[++i]);
        } else if (arg == "--event-bias" && i + 1 < argc) {
            event_bias_pct = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg[0] != '-') {
            max_steps = std::stoull(arg);
        }
    }

    std::cout << "Starting Event-Biased Fuzzer (Event Bias: " << event_bias_pct << "%): max_steps=" << max_steps
              << ", max_games=" << max_games
              << ", seed=" << seed << "..." << std::endl;

    uint64_t fuzzer_prng = seed;
    uint64_t total_steps = 0;
    uint64_t total_games = 0;
    uint64_t total_events_triggered = 0;

    ts::GameState state{};
    ts::Engine::init_game(state, ts::Prng::next_u64(fuzzer_prng));

    uint8_t mask_buf[128];
    size_t mask_size = 0;

    while (true) {
        if (max_games > 0 && total_games >= max_games) break;
        if (max_games == 0 && total_steps >= max_steps) break;

        if (ts::Engine::is_terminal(state)) {
            total_games++;
            ts::Engine::init_game(state, ts::Prng::next_u64(fuzzer_prng));
            if (max_games > 0 && total_games >= max_games) break;
        }

        ts::Engine::get_legal_action_mask(state, mask_buf, &mask_size);

        if (mask_size == 0) {
            std::cerr << "Fuzzer error: empty action mask on non-terminal state!" << std::endl;
            std::exit(1);
        }

        // Collect legal indices
        std::vector<uint8_t> legal_indices;
        for (size_t i = 0; i < mask_size; ++i) {
            if (mask_buf[i]) legal_indices.push_back(static_cast<uint8_t>(i));
        }

        if (legal_indices.empty()) {
            // P17: the canonical legal set is the 212-dim flat mask -- that is what
            // StateMachine::step validates against. The narrow mask has no slot for "decline" at
            // every decision type, so consult the flat one before declaring a dead end: if it
            // offers the shared decline index, confirm/done is a legal move here.
            uint8_t flat[ts::FLAT_ACTION_SPACE_SIZE] = {0};
            ts::ActionMask::generate_flat_mask_212(state, flat);
            if (flat[ts::flat_slots::CONFIRM_DONE] || state.ctx().allow_early_stop ||
                state.ctx().decision_type == ts::DecisionType::POINT_NODE) {
                ts::MicroAction action{};
                action.decision_type = state.ctx().decision_type;
                action.flags = ts::action_flags::CONFIRM_DONE;
                ASSERT_TRUE(ts::Engine::step(state, action));
                total_steps++;
                continue;
            }
            std::cerr << "Fuzzer error: no legal actions found in mask! decision_type="
                      << static_cast<int>(state.ctx().decision_type) << std::endl;
            std::exit(1);
        }

        uint8_t chosen_action_id = legal_indices[0];
        ts::DecisionType d_type = state.ctx().decision_type;

        // Apply Event-Heavy Biasing
        if (d_type == ts::DecisionType::SELECT_PLAY_MODE) {
            // Check if PlayMode::EVENT (0) is legal
            bool event_legal = (std::find(legal_indices.begin(), legal_indices.end(), 0) != legal_indices.end());
            uint32_t roll = ts::Prng::random_index(fuzzer_prng, 100);
            if (event_legal && roll < event_bias_pct) {
                chosen_action_id = 0; // Play as Event!
                total_events_triggered++;
            } else {
                uint32_t idx = ts::Prng::random_index(fuzzer_prng, static_cast<uint32_t>(legal_indices.size()));
                chosen_action_id = legal_indices[idx];
            }
        } else if (d_type == ts::DecisionType::CHOOSE_TIMING_BRANCH) {
            // Check if TimingBranch::EVENT_FIRST (1) is legal
            bool event_first_legal = (std::find(legal_indices.begin(), legal_indices.end(), 1) != legal_indices.end());
            uint32_t roll = ts::Prng::random_index(fuzzer_prng, 100);
            if (event_first_legal && roll < event_bias_pct) {
                chosen_action_id = 1; // Event First!
                total_events_triggered++;
            } else {
                uint32_t idx = ts::Prng::random_index(fuzzer_prng, static_cast<uint32_t>(legal_indices.size()));
                chosen_action_id = legal_indices[idx];
            }
        } else if (d_type == ts::DecisionType::SELECT_CARD && state.ctx().resolving_card == 0) {
            // Prioritize cards with events over pure scoring/ops if possible
            std::vector<uint8_t> event_cards;
            for (uint8_t cid : legal_indices) {
                if (cid >= 1 && cid <= 110) {
                    const auto& c_info = ts::CardData::get_card(cid);
                    if (!c_info.is_scoring) {
                        event_cards.push_back(cid);
                    }
                }
            }
            uint32_t roll = ts::Prng::random_index(fuzzer_prng, 100);
            if (!event_cards.empty() && roll < event_bias_pct) {
                uint32_t idx = ts::Prng::random_index(fuzzer_prng, static_cast<uint32_t>(event_cards.size()));
                chosen_action_id = event_cards[idx];
            } else {
                uint32_t idx = ts::Prng::random_index(fuzzer_prng, static_cast<uint32_t>(legal_indices.size()));
                chosen_action_id = legal_indices[idx];
            }
        } else {
            // Default random choice among legal actions
            uint32_t chosen_idx = ts::Prng::random_index(fuzzer_prng, static_cast<uint32_t>(legal_indices.size()));
            chosen_action_id = legal_indices[chosen_idx];
        }

        ts::MicroAction action{};
        action.decision_type = d_type;
        action.primary_id = chosen_action_id;

        // Step engine
        ASSERT_TRUE(ts::Engine::step(state, action));
        total_steps++;

        // Invariant checks
        if (state.victory_points < -20 || state.victory_points > 20) {
            std::cerr << "Invariant violated: victory_points=" << static_cast<int>(state.victory_points) << std::endl;
            std::exit(1);
        }
        if (state.defcon < 1 || state.defcon > 5) {
            std::cerr << "Invariant violated: defcon=" << static_cast<int>(state.defcon) << std::endl;
            std::exit(1);
        }
        if (state.us_mil_ops > 5 || state.ussr_mil_ops > 5) {
            std::cerr << "Invariant violated: mil_ops US=" << static_cast<int>(state.us_mil_ops)
                      << ", USSR=" << static_cast<int>(state.ussr_mil_ops) << std::endl;
            std::exit(1);
        }
        if (state.us_space_track > 8 || state.ussr_space_track > 8) {
            std::cerr << "Invariant violated: space_track US=" << static_cast<int>(state.us_space_track)
                      << ", USSR=" << static_cast<int>(state.ussr_space_track) << std::endl;
            std::exit(1);
        }
        if (state.ctx_stack_depth > 2) {
            std::cerr << "Invariant violated: ctx_stack_depth=" << static_cast<int>(state.ctx_stack_depth) << std::endl;
            std::exit(1);
        }

        if (total_steps % 100000 == 0) {
            std::cout << "Event Fuzzer progress: " << total_steps << " steps (" << total_games << " games completed, "
                      << total_events_triggered << " events triggered)" << std::endl;
        }
    }

    std::cout << "Event-Biased Fuzzing Completed Successfully: " << total_steps << " steps, "
              << total_games << " full games, " << total_events_triggered << " events triggered." << std::endl;
    return 0;
}
