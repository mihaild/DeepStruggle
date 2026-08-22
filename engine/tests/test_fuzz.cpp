#include "ts/engine.hpp"
#include "ts/prng.hpp"
#include "ts/constants.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
    uint64_t max_steps = 100000;
    uint64_t max_games = 0;
    uint64_t seed = 42;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--steps" && i + 1 < argc) {
            max_steps = std::stoull(argv[++i]);
        } else if (arg == "--games" && i + 1 < argc) {
            max_games = std::stoull(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoull(argv[++i]);
        } else if (arg[0] != '-') {
            max_steps = std::stoull(arg);
        }
    }

    std::cout << "Starting Invariant Fuzzer: max_steps=" << max_steps
              << ", max_games=" << max_games
              << ", seed=" << seed << "..." << std::endl;

    uint64_t fuzzer_prng = seed;
    uint64_t total_steps = 0;
    uint64_t total_games = 0;

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
            std::cerr << "Fuzzer error: no legal actions found in mask! decision_type="
                      << static_cast<int>(state.ctx().decision_type) << std::endl;
            std::exit(1);
        }

        uint32_t chosen_idx = ts::Prng::random_index(fuzzer_prng, static_cast<uint32_t>(legal_indices.size()));
        uint8_t chosen_action_id = legal_indices[chosen_idx];

        ts::MicroAction action{};
        action.decision_type = state.ctx().decision_type;
        action.primary_id = chosen_action_id;

        // Step engine
        ts::Engine::step(state, action);
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
            std::cout << "Fuzzer progress: " << total_steps << " steps (" << total_games << " games completed)" << std::endl;
        }
    }

    std::cout << "Invariant Fuzzing Completed Successfully: " << total_steps << " steps, " << total_games << " full games." << std::endl;
    return 0;
}
