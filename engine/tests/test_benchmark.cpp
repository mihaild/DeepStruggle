#include "ts/engine.hpp"
#include "test_framework.hpp"
#include "ts/prng.hpp"
#include "ts/constants.hpp"
#include <iostream>
#include <chrono>
#include <vector>

int main() {
    constexpr uint64_t BENCHMARK_STEPS = 500000;
    std::cout << "Running Twilight Struggle Engine Benchmark (" << BENCHMARK_STEPS << " steps)..." << std::endl;

    ts::GameState state{};
    uint64_t prng_state = 0xDEADBEEFCAFEULL;
    ts::Engine::init_game(state, ts::Prng::next_u64(prng_state));

    uint8_t mask_buf[128];
    size_t mask_size = 0;
    uint8_t legal_buf[128];

    auto start_time = std::chrono::high_resolution_clock::now();

    uint64_t completed_steps = 0;
    while (completed_steps < BENCHMARK_STEPS) {
        if (ts::Engine::is_terminal(state)) {
            ts::Engine::init_game(state, ts::Prng::next_u64(prng_state));
        }

        ts::Engine::get_legal_action_mask(state, mask_buf, &mask_size);
        if (mask_size == 0) {
            ts::Engine::init_game(state, ts::Prng::next_u64(prng_state));
            continue;
        }

        size_t legal_cnt = 0;
        for (size_t i = 0; i < mask_size; ++i) {
            if (mask_buf[i]) legal_buf[legal_cnt++] = static_cast<uint8_t>(i);
        }

        if (legal_cnt == 0) {
            ts::Engine::init_game(state, ts::Prng::next_u64(prng_state));
            continue;
        }

        uint32_t chosen = ts::Prng::random_index(prng_state, static_cast<uint32_t>(legal_cnt));
        ts::MicroAction action{};
        action.decision_type = state.ctx().decision_type;
        action.primary_id = legal_buf[chosen];

        ASSERT_TRUE(ts::Engine::step(state, action));
        completed_steps++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double duration_sec = std::chrono::duration<double>(end_time - start_time).count();

    double steps_per_sec = static_cast<double>(completed_steps) / duration_sec;
    std::cout << "Completed " << completed_steps << " steps in " << duration_sec << " seconds." << std::endl;
    std::cout << "Throughput: " << static_cast<uint64_t>(steps_per_sec) << " simulation steps/second" << std::endl;

    if (steps_per_sec >= 100000.0) {
        std::cout << "PERFORMANCE TARGET MET (>= 100,000 steps/sec)!" << std::endl;
    } else {
        std::cout << "Warning: performance target not met." << std::endl;
    }

    return 0;
}
