// One digest of whole games, compiled into both the Python module and the WebAssembly build, so
// "the browser engine plays the same game" is a comparison of two numbers.
//
// `games` games from seeds 1..games, choices drawn from a fixed engine PRNG over the flat legal
// mask, chance nodes rolled from the state's own RNG. At every position it hashes the full save
// (every field the rules read), both observations (float bits) and the legal mask. Any
// difference between the two compilations -- a rule, an observation float, an uninitialised
// byte that leaks into a decision -- changes the number.
#pragma once

#include <cstdint>
#include <string>

#include "state_json.hpp"
#include "ts/action_mask.hpp"
#include "ts/engine.hpp"
#include "ts/game_state.hpp"
#include "ts/observation.hpp"
#include "ts/prng.hpp"

namespace ts::selftest {

inline uint64_t fnv1a(uint64_t h, const void* data, size_t n) {
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

inline void drain_chance(GameState& s) {
    while (!Engine::is_terminal(s) && s.ctx().decision_player == Player::NONE &&
           s.ctx().decision_type == DecisionType::ROLL_DIE) {
        if (!Engine::step(s, MicroAction{DecisionType::ROLL_DIE, 0, 0, 0})) return;
    }
}

inline uint32_t digest(int games, uint32_t* steps_out) {
    uint64_t h = 1469598103934665603ULL;
    uint64_t pick = 0x5EED5EEDULL;
    uint32_t steps = 0;
    ObservationBufferV23 ob;
    uint8_t mask[FLAT_ACTION_SPACE_SIZE];
    for (int gi = 1; gi <= games; ++gi) {
        GameState s{};
        Engine::init_game(s, static_cast<uint64_t>(gi));
        drain_chance(s);
        while (!Engine::is_terminal(s) && steps < 50'000'000) {
            const std::string save = state_json::save(s).dump();
            h = fnv1a(h, save.data(), save.size());
            for (Player p : {Player::US, Player::USSR}) {
                Observation::extract(s, p, &ob);
                h = fnv1a(h, &ob, OBS_SIZE_V23 * sizeof(float));
            }
            Engine::get_flat_action_mask(s, mask, false);
            h = fnv1a(h, mask, sizeof(mask));
            uint16_t legal[FLAT_ACTION_SPACE_SIZE];
            uint32_t n = 0;
            for (uint16_t i = 0; i < FLAT_ACTION_SPACE_SIZE; ++i) {
                if (mask[i]) legal[n++] = i;
            }
            if (n == 0) break;
            if (!Engine::step_flat(s, legal[Prng::random_index(pick, n)])) break;
            drain_chance(s);
            ++steps;
        }
        const std::string end = state_json::save(s).dump();
        h = fnv1a(h, end.data(), end.size());
    }
    if (steps_out) *steps_out = steps;
    return static_cast<uint32_t>(h ^ (h >> 32));
}

}  // namespace ts::selftest
