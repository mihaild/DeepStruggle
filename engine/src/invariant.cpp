#include "ts/invariant.hpp"

#include <cstdint>
#include <cstring>

namespace ts {

namespace {

// FNV-1a over the raw state, only to tell "the same decision again" from "a new one".
uint64_t state_fingerprint(const GameState& state) noexcept {
    const auto* bytes = reinterpret_cast<const unsigned char*>(&state);
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < sizeof(GameState); ++i) {
        h ^= bytes[i];
        h *= 1099511628211ULL;
    }
    return h;
}

} // namespace

void report_anomaly(const char* what, const GameState& state) noexcept {
    // Per thread: training runs several environments at once, and a shared counter would need
    // a lock on a path that must never become a bottleneck.
    static thread_local uint64_t last_fingerprint = 0;
    static thread_local uint64_t seen = 0;

    const uint64_t fingerprint = state_fingerprint(state);
    if (fingerprint == last_fingerprint) return;   // same decision, mask asked again
    last_fingerprint = fingerprint;
    ++seen;

    const auto& ctx = state.ctx();
    std::fprintf(stderr,
                 "\n=== ts engine anomaly #%llu: %s ===\n"
                 "turn %d AR %d phase %d phasing %d | decision %d player %d "
                 "resolving_card %d pending_op_card %d remaining_steps %d "
                 "allow_early_stop %d stack_depth %d\n",
                 static_cast<unsigned long long>(seen), what,
                 static_cast<int>(state.turn), static_cast<int>(state.action_round),
                 static_cast<int>(state.current_phase),
                 static_cast<int>(state.phasing_player),
                 static_cast<int>(ctx.decision_type),
                 static_cast<int>(ctx.decision_player),
                 static_cast<int>(ctx.resolving_card),
                 static_cast<int>(ctx.pending_op_card),
                 static_cast<int>(ctx.remaining_steps),
                 static_cast<int>(ctx.allow_early_stop),
                 static_cast<int>(state.ctx_stack_depth));

    // The position itself, byte for byte, so it can be replayed rather than guessed at.
    std::fprintf(stderr, "state=");
    const auto* bytes = reinterpret_cast<const unsigned char*>(&state);
    for (size_t i = 0; i < sizeof(GameState); ++i) {
        std::fprintf(stderr, "%02x", bytes[i]);
    }
    std::fprintf(stderr, "\n=== end anomaly #%llu ===\n",
                 static_cast<unsigned long long>(seen));
    std::fflush(stderr);
}

} // namespace ts
