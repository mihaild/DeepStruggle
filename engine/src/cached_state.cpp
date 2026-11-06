#include "ts/cached_state.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "ts/action_mask.hpp"
#include "ts/engine.hpp"
#include "ts/state_machine.hpp"

namespace ts {

namespace {
std::atomic<bool> g_mask_cache_checks{false};
}  // namespace

void set_mask_cache_checks(bool on) noexcept { g_mask_cache_checks.store(on, std::memory_order_relaxed); }

bool mask_cache_checks() noexcept { return g_mask_cache_checks.load(std::memory_order_relaxed); }

const uint8_t* CachedState::legal_mask() noexcept {
    if (!valid_) {
        ActionMask::generate_flat_mask_212(state_, mask_.data());
        valid_ = true;
    } else if (mask_cache_checks()) {
        verify();
    }
    return mask_.data();
}

void CachedState::verify() const noexcept {
    uint8_t fresh[FLAT_ACTION_SPACE_SIZE];
    ActionMask::generate_flat_mask_212(state_, fresh);
    if (std::memcmp(fresh, mask_.data(), FLAT_ACTION_SPACE_SIZE) == 0) return;
    // A stale cache is a wrong legal set: stop, loudly, with what is needed to reproduce it.
    std::fprintf(stderr, "CachedState: the cached legal mask differs from the state's own mask\n");
    for (size_t a = 0; a < FLAT_ACTION_SPACE_SIZE; ++a)
        if (fresh[a] != mask_[a])
            std::fprintf(stderr, "  action %zu: cached %d, actual %d\n", a, mask_[a], fresh[a]);
    std::fprintf(stderr, "  state=");
    const auto* bytes = reinterpret_cast<const unsigned char*>(&state_);
    for (size_t i = 0; i < sizeof(GameState); ++i) std::fprintf(stderr, "%02x", bytes[i]);
    std::fprintf(stderr, "\n");
    std::abort();
}

bool CachedState::step(const MicroAction& action) noexcept {
    // The mask is read before modify() invalidates it and is not written during the step, so the
    // pointer is good for the validation inside.
    const uint8_t* legal = legal_mask();
    bool ok = false;
    modify([&](GameState& g) { ok = StateMachine::step(g, action, legal); });
    return ok;
}

bool CachedState::step_flat(uint16_t action_idx, bool merged_influence) noexcept {
    if (merged_influence && ActionMask::is_merged_influence_action(state_, action_idx)) {
        bool ok = false;
        modify([&](GameState& g) { ok = Engine::step_flat(g, action_idx, false, true); });
        return ok;
    }
    // Engine::step_flat's E4 path: decode, then StateMachine::step.
    return step(ActionMask::decode_flat_action_212(state_, action_idx));
}

bool CachedState::step_flat(uint16_t action_idx, ActionView view) noexcept {
    if (view == ActionView::MERGED) return step_flat(action_idx, true);
    if (view == ActionView::MACRO && ActionMask::is_macro_action(action_idx)) {
        bool ok = false;
        modify([&](GameState& g) { ok = Engine::step_flat(g, action_idx, false, ActionView::MACRO); });
        return ok;
    }
    return step_flat(action_idx, false);
}

size_t CachedState::auto_advance(size_t max_steps) noexcept {
    size_t advanced = 0;
    bool final_mask = false;
    modify([&](GameState& g) {
        advanced = Engine::auto_advance_step(g, max_steps, mask_.data(), &final_mask);
    });
    // The mask auto-advance generated at the decision it stopped at, with nothing changed since.
    valid_ = final_mask;
    return advanced;
}

} // namespace ts
