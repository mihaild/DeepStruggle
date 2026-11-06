#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "game_state.hpp"
#include "action_mask.hpp"
#include "micro_action.hpp"

namespace ts {

// A GameState that keeps its own flat legal mask, for the containers that own their states and ask
// for that mask over and over: the C++ search tree and the batch runner.
//
// Why: the mask was built three times for every position a search reached -- StateMachine::step
// regenerates it to validate the action it is given, auto-advance generates it to look for a
// forced move, and the search generates it again for the network -- 3.27 times a leaf, ~20% of
// the leaf's CPU. Kept here, it is generated once per position and every later use reads it.
//
// The guarantee is structural. The state is private: reading it is `state()`, a const reference,
// and the ONLY way to a mutable one is `modify(f)`, which marks the cache invalid before f runs
// and again after. step / step_flat / auto_advance go through modify too. So no code outside this
// class can change the state without invalidating the cache -- a write that skipped it would not
// compile. The engine's own rules code (StateMachine, the event handlers) keeps taking a plain
// GameState&, and only ever runs inside one of those calls.
//
// The one write that does not invalidate is `set_rng`: the mask does not depend on the RNG state
// (engine test CachedState.MaskIgnoresRngAndLastRolls), and it is what lets a search child be
// validated against its parent's mask.
//
// Checked, not only argued: with set_mask_cache_checks(true) -- on in the engine tests, the
// fuzzers and the whole Python test suite -- every read of a cached mask regenerates it and aborts
// if the two differ. Python's GameState is the plain struct and has no cache, so nothing Python
// writes can leave one stale.
class CachedState {
public:
    CachedState() noexcept = default;
    explicit CachedState(const GameState& s) noexcept : state_(s) {}

    const GameState& state() const noexcept { return state_; }

    // The only mutable access. `f(GameState&)` returns nothing, so the reference does not outlive
    // the call.
    template <class F>
    void modify(F&& f) {
        valid_ = false;
        f(state_);
        valid_ = false;
    }

    // The chance seed. The mask does not read it, so the cache stays valid.
    void set_rng(uint64_t seed) noexcept { state_.rng_state = seed; }

    // The flat legal mask (FLAT_ACTION_SPACE_SIZE bytes) of the current state, generated on first
    // use after a change and kept until the next one. Valid until this object is next modified.
    const uint8_t* legal_mask() noexcept;
    bool mask_cached() const noexcept { return valid_; }

    // StateMachine::step, validated against the cached mask.
    [[nodiscard]] bool step(const MicroAction& action) noexcept;
    // Engine::step_flat without auto-advance. In the merged-influence view a composed action goes
    // through Engine::step_flat's own path (its mask is another view); everything else is the E4
    // step, validated against the cached mask.
    [[nodiscard]] bool step_flat(uint16_t action_idx, bool merged_influence = false) noexcept;
    // The same in any view; MACRO applies an until-control slot as Engine::step_flat does.
    [[nodiscard]] bool step_flat(uint16_t action_idx, ActionView view) noexcept;
    // Engine::auto_advance_step, leaving the mask of the decision it stops at cached.
    size_t auto_advance(size_t max_steps = 128) noexcept;

private:
    void verify() const noexcept;

    GameState state_{};
    std::array<uint8_t, FLAT_ACTION_SPACE_SIZE> mask_{};
    bool valid_ = false;
};

static_assert(std::is_trivially_copyable_v<CachedState>,
              "CachedState is copied by value into search nodes and runner slots");

// Every read of a cached mask regenerated and compared, aborting on a difference. Off by default
// (it costs a mask generation per read); on in tests and fuzzers.
void set_mask_cache_checks(bool on) noexcept;
bool mask_cache_checks() noexcept;

} // namespace ts
