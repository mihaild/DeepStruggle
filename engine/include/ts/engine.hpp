#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include "types.hpp"
#include "game_state.hpp"
#include "micro_action.hpp"
#include "action_mask.hpp"

namespace ts {

class Engine {
public:
    // Initialize standard game state (shuffles Early War deck, deals hands, sets USSR China Card face up)
    static void init_game(GameState& state, uint64_t seed) noexcept;

    // Writes binary mask of valid actions into mask_out buffer.
    // Max buffer required: 112 bytes for SELECT_CARD, 84 bytes for POINT_NODE.
    static void get_legal_action_mask(const GameState& state, uint8_t* mask_out, size_t* out_size) noexcept;

    // Writes 212-element flat legal action mask
    static void get_flat_action_mask(const GameState& state, uint8_t* mask_212) noexcept;

    // P23 / E4.1: the same, in the merged-influence view when `merged_influence` is set. See
    // ActionMask::generate_flat_mask_merged. `false` is exactly the E4 mask.
    static void get_flat_action_mask(const GameState& state, uint8_t* mask, bool merged_influence) noexcept;

    // Advances game state by 1 validated micro-action. Returns true on success.
    [[nodiscard]] static bool step(GameState& state, const MicroAction& action, bool auto_advance = false) noexcept;

    // Advances game state by 1 flat action index [0..211].
    [[nodiscard]] static bool step_flat(GameState& state, uint16_t action_idx, bool auto_advance = false) noexcept;

    // P23 / E4.1: in the merged-influence view a composed action is applied as the two E4 steps
    // that define it, atomically -- on refusal of either, the state is restored and false returned.
    // `merged_influence == false` is exactly step_flat above.
    [[nodiscard]] static bool step_flat(GameState& state, uint16_t action_idx, bool auto_advance,
                                        bool merged_influence) noexcept;

    // Automatically advances deterministic decisions (chance rolls, single valid action,
    // and deterministic event targets) until a choice requiring player discretion is reached.
    static size_t auto_advance_step(GameState& state, size_t max_steps = 128) noexcept;

    // Fast terminal state evaluation
    static bool is_terminal(const GameState& state) noexcept;
    static float get_terminal_utility(const GameState& state) noexcept; // +1.0 (US), -1.0 (USSR), 0.0 (Draw)

    // Rule 4.4 Held Scoring Card Detection (Turn End only)
    static bool has_held_scoring_card(const GameState& state, Player p) noexcept;
    static bool is_held_scoring_game_over(const GameState& state) noexcept;
    static bool is_held_scoring_loss(const GameState& state, Player p) noexcept;

    // Serialization & Inspection
    static void serialize(const GameState& state, uint8_t* out_bytes, size_t max_bytes) noexcept;
    static void deserialize(GameState& state, const uint8_t* in_bytes, size_t in_size) noexcept;
    static std::string to_json(const GameState& state);
};

} // namespace ts
