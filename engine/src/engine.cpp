#include "ts/engine.hpp"
#include "ts/state_machine.hpp"
#include "ts/action_mask.hpp"
#include "ts/serialization.hpp"
#include "ts/card_data.hpp"
#include "ts/constants.hpp"
#include "ts/map_data.hpp"

#include <array>

namespace ts {

namespace {

constexpr int16_t kCloseEvent = 255;

// The country the resolving event's next removal is forced onto, kCloseEvent when nothing is
// left to remove, or -1 when the player has a real choice. Each of these events removes
// influence from countries until its steps run out, so once every remaining target must be
// taken anyway the order is not a decision.
int16_t forced_event_target(const GameState& state) noexcept {
    const auto& ctx = state.ctx();
    const uint8_t rc = ctx.resolving_card;

    if (rc == card_ids::SUEZ_CRISIS) {
        // Up to 2 from each of the UK, France and Israel: a country's capacity is what it holds,
        // up to what is left of its 2.
        constexpr uint8_t kSuez[] = {countries::UNITED_KINGDOM, countries::FRANCE, countries::ISRAEL};
        uint8_t sum_avail = 0;
        int16_t first = -1;
        for (uint8_t cid : kSuez) {
            const uint8_t used = ctx.node_count(cid);
            const uint8_t cap = (used < 2) ? static_cast<uint8_t>(2 - used) : 0;
            const uint8_t held = state.countries[cid].us_influence;
            const uint8_t avail = (held < cap) ? held : cap;
            sum_avail = static_cast<uint8_t>(sum_avail + avail);
            if (avail > 0 && first < 0) first = cid;
        }
        if (sum_avail > ctx.remaining_steps) return -1;
        return (first >= 0) ? first : kCloseEvent;
    }

    // One removal per country, from those still holding the side's influence, the candidates
    // taken in the order given.
    auto once_each = [&](const auto& candidates, auto is_target) -> int16_t {
        uint8_t valid = 0;
        int16_t first = -1;
        for (uint8_t cid : candidates) {
            if (is_target(cid) && !ctx.is_visited(cid)) {
                valid++;
                if (first < 0) first = cid;
            }
        }
        if (valid > ctx.remaining_steps) return -1;
        return (first >= 0) ? first : kCloseEvent;
    };

    if (rc == card_ids::MUSLIM_REVOLUTION) {
        return once_each(MUSLIM_REVOLUTION_TARGETS, [&](uint8_t cid) {
            return state.countries[cid].us_influence > 0;
        });
    }
    if (rc == card_ids::EAST_EUROPEAN_UNREST) {
        static constexpr auto kAll = [] {
            std::array<uint8_t, 84> ids{};
            for (uint8_t i = 0; i < 84; ++i) ids[i] = i;
            return ids;
        }();
        return once_each(kAll, [&](uint8_t cid) {
            return MapData::get_country(cid).in_eastern_europe &&
                   state.countries[cid].ussr_influence > 0;
        });
    }
    return -1;
}

} // namespace

void Engine::init_game(GameState& state, uint64_t seed) noexcept {
    StateMachine::init_new_game(state, seed);
}

void Engine::get_legal_action_mask(const GameState& state, uint8_t* mask_out, size_t* out_size) noexcept {
    ActionMask::generate_mask(state, mask_out, out_size);
}

void Engine::get_flat_action_mask(const GameState& state, uint8_t* mask_212) noexcept {
    ActionMask::generate_flat_mask_212(state, mask_212);
}

bool Engine::step(GameState& state, const MicroAction& action, bool auto_advance) noexcept {
    bool ok = StateMachine::step(state, action);
    if (ok && auto_advance) {
        auto_advance_step(state);
    }
    return ok;
}

bool Engine::step_flat(GameState& state, uint16_t action_idx, bool auto_advance) noexcept {
    // An index outside the flat space is refused, as the merged overload below refuses it --
    // never decoded into some action that happens to be legal here.
    if (action_idx >= FLAT_ACTION_SPACE_SIZE) return false;
    MicroAction action = ActionMask::decode_flat_action_212(state, action_idx);
    bool ok = StateMachine::step(state, action);
    if (ok && auto_advance) {
        auto_advance_step(state);
    }
    return ok;
}

void Engine::get_flat_action_mask(const GameState& state, uint8_t* mask, bool merged_influence) noexcept {
    if (merged_influence) {
        ActionMask::generate_flat_mask_merged(state, mask);
    } else {
        ActionMask::generate_flat_mask_212(state, mask);
    }
}

bool Engine::step_flat(GameState& state, uint16_t action_idx, bool auto_advance,
                       bool merged_influence) noexcept {
    if (!merged_influence || !ActionMask::is_merged_influence_action(state, action_idx)) {
        return step_flat(state, action_idx, auto_advance);
    }
    // The composed actions are not in the E4 mask StateMachine::step validates against (a NODE
    // slot at an op-choice node), or mean something else there (OPS_INFLUENCE), so validate
    // against the merged view first.
    uint8_t mask[FLAT_ACTION_SPACE_SIZE];
    ActionMask::generate_flat_mask_merged(state, mask);
    if (action_idx >= FLAT_ACTION_SPACE_SIZE || !mask[action_idx]) return false;

    const GameState before = state;
    const MicroAction commit = ActionMask::decode_flat_action_212(state, flat_slots::OPS_INFLUENCE);
    if (!StateMachine::step(state, commit)) { state = before; return false; }
    if (is_terminal(state)) {
        // The commit ended the game (the merged mask offers only OPS_INFLUENCE then): the E4
        // action is complete, with nothing left to compose.
        if (action_idx != flat_slots::OPS_INFLUENCE) { state = before; return false; }
        return true;
    }
    const uint16_t second = (action_idx == flat_slots::OPS_INFLUENCE) ? flat_slots::CONFIRM_DONE
                                                                      : action_idx;
    const MicroAction place = ActionMask::decode_flat_action_212(state, second);
    if (!StateMachine::step(state, place)) { state = before; return false; }
    if (auto_advance) auto_advance_step(state);
    return true;
}

void Engine::get_flat_action_mask(const GameState& state, uint8_t* mask, ActionView view) noexcept {
    switch (view) {
        case ActionView::MERGED: ActionMask::generate_flat_mask_merged(state, mask); return;
        case ActionView::MACRO:  ActionMask::generate_flat_mask_macro(state, mask); return;
        case ActionView::E4:     break;
    }
    ActionMask::generate_flat_mask_212(state, mask);
}

bool Engine::step_flat(GameState& state, uint16_t action_idx, bool auto_advance, ActionView view) noexcept {
    if (view == ActionView::MERGED) return step_flat(state, action_idx, auto_advance, true);
    if (view == ActionView::E4 || !ActionMask::is_macro_action(action_idx)) {
        return step_flat(state, action_idx, auto_advance);
    }
    // An until-control slot is not in the base mask StateMachine::step validates against, so it is
    // validated against the macro view first, then applied as the one-point steps that define it.
    uint8_t mask[FLAT_ACTION_SPACE_SIZE];
    ActionMask::generate_flat_mask_macro(state, mask);
    if (!mask[action_idx]) return false;
    const GameState before = state;
    const int placed = ActionMask::apply_macro_expansion(
        state, static_cast<uint8_t>(action_idx - flat_slots::MACRO_NODE));
    if (placed < 2) { state = before; return false; }
    if (auto_advance) auto_advance_step(state);
    return true;
}

size_t Engine::auto_advance_step(GameState& state, size_t max_steps) noexcept {
    return auto_advance_step(state, max_steps, nullptr, nullptr);
}

size_t Engine::auto_advance_step(GameState& state, size_t max_steps, uint8_t* final_mask,
                                 bool* final_mask_valid) noexcept {
    if (final_mask_valid) *final_mask_valid = false;
    size_t advanced = 0;
    while (advanced < max_steps && !is_terminal(state)) {
        // 1. Chance / Die rolls (unattended rolls, e.g. space race, war, coup)
        if (state.ctx().decision_player == Player::NONE &&
            state.ctx().decision_type == DecisionType::ROLL_DIE) {
            MicroAction chance_ma{DecisionType::ROLL_DIE, 0, 0, 0};
            if (!StateMachine::step(state, chance_ma)) break;
            advanced++;
            continue;
        }

        // 2. Multi-step events whose remaining targets must all be taken: no choice among them,
        // so the first is taken (or the event closed when none is left).
        const int16_t forced = forced_event_target(state);
        if (forced >= 0) {
            const MicroAction ma = (forced == kCloseEvent)
                ? MicroAction{DecisionType::POINT_NODE, 0, 0, action_flags::CONFIRM_DONE}
                : MicroAction{DecisionType::POINT_NODE, static_cast<uint8_t>(forced), 0, 0};
            if (!StateMachine::step(state, ma)) break;
            advanced++;
            continue;
        }

        // 3. Generic single-choice auto-advance
        uint8_t local_mask[FLAT_ACTION_SPACE_SIZE];
        uint8_t* mask_212 = final_mask ? final_mask : local_mask;
        ActionMask::generate_flat_mask_212(state, mask_212);
        uint16_t valid_choices = 0;
        int16_t sole_action = -1;
        // The WHOLE space. This scanned `i < 212`, which after the section 4 repack stops
        // three short of the region head at 214..219 -- a decision whose only legal
        // actions live up there would look like it had none and auto-advance would stop.
        for (int i = 0; i < static_cast<int>(FLAT_ACTION_SPACE_SIZE); ++i) {
            if (mask_212[i]) {
                valid_choices++;
                sole_action = static_cast<int16_t>(i);
            }
        }

        if (valid_choices == 1 && sole_action >= 0) {
            MicroAction ma = ActionMask::decode_flat_action_212(state, static_cast<uint16_t>(sole_action));
            // Validated against the mask just generated for this same state.
            if (!StateMachine::step(state, ma, mask_212)) break;
            advanced++;
            continue;
        }

        // Decision requires non-trivial player choice. mask_212 is this decision's mask.
        if (final_mask_valid) *final_mask_valid = true;
        break;
    }
    return advanced;
}

bool Engine::is_terminal(const GameState& state) noexcept {
    return state.current_phase == Phase::GAME_OVER || state.victory_points >= 20 || state.victory_points <= -20;
}

float Engine::get_terminal_utility(const GameState& state) noexcept {
    if (state.victory_points >= 20) return 1.0f;
    if (state.victory_points <= -20) return -1.0f;
    if (state.victory_points > 0) return 1.0f;
    if (state.victory_points < 0) return -1.0f;
    return 0.0f;
}

bool Engine::has_held_scoring_card(const GameState& state, Player p) noexcept {
    if (p == Player::NONE) return false;
    for (uint8_t i = 1; i <= 110; ++i) {
        if (CardData::is_scoring_card(i) && in_hand_of(state.card_locations[i], p)) {
            return true;
        }
    }
    return false;
}

bool Engine::is_held_scoring_game_over(const GameState& state) noexcept {
    if (state.current_phase != Phase::GAME_OVER) return false;
    if (state.defcon <= 1) return false;
    uint8_t max_ar = (state.turn <= 3) ? 6 : 7;
    if (state.action_round <= max_ar) return false;
    return has_held_scoring_card(state, Player::US) || has_held_scoring_card(state, Player::USSR);
}

bool Engine::is_held_scoring_loss(const GameState& state, Player p) noexcept {
    if (!is_held_scoring_game_over(state)) return false;
    return has_held_scoring_card(state, p);
}

void Engine::serialize(const GameState& state, uint8_t* out_bytes, size_t max_bytes) noexcept {
    Serializer::serialize_binary(state, out_bytes, max_bytes);
}

void Engine::deserialize(GameState& state, const uint8_t* in_bytes, size_t in_size) noexcept {
    Serializer::deserialize_binary(state, in_bytes, in_size);
}

std::string Engine::to_json(const GameState& state) {
    return Serializer::to_json(state);
}

} // namespace ts
