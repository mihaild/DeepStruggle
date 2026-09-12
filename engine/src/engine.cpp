#include "ts/engine.hpp"
#include "ts/state_machine.hpp"
#include "ts/action_mask.hpp"
#include "ts/serialization.hpp"
#include "ts/card_data.hpp"
#include "ts/constants.hpp"
#include "ts/map_data.hpp"

namespace ts {

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
    MicroAction action = ActionMask::decode_flat_action_212(state, action_idx);
    bool ok = StateMachine::step(state, action);
    if (ok && auto_advance) {
        auto_advance_step(state);
    }
    return ok;
}

size_t Engine::auto_advance_step(GameState& state, size_t max_steps) noexcept {
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

        // 2. Specific multi-step event auto-resolutions
        uint8_t rc = state.ctx().resolving_card;

        // 2a. Suez Crisis (#28)
        if (rc == card_ids::SUEZ_CRISIS) {
            uint8_t uk = state.countries[countries::UNITED_KINGDOM].us_influence;
            uint8_t fr = state.countries[countries::FRANCE].us_influence;
            uint8_t isr = state.countries[countries::ISRAEL].us_influence;
            uint8_t used_uk = state.ctx().node_count(countries::UNITED_KINGDOM);
            uint8_t used_fr = state.ctx().node_count(countries::FRANCE);
            uint8_t used_isr = state.ctx().node_count(countries::ISRAEL);
            uint8_t cap_uk = (used_uk < 2) ? static_cast<uint8_t>(2 - used_uk) : 0;
            uint8_t cap_fr = (used_fr < 2) ? static_cast<uint8_t>(2 - used_fr) : 0;
            uint8_t cap_isr = (used_isr < 2) ? static_cast<uint8_t>(2 - used_isr) : 0;
            uint8_t avail_uk = (uk < cap_uk) ? uk : cap_uk;
            uint8_t avail_fr = (fr < cap_fr) ? fr : cap_fr;
            uint8_t avail_isr = (isr < cap_isr) ? isr : cap_isr;
            uint8_t sum_avail = static_cast<uint8_t>(avail_uk + avail_fr + avail_isr);

            if (sum_avail <= state.ctx().remaining_steps) {
                uint8_t chosen_cid = 255;
                if (avail_uk > 0) chosen_cid = countries::UNITED_KINGDOM;
                else if (avail_fr > 0) chosen_cid = countries::FRANCE;
                else if (avail_isr > 0) chosen_cid = countries::ISRAEL;

                if (chosen_cid != 255) {
                    MicroAction ma{DecisionType::POINT_NODE, chosen_cid, 0, 0};
                    if (!StateMachine::step(state, ma)) break;
                    advanced++;
                    continue;
                } else {
                    MicroAction ma{DecisionType::POINT_NODE, 0, 0, 0x80};
                    if (!StateMachine::step(state, ma)) break;
                    advanced++;
                    continue;
                }
            }
        }

        // 2b. Muslim Revolution (#56)
        if (rc == card_ids::MUSLIM_REVOLUTION) {
            uint8_t mr_targets[] = { countries::SUDAN, countries::IRAN, countries::IRAQ,
                                     countries::EGYPT, countries::LIBYA, countries::SAUDI_ARABIA,
                                     countries::SYRIA, countries::JORDAN };
            uint8_t valid_cnt = 0;
            uint8_t first_cid = 255;
            for (uint8_t cid : mr_targets) {
                if (state.countries[cid].us_influence > 0 && !state.ctx().is_visited(cid)) {
                    valid_cnt++;
                    if (first_cid == 255) first_cid = cid;
                }
            }
            if (valid_cnt <= state.ctx().remaining_steps) {
                if (first_cid != 255) {
                    MicroAction ma{DecisionType::POINT_NODE, first_cid, 0, 0};
                    if (!StateMachine::step(state, ma)) break;
                    advanced++;
                    continue;
                } else {
                    MicroAction ma{DecisionType::POINT_NODE, 0, 0, 0x80};
                    if (!StateMachine::step(state, ma)) break;
                    advanced++;
                    continue;
                }
            }
        }

        // 2c. East European Unrest (#29)
        if (rc == card_ids::EAST_EUROPEAN_UNREST) {
            uint8_t valid_cnt = 0;
            uint8_t first_cid = 255;
            for (uint8_t cid = 0; cid < 84; ++cid) {
                if (MapData::get_country(cid).in_eastern_europe &&
                    state.countries[cid].ussr_influence > 0 && !state.ctx().is_visited(cid)) {
                    valid_cnt++;
                    if (first_cid == 255) first_cid = cid;
                }
            }
            if (valid_cnt <= state.ctx().remaining_steps) {
                if (first_cid != 255) {
                    MicroAction ma{DecisionType::POINT_NODE, first_cid, 0, 0};
                    if (!StateMachine::step(state, ma)) break;
                    advanced++;
                    continue;
                } else {
                    MicroAction ma{DecisionType::POINT_NODE, 0, 0, 0x80};
                    if (!StateMachine::step(state, ma)) break;
                    advanced++;
                    continue;
                }
            }
        }

        // 3. Generic single-choice auto-advance
        uint8_t mask_212[212];
        ActionMask::generate_flat_mask_212(state, mask_212);
        uint16_t valid_choices = 0;
        int16_t sole_action = -1;
        for (int i = 0; i < 212; ++i) {
            if (mask_212[i]) {
                valid_choices++;
                sole_action = static_cast<int16_t>(i);
            }
        }

        if (valid_choices == 1 && sole_action >= 0) {
            MicroAction ma = ActionMask::decode_flat_action_212(state, static_cast<uint16_t>(sole_action));
            if (!StateMachine::step(state, ma)) break;
            advanced++;
            continue;
        }

        // Decision requires non-trivial player choice
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
