#include "ts/space_race.hpp"
#include "ts/card_data.hpp"
#include "ts/prng.hpp"
#include <algorithm>

namespace ts {

namespace {

constexpr std::array<SpaceBoxInfo, 9> SPACE_BOXES = {{
    {0, 0, 0, 0}, // Box 0 (Start)
    {2, 3, 2, 1}, // Box 1: Earth Satellite (2 Ops, 1-3, 2/1 VP)
    {2, 4, 0, 0}, // Box 2: Animal in Space (2 Ops, 1-4, 0/0 VP, 2 attempts/turn)
    {2, 4, 2, 0}, // Box 3: Man in Orbit (2 Ops, 1-4, 2/0 VP)
    {2, 3, 0, 0}, // Box 4: Man in Space (2 Ops, 1-3, 0/0 VP, Opponent reveals headline first)
    {3, 4, 3, 1}, // Box 5: Lunar Probe (3 Ops, 1-4, 3/1 VP)
    {3, 3, 0, 0}, // Box 6: Space Walk (3 Ops, 1-3, 0/0 VP, Discard held card)
    {3, 4, 4, 2}, // Box 7: Space Station (3 Ops, 1-4, 4/2 VP)
    {4, 2, 2, 0}  // Box 8: Eagle/Bear Landed (4 Ops, 1-2, 2/0 VP, 8 ARs)
}};

} // namespace

const SpaceBoxInfo& SpaceRace::get_box_info(uint8_t box) noexcept {
    return (box <= 8) ? SPACE_BOXES[box] : SPACE_BOXES[8];
}

bool SpaceRace::has_animal_in_space(const GameState& state, Player p) noexcept {
    if (p == Player::US) {
        return state.us_space_track >= 2 && state.ussr_space_track < 2;
    } else if (p == Player::USSR) {
        return state.ussr_space_track >= 2 && state.us_space_track < 2;
    }
    return false;
}

bool SpaceRace::has_man_in_space(const GameState& state, Player p) noexcept {
    if (p == Player::US) {
        return state.us_space_track >= 4 && state.ussr_space_track < 4;
    } else if (p == Player::USSR) {
        return state.ussr_space_track >= 4 && state.us_space_track < 4;
    }
    return false;
}

bool SpaceRace::has_space_walk(const GameState& state, Player p) noexcept {
    if (p == Player::US) {
        return state.us_space_track >= 6 && state.ussr_space_track < 6;
    } else if (p == Player::USSR) {
        return state.ussr_space_track >= 6 && state.us_space_track < 6;
    }
    return false;
}

bool SpaceRace::has_space_station_ar8(const GameState& state, Player p) noexcept {
    if (p == Player::US) {
        return state.us_space_track >= 8 && state.ussr_space_track < 8;
    } else if (p == Player::USSR) {
        return state.ussr_space_track >= 8 && state.us_space_track < 8;
    }
    return false;
}

bool SpaceRace::can_attempt_space(const GameState& state, Player p, uint8_t card_id) noexcept {
    if (p == Player::NONE || card_id < 1 || card_id > 110) return false;
    uint8_t cur_track = (p == Player::US) ? state.us_space_track : state.ussr_space_track;
    if (cur_track >= 8) return false; // Already reached max box

    uint8_t turns_used = (p == Player::US) ? state.us_space_turns_used : state.ussr_space_turns_used;
    uint8_t max_attempts = has_animal_in_space(state, p) ? 2 : 1;
    if (turns_used >= max_attempts) return false;

    const auto& next_box = SPACE_BOXES[cur_track + 1];
    const auto& c_info = CardData::get_card(card_id);
    return c_info.ops >= next_box.min_ops;
}

bool SpaceRace::attempt_space(GameState& state, Player p, uint8_t card_id, uint8_t forced_roll) noexcept {
    if (!can_attempt_space(state, p, card_id)) return false;

    uint8_t& cur_track = (p == Player::US) ? state.us_space_track : state.ussr_space_track;
    uint8_t opp_track = (p == Player::US) ? state.ussr_space_track : state.us_space_track;
    uint8_t& turns_used = (p == Player::US) ? state.us_space_turns_used : state.ussr_space_turns_used;

    turns_used++;

    // Space race card is moved to discard pile (event never occurs)
    state.card_locations[card_id] = CardLocation::DISCARD_PILE;

    uint8_t next_box_num = cur_track + 1;
    const auto& next_box = SPACE_BOXES[next_box_num];

    uint8_t roll = (forced_roll > 0) ? forced_roll : Prng::roll_d6(state.rng_state);

    if (roll <= next_box.max_roll) {
        // Advance track
        cur_track = next_box_num;

        // Check VP award: 1st if cur_track > opp_track, 2nd if cur_track <= opp_track
        uint8_t vp_award = (cur_track > opp_track) ? next_box.vp_first : next_box.vp_second;
        if (vp_award > 0) {
            int32_t vp_delta = (p == Player::US) ? vp_award : -static_cast<int32_t>(vp_award);
            int32_t new_vp = static_cast<int32_t>(state.victory_points) + vp_delta;
            new_vp = std::clamp(new_vp, -20, 20);
            state.victory_points = static_cast<int8_t>(new_vp);
            if (state.victory_points >= 20 || state.victory_points <= -20) {
                state.current_phase = Phase::GAME_OVER;
            }
        }
        return true;
    }

    return false;
}

} // namespace ts
