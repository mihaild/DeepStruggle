#include "ts/space_race.hpp"
#include "ts/card_data.hpp"
#include "ts/ops.hpp"
#include "ts/prng.hpp"
#include <algorithm>

namespace ts {

namespace {

// {min_ops, max_roll, vp_first, vp_second}. The roll thresholds for boxes 3 to 7 were the
// wrong way round, alternating 4/3/4/3/4 where the game alternates 3/4/3/4/3, which the box
// names alongside them show the cause of: boxes 3 and 4 were labelled in the wrong order, and
// so on up the track. Every one of the 1,190 space attempts in the 287 downloaded human games
// agrees with the values below and disagrees with the old ones. The Ops requirements, the VP
// awards and the box numbers each benefit hangs off were all already right.
constexpr std::array<SpaceBoxInfo, 9> SPACE_BOXES = {{
    {0, 0, 0, 0}, // Box 0 (Start)
    {2, 3, 2, 1}, // Box 1: Earth Satellite (2 Ops, 1-3, 2/1 VP)
    {2, 4, 0, 0}, // Box 2: Animal in Space (2 Ops, 1-4, 2 attempts/turn)
    {2, 3, 2, 0}, // Box 3: Man in Space (2 Ops, 1-3, 2/0 VP)
    {2, 4, 0, 0}, // Box 4: Man in Earth Orbit (2 Ops, 1-4, opponent headlines first)
    {3, 3, 3, 1}, // Box 5: Lunar Orbit (3 Ops, 1-3, 3/1 VP)
    {3, 4, 0, 0}, // Box 6: Eagle/Bear Has Landed (3 Ops, 1-4, discard a held card)
    {3, 3, 4, 2}, // Box 7: Space Shuttle (3 Ops, 1-3, 4/2 VP)
    {4, 2, 2, 0}  // Box 8: Space Station (4 Ops, 1-2, 2/0 VP, 8 action rounds)
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

    uint8_t turns_used = state.get_space_turns_used(p);
    uint8_t max_attempts = has_animal_in_space(state, p) ? 2 : 1;
    if (turns_used >= max_attempts) return false;

    // Effective Ops, not printed: the Ops modifiers that decide what a card can buy on the
    // board decide what it can buy on the space track too. At turn 4 AR3 of ts-replayer game
    // 113 the USSR raced with OAS Founded -- 1 printed Op, but Brezhnev Doctrine was active,
    // making it the 2 that box 3 requires. No region applies: the space track is not on the
    // map, so the China Card's Asia bonus and Vietnam Revolts' Southeast Asia bonus, which pay
    // only for Operations in a region, pay nothing here.
    const auto& next_box = SPACE_BOXES[cur_track + 1];
    return Operations::get_effective_ops(state, card_id, p) >= next_box.min_ops;
}

bool SpaceRace::attempt_space(GameState& state, Player p, uint8_t card_id, uint8_t forced_roll) noexcept {
    if (!can_attempt_space(state, p, card_id)) return false;

    uint8_t& cur_track = (p == Player::US) ? state.us_space_track : state.ussr_space_track;
    uint8_t opp_track = (p == Player::US) ? state.ussr_space_track : state.us_space_track;

    state.record_space_attempt(p);

    // Space race card is moved to discard pile (event never occurs). The China Card is never
    // discarded: whatever it was played for, it passes to the opponent face down, unplayable
    // until the turn after. At turn 10 AR4 of ts-replayer game 247 the US races with it to
    // box 5 -- a poor play, and a legal one.
    if (card_id == card_ids::THE_CHINA_CARD) {
        state.china_card_holder = get_opponent(p);
        state.china_card_playable = 0;
        if (p == Player::US) state.clear_flag(effect_bits::FORMOSAN_RESOLUTION_ACTIVE);
    } else {
        state.card_locations[card_id] = CardLocation::DISCARD_PILE;
    }

    uint8_t next_box_num = cur_track + 1;
    const auto& next_box = SPACE_BOXES[next_box_num];

    uint8_t roll = (forced_roll > 0) ? forced_roll : Prng::roll_d6(state.rng_state);
    state.last_die_roll = roll;

    bool success = (roll <= next_box.max_roll);
    state.last_roll = DieRollRecord{
        .type = RollType::SPACE_RACE,
        .roller = p,
        .card_id = card_id,
        .country_id = next_box_num,
        .roll1 = roll,
        .mod1 = static_cast<int8_t>(next_box.max_roll),
        .roll2 = 0,
        .mod2 = 0,
        .success = success,
        .net_delta = static_cast<int8_t>(success ? 1 : 0)
    };

    if (success) {
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
