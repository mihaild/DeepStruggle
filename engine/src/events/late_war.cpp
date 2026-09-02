#include "ts/card_handlers.hpp"
#include "ts/war_events.hpp"
#include "ts/card_data.hpp"
#include "ts/map_data.hpp"
#include "ts/scoring.hpp"
#include "ts/space_race.hpp"
#include "ts/ops.hpp"
#include "ts/prng.hpp"
#include "ts/defcon.hpp"
#include <algorithm>

namespace ts {

namespace late_war {

bool trigger_iranian_hostage_crisis(GameState& state, Player p) noexcept {
    state.countries[countries::IRAN].us_influence = 0;
    state.countries[countries::IRAN].add_influence(Player::USSR, 2);
    state.set_flag(effect_bits::IRANIAN_HOSTAGE_CRISIS_PLAY);
    return true;
}

bool trigger_iron_lady(GameState& state, Player p) noexcept {
    state.countries[countries::ARGENTINA].add_influence(Player::USSR, 1);
    state.countries[countries::UNITED_KINGDOM].ussr_influence = 0;
    state.victory_points = static_cast<int8_t>(std::min(20, state.victory_points + 1));
    if (state.victory_points >= 20) state.current_phase = Phase::GAME_OVER;
    state.set_flag(effect_bits::IRON_LADY_PLAYED);
    return true;
}

bool trigger_reagan_bombs_libya(GameState& state, Player p) noexcept {
    uint8_t ussr_inf = state.countries[countries::LIBYA].ussr_influence;
    uint8_t pts = ussr_inf / 2;
    if (pts > 0) {
        state.victory_points = static_cast<int8_t>(std::min(20, state.victory_points + pts));
        if (state.victory_points >= 20) state.current_phase = Phase::GAME_OVER;
    }
    return true;
}

bool trigger_star_wars(GameState& state, Player p) noexcept {
    if (state.us_space_track <= state.ussr_space_track) return true;

    // Check if there are non-scoring cards in discard pile
    uint8_t discard_cards[111];
    uint8_t cnt = 0;
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == CardLocation::DISCARD_PILE && !CardData::is_scoring_card(i)) {
            discard_cards[cnt++] = i;
        }
    }
    if (cnt == 0) return true;

    state.ctx().decision_player = Player::US;
    state.ctx().decision_type = DecisionType::SELECT_CARD;
    state.ctx().remaining_steps = 1;
    state.ctx().resolving_card = card_ids::STAR_WARS;
    return false;
}

bool trigger_north_sea_oil(GameState& state, Player p) noexcept {
    state.set_flag(effect_bits::NORTH_SEA_OIL_PLAYED);
    state.set_flag(effect_bits::NORTH_SEA_OIL_ACTIVE);
    return true;
}

bool trigger_the_reformer(GameState& state, Player p) noexcept {
    state.set_flag(effect_bits::THE_REFORMER_PLAYED);
    uint8_t total_inf = (state.victory_points < 0) ? 6 : 4; // USSR ahead in VP

    state.ctx().decision_player = Player::USSR;
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().remaining_steps = total_inf;
    state.ctx().max_per_country = 2;
    state.ctx().allow_early_stop = 1;
    state.ctx().resolving_card = card_ids::THE_REFORMER;
    return false;
}

bool trigger_marine_barracks_bombing(GameState& state, Player p) noexcept {
    state.countries[countries::LEBANON].us_influence = 0;
    state.ctx().decision_player = Player::USSR;
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().remaining_steps = 2;
    state.ctx().max_per_country = 2;
    state.ctx().allow_early_stop = 1;
    state.ctx().resolving_card = card_ids::MARINE_BARRACKS_BOMBING;
    return false;
}

bool trigger_soviets_shoot_down_kal(GameState& state, Player p) noexcept {
    if (state.defcon > 1) {
        state.defcon--;
        if (state.defcon == 2) state.defcon_dropped_to_2_in_ar = 1;
    }
    if (state.defcon == 1) {
        resolve_defcon_one_loss(state, p);
        return true;
    }
    state.victory_points = static_cast<int8_t>(std::min(20, state.victory_points + 2));
    if (state.victory_points >= 20) {
        state.current_phase = Phase::GAME_OVER;
        return true;
    }

    if (Scoring::is_controlled_by(state, countries::SOUTH_KOREA, Player::US)) {
        state.ctx().decision_player = Player::US;
        state.ctx().pending_op_card = card_ids::SOVIETS_SHOOT_DOWN_KAL_007;
        state.ctx().pending_ops_value = Operations::grant_ops(state, 4, Player::US);
        // These Ops are the event's free action, whose restrictions differ from those on the
        // card played for Ops -- the action mask reads this to tell the two apart.
        state.ctx().event_granted_ops = 1;
        state.ctx().decision_type = DecisionType::SELECT_OP_MODE; // Influence or Realign
        state.ctx().resolving_card = 0;
        return false;
    }
    return true;
}

bool trigger_glasnost(GameState& state, Player p) noexcept {
    state.defcon = static_cast<uint8_t>(std::min(5, static_cast<int>(state.defcon) + 1));
    state.victory_points = static_cast<int8_t>(std::max(-20, state.victory_points - 2));
    if (state.victory_points <= -20) {
        state.current_phase = Phase::GAME_OVER;
        return true;
    }

    if (state.has_flag(effect_bits::THE_REFORMER_PLAYED)) {
        state.ctx().decision_player = Player::USSR;
        state.ctx().pending_op_card = card_ids::GLASNOST;
        state.ctx().pending_ops_value = Operations::grant_ops(state, 4, Player::USSR);
        state.ctx().event_granted_ops = 1;
        state.ctx().decision_type = DecisionType::SELECT_OP_MODE;
        state.ctx().resolving_card = 0;
        return false;
    }
    return true;
}

bool trigger_ortega_elected(GameState& state, Player p) noexcept {
    state.countries[countries::NICARAGUA].us_influence = 0;
    // USSR performs free coup in country adjacent to Nicaragua using 2 Ops
    state.ctx().decision_player = Player::USSR;
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().remaining_steps = 1;
    state.ctx().allow_early_stop = 1;
    state.ctx().resolving_card = card_ids::ORTEGA_ELECTED_IN_NICARAGUA;
    return false;
}

bool trigger_terrorism(GameState& state, Player p) noexcept {
    Player opp = get_opponent(p);
    uint8_t discard_count = (opp == Player::US && state.has_flag(effect_bits::IRANIAN_HOSTAGE_CRISIS_PLAY)) ? 2 : 1;

    CardLocation opp_hand = (opp == Player::US) ? CardLocation::HAND_US : CardLocation::HAND_USSR;

    for (uint8_t d = 0; d < discard_count; ++d) {
        uint8_t cards[111];
        uint8_t cnt = 0;
        for (uint8_t i = 1; i <= 110; ++i) {
            if (state.card_locations[i] == opp_hand) {
                cards[cnt++] = i;
            }
        }
        if (cnt == 0) break;
        uint32_t chosen = Prng::random_index(state.rng_state, cnt);
        state.card_locations[cards[chosen]] = CardLocation::DISCARD_PILE;
    }
    return true;
}

bool trigger_iran_contra(GameState& state, Player p) noexcept {
    state.set_flag(effect_bits::IRAN_CONTRA_ACTIVE);
    return true;
}

bool trigger_chernobyl(GameState& state, Player p) noexcept {
    // US designates region (0..5) via CHOOSE_BRANCH
    state.ctx().decision_player = Player::US;
    state.ctx().decision_type = DecisionType::CHOOSE_BRANCH;
    state.ctx().resolving_card = card_ids::CHERNOBYL;
    return false;
}

bool trigger_latin_debt_crisis(GameState& state, Player p) noexcept {
    // Check if US has 3+ Ops card
    bool has_3ops = false;
    for (uint8_t i = 1; i <= 110; ++i) {
        // Effective Ops, as the discard itself is judged.
        if (state.card_locations[i] == CardLocation::HAND_US &&
            Operations::get_effective_ops(state, i, Player::US) >= 3) {
            has_3ops = true;
            break;
        }
    }
    if (!has_3ops) {
        // USSR doubles USSR influence in 2 countries in South America
        state.ctx().decision_player = Player::USSR;
        state.ctx().decision_type = DecisionType::POINT_NODE;
        state.ctx().remaining_steps = 2;
        state.ctx().max_per_country = 1;
        state.ctx().allow_early_stop = 1;
        state.ctx().resolving_card = card_ids::LATIN_AMERICAN_DEBT_CRISIS;
        return false;
    } else {
        // US decides whether to discard 3+ Ops card or allow USSR to double
        state.ctx().decision_player = Player::US;
        state.ctx().decision_type = DecisionType::SELECT_CARD;
        state.ctx().remaining_steps = 1;
        state.ctx().allow_early_stop = 1; // Confirm done means US chooses not to discard and allows USSR to double
        state.ctx().resolving_card = card_ids::LATIN_AMERICAN_DEBT_CRISIS;
        return false;
    }
}

bool trigger_tear_down_this_wall(GameState& state, Player p) noexcept {
    state.countries[countries::EAST_GERMANY].add_influence(Player::US, 3);
    state.set_flag(effect_bits::TEAR_DOWN_THIS_WALL_PLAYED);
    state.clear_flag(effect_bits::WILLY_BRANDT_PLAYED);
    state.clear_flag(effect_bits::NATO_CANCELED_WEST_GERMANY);

    // US performs free Coup or Realignment in Europe with 3 Ops
    state.ctx().decision_player = Player::US;
    state.ctx().pending_op_card = card_ids::TEAR_DOWN_THIS_WALL;
    state.ctx().pending_ops_value = Operations::grant_ops(state, 3, Player::US);
    state.ctx().decision_type = DecisionType::SELECT_OP_MODE;
    state.ctx().resolving_card = 0;
    return false;
}

bool trigger_an_evil_empire(GameState& state, Player p) noexcept {
    state.victory_points = static_cast<int8_t>(std::min(20, state.victory_points + 1));
    if (state.victory_points >= 20) state.current_phase = Phase::GAME_OVER;
    state.set_flag(effect_bits::EVIL_EMPIRE_PLAYED);
    state.clear_flag(effect_bits::FLOWER_POWER_ACTIVE);
    return true;
}

bool trigger_aldrich_ames(GameState& state, Player p) noexcept {
    state.set_flag(effect_bits::ALDRICH_AMES_ACTIVE);
    // USSR chooses card from US hand to discard
    state.ctx().decision_player = Player::USSR;
    state.ctx().decision_type = DecisionType::SELECT_CARD;
    state.ctx().remaining_steps = 1;
    state.ctx().resolving_card = card_ids::ALDRICH_AMES;
    return false;
}

bool trigger_pershing_ii(GameState& state, Player p) noexcept {
    state.victory_points = static_cast<int8_t>(std::max(-20, state.victory_points - 1));
    if (state.victory_points <= -20) {
        state.current_phase = Phase::GAME_OVER;
        return true;
    }
    state.ctx().decision_player = Player::USSR;
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().remaining_steps = 3;
    state.ctx().max_per_country = 1;
    state.ctx().allow_early_stop = 1;
    state.ctx().resolving_card = card_ids::PERSHING_II_DEPLOYED;
    return false;
}

bool trigger_wargames(GameState& state, Player p) noexcept {
    if (state.defcon == 2) {
        // Phasing player chooses: Branch 0 = Give 6 VP to opponent and end game immediately; Branch 1 = Pass
        state.ctx().decision_player = p;
        state.ctx().decision_type = DecisionType::CHOOSE_BRANCH;
        state.ctx().resolving_card = card_ids::WARGAMES;
        return false;
    }
    return true;
}

bool trigger_solidarity(GameState& state, Player p) noexcept {
    if (state.has_flag(effect_bits::JOHN_PAUL_II_PLAYED)) {
        state.countries[countries::POLAND].add_influence(Player::US, 3);
    }
    return true;
}

bool trigger_iran_iraq_war(GameState& state, Player p) noexcept {
    return war_helpers::trigger_war(state, card_ids::IRAN_IRAQ_WAR, p);
}

bool trigger_yuri_and_samantha(GameState& state, Player p) noexcept {
    state.set_flag(effect_bits::YURI_AND_SAMANTHA_ACTIVE);
    return true;
}

bool trigger_awacs_sale(GameState& state, Player p) noexcept {
    state.countries[countries::SAUDI_ARABIA].add_influence(Player::US, 2);
    state.set_flag(effect_bits::AWACS_PLAYED);
    return true;
}

} // namespace late_war

} // namespace ts
