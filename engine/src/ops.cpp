#include "ts/ops.hpp"
#include "ts/map_data.hpp"
#include "ts/card_data.hpp"
#include "ts/scoring.hpp"
#include "ts/prng.hpp"
#include "ts/defcon.hpp"
#include "ts/constants.hpp"
#include <algorithm>

namespace ts {

uint8_t Operations::get_modified_ops(const GameState& state, uint8_t base_ops, Player player, Region target_region) noexcept {
    if (base_ops == 0 || player == Player::NONE) return 0;

    int16_t ops = static_cast<int16_t>(base_ops);

    if (player == Player::US) {
        bool containment = state.has_flag(effect_bits::CONTAINMENT_ACTIVE);
        bool purge = state.has_flag(effect_bits::PURGE_US_ACTIVE);
        if (containment && purge) {
            // Net zero modifier: ops unaffected
        } else if (containment) {
            ops = std::min<int16_t>(4, ops + 1);
        } else if (purge) {
            ops = std::max<int16_t>(1, ops - 1);
        }
    } else if (player == Player::USSR) {
        bool brezhnev = state.has_flag(effect_bits::BREZHNEV_DOCTRINE_ACTIVE);
        bool purge = state.has_flag(effect_bits::PURGE_USSR_ACTIVE);
        if (brezhnev && purge) {
            // Net zero modifier: ops unaffected
        } else if (brezhnev) {
            ops = std::min<int16_t>(4, ops + 1);
        } else if (purge) {
            ops = std::max<int16_t>(1, ops - 1);
        }
        if (state.has_flag(effect_bits::VIETNAM_REVOLTS_ACTIVE) && (target_region == Region::ASIA)) {
            ops += 1;
        }
    }

    return static_cast<uint8_t>(std::clamp<int16_t>(ops, 1, 5));
}

uint8_t Operations::get_effective_ops(const GameState& state, uint8_t card_id, Player player, Region target_region) noexcept {
    if (card_id < 1 || card_id > 110 || player == Player::NONE) return 0;

    uint8_t base_ops = CardData::get_card(card_id).ops;
    uint8_t ops = get_modified_ops(state, base_ops, player, target_region);

    if (card_id == card_ids::THE_CHINA_CARD && target_region == Region::ASIA) {
        ops += 1;
    }

    // Ceiling of 6, not 5: the China Card played in Southeast Asia by the USSR with Vietnam
    // Revolts active is 4 base + 1 (all Ops in Asia) + 1 (all Ops in Southeast Asia). A limit
    // of 5 silently swallowed one of the two bonuses whenever they stacked.
    return static_cast<uint8_t>(std::clamp<int16_t>(ops, 1, 6));
}

uint8_t Operations::grant_ops_for_card(const GameState& state, uint8_t card_id,
                                       Player player) noexcept {
    return get_effective_ops(state, card_id, player, Region::ASIA);
}

uint8_t Operations::grant_ops(const GameState& state, uint8_t base_ops, Player player) noexcept {
    return get_modified_ops(state, base_ops, player, Region::ASIA);
}

bool Operations::can_place_influence(const GameState& state, Player p, uint8_t country_id) noexcept {
    if (country_id >= 84 || p == Player::NONE) return false;

    const auto& c_info = MapData::get_country(country_id);

    // Chernobyl restriction check
    if (p == Player::USSR && state.has_flag(effect_bits::CHERNOBYL_ACTIVE)) {
        uint8_t ch_region = static_cast<uint8_t>((state.persistent_effects & effect_bits::CHERNOBYL_REGION_MASK) >> effect_bits::CHERNOBYL_REGION_SHIFT);
        if (c_info.region == static_cast<Region>(ch_region)) {
            return false;
        }
    }

    // Has influence already
    if (state.countries[country_id].get_influence(p) > 0) return true;

    // Superpower adjacent
    if (c_info.superpower_adjacent == p) return true;

    // Adjacent to friendly influence (must have had influence at start of Op if snapshot exists)
    bool has_snapshot = (state.ctx().start_influence_nodes[0] != 0 || state.ctx().start_influence_nodes[1] != 0);
    for (uint8_t i = 0; i < c_info.num_neighbors; ++i) {
        uint8_t n_id = c_info.neighbors[i];
        if (state.countries[n_id].get_influence(p) > 0) {
            if (!has_snapshot || state.ctx().has_start_influence(n_id)) {
                return true;
            }
        }
    }

    return false;
}

uint8_t Operations::get_influence_cost(const GameState& state, Player p, uint8_t country_id) noexcept {
    if (country_id >= 84 || p == Player::NONE) return 1;
    Player opp = get_opponent(p);
    return Scoring::is_controlled_by(state, country_id, opp) ? 2 : 1;
}

bool Operations::place_influence(GameState& state, Player p, uint8_t country_id) noexcept {
    if (!can_place_influence(state, p, country_id)) return false;
    state.countries[country_id].add_influence(p, 1);
    return true;
}

bool Operations::can_coup_or_realign(const GameState& state, Player p, uint8_t country_id) noexcept {
    if (country_id >= 84 || p == Player::NONE) return false;

    Player opp = get_opponent(p);
    if (state.countries[country_id].get_influence(opp) == 0) return false;

    const auto& c_info = MapData::get_country(country_id);

    // Special effects that circumvent DEFCON restrictions:
    // Tear Down This Wall (#96) allows US free Coup Attempts or Realignment rolls in Europe regardless of DEFCON
    bool defcon_exempt = (p == Player::US &&
                          state.ctx().pending_op_card == card_ids::TEAR_DOWN_THIS_WALL &&
                          state.ctx().event_granted_ops &&
                          c_info.region == Region::EUROPE);

    if (!defcon_exempt) {
        // DEFCON regional restrictions (Rule 8.1.5)
        if (state.defcon <= 4 && c_info.region == Region::EUROPE) return false;
        if (state.defcon <= 3 && c_info.region == Region::ASIA) return false;
        if (state.defcon <= 2 && c_info.region == Region::MIDDLE_EAST) return false;
    }

    // Special restrictions for USSR
    if (p == Player::USSR) {
        // US/Japan Pact: USSR cannot coup or realign in Japan
        if (state.has_flag(effect_bits::US_JAPAN_PACT_ACTIVE) && country_id == countries::JAPAN) {
            return false;
        }

        // NATO protection: USSR cannot coup or realign in US-controlled European countries
        if (state.has_flag(effect_bits::NATO_ACTIVE) && c_info.region == Region::EUROPE) {
            if (Scoring::is_controlled_by(state, country_id, Player::US)) {
                bool exempt = (country_id == countries::FRANCE && state.has_flag(effect_bits::NATO_CANCELED_FRANCE)) ||
                              (country_id == countries::WEST_GERMANY && state.has_flag(effect_bits::NATO_CANCELED_WEST_GERMANY));
                if (!exempt) return false;
            }
        }
    }

    return true;
}

bool Operations::can_coup_or_realign(const GameState& state, Player p) noexcept {
    if (p == Player::NONE) return false;
    for (uint8_t i = 0; i < 84; ++i) {
        if (can_coup_or_realign(state, p, i)) return true;
    }
    return false;
}

bool Operations::can_coup(const GameState& state, Player p, uint8_t country_id) noexcept {
    if (!can_coup_or_realign(state, p, country_id)) return false;

    const auto& c_info = MapData::get_country(country_id);

    // Coup-specific restrictions:
    // The Reformer prevents all USSR coups in Europe
    if (p == Player::USSR && state.has_flag(effect_bits::THE_REFORMER_PLAYED) && c_info.region == Region::EUROPE) {
        return false;
    }

    return true;
}

CoupResult Operations::execute_coup(GameState& state, Player p, uint8_t country_id, uint8_t ops_value, uint8_t forced_roll) noexcept {
    CoupResult res{};
    if (country_id >= 84 || p == Player::NONE) return res;

    Player opp = get_opponent(p);
    const auto& c_info = MapData::get_country(country_id);
    bool defcon_one_loss = false;

    // 1. Cuban Missile Crisis check & cancellation
    if (p == Player::USSR && state.has_flag(effect_bits::CMC_ACTIVE_US)) {
        if (state.countries[countries::CUBA].ussr_influence >= 2) {
            // Cancel CMC by removing 2 USSR influence from Cuba
            state.countries[countries::CUBA].remove_influence(Player::USSR, 2);
            state.clear_flag(effect_bits::CMC_ACTIVE_US);
        } else {
            // USSR loses immediately. DEFCON is deliberately NOT set to 1: this is a Cuban
            // Missile Crisis loss, not a thermonuclear one. The flag records it so the loss
            // is distinguishable from a legitimate 20 VP win, which otherwise looks
            // identical from the outside -- same victory_points, same GAME_OVER, DEFCON
            // untouched.
            state.set_flag(effect_bits::CMC_SUICIDE_LOSS);
            state.victory_points = 20;
            state.current_phase = Phase::GAME_OVER;
            res.caused_defcon_suicide = true;
            return res;
        }
    }
    if (p == Player::US && state.has_flag(effect_bits::CMC_ACTIVE_USSR)) {
        if (state.countries[countries::WEST_GERMANY].us_influence >= 2) {
            state.countries[countries::WEST_GERMANY].remove_influence(Player::US, 2);
            state.clear_flag(effect_bits::CMC_ACTIVE_USSR);
        } else if (state.countries[countries::TURKEY].us_influence >= 2) {
            state.countries[countries::TURKEY].remove_influence(Player::US, 2);
            state.clear_flag(effect_bits::CMC_ACTIVE_USSR);
        } else {
            // US loses immediately. See the note above on why DEFCON stays put.
            state.set_flag(effect_bits::CMC_SUICIDE_LOSS);
            state.victory_points = -20;
            state.current_phase = Phase::GAME_OVER;
            res.caused_defcon_suicide = true;
            return res;
        }
    }

    // 2. Military Operations Credit (uses effective ops_value with modifiers). A coup made
    // with an event's own free attempt earns none -- see free_action::coup_earns_no_military_ops.
    if (free_action::coup_earns_no_military_ops(state.ctx().pending_op_card,
                                                state.ctx().event_granted_ops,
                                                state.ctx().resolving_card)) {
        // nothing spent, nothing to record
    } else if (p == Player::US) {
        state.us_mil_ops = static_cast<uint8_t>(std::min(5, static_cast<int>(state.us_mil_ops) + ops_value));
    } else {
        state.ussr_mil_ops = static_cast<uint8_t>(std::min(5, static_cast<int>(state.ussr_mil_ops) + ops_value));
    }

    // 3. DEFCON Degradation on Battleground Coup
    if (c_info.battleground) {
        bool nuc_subs_immune = (p == Player::US && state.has_flag(effect_bits::NUCLEAR_SUBS_ACTIVE));
        if (!nuc_subs_immune) {
            if (state.defcon > 1) {
                state.defcon--;
                res.defcon_degraded = true;
                if (state.defcon == 2) {
                    state.defcon_dropped_to_2 = 1;
                }
            }

            // The coup that takes DEFCON to 1 still happens: the die is rolled and the
            // influence moves, and only then does the game end with the couping player
            // losing. Returning here instead skipped the roll entirely, so the board froze
            // -- at turn 5's headline of ts-replayer game 102 the USSR's Panama coup removed
            // the two US influence in the log and none in the engine. Deferred to the end.
            defcon_one_loss = (state.defcon == 1);
        }
    }

    // 4. Die Roll & Modifiers
    uint8_t roll = (forced_roll > 0) ? forced_roll : Prng::roll_d6(state.rng_state);
    res.die_roll = roll;
    state.last_die_roll = roll;

    int16_t mod_roll = roll;

    // Latin American Death Squads
    bool in_latin_america = (c_info.region == Region::CENTRAL_AMERICA || c_info.region == Region::SOUTH_AMERICA);
    if (in_latin_america) {
        if (state.has_flag(effect_bits::DEATH_SQUADS_US)) {
            mod_roll += (p == Player::US ? 1 : -1);
        } else if (state.has_flag(effect_bits::DEATH_SQUADS_USSR)) {
            mod_roll += (p == Player::USSR ? 1 : -1);
        }
    }

    // SALT Negotiations
    if (state.has_flag(effect_bits::SALT_ACTIVE)) {
        mod_roll -= 1;
    }

    // 5. Margin Calculation
    int16_t total = mod_roll + ops_value;
    int16_t margin = total - (2 * c_info.stability);
    res.margin = margin;

    if (margin > 0) {
        res.success = true;
        uint8_t opp_inf = state.countries[country_id].get_influence(opp);
        uint8_t to_remove = static_cast<uint8_t>(std::min(static_cast<int16_t>(opp_inf), margin));
        state.countries[country_id].remove_influence(opp, to_remove);
        res.opp_inf_removed = to_remove;

        uint8_t remaining = static_cast<uint8_t>(margin - to_remove);
        if (remaining > 0) {
            state.countries[country_id].add_influence(p, remaining);
            res.player_inf_added = remaining;
        }
    }

    // 6. Yuri and Samantha (#109)
    if (p == Player::US && state.has_flag(effect_bits::YURI_AND_SAMANTHA_ACTIVE)) {
        state.victory_points = static_cast<int8_t>(std::max(-20, state.victory_points - 1));
        if (state.victory_points <= -20) {
            state.current_phase = Phase::GAME_OVER;
        }
    }

    state.last_roll = DieRollRecord{
        .type = RollType::COUP,
        .roller = p,
        .card_id = 0,
        .country_id = country_id,
        .roll1 = roll,
        .mod1 = static_cast<int8_t>(ops_value + (mod_roll - roll)),
        .roll2 = 0,
        .mod2 = static_cast<int8_t>(2 * c_info.stability),
        .success = (margin > 0),
        .net_delta = static_cast<int8_t>(res.opp_inf_removed + res.player_inf_added)
    };

    // Record turn aggregate
    size_t p_idx = player_to_index(p);
    size_t r_idx = static_cast<size_t>(c_info.region);
    if (p_idx < 2 && r_idx < 6) {
        state.turn_aggregates.coups_by_region[p_idx][r_idx]++;
    }

    if (defcon_one_loss) {
        res.caused_defcon_suicide = true;
        resolve_defcon_one_loss(state, p);
    }

    return res;
}

bool Operations::can_realign(const GameState& state, Player p, uint8_t country_id) noexcept {
    return can_coup_or_realign(state, p, country_id);
}

RealignResult Operations::execute_realign(GameState& state, Player p, uint8_t country_id, uint8_t forced_us_roll, uint8_t forced_ussr_roll) noexcept {
    RealignResult res{};
    if (country_id >= 84 || p == Player::NONE) return res;

    const auto& c_info = MapData::get_country(country_id);

    res.us_roll = (forced_us_roll > 0) ? forced_us_roll : Prng::roll_d6(state.rng_state);
    res.ussr_roll = (forced_ussr_roll > 0) ? forced_ussr_roll : Prng::roll_d6(state.rng_state);
    state.last_die_roll = res.us_roll;
    state.last_opp_die_roll = res.ussr_roll;

    // Compute modifiers
    int16_t us_mod = 0;
    int16_t ussr_mod = 0;

    // +1 for each adjacent controlled country
    for (uint8_t i = 0; i < c_info.num_neighbors; ++i) {
        uint8_t n_id = c_info.neighbors[i];
        if (Scoring::is_controlled_by(state, n_id, Player::US)) us_mod++;
        if (Scoring::is_controlled_by(state, n_id, Player::USSR)) ussr_mod++;
    }

    // +1 if player has more influence in target than opponent
    uint8_t us_inf = state.countries[country_id].us_influence;
    uint8_t ussr_inf = state.countries[country_id].ussr_influence;
    if (us_inf > ussr_inf) us_mod++;
    else if (ussr_inf > us_inf) ussr_mod++;

    // +1 if adjacent to superpower home
    if (c_info.superpower_adjacent == Player::US) us_mod++;
    if (c_info.superpower_adjacent == Player::USSR) ussr_mod++;

    // Iran-Contra Scandal: US -1 Realignment
    if (state.has_flag(effect_bits::IRAN_CONTRA_ACTIVE)) {
        us_mod -= 1;
    }

    res.us_mod = us_mod;
    res.ussr_mod = ussr_mod;
    res.us_total = res.us_roll + us_mod;
    res.ussr_total = res.ussr_roll + ussr_mod;

    if (res.us_total > res.ussr_total) {
        res.winner = Player::US;
        uint8_t diff = static_cast<uint8_t>(res.us_total - res.ussr_total);
        uint8_t to_remove = std::min(diff, state.countries[country_id].ussr_influence);
        state.countries[country_id].remove_influence(Player::USSR, to_remove);
        res.inf_removed = to_remove;
    } else if (res.ussr_total > res.us_total) {
        res.winner = Player::USSR;
        uint8_t diff = static_cast<uint8_t>(res.ussr_total - res.us_total);
        uint8_t to_remove = std::min(diff, state.countries[country_id].us_influence);
        state.countries[country_id].remove_influence(Player::US, to_remove);
        res.inf_removed = to_remove;
    } else {
        res.winner = Player::NONE;
    }

    state.last_roll = DieRollRecord{
        .type = RollType::REALIGNMENT,
        .roller = p,
        .card_id = 0,
        .country_id = country_id,
        .roll1 = res.us_roll,
        .mod1 = static_cast<int8_t>(us_mod),
        .roll2 = res.ussr_roll,
        .mod2 = static_cast<int8_t>(ussr_mod),
        .success = (res.winner == p),
        .net_delta = static_cast<int8_t>(res.inf_removed)
    };

    // Record turn aggregate
    size_t p_idx = player_to_index(p);
    size_t r_idx = static_cast<size_t>(c_info.region);
    if (p_idx < 2 && r_idx < 6) {
        state.turn_aggregates.realignments_by_region[p_idx][r_idx]++;
    }

    return res;
}

void Operations::get_influence_placement_mask(const GameState& state, Player p, uint8_t ops_available, uint8_t* out_mask_84) noexcept {
    for (uint8_t i = 0; i < 84; ++i) {
        if (can_place_influence(state, p, i)) {
            uint8_t cost = get_influence_cost(state, p, i);
            out_mask_84[i] = (ops_available >= cost) ? 1 : 0;
        } else {
            out_mask_84[i] = 0;
        }
    }
}

void Operations::get_coup_target_mask(const GameState& state, Player p, uint8_t* out_mask_84) noexcept {
    for (uint8_t i = 0; i < 84; ++i) {
        out_mask_84[i] = can_coup(state, p, i) ? 1 : 0;
    }
}

void Operations::get_realign_target_mask(const GameState& state, Player p, uint8_t* out_mask_84) noexcept {
    for (uint8_t i = 0; i < 84; ++i) {
        out_mask_84[i] = can_realign(state, p, i) ? 1 : 0;
    }
}

} // namespace ts
