#pragma once
#include "ts/engine.hpp"
#include "ts/state_machine.hpp"
#include "ts/card_data.hpp"
#include "ts/scoring.hpp"
#include "ts/map_data.hpp"
#include "ts/ops.hpp"
#include "ts/space_race.hpp"
#include "ts/constants.hpp"
#include "ts/action_mask.hpp"
#include <functional>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <cassert>

namespace ts {

struct ScoringEvent {
    uint8_t turn;
    Phase phase;
    Region region;
    RegionalStatus us_status;
    RegionalStatus ussr_status;
    int16_t us_score;
    int16_t ussr_score;
    int16_t net_delta;
    int8_t vp_before;
    int8_t vp_after;
};

class GameTestWrapper {
public:
    GameState state{};
    uint64_t seed = 0;
    size_t step_count = 0;
    std::vector<ScoringEvent> scoring_events;
    std::vector<MicroAction> action_history;

    using PolicyFn = std::function<MicroAction(const GameState& state, const std::vector<uint8_t>& legal_actions, size_t step_idx)>;

    explicit GameTestWrapper(uint64_t initial_seed = 42) {
        init(initial_seed);
    }

    void init(uint64_t initial_seed) {
        seed = initial_seed;
        step_count = 0;
        scoring_events.clear();
        action_history.clear();
        state = GameState{};
        Engine::init_game(state, seed);
    }

    bool is_terminal() const noexcept {
        return Engine::is_terminal(state);
    }

    uint8_t get_turn() const noexcept {
        return state.turn;
    }

    Phase get_phase() const noexcept {
        return state.current_phase;
    }

    int8_t get_vp() const noexcept {
        return state.victory_points;
    }

    uint8_t get_defcon() const noexcept {
        return state.defcon;
    }

    std::vector<uint8_t> get_legal_actions() const noexcept {
        uint8_t mask_buf[128];
        size_t mask_size = 0;
        Engine::get_legal_action_mask(state, mask_buf, &mask_size);
        std::vector<uint8_t> legal;
        legal.reserve(mask_size);
        for (size_t i = 0; i < mask_size; ++i) {
            if (mask_buf[i]) legal.push_back(static_cast<uint8_t>(i));
        }
        return legal;
    }

    RegionScoreSummary get_region_summary(Region r) const noexcept {
        return Scoring::evaluate_region(state, r);
    }

    void assert_invariants() const {
        assert(state.victory_points >= -20 && state.victory_points <= 20);
        assert(state.defcon >= 1 && state.defcon <= 5);
        assert(state.us_mil_ops <= 5 && state.ussr_mil_ops <= 5);
        assert(state.us_space_track <= 8 && state.ussr_space_track <= 8);
        assert(state.ctx_stack_depth <= 2);
    }

    bool step(const MicroAction& action) {
        if (is_terminal()) return false;

        int8_t vp_before = state.victory_points;
        uint8_t turn_before = state.turn;
        Phase phase_before = state.current_phase;
        DecisionType dt_before = state.ctx().decision_type;
        uint8_t pending_card = state.ctx().pending_op_card;
        uint8_t hl_us = state.headline_us_card;
        uint8_t hl_ussr = state.headline_ussr_card;

        bool ok = Engine::step(state, action);
        if (!ok) return false;

        action_history.push_back(action);
        step_count++;
        assert_invariants();

        auto check_scoring_card = [&](uint8_t card_id) {
            Region r = Region::NONE_REGION;
            if (card_id == card_ids::ASIA_SCORING) r = Region::ASIA;
            else if (card_id == card_ids::EUROPE_SCORING) r = Region::EUROPE;
            else if (card_id == card_ids::MIDDLE_EAST_SCORING) r = Region::MIDDLE_EAST;
            else if (card_id == card_ids::CENTRAL_AMERICA_SCORING) r = Region::CENTRAL_AMERICA;
            else if (card_id == card_ids::SOUTH_AMERICA_SCORING) r = Region::SOUTH_AMERICA;
            else if (card_id == card_ids::AFRICA_SCORING) r = Region::AFRICA;

            if (r != Region::NONE_REGION) {
                auto summary = Scoring::evaluate_region(state, r);
                scoring_events.push_back(ScoringEvent{
                    turn_before, phase_before, r,
                    summary.us_status, summary.ussr_status,
                    summary.us_score, summary.ussr_score,
                    summary.net_delta, vp_before, state.victory_points
                });
            }
        };

        // 1. Check Action Round event play
        if (dt_before == DecisionType::SELECT_PLAY_MODE && action.primary_id == static_cast<uint8_t>(PlayMode::EVENT)) {
            check_scoring_card(pending_card);
        }

        // 2. Check Headline card resolution
        if (phase_before == Phase::HEADLINE && (hl_us != 0 || hl_ussr != 0)) {
            if (CardData::is_scoring_card(hl_us)) check_scoring_card(hl_us);
            if (CardData::is_scoring_card(hl_ussr)) check_scoring_card(hl_ussr);
        }

        // 3. Check Final Scoring at Turn 10 completion
        if (turn_before == 10 && state.current_phase == Phase::GAME_OVER && state.turn >= 10) {
            constexpr Region all_regions[] = {
                Region::EUROPE, Region::ASIA, Region::MIDDLE_EAST,
                Region::AFRICA, Region::CENTRAL_AMERICA, Region::SOUTH_AMERICA
            };
            for (Region r : all_regions) {
                auto summary = Scoring::evaluate_region(state, r);
                scoring_events.push_back(ScoringEvent{
                    10, Phase::GAME_OVER, r,
                    summary.us_status, summary.ussr_status,
                    summary.us_score, summary.ussr_score,
                    summary.net_delta, vp_before, state.victory_points
                });
            }
        }

        return true;
    }

    bool auto_step(const PolicyFn& policy) {
        if (is_terminal()) return false;
        auto legal = get_legal_actions();
        MicroAction action = policy(state, legal, step_count);
        return step(action);
    }

    size_t run_to_completion(const PolicyFn& policy, size_t max_steps = 5000) {
        size_t start_steps = step_count;
        while (!is_terminal() && (step_count - start_steps) < max_steps) {
            if (!auto_step(policy)) break;
        }
        return step_count - start_steps;
    }

    size_t run_turn(const PolicyFn& policy, size_t max_steps = 500) {
        uint8_t cur_turn = state.turn;
        size_t start_steps = step_count;
        while (!is_terminal() && state.turn == cur_turn && (step_count - start_steps) < max_steps) {
            if (!auto_step(policy)) break;
        }
        return step_count - start_steps;
    }

    size_t total_presences() const noexcept {
        size_t count = 0;
        for (const auto& ev : scoring_events) {
            if (ev.us_status == RegionalStatus::PRESENCE || ev.ussr_status == RegionalStatus::PRESENCE) {
                count++;
            }
        }
        return count;
    }

    size_t total_dominations() const noexcept {
        size_t count = 0;
        for (const auto& ev : scoring_events) {
            if (ev.us_status == RegionalStatus::DOMINATION || ev.ussr_status == RegionalStatus::DOMINATION) {
                count++;
            }
        }
        return count;
    }

    size_t total_controls() const noexcept {
        size_t count = 0;
        for (const auto& ev : scoring_events) {
            if (ev.us_status == RegionalStatus::CONTROL || ev.ussr_status == RegionalStatus::CONTROL) {
                count++;
            }
        }
        return count;
    }

    static PolicyFn create_balanced_policy() {
        return [](const GameState& s, const std::vector<uint8_t>& legal, size_t step_idx) -> MicroAction {
            DecisionType dt = s.ctx().decision_type;
            Player p = s.ctx().decision_player;
            if (legal.empty()) {
                return MicroAction{dt, 0, 0, action_flags::CONFIRM_DONE};
            }
            uint8_t chosen_id = legal[0];
            uint8_t flags = 0;

            auto is_defcon_danger_card = [&](uint8_t cid) -> bool {
                if (s.defcon > 2) return false;
                if (p == Player::US && (cid == card_ids::DUCK_AND_COVER || cid == card_ids::CIA_CREATED || cid == card_ids::SOVIETS_SHOOT_DOWN_KAL_007)) return true;
                if (p == Player::USSR && (cid == card_ids::LONE_GUNMAN || cid == card_ids::OLYMPIC_GAMES || cid == card_ids::CHE || cid == card_ids::ORTEGA_ELECTED_IN_NICARAGUA)) return true;
                return false;
            };

            if (dt == DecisionType::SELECT_CARD) {
                uint8_t scoring_card = 0;
                uint8_t safe_card = 0;
                for (uint8_t cid : legal) {
                    if (CardData::is_scoring_card(cid)) scoring_card = cid;
                    else if (!is_defcon_danger_card(cid)) safe_card = cid;
                }
                if (s.current_phase == Phase::HEADLINE) {
                    chosen_id = (safe_card != 0) ? safe_card : legal[0];
                } else {
                    if (scoring_card != 0) chosen_id = scoring_card;
                    else if (safe_card != 0) chosen_id = safe_card;
                    else chosen_id = legal[0];
                }
            } else if (dt == DecisionType::SELECT_PLAY_MODE) {
                uint8_t card = s.ctx().pending_op_card;
                bool can_space = SpaceRace::can_attempt_space(s, p, card);
                if (is_defcon_danger_card(card) && can_space) {
                    chosen_id = static_cast<uint8_t>(PlayMode::SPACE);
                } else if (CardData::is_scoring_card(card)) {
                    chosen_id = static_cast<uint8_t>(PlayMode::EVENT);
                } else {
                    chosen_id = static_cast<uint8_t>(PlayMode::OPS);
                }
            } else if (dt == DecisionType::SELECT_OP_MODE) {
                uint8_t cur_mil = (p == Player::US) ? s.us_mil_ops : s.ussr_mil_ops;
                bool can_coup = false;
                bool can_inf = false;
                for (uint8_t m : legal) {
                    if (m == static_cast<uint8_t>(OpMode::COUP)) can_coup = true;
                    if (m == static_cast<uint8_t>(OpMode::INFLUENCE)) can_inf = true;
                }
                if (cur_mil < s.defcon && can_coup && s.defcon > 2) {
                    chosen_id = static_cast<uint8_t>(OpMode::COUP);
                } else if (can_inf) {
                    chosen_id = static_cast<uint8_t>(OpMode::INFLUENCE);
                } else {
                    chosen_id = legal[0];
                }
            } else if (dt == DecisionType::POINT_NODE) {
                if (legal.empty()) {
                    flags = action_flags::CONFIRM_DONE;
                } else if (s.ctx().resolving_card != 0 && s.ctx().allow_early_stop) {
                    chosen_id = legal[0];
                } else if (s.ctx().op_mode == OpMode::COUP) {
                    if (s.defcon == 2) {
                        bool non_bg_found = false;
                        for (uint8_t cid : legal) {
                            if (!MapData::get_country(cid).battleground) {
                                chosen_id = cid;
                                non_bg_found = true;
                                break;
                            }
                        }
                        if (!non_bg_found) chosen_id = legal[0];
                    } else {
                        chosen_id = legal[step_idx % legal.size()];
                    }
                } else if (s.current_phase == Phase::SETUP) {
                    if (p == Player::USSR) {
                        constexpr uint8_t ussr_setup[] = {countries::POLAND, countries::EAST_GERMANY, countries::AUSTRIA, countries::BULGARIA, countries::CZECHOSLOVAKIA};
                        for (uint8_t target : ussr_setup) {
                            if (std::find(legal.begin(), legal.end(), target) != legal.end()) {
                                chosen_id = target;
                                break;
                            }
                        }
                    } else {
                        constexpr uint8_t us_setup[] = {countries::WEST_GERMANY, countries::FRANCE, countries::ITALY, countries::UNITED_KINGDOM};
                        for (uint8_t target : us_setup) {
                            if (std::find(legal.begin(), legal.end(), target) != legal.end()) {
                                chosen_id = target;
                                break;
                            }
                        }
                    }
                } else {
                    if (p == Player::USSR) {
                        const std::vector<uint8_t> targets = {
                            countries::MEXICO, countries::NICARAGUA, countries::COSTA_RICA, countries::PANAMA, countries::CUBA, countries::GUATEMALA,
                            countries::POLAND, countries::EAST_GERMANY, countries::NORTH_KOREA, countries::IRAQ, countries::ANGOLA, countries::CHILE
                        };
                        bool target_found = false;
                        for (uint8_t t : targets) {
                            if (std::find(legal.begin(), legal.end(), t) != legal.end()) {
                                if (t == countries::PANAMA && s.countries[countries::PANAMA].ussr_influence >= 3) continue;
                                if (t == countries::MEXICO && s.countries[countries::MEXICO].ussr_influence >= 4) continue;
                                if (t == countries::COSTA_RICA && s.countries[countries::COSTA_RICA].ussr_influence >= 3) continue;
                                if (t == countries::NICARAGUA && s.countries[countries::NICARAGUA].ussr_influence >= 1) continue;
                                chosen_id = t;
                                target_found = true;
                                break;
                            }
                        }
                        if (!target_found) chosen_id = legal[step_idx % legal.size()];
                    } else {
                        const std::vector<uint8_t> targets = {
                            countries::ISRAEL, countries::EGYPT, countries::JORDAN, countries::LEBANON, countries::SAUDI_ARABIA,
                            countries::WEST_GERMANY, countries::FRANCE, countries::SOUTH_KOREA, countries::JAPAN, countries::SOUTH_AFRICA, countries::BRAZIL
                        };
                        bool target_found = false;
                        for (uint8_t t : targets) {
                            if (std::find(legal.begin(), legal.end(), t) != legal.end()) {
                                if (t == countries::ISRAEL && s.countries[countries::ISRAEL].us_influence >= 2) continue;
                                if (t == countries::EGYPT && s.countries[countries::EGYPT].us_influence >= 3) continue;
                                if (t == countries::JORDAN && s.countries[countries::JORDAN].us_influence >= 2) continue;
                                if (t == countries::LEBANON && s.countries[countries::LEBANON].us_influence >= 2) continue;
                                if (t == countries::SAUDI_ARABIA && s.countries[countries::SAUDI_ARABIA].us_influence >= 3) continue;
                                chosen_id = t;
                                target_found = true;
                                break;
                            }
                        }
                        if (!target_found) chosen_id = legal[step_idx % legal.size()];
                    }
                }
            } else if (dt == DecisionType::CHOOSE_BRANCH) {
                chosen_id = legal[0];
            }

            MicroAction act{};
            act.decision_type = dt;
            act.primary_id = chosen_id;
            act.flags = flags;
            return act;
        };
    }
};

} // namespace ts
