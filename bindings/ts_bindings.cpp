#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/array.h>

#include "ts/types.hpp"
#include "ts/constants.hpp"
#include "ts/micro_action.hpp"
#include "ts/game_state.hpp"
#include "ts/map_data.hpp"
#include "ts/card_data.hpp"
#include "ts/scoring.hpp"
#include "ts/space_race.hpp"
#include "ts/ops.hpp"
#include "ts/action_mask.hpp"
#include "ts/state_machine.hpp"
#include "ts/serialization.hpp"
#include "ts/engine.hpp"

namespace nb = nanobind;

// Helper: Convert enum types to human-readable string names
static const char* decision_type_to_str(ts::DecisionType dt) {
    switch (dt) {
        case ts::DecisionType::NONE: return "NONE";
        case ts::DecisionType::SELECT_CARD: return "SELECT_CARD";
        case ts::DecisionType::SELECT_PLAY_MODE: return "SELECT_PLAY_MODE";
        case ts::DecisionType::CHOOSE_TIMING_BRANCH: return "CHOOSE_TIMING_BRANCH";
        case ts::DecisionType::SELECT_OP_MODE: return "SELECT_OP_MODE";
        case ts::DecisionType::POINT_NODE: return "POINT_NODE";
        case ts::DecisionType::CHOOSE_BRANCH: return "CHOOSE_BRANCH";
        default: return "UNKNOWN";
    }
}

static const char* phase_to_str(ts::Phase phase) {
    switch (phase) {
        case ts::Phase::SETUP: return "SETUP";
        case ts::Phase::HEADLINE: return "HEADLINE";
        case ts::Phase::ACTION_ROUND: return "ACTION_ROUND";
        case ts::Phase::INTERRUPT: return "INTERRUPT";
        case ts::Phase::DISCARD: return "DISCARD";
        case ts::Phase::END_TURN: return "END_TURN";
        case ts::Phase::GAME_OVER: return "GAME_OVER";
        default: return "UNKNOWN";
    }
}

static const char* play_mode_to_str(uint8_t mode) {
    switch (mode) {
        case static_cast<uint8_t>(ts::PlayMode::EVENT): return "EVENT";
        case static_cast<uint8_t>(ts::PlayMode::OPS): return "OPS";
        case static_cast<uint8_t>(ts::PlayMode::SPACE): return "SPACE";
        case static_cast<uint8_t>(ts::PlayMode::PASS): return "PASS";
        default: return "UNKNOWN";
    }
}

static const char* op_mode_to_str(uint8_t mode) {
    switch (mode) {
        case static_cast<uint8_t>(ts::OpMode::INFLUENCE): return "INFLUENCE";
        case static_cast<uint8_t>(ts::OpMode::COUP): return "COUP";
        case static_cast<uint8_t>(ts::OpMode::REALIGN): return "REALIGN";
        default: return "UNKNOWN";
    }
}

static const char* timing_branch_to_str(uint8_t branch) {
    switch (branch) {
        case static_cast<uint8_t>(ts::TimingBranch::OPS_FIRST): return "OPS_FIRST";
        case static_cast<uint8_t>(ts::TimingBranch::EVENT_FIRST): return "EVENT_FIRST";
        default: return "UNKNOWN";
    }
}

// Helper: Convert entire GameState to a detailed Python dictionary
nb::dict game_state_to_dict(const ts::GameState& state) {
    nb::dict d;

    // 1. Global Tracks
    d["victory_points"] = state.victory_points;
    d["defcon"] = state.defcon;
    nb::dict mil_ops;
    mil_ops["US"] = state.us_mil_ops;
    mil_ops["USSR"] = state.ussr_mil_ops;
    d["mil_ops"] = mil_ops;

    nb::dict space;
    space["US"] = state.us_space_track;
    space["USSR"] = state.ussr_space_track;
    d["space"] = space;

    d["turn"] = state.turn;
    d["action_round"] = state.action_round;

    // 2. Phasing & Priority
    d["phasing_player"] = (state.phasing_player == ts::Player::US ? "US" : (state.phasing_player == ts::Player::USSR ? "USSR" : "NONE"));
    d["current_phase"] = static_cast<int>(state.current_phase);
    d["current_phase_name"] = phase_to_str(state.current_phase);
    d["phase_name"] = phase_to_str(state.current_phase);
    d["headline_us_card"] = state.headline_us_card;
    d["headline_ussr_card"] = state.headline_ussr_card;
    d["forced_card_player"] = (state.forced_card_player == ts::Player::US ? "US" : (state.forced_card_player == ts::Player::USSR ? "USSR" : "NONE"));
    d["forced_card_id"] = state.forced_card_id;

    // 3. Space turns used
    nb::dict space_turns;
    space_turns["US"] = state.us_space_turns_used;
    space_turns["USSR"] = state.ussr_space_turns_used;
    d["space_turns_used"] = space_turns;

    // 4. China Card
    nb::dict china;
    china["holder"] = (state.china_card_holder == ts::Player::US ? "US" : "USSR");
    china["playable"] = (state.china_card_playable != 0);
    d["china_card"] = china;

    // 5. Persistent Flags
    nb::list flags;
    #define CHECK_FLAG(f, name) if (state.has_flag(ts::effect_bits::f)) flags.append(name)
    CHECK_FLAG(NATO_ACTIVE, "NATO_ACTIVE");
    CHECK_FLAG(NATO_CANCELED_FRANCE, "NATO_CANCELED_FRANCE");
    CHECK_FLAG(NATO_CANCELED_WEST_GERMANY, "NATO_CANCELED_WEST_GERMANY");
    CHECK_FLAG(MARSHALL_PLAN_PLAYED, "MARSHALL_PLAN_PLAYED");
    CHECK_FLAG(WARSAW_PACT_PLAYED, "WARSAW_PACT_PLAYED");
    CHECK_FLAG(US_JAPAN_PACT_ACTIVE, "US_JAPAN_PACT_ACTIVE");
    CHECK_FLAG(CONTAINMENT_ACTIVE, "CONTAINMENT_ACTIVE");
    CHECK_FLAG(PURGE_US_ACTIVE, "PURGE_US_ACTIVE");
    CHECK_FLAG(PURGE_USSR_ACTIVE, "PURGE_USSR_ACTIVE");
    CHECK_FLAG(VIETNAM_REVOLTS_ACTIVE, "VIETNAM_REVOLTS_ACTIVE");
    CHECK_FLAG(FORMOSAN_RESOLUTION_ACTIVE, "FORMOSAN_RESOLUTION_ACTIVE");
    CHECK_FLAG(CMC_ACTIVE_US, "CMC_ACTIVE_US");
    CHECK_FLAG(CMC_ACTIVE_USSR, "CMC_ACTIVE_USSR");
    CHECK_FLAG(NUCLEAR_SUBS_ACTIVE, "NUCLEAR_SUBS_ACTIVE");
    CHECK_FLAG(QUAGMIRE_ACTIVE, "QUAGMIRE_ACTIVE");
    CHECK_FLAG(BEAR_TRAP_ACTIVE, "BEAR_TRAP_ACTIVE");
    CHECK_FLAG(SALT_ACTIVE, "SALT_ACTIVE");
    CHECK_FLAG(WE_WILL_BURY_YOU_PENDING, "WE_WILL_BURY_YOU_PENDING");
    CHECK_FLAG(BREZHNEV_DOCTRINE_ACTIVE, "BREZHNEV_DOCTRINE_ACTIVE");
    CHECK_FLAG(FLOWER_POWER_ACTIVE, "FLOWER_POWER_ACTIVE");
    CHECK_FLAG(U2_INCIDENT_ACTIVE, "U2_INCIDENT_ACTIVE");
    CHECK_FLAG(SHUTTLE_DIPLOMACY_ACTIVE, "SHUTTLE_DIPLOMACY_ACTIVE");
    CHECK_FLAG(DEATH_SQUADS_US, "DEATH_SQUADS_US");
    CHECK_FLAG(DEATH_SQUADS_USSR, "DEATH_SQUADS_USSR");
    CHECK_FLAG(CAMP_DAVID_PLAYED, "CAMP_DAVID_PLAYED");
    CHECK_FLAG(IRON_LADY_PLAYED, "IRON_LADY_PLAYED");
    CHECK_FLAG(NORTH_SEA_OIL_PLAYED, "NORTH_SEA_OIL_PLAYED");
    CHECK_FLAG(NORTH_SEA_OIL_ACTIVE, "NORTH_SEA_OIL_ACTIVE");
    CHECK_FLAG(THE_REFORMER_PLAYED, "THE_REFORMER_PLAYED");
    CHECK_FLAG(IRAN_CONTRA_ACTIVE, "IRAN_CONTRA_ACTIVE");
    CHECK_FLAG(EVIL_EMPIRE_PLAYED, "EVIL_EMPIRE_PLAYED");
    CHECK_FLAG(ALDRICH_AMES_ACTIVE, "ALDRICH_AMES_ACTIVE");
    CHECK_FLAG(JOHN_PAUL_II_PLAYED, "JOHN_PAUL_II_PLAYED");
    CHECK_FLAG(NORAD_ACTIVE, "NORAD_ACTIVE");
    CHECK_FLAG(YURI_AND_SAMANTHA_ACTIVE, "YURI_AND_SAMANTHA_ACTIVE");
    CHECK_FLAG(AWACS_PLAYED, "AWACS_PLAYED");
    CHECK_FLAG(IRANIAN_HOSTAGE_CRISIS_PLAY, "IRANIAN_HOSTAGE_CRISIS_PLAY");
    CHECK_FLAG(WILLY_BRANDT_PLAYED, "WILLY_BRANDT_PLAYED");
    CHECK_FLAG(TEAR_DOWN_THIS_WALL_PLAYED, "TEAR_DOWN_THIS_WALL_PLAYED");
    CHECK_FLAG(CHERNOBYL_ACTIVE, "CHERNOBYL_ACTIVE");
    #undef CHECK_FLAG
    d["flags"] = flags;
    d["persistent_effects"] = state.persistent_effects;

    // 6. Countries (84 Nodes)
    nb::dict countries;
    for (uint8_t i = 0; i < 84; ++i) {
        nb::dict c;
        const auto& info = ts::MapData::get_country(i);
        c["id"] = i;
        c["name"] = std::string(info.name);
        c["stability"] = info.stability;
        c["battleground"] = info.battleground;
        c["region"] = static_cast<int>(info.region);
        c["us_influence"] = state.countries[i].us_influence;
        c["ussr_influence"] = state.countries[i].ussr_influence;

        // Control evaluation
        uint8_t us_inf = state.countries[i].us_influence;
        uint8_t ussr_inf = state.countries[i].ussr_influence;
        uint8_t stab = info.stability;
        if (us_inf >= stab && us_inf >= ussr_inf + stab) {
            c["controlled_by"] = "US";
        } else if (ussr_inf >= stab && ussr_inf >= us_inf + stab) {
            c["controlled_by"] = "USSR";
        } else {
            c["controlled_by"] = "NONE";
        }

        countries[std::string(info.name).c_str()] = c;
    }
    d["countries"] = countries;

    // 7. Cards & Locations
    nb::list us_hand;
    nb::list ussr_hand;
    nb::list us_cards;
    nb::list ussr_cards;
    nb::list discard_pile;
    nb::list removed_pile;
    nb::list unavailable_cards;
    uint8_t draw_deck_count = 0;

    for (uint8_t i = 1; i <= 110; ++i) {
        auto loc = state.card_locations[i];
        switch (loc) {
            case ts::CardLocation::HAND_US: {
                us_hand.append(i);
                nb::dict ci;
                ci["id"] = i;
                ci["name"] = std::string(ts::CardData::get_card_name(i));
                ci["ops"] = ts::CardData::get_card(i).ops;
                us_cards.append(ci);
                break;
            }
            case ts::CardLocation::HAND_USSR: {
                ussr_hand.append(i);
                nb::dict ci;
                ci["id"] = i;
                ci["name"] = std::string(ts::CardData::get_card_name(i));
                ci["ops"] = ts::CardData::get_card(i).ops;
                ussr_cards.append(ci);
                break;
            }
            case ts::CardLocation::DISCARD_PILE: discard_pile.append(i); break;
            case ts::CardLocation::REMOVED_FROM_GAME: removed_pile.append(i); break;
            case ts::CardLocation::DRAW_DECK: draw_deck_count++; break;
            case ts::CardLocation::UNAVAILABLE: unavailable_cards.append(i); break;
            default: break;
        }
    }

    nb::dict hands;
    hands["US"] = us_hand;
    hands["USSR"] = ussr_hand;
    hands["US_cards"] = us_cards;
    hands["USSR_cards"] = ussr_cards;
    d["hands"] = hands;
    d["discard_pile"] = discard_pile;
    d["removed_pile"] = removed_pile;
    d["unavailable_cards"] = unavailable_cards;
    d["draw_deck_count"] = draw_deck_count;

    // 8. Decision Context & Stack
    const auto& ctx = state.ctx();
    nb::dict ctx_dict;
    ctx_dict["decision_player"] = (ctx.decision_player == ts::Player::US ? "US" : (ctx.decision_player == ts::Player::USSR ? "USSR" : "NONE"));
    ctx_dict["decision_type"] = static_cast<int>(ctx.decision_type);
    ctx_dict["decision_type_name"] = decision_type_to_str(ctx.decision_type);
    ctx_dict["pending_op_card"] = ctx.pending_op_card;
    ctx_dict["pending_op_card_name"] = (ctx.pending_op_card > 0) ? std::string(ts::CardData::get_card_name(ctx.pending_op_card)) : "";
    ctx_dict["pending_ops_value"] = ctx.pending_ops_value;
    ctx_dict["remaining_steps"] = ctx.remaining_steps;
    ctx_dict["max_per_country"] = ctx.max_per_country;
    ctx_dict["allow_early_stop"] = (ctx.allow_early_stop != 0);
    ctx_dict["resolving_card"] = ctx.resolving_card;
    ctx_dict["resolving_card_name"] = (ctx.resolving_card > 0) ? std::string(ts::CardData::get_card_name(ctx.resolving_card)) : "";
    ctx_dict["stack_depth"] = state.ctx_stack_depth;
    d["decision_context"] = ctx_dict;

    // 9. Legal Action Mask & Human Readable Action Labels
    uint8_t mask[128];
    size_t mask_size = 0;
    ts::Engine::get_legal_action_mask(state, mask, &mask_size);

    nb::list legal_list;
    nb::dict action_labels;
    for (size_t i = 0; i < mask_size; ++i) {
        if (mask[i]) {
            legal_list.append(i);

            // Generate contextual human-readable label
            std::string label;
            switch (ctx.decision_type) {
                case ts::DecisionType::POINT_NODE:
                    if (i < 84) {
                        label = std::string(ts::MapData::get_country_name(static_cast<uint8_t>(i)));
                    }
                    break;
                case ts::DecisionType::SELECT_CARD:
                    if (i >= 1 && i <= 110) {
                        label = std::string(ts::CardData::get_card_name(static_cast<uint8_t>(i)));
                    }
                    break;
                case ts::DecisionType::SELECT_PLAY_MODE:
                    label = play_mode_to_str(static_cast<uint8_t>(i));
                    break;
                case ts::DecisionType::SELECT_OP_MODE:
                    label = op_mode_to_str(static_cast<uint8_t>(i));
                    break;
                case ts::DecisionType::CHOOSE_TIMING_BRANCH:
                    label = timing_branch_to_str(static_cast<uint8_t>(i));
                    break;
                default:
                    label = "Option " + std::to_string(i);
                    break;
            }
            if (!label.empty()) {
                action_labels[std::to_string(i).c_str()] = label;
            }
        }
    }
    nb::dict legal_actions;
    legal_actions["decision_type"] = static_cast<int>(ctx.decision_type);
    legal_actions["decision_type_name"] = decision_type_to_str(ctx.decision_type);
    legal_actions["decision_player"] = (ctx.decision_player == ts::Player::US ? "US" : (ctx.decision_player == ts::Player::USSR ? "USSR" : "NONE"));
    legal_actions["valid_ids"] = legal_list;
    legal_actions["valid_action_labels"] = action_labels;
    legal_actions["allow_early_stop"] = (ctx.allow_early_stop != 0);
    d["legal_actions"] = legal_actions;

    d["is_terminal"] = ts::Engine::is_terminal(state);
    if (ts::Engine::is_terminal(state)) {
        d["terminal_utility"] = ts::Engine::get_terminal_utility(state);
    } else {
        d["terminal_utility"] = 0.0f;
    }

    return d;
}

NB_MODULE(ts_engine, m) {
    m.doc() = "Twilight Struggle C++ Simulation Engine Python Bindings (nanobind)";

    // Enums (with nb::is_arithmetic to support direct integer casting/comparison)
    nb::enum_<ts::Player>(m, "Player", nb::is_arithmetic())
        .value("NONE", ts::Player::NONE)
        .value("US", ts::Player::US)
        .value("USSR", ts::Player::USSR)
        .export_values();

    nb::enum_<ts::Phase>(m, "Phase", nb::is_arithmetic())
        .value("SETUP", ts::Phase::SETUP)
        .value("HEADLINE", ts::Phase::HEADLINE)
        .value("ACTION_ROUND", ts::Phase::ACTION_ROUND)
        .value("INTERRUPT", ts::Phase::INTERRUPT)
        .value("DISCARD", ts::Phase::DISCARD)
        .value("END_TURN", ts::Phase::END_TURN)
        .value("GAME_OVER", ts::Phase::GAME_OVER)
        .export_values();

    nb::enum_<ts::DecisionType>(m, "DecisionType", nb::is_arithmetic())
        .value("NONE", ts::DecisionType::NONE)
        .value("SELECT_CARD", ts::DecisionType::SELECT_CARD)
        .value("SELECT_PLAY_MODE", ts::DecisionType::SELECT_PLAY_MODE)
        .value("CHOOSE_TIMING_BRANCH", ts::DecisionType::CHOOSE_TIMING_BRANCH)
        .value("SELECT_OP_MODE", ts::DecisionType::SELECT_OP_MODE)
        .value("POINT_NODE", ts::DecisionType::POINT_NODE)
        .value("CHOOSE_BRANCH", ts::DecisionType::CHOOSE_BRANCH)
        .export_values();

    nb::enum_<ts::PlayMode>(m, "PlayMode", nb::is_arithmetic())
        .value("EVENT", ts::PlayMode::EVENT)
        .value("OPS", ts::PlayMode::OPS)
        .value("SPACE", ts::PlayMode::SPACE)
        .value("PASS", ts::PlayMode::PASS)
        .export_values();

    nb::enum_<ts::TimingBranch>(m, "TimingBranch", nb::is_arithmetic())
        .value("OPS_FIRST", ts::TimingBranch::OPS_FIRST)
        .value("EVENT_FIRST", ts::TimingBranch::EVENT_FIRST)
        .export_values();

    nb::enum_<ts::OpMode>(m, "OpMode", nb::is_arithmetic())
        .value("INFLUENCE", ts::OpMode::INFLUENCE)
        .value("COUP", ts::OpMode::COUP)
        .value("REALIGN", ts::OpMode::REALIGN)
        .export_values();

    nb::enum_<ts::CardLocation>(m, "CardLocation", nb::is_arithmetic())
        .value("UNAVAILABLE", ts::CardLocation::UNAVAILABLE)
        .value("DRAW_DECK", ts::CardLocation::DRAW_DECK)
        .value("HAND_US", ts::CardLocation::HAND_US)
        .value("HAND_USSR", ts::CardLocation::HAND_USSR)
        .value("DISCARD_PILE", ts::CardLocation::DISCARD_PILE)
        .value("REMOVED_FROM_GAME", ts::CardLocation::REMOVED_FROM_GAME)
        .value("ONGOING_EVENT", ts::CardLocation::ONGOING_EVENT)
        .value("PEEKED_TEMP", ts::CardLocation::PEEKED_TEMP)
        .export_values();

    nb::enum_<ts::WarEra>(m, "WarEra", nb::is_arithmetic())
        .value("EARLY", ts::WarEra::EARLY)
        .value("MID", ts::WarEra::MID)
        .value("LATE", ts::WarEra::LATE)
        .export_values();

    // Structs
    nb::class_<ts::MicroAction>(m, "MicroAction")
        .def(nb::init<>())
        .def(nb::init<ts::DecisionType, uint8_t, uint8_t, uint8_t>(),
             nb::arg("decision_type"), nb::arg("primary_id") = 0, nb::arg("secondary_id") = 0, nb::arg("flags") = 0)
        .def_rw("decision_type", &ts::MicroAction::decision_type)
        .def_rw("primary_id", &ts::MicroAction::primary_id)
        .def_rw("secondary_id", &ts::MicroAction::secondary_id)
        .def_rw("flags", &ts::MicroAction::flags)
        .def("is_confirm_done", &ts::MicroAction::is_confirm_done)
        .def("__repr__", [](const ts::MicroAction& a) {
            return "<MicroAction type=" + std::to_string(static_cast<int>(a.decision_type)) +
                   " primary=" + std::to_string(static_cast<int>(a.primary_id)) +
                   " secondary=" + std::to_string(static_cast<int>(a.secondary_id)) +
                   " flags=" + std::to_string(static_cast<int>(a.flags)) + ">";
        });

    nb::class_<ts::CountryState>(m, "CountryState")
        .def_rw("us_influence", &ts::CountryState::us_influence)
        .def_rw("ussr_influence", &ts::CountryState::ussr_influence);

    nb::class_<ts::DecisionContext>(m, "DecisionContext")
        .def_ro("decision_player", &ts::DecisionContext::decision_player)
        .def_ro("decision_type", &ts::DecisionContext::decision_type)
        .def_ro("pending_op_card", &ts::DecisionContext::pending_op_card)
        .def_ro("pending_ops_value", &ts::DecisionContext::pending_ops_value)
        .def_ro("remaining_steps", &ts::DecisionContext::remaining_steps)
        .def_ro("max_per_country", &ts::DecisionContext::max_per_country)
        .def_ro("allow_early_stop", &ts::DecisionContext::allow_early_stop)
        .def_ro("resolving_card", &ts::DecisionContext::resolving_card)
        .def("is_visited", &ts::DecisionContext::is_visited);

    nb::class_<ts::GameState>(m, "GameState")
        .def(nb::init<>())
        .def_rw("victory_points", &ts::GameState::victory_points)
        .def_rw("defcon", &ts::GameState::defcon)
        .def_rw("us_mil_ops", &ts::GameState::us_mil_ops)
        .def_rw("ussr_mil_ops", &ts::GameState::ussr_mil_ops)
        .def_rw("us_space_track", &ts::GameState::us_space_track)
        .def_rw("ussr_space_track", &ts::GameState::ussr_space_track)
        .def_rw("turn", &ts::GameState::turn)
        .def_rw("action_round", &ts::GameState::action_round)
        .def_rw("phasing_player", &ts::GameState::phasing_player)
        .def_rw("headline_us_card", &ts::GameState::headline_us_card)
        .def_rw("headline_ussr_card", &ts::GameState::headline_ussr_card)
        .def_rw("current_phase", &ts::GameState::current_phase)
        .def_rw("forced_card_player", &ts::GameState::forced_card_player)
        .def_rw("forced_card_id", &ts::GameState::forced_card_id)
        .def_rw("china_card_holder", &ts::GameState::china_card_holder)
        .def_rw("china_card_playable", &ts::GameState::china_card_playable)
        .def_rw("persistent_effects", &ts::GameState::persistent_effects)
        .def_rw("ctx_stack_depth", &ts::GameState::ctx_stack_depth)
        .def_rw("rng_state", &ts::GameState::rng_state)
        .def("ctx", [](ts::GameState& s) -> ts::DecisionContext& { return s.ctx(); }, nb::rv_policy::reference)
        .def("has_flag", &ts::GameState::has_flag)
        .def("get_country", [](const ts::GameState& s, uint8_t idx) -> ts::CountryState {
            if (idx >= 84) throw std::out_of_range("Country ID must be 0..83");
            return s.countries[idx];
        })
        .def("set_country", [](ts::GameState& s, uint8_t idx, uint8_t us, uint8_t ussr) {
            if (idx >= 84) throw std::out_of_range("Country ID must be 0..83");
            s.countries[idx].us_influence = us;
            s.countries[idx].ussr_influence = ussr;
        })
        .def("get_card_location", [](const ts::GameState& s, uint8_t card_id) -> ts::CardLocation {
            if (card_id < 1 || card_id > 110) throw std::out_of_range("Card ID must be 1..110");
            return s.card_locations[card_id];
        })
        .def("set_card_location", [](ts::GameState& s, uint8_t card_id, ts::CardLocation loc) {
            if (card_id < 1 || card_id > 110) throw std::out_of_range("Card ID must be 1..110");
            s.card_locations[card_id] = loc;
        })
        .def("to_dict", &game_state_to_dict)
        .def("to_json", [](const ts::GameState& s) { return ts::Serializer::to_json(s); });

    // Engine class
    nb::class_<ts::Engine>(m, "Engine")
        .def_static("init_game", &ts::Engine::init_game)
        .def_static("step", &ts::Engine::step)
        .def_static("is_terminal", &ts::Engine::is_terminal)
        .def_static("get_terminal_utility", &ts::Engine::get_terminal_utility)
        .def_static("get_legal_action_mask", [](const ts::GameState& state) {
            uint8_t mask[128];
            size_t out_size = 0;
            ts::Engine::get_legal_action_mask(state, mask, &out_size);
            nb::list res;
            for (size_t i = 0; i < out_size; ++i) {
                res.append(static_cast<int>(mask[i]));
            }
            return res;
        })
        .def_static("get_legal_action_indices", [](const ts::GameState& state) {
            uint8_t mask[128];
            size_t out_size = 0;
            ts::Engine::get_legal_action_mask(state, mask, &out_size);
            nb::list res;
            for (size_t i = 0; i < out_size; ++i) {
                if (mask[i]) res.append(static_cast<int>(i));
            }
            return res;
        });

    // Map Metadata helpers
    nb::class_<ts::MapData>(m, "MapData")
        .def_static("get_country_name", [](uint8_t id) { return std::string(ts::MapData::get_country_name(id)); })
        .def_static("get_country_by_name", [](const std::string& name) { return ts::MapData::get_country_by_name(name); })
        .def_static("get_country_info", [](uint8_t id) {
            if (id >= 84) throw std::out_of_range("Country ID must be 0..83");
            const auto& c = ts::MapData::get_country(id);
            nb::dict d;
            d["id"] = c.id;
            d["name"] = std::string(c.name);
            d["stability"] = c.stability;
            d["battleground"] = c.battleground;
            d["region"] = static_cast<int>(c.region);
            d["in_western_europe"] = c.in_western_europe;
            d["in_eastern_europe"] = c.in_eastern_europe;
            d["in_southeast_asia"] = c.in_southeast_asia;
            d["superpower_adjacent"] = (c.superpower_adjacent == ts::Player::US ? "US" : (c.superpower_adjacent == ts::Player::USSR ? "USSR" : "NONE"));
            nb::list neighbors;
            for (uint8_t i = 0; i < c.num_neighbors; ++i) {
                neighbors.append(c.neighbors[i]);
            }
            d["neighbors"] = neighbors;
            return d;
        });

    // Card Metadata helpers
    nb::class_<ts::CardData>(m, "CardData")
        .def_static("get_card_name", [](uint8_t id) { return std::string(ts::CardData::get_card_name(id)); })
        .def_static("get_card_by_name", [](const std::string& name) { return ts::CardData::get_card_by_name(name); })
        .def_static("get_card_info", [](uint8_t id) {
            if (id < 1 || id > 110) throw std::out_of_range("Card ID must be 1..110");
            const auto& c = ts::CardData::get_card(id);
            nb::dict d;
            d["id"] = c.id;
            d["name"] = std::string(c.name);
            d["ops"] = c.ops;
            d["side"] = (c.side == ts::Player::US ? "US" : (c.side == ts::Player::USSR ? "USSR" : "NONE"));
            d["era"] = static_cast<int>(c.era);
            d["one_time"] = c.one_time;
            d["is_scoring"] = c.is_scoring;
            d["is_war_card"] = c.is_war_card;
            return d;
        });

    m.def("state_to_dict", &game_state_to_dict, "Convert GameState to Python dictionary");
}
