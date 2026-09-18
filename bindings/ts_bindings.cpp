#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <nanobind/ndarray.h>
#include "ts/observation.hpp"
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
#include "ts/card_handlers.hpp"
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
        case ts::DecisionType::ROLL_DIE: return "ROLL_DIE";
        default: return "UNKNOWN";
    }
}

// A refused action is a caller bug, never a game event: post-P14 `step` validates against the same
// mask the caller is expected to sample from, so a refusal can only mean the action was chosen
// against a different state. Raising is therefore the right default, and `try_step*` is kept for
// the callers that genuinely probe (the fuzzers, and the mask-vs-step differential sweeps).
[[noreturn]] static void throw_illegal_action(const ts::GameState& state, const std::string& what) {
    const auto& ctx = state.ctx();
    const char* who = (ctx.decision_player == ts::Player::US) ? "US"
                    : (ctx.decision_player == ts::Player::USSR) ? "USSR" : "NONE";
    throw std::runtime_error(
        "engine refused " + what + "; it is asking for " +
        std::string(decision_type_to_str(ctx.decision_type)) + " from " + who +
        ". Use try_step/try_step_flat if you meant to probe legality.");
}

static const char* roll_type_to_str(ts::RollType t) noexcept {
    switch (t) {
        case ts::RollType::COUP: return "COUP";
        case ts::RollType::REALIGNMENT: return "REALIGNMENT";
        case ts::RollType::SPACE_RACE: return "SPACE_RACE";
        case ts::RollType::WAR_EVENT: return "WAR_EVENT";
        case ts::RollType::OLYMPIC_GAMES: return "OLYMPIC_GAMES";
        case ts::RollType::SUMMIT: return "SUMMIT";
        case ts::RollType::TRAP_ESCAPE: return "TRAP_ESCAPE";
        case ts::RollType::TURN_CLEANUP: return "TURN_CLEANUP";
        default: return "NONE";
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
    // P17: a SELECT_PLAY_MODE node carries a Resolution, so this names those.
    switch (mode) {
        case static_cast<uint8_t>(ts::Resolution::EVENT): return "EVENT";
        case static_cast<uint8_t>(ts::Resolution::SPACE): return "SPACE";
        case static_cast<uint8_t>(ts::Resolution::OPS_INFLUENCE): return "OPS_INFLUENCE";
        case static_cast<uint8_t>(ts::Resolution::OPS_COUP): return "OPS_COUP";
        case static_cast<uint8_t>(ts::Resolution::OPS_REALIGN): return "OPS_REALIGN";
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
namespace {

// Named-field save format for a game's STARTING position. See from_save_dict for the scope.
nb::dict game_state_to_save_dict(const ts::GameState& state) {
    nb::dict d;
    d["format"] = "ts_save_v2";   // v1 dropped the decision stack; readers must know which

    d["victory_points"] = state.victory_points;
    d["defcon"] = state.defcon;
    d["turn"] = state.turn;
    d["action_round"] = state.action_round;
    d["us_mil_ops"] = state.us_mil_ops;
    d["ussr_mil_ops"] = state.ussr_mil_ops;
    d["us_space_track"] = state.us_space_track;
    d["ussr_space_track"] = state.ussr_space_track;
    d["phasing_player"] = static_cast<int>(state.phasing_player);
    d["current_phase"] = static_cast<int>(state.current_phase);
    d["headline_us_card"] = state.headline_us_card;
    d["headline_ussr_card"] = state.headline_ussr_card;
    d["headline_first_card"] = state.headline_first_card;
    d["headline_second_card"] = state.headline_second_card;
    d["headline_stage"] = state.headline_stage;
    d["forced_card_player"] = static_cast<int>(state.forced_card_player);
    d["forced_card_id"] = state.forced_card_id;
    d["defcon_dropped_to_2"] = state.defcon_dropped_to_2;
    d["china_card_holder"] = static_cast<int>(state.china_card_holder);
    d["china_card_playable"] = state.china_card_playable;
    d["persistent_effects"] = state.persistent_effects;
    d["rng_state"] = state.rng_state;

    d["headline_first_owner"] = static_cast<int>(state.headline_first_owner);
    d["headline_second_owner"] = static_cast<int>(state.headline_second_owner);
    d["last_die_roll"] = state.last_die_roll;
    d["last_opp_die_roll"] = state.last_opp_die_roll;

    nb::dict roll;
    roll["type"] = static_cast<int>(state.last_roll.type);
    roll["roller"] = static_cast<int>(state.last_roll.roller);
    roll["card_id"] = state.last_roll.card_id;
    roll["country_id"] = state.last_roll.country_id;
    roll["roll1"] = state.last_roll.roll1;
    roll["mod1"] = state.last_roll.mod1;
    roll["roll2"] = state.last_roll.roll2;
    roll["mod2"] = state.last_roll.mod2;
    roll["success"] = state.last_roll.success;
    roll["net_delta"] = state.last_roll.net_delta;
    d["last_roll"] = roll;

    // The decision state machine. Without this the restored state is pointing at a different
    // decision than the saved one, and every legal action and every observation differs.
    nb::list frames;
    for (size_t i = 0; i < state.ctx_stack.size(); ++i) {
        const ts::DecisionContext& c = state.ctx_stack[i];
        nb::dict f;
        f["decision_player"] = static_cast<int>(c.decision_player);
        f["decision_type"] = static_cast<int>(c.decision_type);
        f["op_mode"] = static_cast<int>(c.op_mode);
        f["pending_op_card"] = c.pending_op_card;
        f["pending_ops_value"] = c.pending_ops_value;
        f["remaining_steps"] = c.remaining_steps;
        f["max_per_country"] = c.max_per_country;
        f["allow_early_stop"] = c.allow_early_stop;
        f["resolving_card"] = c.resolving_card;
        f["timing_branch"] = c.timing_branch;
        f["suppress_op_card_event"] = c.suppress_op_card_event;
        f["event_granted_ops"] = c.event_granted_ops;
        f["pending_roll"] = static_cast<int>(c.pending_roll);
        f["roll_target"] = c.roll_target;
        f["roll_actor"] = static_cast<int>(c.roll_actor);
        f["event_stage"] = c.event_stage;
        nb::list si, vn, nc;
        for (size_t k = 0; k < c.start_influence_nodes.size(); ++k) si.append(c.start_influence_nodes[k]);
        for (size_t k = 0; k < c.visited_nodes.size(); ++k) vn.append(c.visited_nodes[k]);
        for (size_t k = 0; k < c.node_count_bits.size(); ++k) nc.append(c.node_count_bits[k]);
        f["start_influence_nodes"] = si;
        f["visited_nodes"] = vn;
        f["node_count_bits"] = nc;
        frames.append(f);
    }
    d["ctx_stack"] = frames;
    d["ctx_stack_depth"] = state.ctx_stack_depth;

    nb::list locs;
    for (int i = 0; i <= 110; ++i) locs.append(static_cast<int>(state.card_locations[i]));
    d["card_locations"] = locs;

    nb::list us_inf, ussr_inf;
    for (int i = 0; i < 84; ++i) {
        us_inf.append(state.countries[i].us_influence);
        ussr_inf.append(state.countries[i].ussr_influence);
    }
    d["us_influence"] = us_inf;
    d["ussr_influence"] = ussr_inf;
    return d;
}

template <typename T>
T save_get(const nb::dict& d, const char* key, T fallback) {
    if (!d.contains(key)) return fallback;   // missing field -> default, so old saves still load
    try { return nb::cast<T>(d[key]); } catch (...) { return fallback; }
}

ts::GameState game_state_from_save_dict(const nb::dict& d) {
    ts::GameState s{};
    ts::Engine::init_game(s, 1);   // a valid baseline; every field below overwrites it

    s.victory_points = save_get<int8_t>(d, "victory_points", s.victory_points);
    s.defcon = save_get<uint8_t>(d, "defcon", s.defcon);
    s.turn = save_get<uint8_t>(d, "turn", s.turn);
    s.action_round = save_get<uint8_t>(d, "action_round", s.action_round);
    s.us_mil_ops = save_get<uint8_t>(d, "us_mil_ops", s.us_mil_ops);
    s.ussr_mil_ops = save_get<uint8_t>(d, "ussr_mil_ops", s.ussr_mil_ops);
    s.us_space_track = save_get<uint8_t>(d, "us_space_track", s.us_space_track);
    s.ussr_space_track = save_get<uint8_t>(d, "ussr_space_track", s.ussr_space_track);
    s.phasing_player = static_cast<ts::Player>(
        save_get<int>(d, "phasing_player", static_cast<int>(s.phasing_player)));
    s.current_phase = static_cast<ts::Phase>(
        save_get<int>(d, "current_phase", static_cast<int>(s.current_phase)));
    s.headline_us_card = save_get<uint8_t>(d, "headline_us_card", s.headline_us_card);
    s.headline_ussr_card = save_get<uint8_t>(d, "headline_ussr_card", s.headline_ussr_card);
    s.headline_first_card = save_get<uint8_t>(d, "headline_first_card", s.headline_first_card);
    s.headline_second_card = save_get<uint8_t>(d, "headline_second_card", s.headline_second_card);
    s.headline_stage = save_get<uint8_t>(d, "headline_stage", s.headline_stage);
    s.forced_card_player = static_cast<ts::Player>(
        save_get<int>(d, "forced_card_player", static_cast<int>(s.forced_card_player)));
    s.forced_card_id = save_get<uint8_t>(d, "forced_card_id", s.forced_card_id);
    s.defcon_dropped_to_2 = save_get<uint8_t>(d, "defcon_dropped_to_2", s.defcon_dropped_to_2);
    s.china_card_holder = static_cast<ts::Player>(
        save_get<int>(d, "china_card_holder", static_cast<int>(s.china_card_holder)));
    s.china_card_playable = save_get<uint8_t>(d, "china_card_playable", s.china_card_playable);
    s.persistent_effects = save_get<uint64_t>(d, "persistent_effects", s.persistent_effects);
    s.rng_state = save_get<uint64_t>(d, "rng_state", s.rng_state);

    s.headline_first_owner = static_cast<ts::Player>(
        save_get<int>(d, "headline_first_owner", static_cast<int>(s.headline_first_owner)));
    s.headline_second_owner = static_cast<ts::Player>(
        save_get<int>(d, "headline_second_owner", static_cast<int>(s.headline_second_owner)));
    s.last_die_roll = save_get<uint8_t>(d, "last_die_roll", s.last_die_roll);
    s.last_opp_die_roll = save_get<uint8_t>(d, "last_opp_die_roll", s.last_opp_die_roll);

    if (d.contains("last_roll")) {
        nb::dict roll = nb::cast<nb::dict>(d["last_roll"]);
        s.last_roll.type = static_cast<ts::RollType>(
            save_get<int>(roll, "type", static_cast<int>(s.last_roll.type)));
        s.last_roll.roller = static_cast<ts::Player>(
            save_get<int>(roll, "roller", static_cast<int>(s.last_roll.roller)));
        s.last_roll.card_id = save_get<uint8_t>(roll, "card_id", s.last_roll.card_id);
        s.last_roll.country_id = save_get<uint8_t>(roll, "country_id", s.last_roll.country_id);
        s.last_roll.roll1 = save_get<uint8_t>(roll, "roll1", s.last_roll.roll1);
        s.last_roll.mod1 = save_get<int8_t>(roll, "mod1", s.last_roll.mod1);
        s.last_roll.roll2 = save_get<uint8_t>(roll, "roll2", s.last_roll.roll2);
        s.last_roll.mod2 = save_get<int8_t>(roll, "mod2", s.last_roll.mod2);
        s.last_roll.success = save_get<bool>(roll, "success", s.last_roll.success);
        s.last_roll.net_delta = save_get<int8_t>(roll, "net_delta", s.last_roll.net_delta);
    }

    if (d.contains("ctx_stack")) {
        nb::list frames = nb::cast<nb::list>(d["ctx_stack"]);
        for (size_t i = 0; i < frames.size() && i < s.ctx_stack.size(); ++i) {
            nb::dict f = nb::cast<nb::dict>(frames[i]);
            ts::DecisionContext& c = s.ctx_stack[i];
            c = ts::DecisionContext{};
            c.decision_player = static_cast<ts::Player>(save_get<int>(f, "decision_player", 0));
            c.decision_type = static_cast<ts::DecisionType>(save_get<int>(f, "decision_type", 0));
            c.op_mode = static_cast<ts::OpMode>(save_get<int>(f, "op_mode", 0));
            c.pending_op_card = save_get<uint8_t>(f, "pending_op_card", 0);
            c.pending_ops_value = save_get<uint8_t>(f, "pending_ops_value", 0);
            c.remaining_steps = save_get<uint8_t>(f, "remaining_steps", 0);
            c.max_per_country = save_get<uint8_t>(f, "max_per_country", 0);
            c.allow_early_stop = save_get<uint8_t>(f, "allow_early_stop", 0);
            c.resolving_card = save_get<uint8_t>(f, "resolving_card", 0);
            c.timing_branch = save_get<uint8_t>(f, "timing_branch", 0);
            c.suppress_op_card_event = save_get<uint8_t>(f, "suppress_op_card_event", 0);
            c.event_granted_ops = save_get<uint8_t>(f, "event_granted_ops", 0);
            c.pending_roll = static_cast<ts::RollType>(save_get<int>(f, "pending_roll", 0));
            c.roll_target = save_get<uint8_t>(f, "roll_target", 0);
            c.roll_actor = static_cast<ts::Player>(save_get<int>(f, "roll_actor", 0));
            c.event_stage = save_get<uint8_t>(f, "event_stage", 0);
            if (f.contains("start_influence_nodes")) {
                nb::list v = nb::cast<nb::list>(f["start_influence_nodes"]);
                for (size_t k = 0; k < v.size() && k < c.start_influence_nodes.size(); ++k)
                    c.start_influence_nodes[k] = nb::cast<uint64_t>(v[k]);
            }
            if (f.contains("visited_nodes")) {
                nb::list v = nb::cast<nb::list>(f["visited_nodes"]);
                for (size_t k = 0; k < v.size() && k < c.visited_nodes.size(); ++k)
                    c.visited_nodes[k] = nb::cast<uint64_t>(v[k]);
            }
            if (f.contains("node_count_bits")) {
                nb::list v = nb::cast<nb::list>(f["node_count_bits"]);
                for (size_t k = 0; k < v.size() && k < c.node_count_bits.size(); ++k)
                    c.node_count_bits[k] = nb::cast<uint64_t>(v[k]);
            }
        }
    }
    s.ctx_stack_depth = save_get<uint8_t>(d, "ctx_stack_depth", s.ctx_stack_depth);

    if (d.contains("card_locations")) {
        nb::list locs = nb::cast<nb::list>(d["card_locations"]);
        for (size_t i = 0; i < locs.size() && i <= 110; ++i) {
            s.card_locations[i] = static_cast<ts::CardLocation>(nb::cast<int>(locs[i]));
        }
    }
    if (d.contains("us_influence") && d.contains("ussr_influence")) {
        nb::list us_inf = nb::cast<nb::list>(d["us_influence"]);
        nb::list ussr_inf = nb::cast<nb::list>(d["ussr_influence"]);
        for (size_t i = 0; i < us_inf.size() && i < 84; ++i) {
            s.countries[i].us_influence = nb::cast<uint8_t>(us_inf[i]);
        }
        for (size_t i = 0; i < ussr_inf.size() && i < 84; ++i) {
            s.countries[i].ussr_influence = nb::cast<uint8_t>(ussr_inf[i]);
        }
    }
    return s;
}

}  // namespace

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
    d["headline_first_card"] = state.headline_first_card;
    d["headline_second_card"] = state.headline_second_card;
    d["headline_stage"] = state.headline_stage;
    d["forced_card_player"] = (state.forced_card_player == ts::Player::US ? "US" : (state.forced_card_player == ts::Player::USSR ? "USSR" : "NONE"));
    d["forced_card_id"] = state.forced_card_id;
    d["last_die_roll"] = state.last_die_roll;
    d["last_opp_die_roll"] = state.last_opp_die_roll;

    nb::dict die_roll;
    die_roll["type"] = roll_type_to_str(state.last_roll.type);
    die_roll["type_id"] = static_cast<uint8_t>(state.last_roll.type);
    die_roll["roller"] = (state.last_roll.roller == ts::Player::US ? "US" : (state.last_roll.roller == ts::Player::USSR ? "USSR" : "NONE"));
    die_roll["card_id"] = state.last_roll.card_id;
    die_roll["card_name"] = (state.last_roll.card_id >= 1 && state.last_roll.card_id <= 110) ? std::string(ts::CardData::get_card(state.last_roll.card_id).name) : "";
    die_roll["country_id"] = state.last_roll.country_id;
    die_roll["country_name"] = (state.last_roll.country_id < 84) ? std::string(ts::MapData::get_country(state.last_roll.country_id).name) : "";
    die_roll["roll1"] = state.last_roll.roll1;
    die_roll["mod1"] = state.last_roll.mod1;
    die_roll["total1"] = state.last_roll.roll1 + state.last_roll.mod1;
    die_roll["roll2"] = state.last_roll.roll2;
    die_roll["mod2"] = state.last_roll.mod2;
    die_roll["total2"] = state.last_roll.roll2 + state.last_roll.mod2;
    die_roll["success"] = state.last_roll.success;
    die_roll["net_delta"] = state.last_roll.net_delta;
    d["die_roll"] = die_roll;

    // 3. Space turns used
    nb::dict space_turns;
    space_turns["US"] = state.get_space_turns_used(ts::Player::US);
    space_turns["USSR"] = state.get_space_turns_used(ts::Player::USSR);
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
    CHECK_FLAG(SPACE_US_ATTEMPT_1, "SPACE_US_ATTEMPT_1");
    CHECK_FLAG(SPACE_US_ATTEMPT_2, "SPACE_US_ATTEMPT_2");
    CHECK_FLAG(SPACE_USSR_ATTEMPT_1, "SPACE_USSR_ATTEMPT_1");
    CHECK_FLAG(SPACE_USSR_ATTEMPT_2, "SPACE_USSR_ATTEMPT_2");
    CHECK_FLAG(DEFCON_SUICIDE_PROVOKED, "DEFCON_SUICIDE_PROVOKED");
    CHECK_FLAG(CMC_SUICIDE_LOSS, "CMC_SUICIDE_LOSS");
    CHECK_FLAG(EUROPE_CONTROL_WIN, "EUROPE_CONTROL_WIN");
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
            case ts::CardLocation::HAND_US_UNKNOWN:
            case ts::CardLocation::HAND_US_KNOWN: {
                us_hand.append(i);
                nb::dict ci;
                ci["id"] = i;
                ci["name"] = std::string(ts::CardData::get_card_name(i));
                ci["ops"] = ts::CardData::get_card(i).ops;
                us_cards.append(ci);
                break;
            }
            case ts::CardLocation::HAND_USSR_UNKNOWN:
            case ts::CardLocation::HAND_USSR_KNOWN: {
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

    nb::dict all_locs;
    for (uint8_t i = 1; i <= 110; ++i) {
        auto loc = state.card_locations[i];
        const char* loc_str = "UNAVAILABLE";
        switch (loc) {
            case ts::CardLocation::HEADLINE_COMMITTED: loc_str = "HEADLINE_COMMITTED"; break;
            case ts::CardLocation::UNAVAILABLE: loc_str = "UNAVAILABLE"; break;
            case ts::CardLocation::DRAW_DECK: loc_str = "DRAW_DECK"; break;
            case ts::CardLocation::HAND_US_UNKNOWN: loc_str = "HAND_US_UNKNOWN"; break;
            case ts::CardLocation::HAND_US_KNOWN: loc_str = "HAND_US_KNOWN"; break;
            case ts::CardLocation::HAND_USSR_UNKNOWN: loc_str = "HAND_USSR_UNKNOWN"; break;
            case ts::CardLocation::HAND_USSR_KNOWN: loc_str = "HAND_USSR_KNOWN"; break;
            case ts::CardLocation::DISCARD_PILE: loc_str = "DISCARD_PILE"; break;
            case ts::CardLocation::REMOVED_FROM_GAME: loc_str = "REMOVED_FROM_GAME"; break;
            case ts::CardLocation::ONGOING_EVENT: loc_str = "ONGOING_EVENT"; break;
            default: break;
        }
        all_locs[nb::cast(i)] = loc_str;
    }
    d["card_locations"] = all_locs;

    // 8. Decision Context & Stack
    const auto& ctx = state.ctx();
    nb::dict ctx_dict;
    ctx_dict["decision_player"] = (ctx.decision_player == ts::Player::US ? "US" : (ctx.decision_player == ts::Player::USSR ? "USSR" : "NONE"));
    ctx_dict["decision_type"] = static_cast<int>(ctx.decision_type);
    ctx_dict["decision_type_name"] = decision_type_to_str(ctx.decision_type);
    ctx_dict["op_mode"] = static_cast<int>(ctx.op_mode);
    ctx_dict["op_mode_name"] = op_mode_to_str(static_cast<uint8_t>(ctx.op_mode));
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
                    } else if (i == 84) {
                        label = "Done / Pass";
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
    nb::enum_<ts::RollType>(m, "RollType", nb::is_arithmetic())
        .value("NONE", ts::RollType::NONE)
        .value("COUP", ts::RollType::COUP)
        .value("REALIGNMENT", ts::RollType::REALIGNMENT)
        .value("SPACE_RACE", ts::RollType::SPACE_RACE)
        .value("WAR_EVENT", ts::RollType::WAR_EVENT)
        .value("OLYMPIC_GAMES", ts::RollType::OLYMPIC_GAMES)
        .value("SUMMIT", ts::RollType::SUMMIT)
        .value("TRAP_ESCAPE", ts::RollType::TRAP_ESCAPE)
        .value("TURN_CLEANUP", ts::RollType::TURN_CLEANUP)
        .export_values();

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
        .value("ROLL_DIE", ts::DecisionType::ROLL_DIE)
        .export_values();

    // P17: `Resolution` is what a SELECT_PLAY_MODE node carries now. `PlayMode` is kept only
    // because a few call sites still name it; it no longer describes any live decision.
    nb::enum_<ts::PlayMode>(m, "PlayMode", nb::is_arithmetic())
        .value("EVENT", ts::PlayMode::EVENT)
        .value("OPS", ts::PlayMode::OPS)
        .value("SPACE", ts::PlayMode::SPACE)
        .value("PASS", ts::PlayMode::PASS)
        .export_values();

    nb::enum_<ts::Resolution>(m, "Resolution", nb::is_arithmetic())
        .value("EVENT", ts::Resolution::EVENT)
        .value("SPACE", ts::Resolution::SPACE)
        .value("OPS_INFLUENCE", ts::Resolution::OPS_INFLUENCE)
        .value("OPS_COUP", ts::Resolution::OPS_COUP)
        .value("OPS_REALIGN", ts::Resolution::OPS_REALIGN)
        .value("COUNT", ts::Resolution::COUNT)
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

    // A hand is four locations now, so "is this card in X's hand" is a question rather than an
    // equality. Exposed so Python asks it the same way the engine does -- there is no compiler
    // here to catch a comparison that silently misses the other variant.
    m.def("reveal_hand", &ts::reveal_hand, nb::arg("state"), nb::arg("player"),
          "Mark every card that player is holding right now as public to the opponent. Cards "
          "drawn afterwards are hidden again -- knowledge attaches to cards, not to players.");
    m.def("reveal_both_hands", &ts::reveal_both_hands, nb::arg("state"),
          "Mark both hands public, as when the draw deck runs out and each side can name the "
          "other's hand as the complement of what it can see.");
    m.def("in_hand_of", &ts::in_hand_of, nb::arg("location"), nb::arg("player"),
          "True when the card is in that player's hand, known to the opponent or not.");
    m.def("known_to_opponent", &ts::known_to_opponent, nb::arg("location"),
          "True when the player who is not holding the card knows it is in that hand.");
    m.def("hand_of", &ts::hand_of, nb::arg("player"), nb::arg("known") = false,
          "The hand location for a player; hidden from the opponent unless known=True.");
    m.def("hand_holder", &ts::hand_holder, nb::arg("location"),
          "Whose hand it is, or Player.NONE when the card is not in one.");
    m.def("revealed", &ts::revealed, nb::arg("location"),
          "The same hand, marked public. Identity for anything not in a hand.");

    nb::enum_<ts::CardLocation>(m, "CardLocation", nb::is_arithmetic())
        .value("UNAVAILABLE", ts::CardLocation::UNAVAILABLE)
        .value("DRAW_DECK", ts::CardLocation::DRAW_DECK)
        .value("HAND_US_UNKNOWN", ts::CardLocation::HAND_US_UNKNOWN)
        .value("HAND_US_KNOWN", ts::CardLocation::HAND_US_KNOWN)
        .value("HAND_USSR_UNKNOWN", ts::CardLocation::HAND_USSR_UNKNOWN)
        .value("HAND_USSR_KNOWN", ts::CardLocation::HAND_USSR_KNOWN)
        .value("DISCARD_PILE", ts::CardLocation::DISCARD_PILE)
        .value("REMOVED_FROM_GAME", ts::CardLocation::REMOVED_FROM_GAME)
        .value("ONGOING_EVENT", ts::CardLocation::ONGOING_EVENT)
        .value("PEEKED_TEMP", ts::CardLocation::PEEKED_TEMP)
        // Not "HEADLINE": export_values() puts every name at module scope, where it would
        // shadow Phase.HEADLINE.
        .value("HEADLINE_COMMITTED", ts::CardLocation::HEADLINE_COMMITTED)
        .export_values();

    nb::enum_<ts::WarEra>(m, "WarEra", nb::is_arithmetic())
        .value("EARLY", ts::WarEra::EARLY)
        .value("MID", ts::WarEra::MID)
        .value("LATE", ts::WarEra::LATE)
        .export_values();

    nb::enum_<ts::Region>(m, "Region", nb::is_arithmetic())
        .value("EUROPE", ts::Region::EUROPE)
        .value("ASIA", ts::Region::ASIA)
        .value("MIDDLE_EAST", ts::Region::MIDDLE_EAST)
        .value("AFRICA", ts::Region::AFRICA)
        .value("CENTRAL_AMERICA", ts::Region::CENTRAL_AMERICA)
        .value("SOUTH_AMERICA", ts::Region::SOUTH_AMERICA)
        .value("NONE_REGION", ts::Region::NONE_REGION)
        .export_values();

    // Structs
    nb::class_<ts::MicroAction>(m, "MicroAction")
        .def(nb::init<>())
        .def("clone", [](const ts::GameState& s) -> ts::GameState { return s; })
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

    // Read-only on purpose. `GameState::get_country` returns a *copy*, so a writable field
    // here made `state.get_country(cid).us_influence = 9` a silent no-op: it mutated a
    // temporary that was discarded on the next line. That is how a test came to set up a
    // US-controlled Iran that was never actually set up. With def_ro the same line raises
    // AttributeError, and `GameState::set_country(cid, us, ussr)` is the one way to change
    // the board -- which is also the only place the engine's own invariants can be enforced.
    nb::class_<ts::CountryState>(m, "CountryState")
        .def_ro("us_influence", &ts::CountryState::us_influence)
        .def_ro("ussr_influence", &ts::CountryState::ussr_influence);

    nb::class_<ts::DecisionContext>(m, "DecisionContext")
        .def_rw("decision_player", &ts::DecisionContext::decision_player)
        .def_rw("decision_type", &ts::DecisionContext::decision_type)
        .def_rw("pending_op_card", &ts::DecisionContext::pending_op_card)
        .def_rw("pending_ops_value", &ts::DecisionContext::pending_ops_value)
        .def_rw("remaining_steps", &ts::DecisionContext::remaining_steps)
        .def_rw("max_per_country", &ts::DecisionContext::max_per_country)
        .def_rw("allow_early_stop", &ts::DecisionContext::allow_early_stop)
        .def_rw("resolving_card", &ts::DecisionContext::resolving_card)
        // Which kind of Operation the pending point decisions belong to. Needed to tell
        // apart a run of point decisions whose order carries no meaning (spreading
        // Influence) from one where it does: a coup or a realignment changes the board
        // between points, so the sequence is itself the decision.
        .def_rw("op_mode", &ts::DecisionContext::op_mode)
        // Which kind of chance node is pending, read straight off the field that holds it.
        //
        // This used to reinterpret a card slot, which was only ever a roll type when a roll was
        // what happened to be staged there. In any frame holding staged cards the same byte is a
        // *card id*, and ids 1..8 alias exactly onto COUP..TURN_CLEANUP -- so the property
        // silently answered "Asia Scoring is a coup". Where the id was above 8 it raised
        // ValueError instead, which is how it was noticed.
        //
        // A caller needs this to tell a die roll from the turn's cleanup
        // (RollType::TURN_CLEANUP), a chance node that rolls nothing and is the last moment the
        // pre-cleanup score is readable.
        .def_prop_ro(
            "pending_roll_type",
            [](const ts::DecisionContext& c) { return c.pending_roll; })
        .def_prop_ro("roll_target", [](const ts::DecisionContext& c) { return c.roll_target; })
        .def_prop_ro("roll_actor", [](const ts::DecisionContext& c) { return c.roll_actor; })
        // Which half of a two-part event is being answered -- Che's second coup,
        // De-Stalinization's placement phase.
        .def_rw("event_stage", &ts::DecisionContext::event_stage)
        // temp_cards is gone. It was exposed so a replay could steer a peeked set, which is now
        // done by setting card locations: a peeked card sits at PEEKED_TEMP, which is both what
        // the mask reads and what the observation shows.
        .def("is_visited", &ts::DecisionContext::is_visited)
        // How much this event has already moved in one country. Packed two bits per country
        // behind these, so callers see a count and not the packing.
        .def("node_count", &ts::DecisionContext::node_count)
        .def_ro_static("NODE_COUNT_MAX", &ts::DecisionContext::NODE_COUNT_MAX);

    nb::class_<ts::GameState>(m, "GameState")
        .def("clone", [](const ts::GameState& s) -> ts::GameState { return s; })
        .def(nb::init<>())
        .def_rw("victory_points", &ts::GameState::victory_points)
        .def_rw("defcon", &ts::GameState::defcon)
        .def_rw("us_mil_ops", &ts::GameState::us_mil_ops)
        .def_rw("ussr_mil_ops", &ts::GameState::ussr_mil_ops)
        .def_rw("us_space_track", &ts::GameState::us_space_track)
        .def_rw("ussr_space_track", &ts::GameState::ussr_space_track)
        .def_rw("turn", &ts::GameState::turn)
        .def_rw("action_round", &ts::GameState::action_round)
        .def_rw("defcon_dropped_to_2", &ts::GameState::defcon_dropped_to_2)
        .def_rw("phasing_player", &ts::GameState::phasing_player)
        .def_rw("headline_us_card", &ts::GameState::headline_us_card)
        .def_rw("headline_ussr_card", &ts::GameState::headline_ussr_card)
        .def_rw("headline_first_card", &ts::GameState::headline_first_card)
        .def_rw("headline_second_card", &ts::GameState::headline_second_card)
        .def_rw("headline_stage", &ts::GameState::headline_stage)
        .def_rw("current_phase", &ts::GameState::current_phase)
        .def_rw("forced_card_player", &ts::GameState::forced_card_player)
        .def_rw("forced_card_id", &ts::GameState::forced_card_id)
        .def_rw("china_card_holder", &ts::GameState::china_card_holder)
        .def_rw("china_card_playable", &ts::GameState::china_card_playable)
        .def_rw("persistent_effects", &ts::GameState::persistent_effects)
        .def_rw("ctx_stack_depth", &ts::GameState::ctx_stack_depth)
        .def_rw("rng_state", &ts::GameState::rng_state)
        .def("set_flag", &ts::GameState::set_flag)
        .def("clear_flag", &ts::GameState::clear_flag)
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
        .def("get_space_turns_used", &ts::GameState::get_space_turns_used)
        .def("set_space_turns_used", &ts::GameState::set_space_turns_used)
        .def("record_space_attempt", &ts::GameState::record_space_attempt)
        .def("get_card_location", [](const ts::GameState& s, uint8_t card_id) -> ts::CardLocation {
            if (card_id < 1 || card_id > 110) throw std::out_of_range("Card ID must be 1..110");
            return s.card_locations[card_id];
        })
        .def("set_card_location", [](ts::GameState& s, uint8_t card_id, ts::CardLocation loc) {
            if (card_id < 1 || card_id > 110) throw std::out_of_range("Card ID must be 1..110");
            s.card_locations[card_id] = loc;
        })
        .def("to_dict", &game_state_to_dict)
        .def("to_save_dict", &game_state_to_save_dict,
             "Named-field save of a position, mid-game included: scalars, RNG, card locations, board "
             "influence, the headline owners, the die-roll record AND the decision-context stack "
             "(ctx_stack + depth). Round-trips with state_from_save_dict -- verified over every "
             "position of whole games, comparing the pending decision, the legal mask and the "
             "observation. Does NOT carry action_history or turn_aggregates, which nothing reads "
             "for rules or for the observation; they are diagnostics.")
        .def("to_json", [](const ts::GameState& s) { return ts::Serializer::to_json(s); });

    // Engine class
        m.def("has_held_scoring_card", &ts::Engine::has_held_scoring_card);
    m.def("is_held_scoring_game_over", &ts::Engine::is_held_scoring_game_over);
    m.def("is_held_scoring_loss", &ts::Engine::is_held_scoring_loss);

    nb::class_<ts::StateMachine>(m, "StateMachine")
        .def_static("advance_headline_step", &ts::StateMachine::advance_headline_step)
        .def_static("advance_after_action_round", &ts::StateMachine::advance_after_action_round)
        // Exposed so the deck-exhaustion reveal can be tested where it happens: the deduction
        // belongs to dealing, not to the shuffle, and the two are only separable from here.
        .def_static("deal_cards_to_hands", &ts::StateMachine::deal_cards_to_hands)
        .def_static("reshuffle_discard_into_draw", &ts::StateMachine::reshuffle_discard_into_draw);

    nb::class_<ts::Engine>(m, "Engine")
        .def_static("init_game", &ts::Engine::init_game)
        .def_static("try_step", &ts::Engine::step, nb::arg("state"), nb::arg("action"), nb::arg("auto_advance") = false,
                    "Advance one action, returning False if the engine refuses it. For probing only.")
        .def_static("step", [](ts::GameState& state, const ts::MicroAction& action, bool auto_advance) {
            if (!ts::Engine::step(state, action, auto_advance)) {
                throw_illegal_action(state, "MicroAction(decision_type=" +
                    std::string(decision_type_to_str(action.decision_type)) +
                    ", primary_id=" + std::to_string(static_cast<int>(action.primary_id)) + ")");
            }
        }, nb::arg("state"), nb::arg("action"), nb::arg("auto_advance") = false,
           "Advance one action, raising RuntimeError if the engine refuses it.")
        .def_static("is_terminal", &ts::Engine::is_terminal)
        .def_static("get_terminal_utility", &ts::Engine::get_terminal_utility)
        .def_static("has_held_scoring_card", &ts::Engine::has_held_scoring_card)
        .def_static("is_held_scoring_game_over", &ts::Engine::is_held_scoring_game_over)
        .def_static("is_held_scoring_loss", &ts::Engine::is_held_scoring_loss)
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
        })
        .def_static("get_flat_action_mask", [](const ts::GameState& state) {
            size_t shape[1] = { 212 };
            uint8_t* data = new uint8_t[212];
            ts::ActionMask::generate_flat_mask_212(state, data);
            nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });
            return nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>(data, 1, shape, owner);
        })
        .def_static("try_step_flat", &ts::Engine::step_flat, nb::arg("state"), nb::arg("action_idx"), nb::arg("auto_advance") = false,
                    "Advance one flat action, returning False if the engine refuses it. For probing only.")
        .def_static("step_flat", [](ts::GameState& state, uint16_t action_idx, bool auto_advance) {
            if (!ts::Engine::step_flat(state, action_idx, auto_advance)) {
                throw_illegal_action(state, "flat action " + std::to_string(static_cast<int>(action_idx)));
            }
        }, nb::arg("state"), nb::arg("action_idx"), nb::arg("auto_advance") = false,
           "Advance one flat action, raising RuntimeError if the engine refuses it.")
        .def_static("auto_advance_step", &ts::Engine::auto_advance_step, nb::arg("state"), nb::arg("max_steps") = 128);

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
            // The enum, not static_cast<int>. `Region` is bound with nb::is_arithmetic(), so
            // int(), `== 0`, numpy conversion and sorting all still work -- but
            // `info["region"] == ts.Region.EUROPE` now works too, where against a bare int it
            // was silently always False.
            d["region"] = c.region;
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

    // Card Handlers & Event execution helpers
    nb::class_<ts::CardHandlers>(m, "CardHandlers")
        .def_static("trigger_event", [](ts::GameState& s, uint8_t card_id, ts::Player p, uint8_t forced_roll) {
            return ts::CardHandlers::trigger_event(s, card_id, p, forced_roll);
        }, nb::arg("state"), nb::arg("card_id"), nb::arg("player"), nb::arg("forced_roll") = 0)
        .def_static("can_trigger_event", &ts::CardHandlers::can_trigger_event,
                    nb::arg("state"), nb::arg("card_id"), nb::arg("player"))
        .def_static("handle_event_step", &ts::CardHandlers::handle_event_step);

    // Regional Status Enum & Score Summary
    nb::enum_<ts::RegionalStatus>(m, "RegionalStatus", nb::is_arithmetic())
        .value("NONE", ts::RegionalStatus::NONE)
        .value("PRESENCE", ts::RegionalStatus::PRESENCE)
        .value("DOMINATION", ts::RegionalStatus::DOMINATION)
        .value("CONTROL", ts::RegionalStatus::CONTROL);

    nb::class_<ts::RegionScoreSummary>(m, "RegionScoreSummary")
        .def_ro("us_status", &ts::RegionScoreSummary::us_status)
        .def_ro("ussr_status", &ts::RegionScoreSummary::ussr_status)
        .def_ro("us_countries", &ts::RegionScoreSummary::us_countries)
        .def_ro("ussr_countries", &ts::RegionScoreSummary::ussr_countries)
        .def_ro("us_battlegrounds", &ts::RegionScoreSummary::us_battlegrounds)
        .def_ro("ussr_battlegrounds", &ts::RegionScoreSummary::ussr_battlegrounds)
        .def_ro("us_superpower_adjacent", &ts::RegionScoreSummary::us_superpower_adjacent)
        .def_ro("ussr_superpower_adjacent", &ts::RegionScoreSummary::ussr_superpower_adjacent)
        .def_ro("us_score", &ts::RegionScoreSummary::us_score)
        .def_ro("ussr_score", &ts::RegionScoreSummary::ussr_score)
        .def_ro("net_delta", &ts::RegionScoreSummary::net_delta);

    // Operations helpers
    nb::class_<ts::Operations>(m, "Operations")
        .def_static("can_place_influence", &ts::Operations::can_place_influence);

    // Scoring helpers
    nb::class_<ts::Scoring>(m, "Scoring")
        .def_static("score_region", &ts::Scoring::score_region)
        .def_static("score_southeast_asia", &ts::Scoring::score_southeast_asia)
        .def_static("execute_final_scoring", &ts::Scoring::execute_final_scoring)
        .def_static("evaluate_military_ops", &ts::Scoring::evaluate_military_ops)
        .def_static("evaluate_region", &ts::Scoring::evaluate_region, nb::arg("state"), nb::arg("region"), nb::arg("is_final_scoring") = false)
        .def_static("get_country_control", &ts::Scoring::get_country_control)
        .def_static("is_controlled_by", &ts::Scoring::is_controlled_by)
        .def_static("compute_useful_actions_potential", &ts::Scoring::compute_useful_actions_potential);

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
    m.def("state_from_save_dict", &game_state_from_save_dict,
          "Rebuild a position from to_save_dict output, mid-game included. Missing keys take their "
          "default, so a save written before a field existed still loads, and an unknown key is "
          "ignored, so a save from a newer build opens minus what it cannot use. Restores the "
          "decision-context stack; does not restore action_history or turn_aggregates, which are "
          "diagnostics that no rule and no observation reads.");


    // Flat Action Mask & Codec exports
    m.def("get_flat_action_mask", [](const ts::GameState& state) {
        size_t shape[1] = { 212 };
        uint8_t* data = new uint8_t[212];
        ts::ActionMask::generate_flat_mask_212(state, data);
        nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });
        return nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>(data, 1, shape, owner);
    });

    m.def("decode_flat_action", &ts::ActionMask::decode_flat_action_212);
    m.def("encode_micro_action", &ts::ActionMask::encode_micro_action_212);

    // One layout, v2.3, 3824 floats -- so there is no `layout` argument to default wrongly.
    //
    // There used to be three, selected by name, defaulting to "legacy". A model reads fixed
    // slices at fixed offsets, so handing a v2.x network the 4293-float legacy vector did not
    // raise: it returned a number computed from the wrong floats. That happened four separate
    // times, once costing a published diagnostic. The argument is gone rather than re-defaulted.
    m.def("extract_observation", [](const ts::GameState& state, ts::Player perspective) {
        constexpr size_t n = ts::OBS_SIZE_V23;
        float* data = new float[n];
        ts::ObservationBufferV23 buf;
        ts::Observation::extract(state, perspective, &buf);
        std::memcpy(data, reinterpret_cast<const float*>(&buf), n * sizeof(float));
        size_t shape[1] = { n };
        nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
        return nb::ndarray<nb::numpy, float, nb::ndim<1>>(data, 1, shape, owner);
    }, nb::arg("state"), nb::arg("perspective"));

    m.attr("OBS_FLAG_STAGED_CARDS") = static_cast<uint32_t>(ts::obs_flags::STAGED_CARDS);

    m.attr("OBS_SIZE_V23") = static_cast<int>(ts::OBS_SIZE_V23);
    //: The width under a name that carries no version. There is one layout, and code saying
    //: OBS_SIZE cannot be misread as choosing between several.
    m.attr("OBS_SIZE") = static_cast<int>(ts::OBS_SIZE_V23);

    nb::class_<ts::ActionMask>(m, "ActionMask")
        .def_static("generate_flat_mask", [](const ts::GameState& state) {
            size_t shape[1] = { 212 };
            uint8_t* data = new uint8_t[212];
            ts::ActionMask::generate_flat_mask_212(state, data);
            nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });
            return nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>(data, 1, shape, owner);
        })
        .def_static("decode_flat_action", &ts::ActionMask::decode_flat_action_212)
        .def_static("encode_micro_action", &ts::ActionMask::encode_micro_action_212);

    // Vectorized Batch Runner for fast parallel self-play rollouts
    struct VectorizedBatchRunner {
        std::vector<ts::GameState> states;
        std::vector<float> obs_buffer;
        std::vector<uint8_t> mask_buffer;
        size_t num_envs;
        const size_t obs_width = ts::OBS_SIZE_V23;

        VectorizedBatchRunner(size_t n, uint64_t base_seed)
            : num_envs(n) {
            states.resize(n);
            obs_buffer.resize(n * obs_width);
            mask_buffer.resize(n * 212);
            for (size_t i = 0; i < n; ++i) {
                ts::StateMachine::init_new_game(states[i], base_seed + i * 10007 + 1);
            }
            refresh_all();
        }

        void reset_game(size_t idx, uint64_t seed) {
            if (idx >= num_envs) return;
            ts::StateMachine::init_new_game(states[idx], seed);
            refresh_single(idx);
        }

        void refresh_single(size_t idx) {
            while (states[idx].current_phase != ts::Phase::GAME_OVER &&
                   states[idx].victory_points < 20 && states[idx].victory_points > -20 &&
                   states[idx].ctx().decision_player == ts::Player::NONE &&
                   states[idx].ctx().decision_type == ts::DecisionType::ROLL_DIE) {
                ts::MicroAction chance_ma{ts::DecisionType::ROLL_DIE, 0, 0, 0};
                // Must match step_flat_all's copy of this drain: without the check a refused
                // ROLL_DIE leaves the loop condition unchanged and this spins forever. Breaking
                // leaves the chance node in place, where the empty mask reports it loudly.
                if (!ts::StateMachine::step(states[idx], chance_ma)) break;
            }
            ts::Player p = (states[idx].ctx().decision_player != ts::Player::NONE)
                ? states[idx].ctx().decision_player : states[idx].phasing_player;
            ts::ObservationBufferV23 ob;
            ts::Observation::extract(states[idx], p, &ob);
            std::memcpy(&obs_buffer[idx * obs_width], reinterpret_cast<const float*>(&ob),
                        obs_width * sizeof(float));
            ts::ActionMask::generate_flat_mask_212(states[idx], &mask_buffer[idx * 212]);
        }

        void refresh_all() {
#pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(num_envs); ++i) {
                refresh_single(static_cast<size_t>(i));
            }
        }

        std::vector<int> step_flat_all(const std::vector<uint16_t>& actions, bool auto_advance = false) {
            std::vector<int> results(num_envs, 0);
            const size_t act_count = actions.size();
#pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(num_envs); ++i) {
                if (static_cast<size_t>(i) >= act_count) continue;
                if (states[i].current_phase == ts::Phase::GAME_OVER ||
                    states[i].victory_points >= 20 ||
                    states[i].victory_points <= -20) {
                    results[i] = 2; // Terminal
                    continue;
                }
                ts::MicroAction ma = ts::ActionMask::decode_flat_action_212(states[i], actions[i]);
                bool ok = ts::StateMachine::step(states[i], ma);
                if (ok) {
                    if (auto_advance) {
                        ts::Engine::auto_advance_step(states[i]);
                    } else {
                        while (states[i].current_phase != ts::Phase::GAME_OVER &&
                               states[i].victory_points < 20 && states[i].victory_points > -20 &&
                               states[i].ctx().decision_player == ts::Player::NONE &&
                               states[i].ctx().decision_type == ts::DecisionType::ROLL_DIE) {
                            ts::MicroAction chance_ma{ts::DecisionType::ROLL_DIE, 0, 0, 0};
                            if (!ts::StateMachine::step(states[i], chance_ma)) break;
                        }
                    }
                }
                results[i] = ok ? 1 : 0;
                refresh_single(static_cast<size_t>(i));
            }
            return results;
        }

        nb::ndarray<nb::numpy, float, nb::ndim<2>> get_observations() {
            size_t shape[2] = { num_envs, obs_width };
            return nb::ndarray<nb::numpy, float, nb::ndim<2>>(obs_buffer.data(), 2, shape);
        }

        nb::ndarray<nb::numpy, uint8_t, nb::ndim<2>> get_action_masks() {
            size_t shape[2] = { num_envs, 212 };
            return nb::ndarray<nb::numpy, uint8_t, nb::ndim<2>>(mask_buffer.data(), 2, shape);
        }

        std::vector<int8_t> get_decision_players() const {
            std::vector<int8_t> res(num_envs);
            for (size_t i = 0; i < num_envs; ++i) {
                ts::Player p = (states[i].ctx().decision_player != ts::Player::NONE)
                    ? states[i].ctx().decision_player : states[i].phasing_player;
                res[i] = static_cast<int8_t>(p);
            }
            return res;
        }

        std::vector<bool> get_terminals() const {
            std::vector<bool> res(num_envs);
            for (size_t i = 0; i < num_envs; ++i) {
                res[i] = (states[i].current_phase == ts::Phase::GAME_OVER ||
                          states[i].victory_points >= 20 ||
                          states[i].victory_points <= -20);
            }
            return res;
        }

        std::vector<float> get_terminal_utilities() const {
            std::vector<float> res(num_envs, 0.0f);
            for (size_t i = 0; i < num_envs; ++i) {
                if (states[i].victory_points >= 20) res[i] = 1.0f;
                else if (states[i].victory_points <= -20) res[i] = -1.0f;
                else if (states[i].victory_points > 0) res[i] = 1.0f;
                else if (states[i].victory_points < 0) res[i] = -1.0f;
            }
            return res;
        }

        std::vector<int8_t> get_victory_points() const {
            std::vector<int8_t> res(num_envs);
            for (size_t i = 0; i < num_envs; ++i) {
                res[i] = states[i].victory_points;
            }
            return res;
        }

        // The opponent's hand as a 110-wide indicator, per env. Nothing in the tree calls this:
        // its consumer was the privileged oracle critic, which was removed with ColdWarNetV4.
        // Kept deliberately -- it is read-only, costs nothing unless called, and is the one part
        // of that feature that never needs migrating when the observation changes, and a future
        // privileged critic will want it back.
        std::vector<float> get_opponent_hands(const std::vector<int8_t>& acting_players) const {
            std::vector<float> res(num_envs * 110, 0.0f);
            for (size_t i = 0; i < num_envs; ++i) {
                ts::Player active_p = (i < acting_players.size()) ? static_cast<ts::Player>(acting_players[i]) : ts::Player::US;
                const ts::Player opp_p = (active_p == ts::Player::US) ? ts::Player::USSR : ts::Player::US;
                for (size_t c = 1; c <= 110; ++c) {
                    if (ts::in_hand_of(states[i].card_locations[c], opp_p)) {
                        res[i * 110 + (c - 1)] = 1.0f;
                    }
                }
            }
            return res;
        }

        std::vector<int8_t> get_turns() const {
            std::vector<int8_t> res(num_envs);
            for (size_t i = 0; i < num_envs; ++i) {
                res[i] = static_cast<int8_t>(states[i].turn);
            }
            return res;
        }
    };

    nb::class_<VectorizedBatchRunner>(m, "VectorizedBatchRunner")
        .def(nb::init<size_t, uint64_t>(), nb::arg("num_envs"), nb::arg("base_seed") = 12345)
        .def_ro("obs_width", &VectorizedBatchRunner::obs_width)
        .def("reset_game", &VectorizedBatchRunner::reset_game)
        .def("refresh_all", &VectorizedBatchRunner::refresh_all)
        .def("step_flat_all", &VectorizedBatchRunner::step_flat_all, nb::arg("actions"), nb::arg("auto_advance") = false)
        .def("get_observations", &VectorizedBatchRunner::get_observations, nb::rv_policy::reference_internal)
        .def("get_action_masks", &VectorizedBatchRunner::get_action_masks, nb::rv_policy::reference_internal)
        .def("get_decision_players", &VectorizedBatchRunner::get_decision_players)
        .def("get_terminals", &VectorizedBatchRunner::get_terminals)
        .def("get_terminal_utilities", &VectorizedBatchRunner::get_terminal_utilities)
        .def("get_victory_points", &VectorizedBatchRunner::get_victory_points)
        .def("get_opponent_hands", &VectorizedBatchRunner::get_opponent_hands)
        .def("get_turns", &VectorizedBatchRunner::get_turns)
        .def("get_state", [](VectorizedBatchRunner& self, size_t idx) -> ts::GameState& { return self.states.at(idx); }, nb::rv_policy::reference_internal)
        // Write a whole GameState into one slot, so an environment can be started from a
        // saved mid-game position instead of a fresh deal. GameState is trivially
        // copyable, so this is a plain struct assignment. get_state hands back a mutable
        // reference, but only the top DecisionContext is exposed -- not the ctx_stack --
        // so copying field by field from Python cannot restore a position captured during
        // nested card resolution, and would corrupt it silently.
        .def("set_state", [](VectorizedBatchRunner& self, size_t idx, const ts::GameState& s) {
            self.states.at(idx) = s;
        }, nb::arg("idx"), nb::arg("state"))
        .def("compute_useful_actions_potentials", [](VectorizedBatchRunner& self, const std::vector<int8_t>& acting_players) {
            size_t n = self.num_envs;
            std::vector<float> potentials(n);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                // Strategic potential Phi(s) strictly defined from US perspective
                potentials[i] = ts::Scoring::compute_useful_actions_potential(self.states[i], ts::Player::US);
            }
            return potentials;
        });

    // Effect Bits module constants
    auto eb = m.def_submodule("EffectBits");
    eb.attr("NATO_ACTIVE") = ts::effect_bits::NATO_ACTIVE;
    eb.attr("NATO_CANCELED_FRANCE") = ts::effect_bits::NATO_CANCELED_FRANCE;
    eb.attr("NATO_CANCELED_WEST_GERMANY") = ts::effect_bits::NATO_CANCELED_WEST_GERMANY;
    eb.attr("MARSHALL_PLAN_PLAYED") = ts::effect_bits::MARSHALL_PLAN_PLAYED;
    eb.attr("WARSAW_PACT_PLAYED") = ts::effect_bits::WARSAW_PACT_PLAYED;
    eb.attr("US_JAPAN_PACT_ACTIVE") = ts::effect_bits::US_JAPAN_PACT_ACTIVE;
    eb.attr("CONTAINMENT_ACTIVE") = ts::effect_bits::CONTAINMENT_ACTIVE;
    eb.attr("PURGE_US_ACTIVE") = ts::effect_bits::PURGE_US_ACTIVE;
    eb.attr("PURGE_USSR_ACTIVE") = ts::effect_bits::PURGE_USSR_ACTIVE;
    eb.attr("VIETNAM_REVOLTS_ACTIVE") = ts::effect_bits::VIETNAM_REVOLTS_ACTIVE;
    eb.attr("FORMOSAN_RESOLUTION_ACTIVE") = ts::effect_bits::FORMOSAN_RESOLUTION_ACTIVE;
    eb.attr("CMC_ACTIVE_US") = ts::effect_bits::CMC_ACTIVE_US;
    eb.attr("CMC_ACTIVE_USSR") = ts::effect_bits::CMC_ACTIVE_USSR;
    eb.attr("NUCLEAR_SUBS_ACTIVE") = ts::effect_bits::NUCLEAR_SUBS_ACTIVE;
    eb.attr("QUAGMIRE_ACTIVE") = ts::effect_bits::QUAGMIRE_ACTIVE;
    eb.attr("BEAR_TRAP_ACTIVE") = ts::effect_bits::BEAR_TRAP_ACTIVE;
    eb.attr("SALT_ACTIVE") = ts::effect_bits::SALT_ACTIVE;
    eb.attr("WE_WILL_BURY_YOU_PENDING") = ts::effect_bits::WE_WILL_BURY_YOU_PENDING;
    eb.attr("BREZHNEV_DOCTRINE_ACTIVE") = ts::effect_bits::BREZHNEV_DOCTRINE_ACTIVE;
    eb.attr("FLOWER_POWER_ACTIVE") = ts::effect_bits::FLOWER_POWER_ACTIVE;
    eb.attr("U2_INCIDENT_ACTIVE") = ts::effect_bits::U2_INCIDENT_ACTIVE;
    eb.attr("SHUTTLE_DIPLOMACY_ACTIVE") = ts::effect_bits::SHUTTLE_DIPLOMACY_ACTIVE;
    eb.attr("DEATH_SQUADS_US") = ts::effect_bits::DEATH_SQUADS_US;
    eb.attr("DEATH_SQUADS_USSR") = ts::effect_bits::DEATH_SQUADS_USSR;
    eb.attr("CAMP_DAVID_PLAYED") = ts::effect_bits::CAMP_DAVID_PLAYED;
    eb.attr("IRON_LADY_PLAYED") = ts::effect_bits::IRON_LADY_PLAYED;
    eb.attr("NORTH_SEA_OIL_PLAYED") = ts::effect_bits::NORTH_SEA_OIL_PLAYED;
    eb.attr("NORTH_SEA_OIL_ACTIVE") = ts::effect_bits::NORTH_SEA_OIL_ACTIVE;
    eb.attr("THE_REFORMER_PLAYED") = ts::effect_bits::THE_REFORMER_PLAYED;
    eb.attr("IRAN_CONTRA_ACTIVE") = ts::effect_bits::IRAN_CONTRA_ACTIVE;
    eb.attr("EVIL_EMPIRE_PLAYED") = ts::effect_bits::EVIL_EMPIRE_PLAYED;
    eb.attr("ALDRICH_AMES_ACTIVE") = ts::effect_bits::ALDRICH_AMES_ACTIVE;
    eb.attr("JOHN_PAUL_II_PLAYED") = ts::effect_bits::JOHN_PAUL_II_PLAYED;
    eb.attr("NORAD_ACTIVE") = ts::effect_bits::NORAD_ACTIVE;
    eb.attr("YURI_AND_SAMANTHA_ACTIVE") = ts::effect_bits::YURI_AND_SAMANTHA_ACTIVE;
    eb.attr("AWACS_PLAYED") = ts::effect_bits::AWACS_PLAYED;
    eb.attr("IRANIAN_HOSTAGE_CRISIS_PLAY") = ts::effect_bits::IRANIAN_HOSTAGE_CRISIS_PLAY;
    eb.attr("WILLY_BRANDT_PLAYED") = ts::effect_bits::WILLY_BRANDT_PLAYED;
    eb.attr("TEAR_DOWN_THIS_WALL_PLAYED") = ts::effect_bits::TEAR_DOWN_THIS_WALL_PLAYED;
    eb.attr("CHERNOBYL_ACTIVE") = ts::effect_bits::CHERNOBYL_ACTIVE;
    eb.attr("CHERNOBYL_REGION_SHIFT") = ts::effect_bits::CHERNOBYL_REGION_SHIFT;
    eb.attr("CHERNOBYL_REGION_MASK") = ts::effect_bits::CHERNOBYL_REGION_MASK;
    eb.attr("SPACE_US_ATTEMPT_1") = ts::effect_bits::SPACE_US_ATTEMPT_1;
    eb.attr("SPACE_US_ATTEMPT_2") = ts::effect_bits::SPACE_US_ATTEMPT_2;
    eb.attr("SPACE_USSR_ATTEMPT_1") = ts::effect_bits::SPACE_USSR_ATTEMPT_1;
    eb.attr("SPACE_USSR_ATTEMPT_2") = ts::effect_bits::SPACE_USSR_ATTEMPT_2;
    eb.attr("DEFCON_SUICIDE_PROVOKED") = ts::effect_bits::DEFCON_SUICIDE_PROVOKED;
    eb.attr("CMC_SUICIDE_LOSS") = ts::effect_bits::CMC_SUICIDE_LOSS;
    eb.attr("EUROPE_CONTROL_WIN") = ts::effect_bits::EUROPE_CONTROL_WIN;
}
