#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
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
#include "ts/cached_state.hpp"
#include "ts/map_data.hpp"
#include "ts/card_data.hpp"
#include "ts/card_handlers.hpp"
#include "ts/scoring.hpp"
#include "ts/space_race.hpp"
#include "ts/ops.hpp"
#include "ts/action_mask.hpp"
#include "ts/canonical.hpp"
#include "ts/state_machine.hpp"
#include "ts/serialization.hpp"
#include "ts/engine.hpp"
#include "ts/invariant.hpp"
#include "state_json.hpp"
#include "selftest.hpp"
#include "batched_search.hpp"
#include "belief.hpp"
#include "rollout_batch.hpp"
#include "playout_labels.hpp"
#include "keyed_chance.hpp"
#include <nanobind/stl/pair.h>
#include <nanobind/stl/optional.h>

#include "gomp_parallel.hpp"  // gomp_parallel_for: the one OpenMP pool, shared with torch

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

// ---- the state as Python objects ------------------------------------------------------------
//
// The display state and the save are written once, as JSON trees, in state_json.cpp -- shared
// with the WebAssembly build the browser workbench runs, so the two cannot drift. Here the tree
// is only converted to and from Python objects.
namespace {

namespace sj = ts::state_json;

nb::object to_py(const sj::Value& v) {
    switch (v.kind()) {
        case sj::Value::Kind::Null: return nb::none();
        case sj::Value::Kind::Bool: return nb::bool_(v.as_bool());
        case sj::Value::Kind::Int: return nb::cast(v.as_int());
        case sj::Value::Kind::UInt: return nb::cast(v.as_uint());
        case sj::Value::Kind::Double: return nb::float_(v.as_double());
        case sj::Value::Kind::String: return nb::str(v.as_string().c_str(), v.as_string().size());
        case sj::Value::Kind::Array: {
            nb::list out;
            for (const auto& item : v.items()) out.append(to_py(item));
            return out;
        }
        case sj::Value::Kind::Object: {
            nb::dict out;
            for (const auto& [key, item] : v.members()) out[nb::str(key.c_str(), key.size())] = to_py(item);
            return out;
        }
    }
    return nb::none();
}

// Python -> tree. A value the tree has no kind for, or an int past 64 bits, becomes null, which
// the loader treats as a wrong-type scalar: its default, as the old nb::cast try/catch did.
sj::Value from_py(nb::handle h) {
    PyObject* o = h.ptr();
    if (o == Py_None) return sj::Value();
    if (PyBool_Check(o)) return sj::Value(o == Py_True);
    if (PyLong_Check(o)) {
        int overflow = 0;
        const long long s = PyLong_AsLongLongAndOverflow(o, &overflow);
        if (overflow == 0 && !(s == -1 && PyErr_Occurred())) return sj::Value(static_cast<int64_t>(s));
        PyErr_Clear();
        if (overflow > 0) {
            const unsigned long long u = PyLong_AsUnsignedLongLong(o);
            if (!(u == static_cast<unsigned long long>(-1) && PyErr_Occurred())) return sj::Value(static_cast<uint64_t>(u));
            PyErr_Clear();
        }
        return sj::Value();
    }
    if (PyFloat_Check(o)) return sj::Value(PyFloat_AsDouble(o));
    if (PyUnicode_Check(o)) return sj::Value(std::string(nb::cast<std::string_view>(h)));
    if (PyDict_Check(o)) {
        sj::Value out = sj::Value::object();
        for (auto [k, v] : nb::borrow<nb::dict>(h)) {
            if (!PyUnicode_Check(k.ptr())) continue;
            out.set(std::string(nb::cast<std::string_view>(k)), from_py(v));
        }
        return out;
    }
    if (PyList_Check(o) || PyTuple_Check(o)) {
        sj::Value out = sj::Value::array();
        for (nb::handle item : nb::borrow<nb::sequence>(h)) out.push(from_py(item));
        return out;
    }
    return sj::Value();
}

// Named-field save format of a position. See state_json.hpp for the scope.
nb::dict game_state_to_save_dict(const ts::GameState& state) {
    return nb::borrow<nb::dict>(to_py(sj::save(state)));
}

ts::GameState game_state_from_save_dict(const nb::dict& d) {
    ts::GameState s{};
    std::string error;
    if (!sj::load_save(from_py(d), s, &error)) throw std::invalid_argument("bad save: " + error);
    return s;
}

std::string game_state_to_save_json(const ts::GameState& state) { return sj::save(state).dump(); }

ts::GameState game_state_from_save_json(const std::string& text) {
    sj::Value v;
    std::string error;
    if (!sj::Value::parse(text, v, &error)) throw std::invalid_argument("save is not JSON: " + error);
    ts::GameState s{};
    if (!sj::load_save(v, s, &error)) throw std::invalid_argument("bad save: " + error);
    return s;
}

// The flat legal mask as an owned numpy array. One implementation behind the three spellings
// Python has for it (ts.get_flat_action_mask, Engine.get_flat_action_mask,
// ActionMask.generate_flat_mask).
// The action view a Python call names: merged_influence is P23's E4.1, macro_influence P42's; at most one.
ts::ActionView view_of(bool merged_influence, bool macro_influence) {
    if (merged_influence && macro_influence) {
        throw std::invalid_argument("merged_influence and macro_influence are separate views; pass at most one");
    }
    return merged_influence ? ts::ActionView::MERGED : macro_influence ? ts::ActionView::MACRO : ts::ActionView::E4;
}

nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>> flat_mask_array(const ts::GameState& state,
                                                             bool merged_influence, bool macro_influence) {
    size_t shape[1] = { ts::FLAT_ACTION_SPACE_SIZE };
    const ts::ActionView view = view_of(merged_influence, macro_influence);
    uint8_t* data = new uint8_t[ts::FLAT_ACTION_SPACE_SIZE];
    ts::Engine::get_flat_action_mask(state, data, view);
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });
    return nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>(data, 1, shape, owner);
}

// Python's decode/encode, refusing what the engine cannot represent instead of handing back a
// NONE action or a -1 that a numpy index would read as the last slot.
ts::MicroAction checked_decode(const ts::GameState& state, int64_t action_idx) {
    if (action_idx < 0 || action_idx >= static_cast<int64_t>(ts::FLAT_ACTION_SPACE_SIZE)) {
        throw nb::value_error(("flat action index out of range: " + std::to_string(action_idx)).c_str());
    }
    return ts::ActionMask::decode_flat_action_212(state, static_cast<uint16_t>(action_idx));
}

int checked_encode(const ts::GameState& state, const ts::MicroAction& action) {
    const int16_t flat = ts::ActionMask::encode_micro_action_212(state, action);
    if (flat < 0) {
        throw nb::value_error(("no flat action encodes this MicroAction (decision type " +
                               std::to_string(static_cast<int>(action.decision_type)) + ", primary " +
                               std::to_string(static_cast<int>(action.primary_id)) + ")").c_str());
    }
    return flat;
}

// A NaN from the network has no order: the masked argmax would quietly take the lowest legal
// index and the playouts would run on garbage. Refused where the floats come in. -inf is allowed
// (a masked logit).
void require_no_nan(const float* data, size_t n, const char* what) {
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(data[i])) {
            throw nb::value_error((std::string(what) + ": NaN at index " + std::to_string(i)).c_str());
        }
    }
}

}  // namespace

nb::dict game_state_to_dict(const ts::GameState& state) {
    return nb::borrow<nb::dict>(to_py(sj::display_state(state)));
}

std::string game_state_to_display_json(const ts::GameState& state) {
    return sj::display_state(state).dump();
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

    // P17: `Resolution` is what a SELECT_PLAY_MODE node carries. The old `PlayMode` enum was
    // removed 2026-09-21 rather than left exported: its values had silently changed meaning --
    // `PlayMode.OPS == 1` is `Resolution::SPACE` -- and it produced three separate mislabelling
    // bugs while it sat here unused.
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

    // The legal-mask cache of the search tree and the batch runner (ts::CachedState): with checks
    // on, every read of a cached mask regenerates it and aborts the process if the two differ.
    // tests/conftest.py turns them on for the whole suite.
    m.def("anomalies_reported", &ts::anomalies_reported,
          "How many anomalies (ts::report_anomaly: a state that should not arise, logged to "
          "stderr) this thread has reported. A caller that must not continue past one reads it "
          "before and after a step.");
    m.def("set_mask_cache_checks", &ts::set_mask_cache_checks, nb::arg("on"),
          "Check every cached legal mask against a fresh one (aborting on a difference).");
    m.def("mask_cache_checks", &ts::mask_cache_checks);

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
        // The rule bits each seat plays (obs_features::RULES; GameState::seat_rules). A harness
        // seats the model it lets decide at `player`; a seat no one seated plays feature set 0's.
        .def("seat_obs_features", [](ts::GameState& s, ts::Player player, uint32_t features) {
            if (player != ts::Player::US && player != ts::Player::USSR)
                throw std::invalid_argument("seat_obs_features: a seat is US or USSR");
            if (features & ~ts::obs_features::ALL)
                throw std::invalid_argument("seat_obs_features: unknown observation feature bits");
            ts::seat_obs_features(s, player, features);
        }, nb::arg("player"), nb::arg("features"),
           "Seat a model of feature set `features` at `player`: that seat plays the rule bits the set "
           "names and the pre-fix rule of each it lacks. Observation bits are ignored.")
        // P43 V3 (CANONICAL_LEARNED): the run-priority handshake. A harness that lets a V3 network decide asks
        // at every decision and, where the run wants its order, gives the network's own priority (its logits over
        // the 84 countries, ranked highest first) before stepping.
        .def("run_priority_wanted", [](const ts::GameState& s) { return ts::canonical::priority_wanted(s); },
             "P43 V3: whether the pending decision opens a CANONICAL_LEARNED run whose order has not been given.")
        .def("set_run_priority", [](ts::GameState& s, const std::vector<int>& ranks) {
            if (ranks.size() != 84) throw std::invalid_argument("set_run_priority: 84 ranks, one per country");
            uint8_t r[84];
            for (int c = 0; c < 84; ++c) {
                if (ranks[c] < 0 || ranks[c] > 83) throw std::invalid_argument("set_run_priority: ranks are 0..83");
                r[c] = static_cast<uint8_t>(ranks[c]);
            }
            if (!ts::canonical::set_run_priority(s, r))
                throw std::invalid_argument("set_run_priority: no CANONICAL_LEARNED run pending, or not a permutation");
        }, nb::arg("ranks"),
           "P43 V3: give the pending run its order -- ranks[c] is country c's rank (0 = highest priority).")
        .def("seat_plays_rule", [](const ts::GameState& s, ts::Player player, uint32_t rule) {
            return ts::seat_plays_rule(s, player, rule);
        }, nb::arg("player"), nb::arg("rule"),
           "Whether the seat `player` plays rule bit `rule` (the fix) rather than the rule before it.")
        .def("set_flag", &ts::GameState::set_flag)
        .def("clear_flag", &ts::GameState::clear_flag)
        // reference_internal: the context lives inside the state, so the state must outlive it.
        // With plain `reference`, `state.clone().ctx().decision_type` read freed memory -- the
        // temporary state was gone before the field was read (found by ASan once
        // VectorizedBatchRunner.get_state started returning a copy).
        .def("ctx", [](ts::GameState& s) -> ts::DecisionContext& { return s.ctx(); }, nb::rv_policy::reference_internal)
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
        .def("set_forced_deal",
             [](ts::GameState& s, ts::Player p, const std::vector<uint8_t>& cards) {
                 const bool us = (p == ts::Player::US);
                 uint8_t* q = us ? s.forced_deal_us : s.forced_deal_ussr;
                 uint8_t& n = us ? s.forced_deal_us_count : s.forced_deal_ussr_count;
                 uint8_t& pos = us ? s.forced_deal_us_pos : s.forced_deal_ussr_pos;
                 if (cards.size() > ts::GameState::FORCED_DEAL_MAX) {
                     throw std::invalid_argument(
                         "a deal gives at most " +
                         std::to_string(ts::GameState::FORCED_DEAL_MAX) +
                         " cards to a player; got " + std::to_string(cards.size()));
                 }
                 for (size_t i = 0; i < cards.size(); ++i) {
                     if (cards[i] < 1 || cards[i] > 110) {
                         throw std::invalid_argument(
                             "forced deal card ids are 1..110; got " +
                             std::to_string(cards[i]));
                     }
                     q[i] = cards[i];
                 }
                 n = static_cast<uint8_t>(cards.size());
                 pos = 0;
             },
             nb::arg("player"), nb::arg("cards"),
             "Name the cards the NEXT deal gives this player, for re-driving a recording.\n\n"
             "A die roll can already be forced through its own action; a deal cannot, because a\n"
             "deal is not a decision. Without this a replay is not self-contained: re-driving it\n"
             "on an engine whose shuffle or draw order changed diverges at the first deal,\n"
             "silently, producing a different game rather than an error.\n\n"
             "Per player on purpose -- what is recorded is WHICH CARDS each side received, so a\n"
             "recording survives a change to the deal algorithm and not merely to the shuffle.\n"
             "Consumed by one deal and then cleared; a source driving several refills before\n"
             "each. Empty in normal play, which costs one comparison per draw. A named card that\n"
             "is not in the draw deck when the deal reaches it is reported as an anomaly and that\n"
             "draw falls back to the RNG, rather than being silently mis-dealt.")
        .def("get_forced_deal_remaining",
             [](const ts::GameState& s, ts::Player p) {
                 const bool us = (p == ts::Player::US);
                 const uint8_t n = us ? s.forced_deal_us_count : s.forced_deal_ussr_count;
                 const uint8_t pos = us ? s.forced_deal_us_pos : s.forced_deal_ussr_pos;
                 return static_cast<int>(n) - static_cast<int>(pos);
             },
             nb::arg("player"),
             "How many named cards this player's queue still holds. 0 in normal play.")
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
        .def("to_save_json", &game_state_to_save_json,
             "The save as canonical JSON text: keys sorted, compact -- the same bytes as "
             "json.dumps(self.to_save_dict(), sort_keys=True, separators=(',', ':')). Written in "
             "C++ and shared with the WebAssembly engine, so a position saved by either opens in "
             "the other.")
        .def("to_display_json", &game_state_to_display_json,
             "to_dict() as JSON text, as the browser workbench's engine produces it.")
        .def("to_json", [](const ts::GameState& s) { return ts::Serializer::to_json(s); })
        .def("raw_bytes", [](const ts::GameState& s) {
            return nb::bytes(reinterpret_cast<const char*>(&s), sizeof(ts::GameState));
        }, "The GameState's memory, for exact equality tests. It is trivially copyable, so two "
           "states reached from copies of one origin by the same steps compare equal byte for byte "
           "(padding included) -- which is how P23 checks a composed step against the two E4 steps.");

    // Engine class

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
        .def_static("get_flat_action_mask", &flat_mask_array,
           nb::arg("state"), nb::arg("merged_influence") = false, nb::arg("macro_influence") = false,
           "The flat legal mask. merged_influence=True is the E4.1 view (P23): at an op-choice "
           "node, NODE slot X means 'influence, first point in X'. macro_influence=True is P42's: at a "
           "POINT_NODE, MACRO_NODE + c means 'place in c until control', offered where it differs from one point.")
        .def_static("try_step_flat", [](ts::GameState& state, uint16_t action_idx, bool auto_advance,
                                        bool merged_influence, bool macro_influence) {
            return ts::Engine::step_flat(state, action_idx, auto_advance, view_of(merged_influence, macro_influence));
        }, nb::arg("state"), nb::arg("action_idx"), nb::arg("auto_advance") = false,
           nb::arg("merged_influence") = false, nb::arg("macro_influence") = false,
           "Advance one flat action, returning False if the engine refuses it. For probing only.")
        .def_static("step_flat", [](ts::GameState& state, uint16_t action_idx, bool auto_advance,
                                    bool merged_influence, bool macro_influence) {
            if (!ts::Engine::step_flat(state, action_idx, auto_advance, view_of(merged_influence, macro_influence))) {
                throw_illegal_action(state, "flat action " + std::to_string(static_cast<int>(action_idx)));
            }
        }, nb::arg("state"), nb::arg("action_idx"), nb::arg("auto_advance") = false,
           nb::arg("merged_influence") = false, nb::arg("macro_influence") = false,
           "Advance one flat action, raising RuntimeError if the engine refuses it. "
           "merged_influence=True applies a composed E4.1 action as the two E4 steps it names; "
           "macro_influence=True applies an until-control slot as the one-point steps that define it.")
        .def_static("auto_advance_step",
                    nb::overload_cast<ts::GameState&, size_t>(&ts::Engine::auto_advance_step),
                    nb::arg("state"), nb::arg("max_steps") = 128);

    // Map Metadata helpers
    nb::class_<ts::MapData>(m, "MapData")
        .def_static("get_country_name", [](uint8_t id) { return std::string(ts::MapData::get_country_name(id)); })
        .def_static("get_country_by_name", [](const std::string& name) { return ts::MapData::get_country_by_name(name); })
        .def_static("placement_rank", [](uint8_t id) {
            if (id >= 84) throw std::out_of_range("Country ID must be 0..83");
            return ts::MapData::placement_rank(id);
        }, "P43: the important-last rank a canonical placement run follows (CANONICAL_LAST; CANONICAL_FIRST "
           "is 83 minus it), a permutation of 0..83.")
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
        .def_static("trigger_event", &ts::CardHandlers::trigger_event,
                    nb::arg("state"), nb::arg("card_id"), nb::arg("player"))
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
        .def_ro("net_delta", &ts::RegionScoreSummary::net_delta)
        .def_ro("ussr_countries_scored", &ts::RegionScoreSummary::ussr_countries_scored)
        .def_ro("ussr_battlegrounds_scored", &ts::RegionScoreSummary::ussr_battlegrounds_scored)
        .def_ro("ussr_superpower_adjacent_scored", &ts::RegionScoreSummary::ussr_superpower_adjacent_scored)
        .def_ro("shuttle_diplomacy_applied", &ts::RegionScoreSummary::shuttle_diplomacy_applied);

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
    m.def("selftest_digest", [](int games) {
              uint32_t steps = 0;
              const uint32_t d = ts::selftest::digest(games, &steps);
              return std::make_pair(d, steps);
          }, nb::arg("games"),
          "(digest, steps) over `games` whole games (bindings/selftest.hpp). The WebAssembly "
          "build exports the same function, so equal numbers mean the two engines play the same "
          "games, position, observation and legal mask alike.");

    // ---- P34's playout labeller (bindings/playout_labels.hpp; native only) ----
    m.def("keyed_key", [](const ts::GameState& s, uint64_t world_key, bool keyed2) {
              ts::GameState out = s;
              ts::keyed::key(out, world_key, keyed2);
              return out;
          }, nb::arg("state"), nb::arg("world_key"), nb::arg("keyed2") = false,
          "A copy of `state` keyed for its next engine step (bindings/keyed_chance.hpp): rng_state "
          "set to hash(world_key, situation) and, at the turn's end, the next deal forced from the "
          "world's deck order. What the batches' keyed=True does before every step (keyed2=True: "
          "keyed2's packing); a reference playout calls it the same way.");
    m.def("state_size", []() { return sizeof(ts::GameState); },
          "Bytes of one GameState: the row width of raw_bytes() arrays.");
    m.def("state_from_raw_bytes", [](nb::bytes b) {
              if (b.size() != sizeof(ts::GameState))
                  throw std::invalid_argument("raw state: expected " + std::to_string(sizeof(ts::GameState)) + " bytes");
              ts::GameState s;
              std::memcpy(&s, b.c_str(), sizeof(ts::GameState));
              return s;
          }, nb::arg("data"),
          "The inverse of GameState.raw_bytes(), for the SAME engine build only: the bytes are the "
          "struct's memory, so a dataset of them records the engine fingerprint it was written by.");
    nb::class_<ts::labels::ObsCodebook>(m, "ObsCodebook",
        "Observation floats as exact one-byte codes (bindings/playout_labels.hpp): encode_raw() "
        "writes stored states' observations, each from its acting side, as codes; values()[codes] "
        "decodes them. One codebook per dataset: codes only mean something with their own table.")
        .def(nb::init<>())
        .def("encode_raw", [](ts::labels::ObsCodebook& self,
                              nb::ndarray<const uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> states,
                              uint32_t features) {
                 if (states.shape(1) != sizeof(ts::GameState))
                     throw std::invalid_argument("encode_raw: rows of state_size() bytes");
                 if (features & ~ts::obs_features::ALL) throw std::invalid_argument("unknown obs_features bit");
                 const size_t n = states.shape(0);
                 const size_t width = ts::OBS_SIZE_V23 + ts::obs_features::extra_width(features);
                 std::vector<ts::GameState> st(n);
                 if (n) std::memcpy(static_cast<void*>(st.data()), states.data(), n * sizeof(ts::GameState));
                 uint8_t* codes = new uint8_t[std::max<size_t>(1, n * width)];
                 nb::capsule own(codes, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });
                 bool ok;
                 {
                     nb::gil_scoped_release release;
                     ok = ts::labels::encode_observations(st.data(), n, features, self, codes);
                 }
                 if (!ok) throw std::runtime_error("encode_raw: more than 256 distinct observation values");
                 size_t sh[2] = { n, width };
                 return nb::ndarray<nb::numpy, uint8_t, nb::ndim<2>>(codes, 2, sh, own);
             }, nb::arg("states"), nb::arg("features") = 0u,
             "(n, obs width) uint8 codes of raw states' observations, each from its acting side.")
        .def("values", [](const ts::labels::ObsCodebook& self) {
                 float* d = new float[256];
                 const size_t n = self.values(d);
                 nb::capsule own(d, [](void* p) noexcept { delete[] static_cast<float*>(p); });
                 size_t sh[1] = { n };
                 return nb::ndarray<nb::numpy, float, nb::ndim<1>>(d, 1, sh, own);
             }, "code -> value (code 0 = 0.0); it only grows.")
        .def("size", &ts::labels::ObsCodebook::size);
    {
        using PL = ts::labels::PlayoutLabels;
        using RawStates = nb::ndarray<const uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
        using Ids = nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
        using Cands = nb::ndarray<const int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
        auto f32 = [](size_t n, size_t w, const std::vector<float>& v) {
            float* d = new float[std::max<size_t>(1, n * w)];
            std::copy(v.begin(), v.end(), d);
            nb::capsule own(d, [](void* p) noexcept { delete[] static_cast<float*>(p); });
            size_t sh[2] = { n, w };
            return nb::ndarray<nb::numpy, float, nb::ndim<2>>(d, 2, sh, own);
        };
        auto i64 = [](const std::vector<int64_t>& v) {
            int64_t* d = new int64_t[std::max<size_t>(1, v.size())];
            std::copy(v.begin(), v.end(), d);
            nb::capsule own(d, [](void* p) noexcept { delete[] static_cast<int64_t*>(p); });
            size_t sh[1] = { v.size() };
            return nb::ndarray<nb::numpy, int64_t, nb::ndim<1>>(d, 1, sh, own);
        };
        auto i32 = [](size_t n, size_t w, const std::vector<int32_t>& v) {
            int32_t* d = new int32_t[std::max<size_t>(1, n * w)];
            std::copy(v.begin(), v.end(), d);
            nb::capsule own(d, [](void* p) noexcept { delete[] static_cast<int32_t*>(p); });
            size_t sh[2] = { n, w };
            return nb::ndarray<nb::numpy, int32_t, nb::ndim<2>>(d, 2, sh, own);
        };
        nb::class_<PL>(m, "PlayoutLabels",
            "P34's playout labeller: queued positions dealt into `worlds` worlds from the mover's "
            "view, fresh dice each, played greedily to the game's end by the caller's network under "
            "the safe-playout rule, with outcome capture (bindings/playout_labels.hpp). Batching is "
            "continuous: fill_live(obs) -> logits -> step(logits) until idle(), collecting "
            "take_positions() / take_siblings() as they finish.")
            .def("__init__", [](PL* self, uint32_t features, int worlds, size_t capacity, uint64_t seed,
                                bool safe, int leaf_horizon, uint32_t max_plies, bool keyed, bool keyed2) {
                     if (features & ~ts::obs_features::ALL) throw std::invalid_argument("unknown obs_features bit");
                     if (worlds <= 0 || worlds > 65535) throw std::invalid_argument("worlds must be in 1..65535");
                     if (capacity == 0) throw std::invalid_argument("capacity must be positive");
                     if (leaf_horizon < 0 || leaf_horizon > 255) throw std::invalid_argument("leaf_horizon must be in 0..255");
                     ts::labels::Config cfg;
                     cfg.features = features;
                     cfg.worlds = static_cast<uint16_t>(worlds);
                     cfg.safe = safe;
                     cfg.leaf_horizon = leaf_horizon;
                     cfg.max_plies = max_plies;
                     cfg.keyed = keyed || keyed2;
                     cfg.keyed2 = keyed2;
                     new (self) PL(cfg, capacity, seed);
                 }, nb::arg("features") = 0u, nb::arg("worlds") = 32, nb::arg("capacity") = 16384,
                 nb::arg("seed") = 0ull, nb::arg("safe") = true, nb::arg("leaf_horizon") = 0,
                 nb::arg("max_plies") = 8000u, nb::arg("keyed") = false, nb::arg("keyed2") = false,
                 "keyed=True ties every draw to the situation it is made in (bindings/keyed_chance.hpp): "
                 "a world's candidates draw the same dice and cards wherever they reach the same situation. "
                 "keyed2=True is keyed chance with the player fields packed so a USSR side cannot hide the "
                 "turn; keyed alone keeps the original packing.")
            .def("enqueue_raw", [](PL& self, RawStates states, Ids ids, std::optional<Cands> candidates) {
                     const size_t n = states.shape(0);
                     if (states.shape(1) != sizeof(ts::GameState))
                         throw std::invalid_argument("enqueue_raw: rows of state_size() bytes");
                     if (ids.shape(0) != n) throw std::invalid_argument("enqueue_raw: one id per state");
                     std::vector<ts::GameState> st(n);
                     if (n) std::memcpy(static_cast<void*>(st.data()), states.data(), n * sizeof(ts::GameState));
                     std::vector<uint16_t> c;
                     size_t k = 0;
                     if (candidates) {
                         if (candidates->shape(0) != n) throw std::invalid_argument("enqueue_raw: one candidate row per state");
                         k = candidates->shape(1);
                         c.resize(n * k);
                         for (size_t i = 0; i < n * k; ++i) {
                             const int32_t a = candidates->data()[i];
                             if (a < 0 || a >= static_cast<int32_t>(ts::FLAT_ACTION_SPACE_SIZE))
                                 throw std::invalid_argument("a candidate is outside the flat action space");
                             c[i] = static_cast<uint16_t>(a);
                         }
                     }
                     const std::string err = self.enqueue(st.data(), ids.data(), n, k ? c.data() : nullptr, k);
                     if (!err.empty()) throw std::invalid_argument(err);
                 }, nb::arg("states"), nb::arg("ids"), nb::arg("candidates") = nb::none(),
                 "Queue positions: `states` is (n, state_size()) uint8 of GameState.raw_bytes() from "
                 "this engine build, `ids` (n,) int64, and `candidates`, if given, (n, k) int32 flat "
                 "actions legal at each position -- the sibling slice. Nothing is queued on an error.")
            .def("enqueue", [](PL& self, const std::vector<ts::GameState>& states, const std::vector<int64_t>& ids,
                               const std::vector<std::vector<int>>& candidates) {
                     if (ids.size() != states.size()) throw std::invalid_argument("enqueue: one id per state");
                     const size_t k = candidates.empty() ? 0 : candidates[0].size();
                     if (!candidates.empty() && candidates.size() != states.size())
                         throw std::invalid_argument("enqueue: one candidate list per state, or none");
                     std::vector<uint16_t> c;
                     for (const auto& row : candidates) {
                         if (row.size() != k) throw std::invalid_argument("enqueue: every position needs the same number of candidates");
                         for (int a : row) {
                             if (a < 0 || a >= static_cast<int>(ts::FLAT_ACTION_SPACE_SIZE))
                                 throw std::invalid_argument("a candidate is outside the flat action space");
                             c.push_back(static_cast<uint16_t>(a));
                         }
                     }
                     const std::string err = self.enqueue(states.data(), ids.data(), states.size(), k ? c.data() : nullptr, k);
                     if (!err.empty()) throw std::invalid_argument(err);
                 }, nb::arg("states"), nb::arg("ids"), nb::arg("candidates") = std::vector<std::vector<int>>{},
                 "enqueue_raw for GameState objects.")
            .def("world", [](const PL& self, const ts::GameState& base, int64_t id, int w) {
                     if (w < 0 || w > 65535) throw std::invalid_argument("world index");
                     return self.world(base, id, static_cast<uint16_t>(w));
                 }, nb::arg("base"), nb::arg("id"), nb::arg("w"),
                 "World `w` of position `id` exactly as the labeller deals it: the mover's unseen "
                 "cards redealt and the world's dice, from the (seed, id) stream.")
            .def("forced", &PL::forced, "Decisions of finished rows with one legal option, played without the network.")
            .def("chosen", &PL::chosen, "Decisions of finished rows the network chose.")
            .def("queued", &PL::queued)
            .def("occupied", &PL::occupied)
            .def("capacity", &PL::capacity)
            .def("obs_width", &PL::obs_width)
            .def("idle", &PL::idle)
            .def("fill_live", [](PL& self, nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> obs) {
                     if (obs.shape(0) < self.capacity() || obs.shape(1) != self.obs_width())
                         throw std::invalid_argument("fill_live: obs must be (capacity, obs_width) float32");
                     float* out = obs.data();
                     nb::gil_scoped_release release;
                     return self.fill_live(out);
                 }, nb::arg("obs"),
                 "Refill free rows from the queue and write the live rows' observations, each from "
                 "its acting side, into the first n rows of `obs`. Returns n, step()'s row count.")
            .def("fill_live_codes", [](PL& self, nb::ndarray<uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> codes) {
                     if (codes.shape(0) < self.capacity() || codes.shape(1) != self.obs_width())
                         throw std::invalid_argument("fill_live_codes: codes must be (capacity, obs_width) uint8");
                     uint8_t* out = codes.data();
                     size_t n;
                     {
                         nb::gil_scoped_release release;
                         n = self.fill_live_codes(out);
                     }
                     if (n == SIZE_MAX) throw std::runtime_error("fill_live_codes: more than 256 distinct observation values");
                     return n;
                 }, nb::arg("codes"),
                 "fill_live with each observation float written as its uint8 code in codebook(): "
                 "decode as codebook()[codes]. Exact.")
            .def("codebook", [](const PL& self) {
                     float* d = new float[256];
                     const size_t n = self.codebook(d);
                     nb::capsule own(d, [](void* p) noexcept { delete[] static_cast<float*>(p); });
                     size_t sh[1] = { n };
                     return nb::ndarray<nb::numpy, float, nb::ndim<1>>(d, 1, sh, own);
                 }, "The values the codes of fill_live_codes stand for (code 0 = 0.0); it only grows.")
            .def("step", [](PL& self, nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> logits,
                            std::optional<nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>> forced) {
                     if (logits.shape(0) < self.listed() || logits.shape(1) != ts::FLAT_ACTION_SPACE_SIZE)
                         throw std::invalid_argument("step: logits must cover the rows of the last fill_live");
                     if (forced && forced->shape(0) < self.listed())
                         throw std::invalid_argument("step: one forced action (or -1) per listed row");
                     const float* lg = logits.data();
                     require_no_nan(lg, self.listed() * ts::FLAT_ACTION_SPACE_SIZE, "step: logits");
                     const int32_t* fa = forced ? forced->data() : nullptr;
                     nb::gil_scoped_release release;
                     self.step(lg, fa);
                 }, nb::arg("logits"), nb::arg("forced") = nb::none(),
                 "Advance the listed rows by the masked argmax of their logits rows (the first n), "
                 "under the safe rule, and harvest the worlds that ended. `forced` (int32, -1 = the "
                 "argmax) gives a row another player's action, taken unless the safe rule refuses it.")
            .def("listed_info", [](const PL& self) {
                     const size_t n = self.listed();
                     int32_t* d = new int32_t[std::max<size_t>(1, n * 4)];
                     self.listed_info(d);
                     nb::capsule own(d, [](void* p) noexcept { delete[] static_cast<int32_t*>(p); });
                     size_t sh[2] = { n, 4 };
                     return nb::ndarray<nb::numpy, int32_t, nb::ndim<2>>(d, 2, sh, own);
                 }, "Per listed row: decision type, 1 if a headline card choice, its turn, its position's turn.")
            .def("listed_state", [](const PL& self, size_t j) -> ts::GameState {
                     if (j >= self.listed()) throw std::out_of_range("listed row");
                     return self.listed_state(j);
                 }, nb::arg("j"), "A copy of listed row j's position (rows in fill_live's order).")
            .def("take_positions", [f32, i64, i32](PL& self) {
                     const auto got = self.take_positions();
                     const size_t n = got.size(), R = ts::labels::kRegions;
                     std::vector<int64_t> id(n);
                     std::vector<int32_t> cnt(n * 2), swn(n * R);
                     std::vector<float> sums(n * 5), sw(n * R);
                     for (size_t i = 0; i < n; ++i) {
                         const auto& p = got[i];
                         id[i] = p.id;
                         cnt[i * 2] = static_cast<int32_t>(p.n);
                         cnt[i * 2 + 1] = static_cast<int32_t>(p.failed);
                         const float s[5] = { p.score, p.score_sq, p.final_vp, p.turn_vp, p.turn_defcon };
                         std::copy(s, s + 5, sums.begin() + static_cast<std::ptrdiff_t>(i * 5));
                         for (size_t g = 0; g < R; ++g) {
                             sw[i * R + g] = p.swing[g];
                             swn[i * R + g] = static_cast<int32_t>(p.swing_n[g]);
                         }
                     }
                     nb::dict d;
                     d["id"] = i64(id);
                     d["count"] = i32(n, 2, cnt);
                     d["sums"] = f32(n, 5, sums);
                     d["swing_sum"] = f32(n, R, sw);
                     d["swing_n"] = i32(n, R, swn);
                     return d;
                 }, "The main-label positions finished since the last call: id (n,); count (n, 2) = "
                    "worlds finished, worlds failed; sums (n, 5) over the finished worlds of score, "
                    "score^2, final VP, turn-end VP, turn-end DEFCON (mover's side); swing_sum / swing_n "
                    "(n, 7) per region (Europe, Asia, Middle East, Africa, Central America, South "
                    "America, Southeast Asia): summed swing at the next scoring and worlds that scored it.")
            .def("take_siblings", [f32, i64, i32](PL& self) {
                     std::vector<ts::GameState> leaves;
                     const auto got = self.take_siblings(leaves);
                     const size_t n = got.size(), R = ts::labels::kRegions;
                     std::vector<int64_t> id(n);
                     std::vector<int32_t> meta(n * 5), swing(n * R);
                     std::vector<float> cols(n * 4);
                     for (size_t i = 0; i < n; ++i) {
                         const auto& s = got[i];
                         id[i] = s.id;
                         const int32_t m5[5] = { s.candidate, s.world, s.failed ? 1 : 0, s.has_leaf ? 1 : 0, s.out.scored };
                         std::copy(m5, m5 + 5, meta.begin() + static_cast<std::ptrdiff_t>(i * 5));
                         const float c4[4] = { s.out.score, static_cast<float>(s.out.final_vp),
                                               static_cast<float>(s.out.turn_vp), static_cast<float>(s.out.turn_defcon) };
                         std::copy(c4, c4 + 4, cols.begin() + static_cast<std::ptrdiff_t>(i * 4));
                         for (size_t g = 0; g < R; ++g) swing[i * R + g] = s.out.swing[g];
                     }
                     uint8_t* raw = new uint8_t[std::max<size_t>(1, n * sizeof(ts::GameState))];
                     if (n) std::memcpy(raw, static_cast<const void*>(leaves.data()), n * sizeof(ts::GameState));
                     nb::capsule own(raw, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });
                     size_t sh[2] = { n, sizeof(ts::GameState) };
                     nb::dict d;
                     d["id"] = i64(id);
                     d["meta"] = i32(n, 5, meta);
                     d["cols"] = f32(n, 4, cols);
                     d["swing"] = i32(n, R, swing);
                     d["leaf"] = nb::ndarray<nb::numpy, uint8_t, nb::ndim<2>>(raw, 2, sh, own);
                     return d;
                 }, "The sibling rows finished since the last call: id (n,); meta (n, 5) = candidate "
                    "index, world, failed, has_leaf, scored-region bits; cols (n, 4) = score, final VP, "
                    "turn-end VP, turn-end DEFCON; swing (n, 7); leaf (n, state_size()) raw states, "
                    "zeros where has_leaf is 0.");
    }

    // ---- the rollout root's playouts (bindings/rollout_batch.hpp, shared with the browser) ----
    m.def("determinize", [](const ts::GameState& s, ts::Player me, uint64_t seed, bool keep_hand, bool keep_revealed) {
              ts::GameState out = s;
              uint64_t rng = seed;
              ts::rollout::determinize(out, me, rng, keep_hand, keep_revealed);
              return out;
          }, nb::arg("state"), nb::arg("me"), nb::arg("seed"), nb::arg("keep_hand") = false,
          nb::arg("keep_revealed") = false,
          "A clone with the cards `me` cannot see -- the opponent's unknown hand and the draw deck "
          "-- reshuffled between those two places from SplitMix64 state `seed`, every count kept, "
          "every seen card left; an unrevealed opposing headline is forgotten. The C++ port of "
          "ai.search.dmcts.determinize that the rollout root uses in both builds. keep_hand=True keeps the "
          "opponent's true hand and reshuffles only the draw deck (the oracle's world). keep_revealed=True keeps "
          "an opposing headline `me` has seen under Space Race box 4 (revealed_headline), chosen and in the hand.");
    m.def("revealed_headline", [](const ts::GameState& s, ts::Player me) { return ts::SpaceRace::revealed_headline(s, me); },
          nb::arg("state"), nb::arg("me"),
          "The opposing headline `me` has seen, else 0: with Man in Earth Orbit (Space Race box 4) held by `me` alone "
          "the opponent chooses first and reveals it.");
    using Hands = nb::ndarray<const uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
    m.def("place_hand", [](const ts::GameState& s, ts::Player me, nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> held) {
              if (held.shape(0) != 110) throw std::invalid_argument("held must have 110 entries");
              ts::GameState out = s;
              ts::rollout::place_hand(out, me, held.data());
              return out;
          }, nb::arg("state"), nb::arg("me"), nb::arg("held"),
          "A clone with `me`'s hidden pool (the opponent's unknown hand and the deck) split by `held` "
          "(110 bytes, card c at c - 1): set -> the opponent's unknown hand, else the deck.");
    m.def("hypothesis_rows", [](const ts::GameState& s, ts::Player me, Hands held, uint32_t features) {
              if (held.shape(1) != 110) throw std::invalid_argument("held must be n x 110");
              if (features & ~ts::obs_features::ALL) throw std::invalid_argument("unknown obs_features bit");
              const size_t n = held.shape(0);
              const size_t width = ts::OBS_SIZE_V23 + ts::obs_features::extra_width(features);
              size_t so[2] = { n, width }, sm[2] = { n, ts::FLAT_ACTION_SPACE_SIZE };
              float* obs = new float[std::max<size_t>(1, n * width)];
              uint8_t* masks = new uint8_t[std::max<size_t>(1, n * ts::FLAT_ACTION_SPACE_SIZE)];
              nb::capsule own_o(obs, [](void* p) noexcept { delete[] static_cast<float*>(p); });
              nb::capsule own_m(masks, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });
              {
                  nb::gil_scoped_release release;
                  ts::rollout::hypothesis_rows(s, me, held.data(), n, features, obs, masks);
              }
              return nb::make_tuple(nb::ndarray<nb::numpy, float, nb::ndim<2>>(obs, 2, so, own_o),
                                    nb::ndarray<nb::numpy, uint8_t, nb::ndim<2>>(masks, 2, sm, own_m));
          }, nb::arg("state"), nb::arg("me"), nb::arg("held"), nb::arg("features") = 0u,
          "A past decision of `me`'s opponent at `state` under n hypothesised hands: for each row of `held` "
          "(n x 110) the position with `me`'s hidden pool split by it (place_hand), its observation from the "
          "opponent's side and its legal mask -> (obs n x width, masks n x 220). The policy-likelihood "
          "belief's inner loop (ai/search/belief.py).");
    m.def("hypothesis_masks", [](const ts::GameState& s, ts::Player me, Hands held) {
              if (held.shape(1) != 110) throw std::invalid_argument("held must be n x 110");
              const size_t n = held.shape(0);
              size_t sm[2] = { n, ts::FLAT_ACTION_SPACE_SIZE };
              uint8_t* masks = new uint8_t[std::max<size_t>(1, n * ts::FLAT_ACTION_SPACE_SIZE)];
              nb::capsule own_m(masks, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });
              {
                  nb::gil_scoped_release release;
                  ts::rollout::hypothesis_masks(s, me, held.data(), n, masks);
              }
              return nb::ndarray<nb::numpy, uint8_t, nb::ndim<2>>(masks, 2, sm, own_m);
          }, nb::arg("state"), nb::arg("me"), nb::arg("held"),
          "hypothesis_rows' legal masks alone (n x 220): the observations differ only in each hidden "
          "card's own-hand / deck slot, which the belief sets on the device (ai/search/belief.py).");
    {
        using BT = ts::belief::Tracker;
        nb::class_<BT>(m, "BeliefTracker",
            "The policy-likelihood belief over the opponent's unseen hand (bindings/belief.hpp, shared with "
            "the browser): observe() every decision of the games; at root decisions prepare(), evaluate the "
            "rows it lists (rows_patch or rows_full) at every temperature, set_log_likelihoods(), worlds().")
            .def(nb::init<uint32_t, bool, uint64_t, bool>(), nb::arg("particles") = 2048, nb::arg("all_decisions") = true,
                 nb::arg("seed") = 0ull, nb::arg("reveal") = false)
            .def_static("temperatures", []() {
                std::vector<float> t(ts::belief::kTemperatures, ts::belief::kTemperatures + ts::belief::kTemps);
                return t;
            })
            .def_prop_ro("particles", &BT::particles)
            .def("reseed", &BT::reseed, nb::arg("seed"))
            .def("new_games", &BT::new_games)
            .def("observe", [](BT& self, size_t game, const ts::GameState& s, ts::Player decider, int action) {
                     if (action < 0 || action >= static_cast<int>(ts::FLAT_ACTION_SPACE_SIZE))
                         throw std::invalid_argument("action outside the flat action space");
                     self.observe(game, s, decider, static_cast<uint16_t>(action));
                 }, nb::arg("game"), nb::arg("state"), nb::arg("decider"), nb::arg("action"))
            .def("prepare", [](BT& self, const std::vector<size_t>& games, const std::vector<const ts::GameState*>& states) {
                     if (games.size() != states.size()) throw std::invalid_argument("games and states differ in length");
                     return self.prepare(games, states);
                 }, nb::arg("games"), nb::arg("states"))
            .def("job_count", [](const BT& self) { return self.jobs().size(); })
            .def("job_decisions", [](const BT& self, size_t j) {
                     if (j >= self.jobs().size()) throw std::out_of_range("job");
                     return self.jobs()[j].decisions.size();
                 }, nb::arg("job"))
            .def("job_position", [](const BT& self, size_t j) {
                     if (j >= self.jobs().size()) throw std::out_of_range("job");
                     return self.jobs()[j].position;
                 }, nb::arg("job"), "The index, among the last prepare() call's positions, of job j.")
            .def("job_hands", [](const BT& self, size_t j) {
                     if (j >= self.jobs().size()) throw std::out_of_range("job");
                     const auto& h = self.jobs()[j].hands;
                     size_t sh[2] = { h.size() / 110, 110 };
                     uint8_t* buf = new uint8_t[std::max<size_t>(1, h.size())];
                     std::copy(h.begin(), h.end(), buf);
                     nb::capsule own(buf, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });
                     return nb::ndarray<nb::numpy, uint8_t, nb::ndim<2>>(buf, 2, sh, own);
                 }, nb::arg("job"))
            .def("rows_patch", [](const BT& self, uint32_t features) {
                     // Every job's recorded decisions as one block each: the decision's own observation from
                     // the opponent's side, its hidden pool then, the cards departed since, its job, the move,
                     // and each hand's legal mask bit-packed (np.packbits order) -- the device builds the rows.
                     const auto& jobs = self.jobs();
                     const size_t N = self.particles();
                     const size_t W = ts::OBS_SIZE_V23 + ts::obs_features::extra_width(features);
                     size_t R = 0;
                     for (const auto& j : jobs) R += j.decisions.size();
                     const size_t PB = (ts::FLAT_ACTION_SPACE_SIZE + 7) / 8;
                     // Owned until the capsules take them at the end, so a throw on the way frees them.
                     auto bases_p = std::make_unique<float[]>(std::max<size_t>(1, R * W));
                     auto pools_p = std::make_unique<uint8_t[]>(std::max<size_t>(1, R * 110));
                     auto lefts_p = std::make_unique<uint8_t[]>(std::max<size_t>(1, R * 110));
                     auto job_of_p = std::make_unique<int64_t[]>(std::max<size_t>(1, R));
                     auto acts_p = std::make_unique<int64_t[]>(std::max<size_t>(1, R));
                     auto hands_p = std::make_unique<uint8_t[]>(std::max<size_t>(1, jobs.size() * N * 110));
                     auto packed_p = std::make_unique<uint8_t[]>(std::max<size_t>(1, R * N * PB));
                     float* bases = bases_p.get();
                     uint8_t* pools = pools_p.get();
                     uint8_t* lefts = lefts_p.get();
                     int64_t* job_of = job_of_p.get();
                     int64_t* acts = acts_p.get();
                     uint8_t* hands = hands_p.get();
                     uint8_t* packed = packed_p.get();
                     {
                         nb::gil_scoped_release release;
                         size_t r = 0;
                         std::vector<uint8_t> held(N * 110), masks(N * ts::FLAT_ACTION_SPACE_SIZE);
                         for (size_t ji = 0; ji < jobs.size(); ++ji) {
                             const auto& j = jobs[ji];
                             std::copy(j.hands.begin(), j.hands.end(), hands + ji * N * 110);
                             const ts::Player opp = j.me == ts::Player::US ? ts::Player::USSR : ts::Player::US;
                             const uint8_t ou = static_cast<uint8_t>(opp == ts::Player::US ? ts::CardLocation::HAND_US_UNKNOWN
                                                                                            : ts::CardLocation::HAND_USSR_UNKNOWN);
                             for (size_t d = 0; d < j.decisions.size(); ++d, ++r) {
                                 const auto& dec = self.decision(ji, d);
                                 ts::extract_observation_features(dec.state, opp, features, bases + r * W);
                                 for (size_t c = 0; c < 110; ++c) {
                                     const uint8_t l = dec.locs[c + 1];
                                     pools[r * 110 + c] = (l == static_cast<uint8_t>(ts::CardLocation::DRAW_DECK) || l == ou) ? 1 : 0;
                                     lefts[r * 110 + c] = j.lefts[d][c];
                                 }
                                 job_of[r] = static_cast<int64_t>(ji);
                                 acts[r] = dec.action;
                                 for (size_t k = 0; k < N; ++k)
                                     for (size_t c = 0; c < 110; ++c)
                                         held[k * 110 + c] = static_cast<uint8_t>(j.hands[k * 110 + c] | j.lefts[d][c]);
                                 ts::rollout::hypothesis_masks(dec.state, j.me, held.data(), N, masks.data());
                                 uint8_t* out = packed + r * N * PB;
                                 std::fill(out, out + N * PB, 0);
                                 for (size_t k = 0; k < N; ++k)
                                     for (size_t a = 0; a < ts::FLAT_ACTION_SPACE_SIZE; ++a)
                                         if (masks[k * ts::FLAT_ACTION_SPACE_SIZE + a])
                                             out[k * PB + a / 8] |= static_cast<uint8_t>(0x80u >> (a % 8));
                             }
                         }
                     }
                     auto own_f = [](std::unique_ptr<float[]>& p) { return nb::capsule(p.release(), [](void* q) noexcept { delete[] static_cast<float*>(q); }); };
                     auto own_u = [](std::unique_ptr<uint8_t[]>& p) { return nb::capsule(p.release(), [](void* q) noexcept { delete[] static_cast<uint8_t*>(q); }); };
                     auto own_i = [](std::unique_ptr<int64_t[]>& p) { return nb::capsule(p.release(), [](void* q) noexcept { delete[] static_cast<int64_t*>(q); }); };
                     size_t s_b[2] = { R, W }, s_p[2] = { R, 110 }, s_r[1] = { R }, s_h[3] = { jobs.size(), N, 110 },
                            s_m[3] = { R, N, PB };
                     return nb::make_tuple(
                         nb::ndarray<nb::numpy, float, nb::ndim<2>>(bases, 2, s_b, own_f(bases_p)),
                         nb::ndarray<nb::numpy, uint8_t, nb::ndim<2>>(pools, 2, s_p, own_u(pools_p)),
                         nb::ndarray<nb::numpy, uint8_t, nb::ndim<2>>(lefts, 2, s_p, own_u(lefts_p)),
                         nb::ndarray<nb::numpy, int64_t, nb::ndim<1>>(job_of, 1, s_r, own_i(job_of_p)),
                         nb::ndarray<nb::numpy, int64_t, nb::ndim<1>>(acts, 1, s_r, own_i(acts_p)),
                         nb::ndarray<nb::numpy, uint8_t, nb::ndim<3>>(hands, 3, s_h, own_u(hands_p)),
                         nb::ndarray<nb::numpy, uint8_t, nb::ndim<3>>(packed, 3, s_m, own_u(packed_p)));
                 }, nb::arg("features") = 0u,
                 "(bases R x W, pools R x 110, lefts R x 110, job_of R, actions R, hands J x N x 110, packed masks "
                 "R x N x 28): every job's recorded decisions, for the caller to build the rows on its device.")
            .def("set_log_likelihoods", [](BT& self, nb::ndarray<const float, nb::ndim<3>, nb::c_contig, nb::device::cpu> lp) {
                     if (lp.shape(0) != ts::belief::kTemps || lp.shape(2) != self.particles())
                         throw std::invalid_argument("log-likelihoods must be temperatures x blocks x particles");
                     size_t R = 0;
                     for (const auto& j : self.jobs()) R += j.decisions.size();
                     if (lp.shape(1) != R) throw std::invalid_argument("one block per recorded decision of every job");
                     self.set_log_likelihoods(lp.data(), R);
                 }, nb::arg("log_likelihoods"))
            .def("worlds", [](const BT& self, size_t p, size_t n) -> nb::object {
                     uint8_t* buf = new uint8_t[std::max<size_t>(1, n * 110)];
                     if (!self.worlds(p, n, buf)) { delete[] buf; return nb::none(); }
                     size_t sh[2] = { n, 110 };
                     nb::capsule own(buf, [](void* q) noexcept { delete[] static_cast<uint8_t*>(q); });
                     return nb::cast(nb::ndarray<nb::numpy, uint8_t, nb::ndim<2>>(buf, 2, sh, own));
                 }, nb::arg("position"), nb::arg("n"),
                 "n hands (n x 110 uint8) resampled from the prepared position's belief, or None where it has none.")
            .def("temperature_weights", [](const BT& self, size_t p) {
                     const auto w = self.temperature_weights(p);
                     return std::vector<float>(w.begin(), w.end());
                 }, nb::arg("position"));
    }
    {
        using RB = ts::rollout::RolloutBatch;
        auto arrays = [](size_t n, size_t width) {
            size_t so[2] = { n, width }, sm[2] = { n, ts::FLAT_ACTION_SPACE_SIZE };
            float* obs = new float[std::max<size_t>(1, n * width)];
            uint8_t* masks = new uint8_t[std::max<size_t>(1, n * ts::FLAT_ACTION_SPACE_SIZE)];
            nb::capsule own_o(obs, [](void* p) noexcept { delete[] static_cast<float*>(p); });
            nb::capsule own_m(masks, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });
            return std::make_tuple(obs, masks,
                                   nb::ndarray<nb::numpy, float, nb::ndim<2>>(obs, 2, so, own_o),
                                   nb::ndarray<nb::numpy, uint8_t, nb::ndim<2>>(masks, 2, sm, own_m));
        };
        nb::class_<RB>(m, "RolloutBatch",
            "The rollout root's playouts for one or more decisions: the sampled worlds of each "
            "candidate move, their greedy playouts to a horizon of action-round boundaries (stop="
            "'boundaries', the default), to the end of the decision's turn ('turn'), or to the later of "
            "the two ('later'), and the "
            "leaves the critic reads. The network is the caller's: fill_live() -> logits -> step(), "
            "until all_done(); fill_leaves() -> critic -> set_leaf_values(); then values(). "
            "Written once for the Python searcher and the browser (bindings/rollout_batch.hpp).")
            .def("__init__", [](RB* self, int horizon, bool shared_dice, bool guard, uint32_t features,
                                uint64_t seed, const std::string& stop, const std::string& leaf, bool keyed, bool oracle,
                                bool aligned, bool box4, bool keyed2) {
                     if (horizon < 0) throw std::invalid_argument("horizon must be >= 0");
                     if (features & ~ts::obs_features::ALL) throw std::invalid_argument("unknown obs_features bit");
                     ts::rollout::Stop s;
                     if (stop == "boundaries") s = ts::rollout::Stop::kBoundaries;
                     else if (stop == "turn") s = ts::rollout::Stop::kTurnEnd;
                     else if (stop == "later") s = ts::rollout::Stop::kLater;
                     else if (stop == "earlier") s = ts::rollout::Stop::kEarlier;
                     else if (stop == "predeal") s = ts::rollout::Stop::kPredeal;
                     else if (stop == "earlier_predeal") s = ts::rollout::Stop::kEarlierPredeal;
                     else throw std::invalid_argument("stop must be 'boundaries', 'turn', 'later', 'earlier', "
                                                      "'predeal' or 'earlier_predeal'");
                     ts::rollout::Leaf l;
                     if (leaf == "mover") l = ts::rollout::Leaf::kMover;
                     else if (leaf == "decider") l = ts::rollout::Leaf::kDecider;
                     else if (leaf == "own") l = ts::rollout::Leaf::kOwn;
                     else throw std::invalid_argument("leaf must be 'mover', 'decider' or 'own'");
                     new (self) RB(ts::rollout::Config{horizon, shared_dice, guard, features, s, l, keyed || keyed2,
                                                       oracle, aligned, box4, keyed2}, seed);
                 }, nb::arg("horizon") = 4, nb::arg("shared_dice") = true, nb::arg("guard") = true,
                 nb::arg("features") = 0u, nb::arg("seed") = 0ull, nb::arg("stop") = "boundaries",
                 nb::arg("leaf") = "mover", nb::arg("keyed") = false, nb::arg("oracle") = false,
                 nb::arg("aligned") = false, nb::arg("box4") = false, nb::arg("keyed2") = false)
            .def("add", [](RB& self, const ts::GameState& base, const std::vector<int>& candidates, size_t worlds,
                           std::optional<Hands> hands) {
                     std::vector<uint16_t> c;
                     for (int a : candidates) {
                         if (a < 0 || a >= static_cast<int>(ts::FLAT_ACTION_SPACE_SIZE))
                             throw std::invalid_argument("a candidate is outside the flat action space");
                         c.push_back(static_cast<uint16_t>(a));
                     }
                     if (hands && (hands->shape(0) != worlds || hands->shape(1) != 110))
                         throw std::invalid_argument("hands must be worlds x 110");
                     const size_t n = self.add(base, c.data(), c.size(), worlds, hands ? hands->data() : nullptr);
                     if (n == 0 && self.last_error()[0]) throw std::invalid_argument(self.last_error());
                     return n;
                 }, nb::arg("base"), nb::arg("candidates"), nb::arg("worlds"), nb::arg("hands") = nb::none(),
                 "One decision: `base` with the mover's flat `candidates` (all legal there), in "
                 "`worlds` worlds. Returns the rows added -- len(candidates) times the worlds kept. "
                 "`hands` (uint8, worlds x 110), when given, is each world's opposing unknown hand "
                 "(a belief's worlds, ai/search/belief.py) instead of a random redeal.")
            .def("race", [](RB& self, size_t position, size_t worlds, std::optional<Hands> hands) {
                     if (hands && (hands->shape(0) != worlds || hands->shape(1) != 110))
                         throw std::invalid_argument("hands must be worlds x 110");
                     const size_t n = self.race(position, worlds, hands ? hands->data() : nullptr);
                     if (n == 0 && self.last_error()[0]) throw std::invalid_argument(self.last_error());
                     return n;
                 }, nb::arg("position"), nb::arg("worlds"), nb::arg("hands") = nb::none(),
                 "The second stage of a raced decision, once its rows are valued: every challenger "
                 "whose paired lead over candidate 0 is below 0 leaves the race, and `worlds` fresh "
                 "worlds are added for candidate 0 and the challengers left (none when none is). "
                 "Returns the rows added; `hands` as add's.")
            .def("in_race", [](const RB& self, size_t p, size_t c) {
                     if (p >= self.positions()) throw std::out_of_range("position");
                     return self.in_race(p, c);
                 }, nb::arg("position"), nb::arg("candidate"),
                 "Whether candidate `candidate` (an index into the position's list) is still in the race.")
            .def("positions", &RB::positions)
            .def("mover", [](const RB& self, size_t p) {
                     if (p >= self.positions()) throw std::out_of_range("position");
                     return self.mover(p);
                 }, nb::arg("position"))
            .def("worlds_used", [](const RB& self, size_t p) {
                     if (p >= self.positions()) throw std::out_of_range("position");
                     return self.worlds_used(p);
                 }, nb::arg("position"))
            .def("rows", &RB::rows)
            .def("live", &RB::live)
            .def("all_done", &RB::all_done)
            .def("obs_width", &RB::obs_width)
            .def("refused", &RB::refused)
            .def("fill_live", [arrays](RB& self) {
                     auto [obs, masks, o, mk] = arrays(self.live(), self.obs_width());
                     self.fill_live(obs, masks);
                     return nb::make_tuple(o, mk);
                 }, "(observations, masks) of the live rows, each row from its acting side, in the "
                    "order step() expects.")
            .def("step", [](RB& self, nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> logits) {
                     if (logits.shape(0) != self.listed() || logits.shape(1) != ts::FLAT_ACTION_SPACE_SIZE)
                         throw std::invalid_argument("step: logits must be (rows of the last fill_live, action space)");
                     require_no_nan(logits.data(), logits.size(), "step: logits");
                     self.step(logits.data());
                 }, nb::arg("logits"),
                 "Advance the rows the last fill_live listed by the masked argmax of each logits row.")
            .def("fill_leaves", [arrays](RB& self) {
                     auto [obs, masks, o, mk] = arrays(self.leaves_pending(), self.obs_width());
                     self.fill_leaves(obs, masks);
                     return nb::make_tuple(o, mk);
                 }, "(observations, masks) of the rows at their horizon and not yet valued, each from "
                    "the MOVER's side, in the order set_leaf_values() expects.")
            .def("set_leaf_values", [](RB& self, nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu> v) {
                     if (v.shape(0) != self.leaves_listed())
                         throw std::invalid_argument("set_leaf_values: one value per row of the last fill_leaves");
                     require_no_nan(v.data(), v.size(), "set_leaf_values");
                     self.set_leaf_values(v.data());
                 }, nb::arg("values"))
            .def("values", [](const RB& self) {
                     size_t shape[1] = { self.rows() };
                     float* data = new float[std::max<size_t>(1, self.rows())];
                     self.values(data);
                     nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
                     return nb::ndarray<nb::numpy, float, nb::ndim<1>>(data, 1, shape, owner);
                 }, "Every row's value from its mover's side; NaN while live or unvalued.")
            .def("row_position", [](const RB& self) { std::vector<int> v; for (size_t r = 0; r < self.rows(); ++r) v.push_back(static_cast<int>(self.row(r).position)); return v; })
            .def("row_world", [](const RB& self) { std::vector<int> v; for (size_t r = 0; r < self.rows(); ++r) v.push_back(self.row(r).world); return v; })
            .def("row_candidate", [](const RB& self) { std::vector<int> v; for (size_t r = 0; r < self.rows(); ++r) v.push_back(self.row(r).candidate); return v; })
            .def("row_done", [](const RB& self) { std::vector<bool> v; for (size_t r = 0; r < self.rows(); ++r) v.push_back(self.row(r).done); return v; })
            .def("row_terminal", [](const RB& self) { std::vector<bool> v; for (size_t r = 0; r < self.rows(); ++r) v.push_back(self.row(r).terminal); return v; })
            .def("row_crossed", [](const RB& self) { std::vector<int> v; for (size_t r = 0; r < self.rows(); ++r) v.push_back(self.row(r).crossed); return v; })
            .def("row_state", [](const RB& self, size_t r) -> ts::GameState {
                     if (r >= self.rows()) throw std::out_of_range("row");
                     return self.row(r).state;
                 }, nb::arg("row"), "A copy of the row's position, for tests.");

        // The root's choice (bindings/rollout_batch.hpp), the same code the page runs.
        auto rule_of = [](const std::string& text) {
            ts::rollout::Rule rule;
            if (!ts::rollout::parse_rule(text.c_str(), rule))
                throw std::invalid_argument("unknown rule '" + text + "': argmax, z<x> or kl<t>");
            return rule;
        };
        m.def("rollout_parse_rule", [rule_of](const std::string& text) {
                  const ts::rollout::Rule r = rule_of(text);
                  const char* kind = r.kind == ts::rollout::Rule::ARGMAX ? "argmax" : r.kind == ts::rollout::Rule::Z ? "z" : "kl";
                  return std::make_pair(std::string(kind), r.param);
              }, nb::arg("rule"), "(kind, parameter) of a rollout rule -- argmax, z<x> or kl<t>; ValueError otherwise.");
        m.def("rollout_candidates",
              [](nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu> logits,
                 nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> mask, size_t k) {
                  if (logits.shape(0) != ts::FLAT_ACTION_SPACE_SIZE || mask.shape(0) != ts::FLAT_ACTION_SPACE_SIZE)
                      throw std::invalid_argument("rollout_candidates: logits and mask are over the flat action space");
                  require_no_nan(logits.data(), logits.size(), "rollout_candidates: logits");
                  std::vector<uint16_t> out(k);
                  const size_t n = ts::rollout::top_candidates(logits.data(), mask.data(), k, out.data());
                  return std::vector<int>(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(n));
              }, nb::arg("logits"), nb::arg("mask"), nb::arg("k"),
              "The k most probable legal moves, best first (fewer where fewer are legal).");
        m.def("rollout_decide",
              [rule_of](const RB& batch, size_t position, const std::vector<int>& candidates,
                        nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu> logits,
                        nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> mask,
                        const std::string& rule) {
                  if (logits.shape(0) != ts::FLAT_ACTION_SPACE_SIZE || mask.shape(0) != ts::FLAT_ACTION_SPACE_SIZE)
                      throw std::invalid_argument("rollout_decide: logits and mask are over the flat action space");
                  require_no_nan(logits.data(), logits.size(), "rollout_decide: logits");
                  if (position >= batch.positions()) throw std::out_of_range("position");
                  std::vector<uint16_t> c;
                  for (int a : candidates) {
                      if (a < 0 || a >= static_cast<int>(ts::FLAT_ACTION_SPACE_SIZE))
                          throw std::invalid_argument("a candidate is outside the flat action space");
                      c.push_back(static_cast<uint16_t>(a));
                  }
                  const ts::rollout::Decision d = ts::rollout::decide(batch, position, c.data(), c.size(),
                                                                      logits.data(), mask.data(), rule_of(rule));
                  nb::dict out;
                  out["pick"] = static_cast<int>(d.pick);
                  out["worlds"] = static_cast<int>(d.worlds);
                  out["q"] = d.q;
                  out["lead"] = d.lead;
                  out["lead_se"] = d.lead_se;
                  return out;
              }, nb::arg("batch"), nb::arg("position"), nb::arg("candidates"), nb::arg("logits"), nb::arg("mask"),
              nb::arg("rule"),
              "One position's decision from a valued batch: {pick, worlds, q, lead, lead_se} -- each "
              "candidate's mean value from the mover's side, its paired lead over the network's own move "
              "(candidate 0) with the standard error, and the pick under `rule`.");
    }
    m.def("game_ending_reason", [](const ts::GameState& s) { return std::string(ts::state_json::ending_reason(s)); },
          nb::arg("state"),
          "Why a finished game ended: '20 VP', 'Europe Control', 'DEFCON 1 (own decision)', "
          "'DEFCON 1 (opponent decision)', 'final scoring' or 'wargames'. Shared with the browser "
          "workbench's engine; tools/lib/tournament_evaluator delegates to it.");
    m.def("state_from_save_json", &game_state_from_save_json, nb::arg("text"),
          "state_from_save_dict for the JSON text to_save_json writes (or any JSON of that "
          "shape). Raises ValueError for text that is not JSON or not a save.");


    // Flat Action Mask & Codec exports
    m.def("get_flat_action_mask", &flat_mask_array,
          nb::arg("state"), nb::arg("merged_influence") = false, nb::arg("macro_influence") = false);

    m.def("decode_flat_action", &checked_decode,
          "The MicroAction a flat index stands for here. Raises ValueError outside the space.");
    m.def("encode_micro_action", &checked_encode,
          "The flat index of a MicroAction. Raises ValueError for one no slot encodes.");

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

    // The view spec's observation half (owner, 2026-10-01): optional feature blocks appended to
    // the base layout. `features` has no default anywhere -- a caller names the set its model was
    // trained with (0 for the base) -- for the reason the old `layout` argument was removed.
    auto check_features = [](uint32_t features) {
        if (features & ~ts::obs_features::ALL)
            throw std::invalid_argument("unknown observation feature bits: " + std::to_string(features));
    };
    m.def("obs_size_for", [check_features](uint32_t features) {
        check_features(features);
        return static_cast<int>(ts::OBS_SIZE_V23 + ts::obs_features::extra_width(features));
    }, nb::arg("features"),
       "Observation width for a feature set: the base 3,824 plus each appended block.");
    m.def("extract_observation_features",
          [check_features](const ts::GameState& state, ts::Player perspective, uint32_t features) {
        check_features(features);
        const size_t n = ts::OBS_SIZE_V23 + ts::obs_features::extra_width(features);
        float* data = new float[n];
        ts::extract_observation_features(state, perspective, features, data);
        size_t shape[1] = { n };
        nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
        return nb::ndarray<nb::numpy, float, nb::ndim<1>>(data, 1, shape, owner);
    }, nb::arg("state"), nb::arg("perspective"), nb::arg("features"),
       "The base observation followed by the blocks `features` appends. features=0 is exactly "
       "extract_observation(state, perspective).");
    m.attr("OBS_FEATURE_OPS_BUDGET") = static_cast<uint32_t>(ts::obs_features::OPS_BUDGET);
    m.attr("OBS_FEATURE_HEADLINE_REVEAL") = static_cast<uint32_t>(ts::obs_features::HEADLINE_REVEAL);
    m.attr("OBS_FEATURE_OPP_EVENT_GATED") = static_cast<uint32_t>(ts::obs_features::OPP_EVENT_GATED);
    m.attr("OBS_FEATURE_CANONICAL_FIRST") = static_cast<uint32_t>(ts::obs_features::CANONICAL_FIRST);
    m.attr("OBS_FEATURE_CANONICAL_LAST") = static_cast<uint32_t>(ts::obs_features::CANONICAL_LAST);
    m.attr("OBS_FEATURE_CANONICAL_LEARNED") = static_cast<uint32_t>(ts::obs_features::CANONICAL_LEARNED);
    m.attr("OBS_FEATURES_RULES") = static_cast<uint32_t>(ts::obs_features::RULES);
    m.attr("OBS_FEATURES_PERSON_RULES") = static_cast<uint32_t>(ts::obs_features::PERSON_RULES);
    m.attr("OBS_FEATURES_ALL") = static_cast<uint32_t>(ts::obs_features::ALL);

    m.attr("OBS_FLAG_STAGED_CARDS") = static_cast<uint32_t>(ts::obs_flags::STAGED_CARDS);

    m.attr("OBS_SIZE_V23") = static_cast<int>(ts::OBS_SIZE_V23);
    //: The width under a name that carries no version. There is one layout, and code saying
    //: OBS_SIZE cannot be misread as choosing between several.
    m.attr("OBS_SIZE") = static_cast<int>(ts::OBS_SIZE_V23);

    //: The flat action space's slots (ts::flat_slots), so Python reads them instead of keeping
    //: a copy (bindings/action_encoder.py).
    m.attr("FLAT_ACTION_SPACE_SIZE") = static_cast<int>(ts::flat_slots::SIZE);
    m.attr("FLAT_CARD") = static_cast<int>(ts::flat_slots::CARD);
    m.attr("FLAT_RESOLUTION") = static_cast<int>(ts::flat_slots::RESOLUTION);
    m.attr("FLAT_OP_MODE") = static_cast<int>(ts::flat_slots::OP_MODE);
    m.attr("FLAT_ROLL_DIE") = static_cast<int>(ts::flat_slots::ROLL_DIE);
    m.attr("FLAT_NODE") = static_cast<int>(ts::flat_slots::NODE);
    m.attr("FLAT_BRANCH") = static_cast<int>(ts::flat_slots::BRANCH);
    m.attr("FLAT_CONFIRM_DONE") = static_cast<int>(ts::flat_slots::CONFIRM_DONE);
    m.attr("FLAT_DEFCON_VALUE") = static_cast<int>(ts::flat_slots::DEFCON_VALUE);
    m.attr("FLAT_REGION") = static_cast<int>(ts::flat_slots::REGION);
    m.attr("FLAT_BASE_SIZE") = static_cast<int>(ts::flat_slots::BASE_SIZE);
    m.attr("FLAT_MACRO_NODE") = static_cast<int>(ts::flat_slots::MACRO_NODE);

    //: Which compiler built this extension, e.g. "clang 21.1.8". The engine is built with clang
    //: (root CMakeLists.txt); tests/bindings/test_build_toolchain.py checks the module actually
    //: imported, which is the one a stale build directory can quietly get wrong.
#if defined(__clang__)
    m.attr("BUILD_COMPILER") = std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    m.attr("BUILD_COMPILER") = std::string("gcc ") + __VERSION__;
#else
    m.attr("BUILD_COMPILER") = std::string("unknown");
#endif

    nb::class_<ts::ActionMask>(m, "ActionMask")
        .def_static("generate_flat_mask", &flat_mask_array,
                    nb::arg("state"), nb::arg("merged_influence") = false, nb::arg("macro_influence") = false)
        .def_static("is_merged_influence_action", &ts::ActionMask::is_merged_influence_action)
        .def_static("is_macro_action", &ts::ActionMask::is_macro_action)
        .def_static("decode_flat_action", &checked_decode)
        .def_static("encode_micro_action", &checked_encode);

    // Vectorized Batch Runner for fast parallel self-play rollouts
    struct VectorizedBatchRunner {
        // Each env's position with its own legal mask (ts::CachedState): the agent's action is
        // validated against it, and the mask auto-advance generated where it stopped is the one
        // handed out -- one generation per decision where there were three. Read through st(i);
        // written only through the slot's own calls, each of which invalidates the mask.
        std::vector<ts::CachedState> slots;
        const ts::GameState& st(size_t i) const { return slots[i].state(); }
        std::vector<float> obs_buffer;
        std::vector<uint8_t> mask_buffer;
        // P23 / E4.1: which action view each side of each env decides in. Per side, not per env,
        // because a tournament puts an E4 agent and an E4.1 agent in the same game. The cached mask
        // is built for whoever decides next, and a step is applied in the view of whoever made it.
        // All zero is exactly E4.
        // The action view each side decides in, per env (ts::ActionView as a byte: E4, MERGED, MACRO).
        std::vector<uint8_t> view_us;
        std::vector<uint8_t> view_ussr;
        size_t num_envs;
        //: The widest observation any env's decider is configured for: the base layout plus the
        //: blocks of the largest feature set in use. Every row is written in its own decider's
        //: feature set and zero-padded to this, so a model reads the first obs_size_for(its set)
        //: floats of the rows where it decides.
        size_t obs_width = ts::OBS_SIZE_V23;
        std::vector<uint32_t> feat_us;
        std::vector<uint32_t> feat_ussr;
        uint32_t features_for(size_t idx, ts::Player p) const {
            if (p == ts::Player::US) return feat_us[idx];
            if (p == ts::Player::USSR) return feat_ussr[idx];
            return 0u;
        }
        // Each side plays the rule bits of the feature set it is observed in; reapplied wherever a
        // slot's state is replaced (init_new_game leaves none, set_state the saver's).
        void seat(size_t idx) {
            slots[idx].modify([&](ts::GameState& g) {
                ts::seat_obs_features(g, ts::Player::US, feat_us[idx]);
                ts::seat_obs_features(g, ts::Player::USSR, feat_ussr[idx]);
            });
        }
        void resize_obs() {
            size_t w = ts::OBS_SIZE_V23;
            for (size_t i = 0; i < num_envs; ++i) {
                w = std::max(w, ts::OBS_SIZE_V23 + ts::obs_features::extra_width(feat_us[i]));
                w = std::max(w, ts::OBS_SIZE_V23 + ts::obs_features::extra_width(feat_ussr[i]));
            }
            obs_width = w;
            obs_buffer.assign(num_envs * obs_width, 0.0f);
            alloc_predeal();
        }
        // P33, the pre-deal critic: with capture on, an env that passes the turn-end park during a
        // step's auto-advance (ts::rollout::at_turn_cleanup) has that position's observation recorded
        // from both sides, each in its side's feature set, zero-padded to obs_width. The decision stream
        // is unchanged: the drain runs one chance node at a time and stops where it always stopped.
        bool capture_predeal = false;
        std::vector<uint8_t> predeal_flag;
        std::vector<float> predeal_us, predeal_ussr;
        void alloc_predeal() {
            predeal_flag.assign(num_envs, 0);
            predeal_us.assign(capture_predeal ? num_envs * obs_width : 0, 0.0f);
            predeal_ussr.assign(capture_predeal ? num_envs * obs_width : 0, 0.0f);
        }
        void set_capture_predeal(bool on) { capture_predeal = on; alloc_predeal(); }
        void record_predeal(size_t idx) {
            for (int side = 0; side < 2; ++side) {
                const ts::Player p = side == 0 ? ts::Player::US : ts::Player::USSR;
                float* row = (side == 0 ? predeal_us.data() : predeal_ussr.data()) + idx * obs_width;
                const size_t w = ts::extract_observation_features(st(idx), p, features_for(idx, p), row);
                if (w < obs_width) std::memset(row + w, 0, (obs_width - w) * sizeof(float));
            }
            predeal_flag[idx] = 1;
        }
        void set_obs_features(const std::vector<uint32_t>& us, const std::vector<uint32_t>& ussr) {
            if (us.size() != num_envs || ussr.size() != num_envs)
                throw std::invalid_argument("set_obs_features: one feature set per env for each side");
            for (size_t i = 0; i < num_envs; ++i) {
                if ((us[i] | ussr[i]) & ~ts::obs_features::ALL)
                    throw std::invalid_argument("set_obs_features: unknown observation feature bits");
                feat_us[i] = us[i];
                feat_ussr[i] = ussr[i];
                seat(i);
            }
            resize_obs();
            refresh_all();
        }
        void set_obs_features_env(size_t idx, uint32_t us, uint32_t ussr) {
            if (idx >= num_envs) throw std::out_of_range("set_obs_features_env: env index");
            if ((us | ussr) & ~ts::obs_features::ALL)
                throw std::invalid_argument("set_obs_features_env: unknown observation feature bits");
            const size_t before = obs_width;
            feat_us[idx] = us;
            feat_ussr[idx] = ussr;
            seat(idx);
            size_t w = ts::OBS_SIZE_V23;
            for (size_t i = 0; i < num_envs; ++i) {
                w = std::max(w, ts::OBS_SIZE_V23 + ts::obs_features::extra_width(feat_us[i]));
                w = std::max(w, ts::OBS_SIZE_V23 + ts::obs_features::extra_width(feat_ussr[i]));
            }
            if (w != before) { obs_width = w; obs_buffer.assign(num_envs * obs_width, 0.0f); alloc_predeal(); refresh_all(); }
            else refresh_single(idx);
        }

        ts::ActionView view_for(size_t idx, ts::Player p) const {
            if (p == ts::Player::US) return static_cast<ts::ActionView>(view_us[idx]);
            if (p == ts::Player::USSR) return static_cast<ts::ActionView>(view_ussr[idx]);
            return ts::ActionView::E4;
        }

        ts::Player decider(size_t idx) const {
            return (st(idx).ctx().decision_player != ts::Player::NONE)
                ? st(idx).ctx().decision_player : st(idx).phasing_player;
        }

        void set_merged_influence(const std::vector<bool>& us, const std::vector<bool>& ussr) {
            if (us.size() != num_envs || ussr.size() != num_envs) {
                throw std::invalid_argument("set_merged_influence: one flag per env for each side");
            }
            for (size_t i = 0; i < num_envs; ++i) {
                view_us[i] = static_cast<uint8_t>(us[i] ? ts::ActionView::MERGED : ts::ActionView::E4);
                view_ussr[i] = static_cast<uint8_t>(ussr[i] ? ts::ActionView::MERGED : ts::ActionView::E4);
            }
            refresh_all();  // cached masks were built in the previous view
        }

        void set_merged_influence_env(size_t idx, bool us, bool ussr) {
            if (idx >= num_envs) throw std::out_of_range("set_merged_influence_env: env index");
            view_us[idx] = static_cast<uint8_t>(us ? ts::ActionView::MERGED : ts::ActionView::E4);
            view_ussr[idx] = static_cast<uint8_t>(ussr ? ts::ActionView::MERGED : ts::ActionView::E4);
            refresh_single(idx);
        }

        static uint8_t checked_view(int v) {
            if (v < 0 || v > static_cast<int>(ts::ActionView::MACRO)) {
                throw std::invalid_argument("action view must be 0 (E4), 1 (merged, E4.1) or 2 (macro, P42)");
            }
            return static_cast<uint8_t>(v);
        }

        void set_action_views(const std::vector<int>& us, const std::vector<int>& ussr) {
            if (us.size() != num_envs || ussr.size() != num_envs) {
                throw std::invalid_argument("set_action_views: one view per env for each side");
            }
            for (size_t i = 0; i < num_envs; ++i) {
                view_us[i] = checked_view(us[i]);
                view_ussr[i] = checked_view(ussr[i]);
            }
            refresh_all();
        }

        void set_action_views_env(size_t idx, int us, int ussr) {
            if (idx >= num_envs) throw std::out_of_range("set_action_views_env: env index");
            view_us[idx] = checked_view(us);
            view_ussr[idx] = checked_view(ussr);
            refresh_single(idx);
        }

        VectorizedBatchRunner(size_t n, uint64_t base_seed)
            : num_envs(n) {
            slots.resize(n);
            obs_buffer.resize(n * obs_width);
            mask_buffer.resize(n * ts::FLAT_ACTION_SPACE_SIZE);
            view_us.assign(n, static_cast<uint8_t>(ts::ActionView::E4));
            view_ussr.assign(n, static_cast<uint8_t>(ts::ActionView::E4));
            feat_us.assign(n, 0u);
            feat_ussr.assign(n, 0u);
            for (size_t i = 0; i < n; ++i) {
                const uint64_t seed = base_seed + i * 10007 + 1;
                slots[i].modify([&](ts::GameState& g) { ts::StateMachine::init_new_game(g, seed); });
                seat(i);
            }
            refresh_all();
        }

        void reset_game(size_t idx, uint64_t seed) {
            if (idx >= num_envs) throw std::out_of_range("reset_game: env index out of range");
            slots[idx].modify([&](ts::GameState& g) { ts::StateMachine::init_new_game(g, seed); });
            seat(idx);
            refresh_single(idx);
        }

        // Rolls every unattended die in front of env `idx`. Without the check a refused ROLL_DIE
        // leaves the loop condition unchanged and this spins forever; breaking leaves the chance
        // node in place, where the empty mask reports it loudly.
        void drain_chance(size_t idx) {
            while (!ts::Engine::is_terminal(st(idx)) &&
                   st(idx).ctx().decision_player == ts::Player::NONE &&
                   st(idx).ctx().decision_type == ts::DecisionType::ROLL_DIE) {
                const ts::MicroAction chance_ma{ts::DecisionType::ROLL_DIE, 0, 0, 0};
                if (!slots[idx].step(chance_ma)) break;
            }
        }

        void refresh_single(size_t idx) {
            drain_chance(idx);
            ts::Player p = (st(idx).ctx().decision_player != ts::Player::NONE)
                ? st(idx).ctx().decision_player : st(idx).phasing_player;
            float* row = &obs_buffer[idx * obs_width];
            const size_t w = ts::extract_observation_features(st(idx), p, features_for(idx, p), row);
            if (w < obs_width) std::memset(row + w, 0, (obs_width - w) * sizeof(float));
            uint8_t* mask_row = &mask_buffer[idx * ts::FLAT_ACTION_SPACE_SIZE];
            const ts::ActionView view = view_for(idx, p);
            if (view != ts::ActionView::E4) {
                ts::Engine::get_flat_action_mask(st(idx), mask_row, view);   // another view: not cached
            } else {
                std::memcpy(mask_row, slots[idx].legal_mask(), ts::FLAT_ACTION_SPACE_SIZE);
            }
        }

        void refresh_all() {
            gomp_parallel_for(static_cast<int64_t>(num_envs), [&](int64_t i) {
                refresh_single(static_cast<size_t>(i));
            });
        }

        std::vector<int> step_flat_all(const std::vector<uint16_t>& actions, bool auto_advance = false) {
            std::vector<int> results(num_envs, 0);
            const size_t act_count = actions.size();
            gomp_parallel_for(static_cast<int64_t>(num_envs), [&](int64_t i) {
                if (capture_predeal) predeal_flag[i] = 0;
                if (static_cast<size_t>(i) >= act_count) return;
                if (ts::Engine::is_terminal(st(i))) {
                    results[i] = 2; // Terminal
                    return;
                }
                // A composed E4.1 action (the decider in the merged view) is both E4 steps,
                // atomically; anything else is the E4 step, validated against the cached mask.
                // Chance nodes are drained below exactly as for any other action.
                const bool ok = slots[i].step_flat(
                    actions[i], view_for(static_cast<size_t>(i), decider(static_cast<size_t>(i))));
                if (ok) {
                    if (auto_advance) {
                        if (capture_predeal) {
                            for (size_t k = 0; k < 256; ++k) {
                                if (ts::rollout::at_turn_cleanup(st(i))) record_predeal(static_cast<size_t>(i));
                                if (slots[i].auto_advance(1) == 0) break;
                            }
                        }
                        slots[i].auto_advance();
                    } else {
                        drain_chance(static_cast<size_t>(i));
                    }
                }
                results[i] = ok ? 1 : 0;
                refresh_single(static_cast<size_t>(i));
            });
            return results;
        }

        nb::ndarray<nb::numpy, float, nb::ndim<2>> get_observations() {
            size_t shape[2] = { num_envs, obs_width };
            return nb::ndarray<nb::numpy, float, nb::ndim<2>>(obs_buffer.data(), 2, shape);
        }

        nb::ndarray<nb::numpy, uint8_t, nb::ndim<2>> get_action_masks() {
            size_t shape[2] = { num_envs, ts::FLAT_ACTION_SPACE_SIZE };
            return nb::ndarray<nb::numpy, uint8_t, nb::ndim<2>>(mask_buffer.data(), 2, shape);
        }

        // P43 V3: which envs' pending decisions open a CANONICAL_LEARNED run whose order has not been given.
        std::vector<bool> priority_wanted() const {
            std::vector<bool> res(num_envs);
            for (size_t i = 0; i < num_envs; ++i) res[i] = ts::canonical::priority_wanted(st(i));
            return res;
        }

        // P43 V3: give env `idx`'s pending run its order (ranks[c]: country c's rank, 0 = highest priority).
        void set_run_priority(size_t idx, const std::vector<int>& ranks) {
            if (idx >= num_envs) throw std::out_of_range("set_run_priority: env index");
            if (ranks.size() != 84) throw std::invalid_argument("set_run_priority: 84 ranks, one per country");
            uint8_t r[84];
            for (int c = 0; c < 84; ++c) {
                if (ranks[c] < 0 || ranks[c] > 83) throw std::invalid_argument("set_run_priority: ranks are 0..83");
                r[c] = static_cast<uint8_t>(ranks[c]);
            }
            bool ok = false;
            slots[idx].modify([&](ts::GameState& g) { ok = ts::canonical::set_run_priority(g, r); });
            if (!ok) throw std::invalid_argument("set_run_priority: no CANONICAL_LEARNED run pending, or not a permutation");
        }

        std::vector<int8_t> get_decision_players() const {
            std::vector<int8_t> res(num_envs);
            for (size_t i = 0; i < num_envs; ++i) {
                ts::Player p = (st(i).ctx().decision_player != ts::Player::NONE)
                    ? st(i).ctx().decision_player : st(i).phasing_player;
                res[i] = static_cast<int8_t>(p);
            }
            return res;
        }

        std::vector<bool> get_terminals() const {
            std::vector<bool> res(num_envs);
            for (size_t i = 0; i < num_envs; ++i) {
                res[i] = ts::Engine::is_terminal(st(i));
            }
            return res;
        }

        std::vector<float> get_terminal_utilities() const {
            std::vector<float> res(num_envs, 0.0f);
            for (size_t i = 0; i < num_envs; ++i) {
                res[i] = ts::Engine::get_terminal_utility(st(i));
            }
            return res;
        }

        std::vector<int8_t> get_victory_points() const {
            std::vector<int8_t> res(num_envs);
            for (size_t i = 0; i < num_envs; ++i) {
                res[i] = st(i).victory_points;
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
                    if (ts::in_hand_of(st(i).card_locations[c], opp_p)) {
                        res[i * 110 + (c - 1)] = 1.0f;
                    }
                }
            }
            return res;
        }

        std::vector<int8_t> get_turns() const {
            std::vector<int8_t> res(num_envs);
            for (size_t i = 0; i < num_envs; ++i) {
                res[i] = static_cast<int8_t>(st(i).turn);
            }
            return res;
        }
    };

    // The tree half of ai/search/batched_mcts.py (bindings/batched_search.hpp). Python runs the
    // network on each round's leaves; everything between two forward passes happens here.
    using ts_search::BatchedSearch;
    nb::class_<BatchedSearch>(m, "BatchedSearch")
        .def(nb::init<size_t, double, bool, uint32_t, bool, double>(), nb::arg("capacity"),
             nb::arg("c_puct") = 1.5, nb::arg("auto_advance") = true, nb::arg("obs_features") = 0u,
             nb::arg("merged_influence") = false, nb::arg("fpu_reduction") = 0.0,
             "Up to `capacity` independent searches, stepped together. obs_features and "
             "merged_influence are the evaluating model's view; fpu_reduction values an unvisited "
             "move at its node's value less this much, from the mover's side (0: the node's value).")
        .def_prop_ro("capacity", &BatchedSearch::capacity)
        .def_prop_ro("obs_width", &BatchedSearch::obs_width)
        .def_prop_ro("num_trees", &BatchedSearch::num_trees)
        .def_prop_ro("num_groups", &BatchedSearch::num_groups)
        .def_prop_ro("roots_phase", &BatchedSearch::roots_phase)
        .def("reset",
             nb::overload_cast<const std::vector<ts::GameState>&, int64_t,
                               const std::vector<uint64_t>&, size_t>(&BatchedSearch::reset),
             nb::arg("roots"), nb::arg("simulations"), nb::arg("rng_state"),
             nb::arg("num_groups") = 1,
             "Start one search per root, each with its own budget of `simulations`, the trees split "
             "into `num_groups` contiguous groups evaluated separately. `rng_state` is "
             "random.Random.getstate()[1]: the children's chance seeds are drawn from it exactly "
             "as the Python tree draws them.")
        .def("reset",
             nb::overload_cast<const std::vector<ts::GameState>&, const std::vector<int64_t>&,
                               const std::vector<uint64_t>&, size_t>(&BatchedSearch::reset),
             nb::arg("roots"), nb::arg("simulations"), nb::arg("rng_state"),
             nb::arg("num_groups") = 1,
             "As above with one budget per root, so searches of different sizes share their "
             "rounds.")
        .def("set_buffers",
             [](BatchedSearch& self,
                nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> obs,
                nb::ndarray<uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> masks) {
                 if (obs.shape(0) != self.capacity() || obs.shape(1) != self.obs_width())
                     throw std::invalid_argument("set_buffers: obs must be (capacity, obs_width)");
                 if (masks.shape(0) != self.capacity() || masks.shape(1) != ts::FLAT_ACTION_SPACE_SIZE)
                     throw std::invalid_argument("set_buffers: masks must be (capacity, action space)");
                 self.set_buffers(obs.data(), masks.data());
             }, nb::arg("obs"), nb::arg("masks"),
             "Write leaves into these caller-owned arrays (e.g. page-locked) instead of this "
             "object's own. The caller keeps them alive.")
        .def("group_offset", &BatchedSearch::group_offset, nb::arg("group"))
        .def("group_has_budget", &BatchedSearch::group_has_budget, nb::arg("group"))
        .def("mt_state", &BatchedSearch::mt_state,
             "The random.Random state after this search's draws, for setstate().")
        .def("select_leaves", &BatchedSearch::select_leaves, nb::arg("group"),
             "Write group `group`'s leaves awaiting evaluation into rows [group_offset, +k) of "
             "observations() and masks() and return k: its roots on the first call, then one "
             "simulation round per call. Take the groups' rounds in turn, 0, 1, ...")
        .def("expand_and_backup",
             [](BatchedSearch& self, size_t group,
                nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> probs,
                nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu> values) {
                 if (probs.shape(1) != ts::FLAT_ACTION_SPACE_SIZE)
                     throw std::invalid_argument("expand_and_backup: probs must be (k, action space)");
                 if (values.shape(0) != probs.shape(0))
                     throw std::invalid_argument("expand_and_backup: one value per row of probs");
                 self.expand_and_backup(group, probs.data(), values.data(), probs.shape(0));
             },
             nb::arg("group"), nb::arg("probs"), nb::arg("values"),
             "The network's answer for rows [0, k): softmax over masked logits, and v_win from each "
             "leaf mover's side. Fills the leaves in and backs their values up.")
        .def("observations",
             [](BatchedSearch& self) {
                 size_t shape[2] = { self.capacity(), self.obs_width() };
                 return nb::ndarray<nb::numpy, float, nb::ndim<2>>(self.obs_data(), 2, shape);
             }, nb::rv_policy::reference_internal)
        .def("masks",
             [](BatchedSearch& self) {
                 size_t shape[2] = { self.capacity(), ts::FLAT_ACTION_SPACE_SIZE };
                 return nb::ndarray<nb::numpy, uint8_t, nb::ndim<2>>(self.mask_data(), 2, shape);
             }, nb::rv_policy::reference_internal)
        .def("set_root_priors", &BatchedSearch::set_root_priors, nb::arg("index"), nb::arg("priors"))
        .def("root",
             [](BatchedSearch& self, size_t i) {
                 const ts_search::Node& r = self.root(i);
                 std::vector<int> actions;
                 std::vector<double> priors, n, w;
                 for (size_t e = 0; e < r.edge_count; ++e) {
                     const ts_search::Edge& ed = self.root_edge(i, e);
                     actions.push_back(static_cast<int>(ed.action));
                     priors.push_back(ed.prior);
                     n.push_back(ed.n);
                     w.push_back(ed.w);
                 }
                 return nb::make_tuple(r.terminal, static_cast<int>(r.mover), r.value_us,
                                       actions, priors, n, w);
             }, nb::arg("index"),
             "(terminal, mover, value_us, actions, priors, visits, value sums) of a root.")
        .def("tree_size", &BatchedSearch::tree_size, nb::arg("index"));

    nb::class_<VectorizedBatchRunner>(m, "VectorizedBatchRunner")
        .def(nb::init<size_t, uint64_t>(), nb::arg("num_envs"), nb::arg("base_seed") = 12345)
        .def_prop_ro("obs_width", [](const VectorizedBatchRunner& self) { return self.obs_width; })
        .def("set_obs_features", &VectorizedBatchRunner::set_obs_features, nb::arg("us"), nb::arg("ussr"),
             "Per env and side, the observation feature set (obs_features bits) that side's model "
             "reads. Rows are written in the decider's set and zero-padded to obs_width.")
        .def("set_obs_features_env", &VectorizedBatchRunner::set_obs_features_env,
             nb::arg("env_index"), nb::arg("us"), nb::arg("ussr"))
        .def("reset_game", &VectorizedBatchRunner::reset_game)
        .def("refresh_all", &VectorizedBatchRunner::refresh_all)
        .def("set_merged_influence", &VectorizedBatchRunner::set_merged_influence, nb::arg("us"), nb::arg("ussr"),
             "P23 / E4.1: per env and side, whether that side decides in the merged-influence view. "
             "Rebuilds the cached masks.")
        .def("set_action_views", &VectorizedBatchRunner::set_action_views, nb::arg("us"), nb::arg("ussr"),
             "Per env and side, the action view that side decides in: 0 E4, 1 merged (E4.1, P23), "
             "2 macro (P42: the until-control slots). Masks are rebuilt; steps apply each view's composed actions.")
        .def("set_action_views_env", &VectorizedBatchRunner::set_action_views_env, nb::arg("idx"), nb::arg("us"),
             nb::arg("ussr"))
        .def("set_merged_influence_env", &VectorizedBatchRunner::set_merged_influence_env,
             nb::arg("env_index"), nb::arg("us"), nb::arg("ussr"),
             "P23 / E4.1: the same for one env; rebuilds only that env's cached mask.")
        .def("step_flat_all", &VectorizedBatchRunner::step_flat_all, nb::arg("actions"), nb::arg("auto_advance") = false)
        .def("set_capture_predeal", &VectorizedBatchRunner::set_capture_predeal, nb::arg("on"),
             "P33: record each env's turn-end park (after a turn's last action, before the deal) from both "
             "sides during step_flat_all's auto-advance; read it with get_predeal after the step.")
        .def("get_predeal", [](VectorizedBatchRunner& self) {
                 if (!self.capture_predeal) throw std::runtime_error("get_predeal: capture is off (set_capture_predeal)");
                 size_t shape[2] = { self.num_envs, self.obs_width };
                 size_t fshape[1] = { self.num_envs };
                 // The runner owns the buffers, so the views hold a reference to it: dropping the
                 // runner while a view lives would otherwise leave it reading freed memory.
                 nb::handle owner = nb::find(&self);
                 return nb::make_tuple(
                     nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>(self.predeal_flag.data(), 1, fshape, owner).cast(),
                     nb::ndarray<nb::numpy, float, nb::ndim<2>>(self.predeal_us.data(), 2, shape, owner).cast(),
                     nb::ndarray<nb::numpy, float, nb::ndim<2>>(self.predeal_ussr.data(), 2, shape, owner).cast());
             },
             "(flags, obs_us, obs_ussr) of the last step: per env, whether it passed its turn-end park, and "
             "that position's observation from each side. Views into the runner: copy before the next step.")
        .def("get_observations", &VectorizedBatchRunner::get_observations, nb::rv_policy::reference_internal)
        .def("get_action_masks", &VectorizedBatchRunner::get_action_masks, nb::rv_policy::reference_internal)
        .def("get_decision_players", &VectorizedBatchRunner::get_decision_players)
        .def("priority_wanted", &VectorizedBatchRunner::priority_wanted,
             "P43 V3: per env, whether its decision opens a CANONICAL_LEARNED run whose order has not been given.")
        .def("set_run_priority", &VectorizedBatchRunner::set_run_priority, nb::arg("env_index"), nb::arg("ranks"),
             "P43 V3: give env `env_index`'s pending run its order (ranks[c]: country c's rank, 0 = highest).")
        .def("get_terminals", &VectorizedBatchRunner::get_terminals)
        .def("get_terminal_utilities", &VectorizedBatchRunner::get_terminal_utilities)
        .def("get_victory_points", &VectorizedBatchRunner::get_victory_points)
        .def("get_opponent_hands", &VectorizedBatchRunner::get_opponent_hands)
        .def("get_turns", &VectorizedBatchRunner::get_turns)
        // A copy of one env's position. It used to be a reference into the runner, through which
        // Python could change an env's state behind the runner's back (and saw it change when the
        // runner stepped); the slot's legal mask is only trustworthy if every write goes through
        // the slot. Write a position back with set_state.
        .def("get_state", [](const VectorizedBatchRunner& self, size_t idx) -> ts::GameState {
            if (idx >= self.num_envs) throw nb::index_error("env index out of range");
            return self.st(idx);
        }, nb::arg("idx"))
        // Write a whole GameState into one slot, so an environment can be started from a
        // saved mid-game position instead of a fresh deal. GameState is trivially
        // copyable, so this is a plain struct assignment. get_state hands back a mutable
        // reference, but only the top DecisionContext is exposed -- not the ctx_stack --
        // so copying field by field from Python cannot restore a position captured during
        // nested card resolution, and would corrupt it silently.
        .def("set_state", [](VectorizedBatchRunner& self, size_t idx, const ts::GameState& s) {
            if (idx >= self.num_envs) throw nb::index_error("env index out of range");
            self.slots[idx] = ts::CachedState(s);
            self.seat(idx);   // the position's seats are this env's models, whoever saved it
        }, nb::arg("idx"), nb::arg("state"))
        .def("compute_useful_actions_potentials", [](VectorizedBatchRunner& self, const std::vector<int8_t>& acting_players) {
            size_t n = self.num_envs;
            std::vector<float> potentials(n);
            gomp_parallel_for(static_cast<int64_t>(n), [&](int64_t i) {
                // Strategic potential Phi(s) strictly defined from US perspective
                potentials[i] = ts::Scoring::compute_useful_actions_potential(self.st(static_cast<size_t>(i)), ts::Player::US);
            });
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
