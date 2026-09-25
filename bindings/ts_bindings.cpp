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
#include "state_json.hpp"

// The batch runner's parallel loops go through libgomp's own entry point, not `#pragma omp`.
//
// Every process that steps the engine also runs torch, and torch ships libgomp. The dynamic
// loader shares a library by soname, so an extension that needs `libgomp.so.1` gets the very
// copy torch loaded: one OpenMP runtime, one thread pool. `#pragma omp` ties that to the
// compiler -- GCC emits GOMP_* calls, clang emits __kmpc_* calls into LLVM's libomp, and clang
// with `-fopenmp=libgomp` silently emits *serial* code -- so under clang the engine brought a
// second pool of spinning workers next to torch's and lost 5-7% on a rollout loop even though
// its own code was ~20% faster. Calling GOMP_parallel directly keeps the single pool under any
// compiler. It is the ABI every GCC-compiled OpenMP binary calls, so it cannot change under us.
// tests/bindings/test_build_toolchain.py fails if a second pool ever comes back.
extern "C" {
void GOMP_parallel(void (*fn)(void*), void* data, unsigned num_threads, unsigned flags);
int omp_get_thread_num(void);
int omp_get_num_threads(void);
}

namespace {

// `for (i = 0; i < n; ++i) body(i)` across the OpenMP team, split into contiguous equal chunks
// -- what `schedule(static)` does. Each iteration must be independent (they are: one env each).
template <class F>
void gomp_parallel_for(int64_t n, F&& body) {
    struct Job {
        int64_t n;
        F* body;
    };
    Job job{n, &body};
    GOMP_parallel(
        [](void* p) {
            const Job* j = static_cast<const Job*>(p);
            const int64_t team = omp_get_num_threads();
            const int64_t chunk = (j->n + team - 1) / team;
            const int64_t lo = static_cast<int64_t>(omp_get_thread_num()) * chunk;
            const int64_t hi = std::min(j->n, lo + chunk);
            for (int64_t i = lo; i < hi; ++i) (*j->body)(i);
        },
        &job, /*num_threads=*/0 /* OMP_NUM_THREADS / the team torch sized */, /*flags=*/0);
}

}  // namespace

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
        .def_static("get_flat_action_mask", [](const ts::GameState& state, bool merged_influence) {
            size_t shape[1] = { ts::FLAT_ACTION_SPACE_SIZE };
            uint8_t* data = new uint8_t[ts::FLAT_ACTION_SPACE_SIZE];
            ts::Engine::get_flat_action_mask(state, data, merged_influence);
            nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });
            return nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>(data, 1, shape, owner);
        }, nb::arg("state"), nb::arg("merged_influence") = false,
           "The flat legal mask. merged_influence=True is the E4.1 view (P23): at an op-choice "
           "node, NODE slot X means 'influence, first point in X'.")
        .def_static("try_step_flat", [](ts::GameState& state, uint16_t action_idx, bool auto_advance,
                                        bool merged_influence) {
            return ts::Engine::step_flat(state, action_idx, auto_advance, merged_influence);
        }, nb::arg("state"), nb::arg("action_idx"), nb::arg("auto_advance") = false,
           nb::arg("merged_influence") = false,
           "Advance one flat action, returning False if the engine refuses it. For probing only.")
        .def_static("step_flat", [](ts::GameState& state, uint16_t action_idx, bool auto_advance,
                                    bool merged_influence) {
            if (!ts::Engine::step_flat(state, action_idx, auto_advance, merged_influence)) {
                throw_illegal_action(state, "flat action " + std::to_string(static_cast<int>(action_idx)));
            }
        }, nb::arg("state"), nb::arg("action_idx"), nb::arg("auto_advance") = false,
           nb::arg("merged_influence") = false,
           "Advance one flat action, raising RuntimeError if the engine refuses it. "
           "merged_influence=True applies a composed E4.1 action as the two E4 steps it names.")
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
    m.def("state_from_save_json", &game_state_from_save_json, nb::arg("text"),
          "state_from_save_dict for the JSON text to_save_json writes (or any JSON of that "
          "shape). Raises ValueError for text that is not JSON or not a save.");


    // Flat Action Mask & Codec exports
    m.def("get_flat_action_mask", [](const ts::GameState& state, bool merged_influence) {
        size_t shape[1] = { ts::FLAT_ACTION_SPACE_SIZE };
        uint8_t* data = new uint8_t[ts::FLAT_ACTION_SPACE_SIZE];
        ts::Engine::get_flat_action_mask(state, data, merged_influence);
        nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });
        return nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>(data, 1, shape, owner);
    }, nb::arg("state"), nb::arg("merged_influence") = false);

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
        .def_static("generate_flat_mask", [](const ts::GameState& state, bool merged_influence) {
            size_t shape[1] = { ts::FLAT_ACTION_SPACE_SIZE };
            uint8_t* data = new uint8_t[ts::FLAT_ACTION_SPACE_SIZE];
            ts::Engine::get_flat_action_mask(state, data, merged_influence);
            nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });
            return nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>(data, 1, shape, owner);
        }, nb::arg("state"), nb::arg("merged_influence") = false)
        .def_static("is_merged_influence_action", &ts::ActionMask::is_merged_influence_action)
        .def_static("decode_flat_action", &ts::ActionMask::decode_flat_action_212)
        .def_static("encode_micro_action", &ts::ActionMask::encode_micro_action_212);

    // Vectorized Batch Runner for fast parallel self-play rollouts
    struct VectorizedBatchRunner {
        std::vector<ts::GameState> states;
        std::vector<float> obs_buffer;
        std::vector<uint8_t> mask_buffer;
        // P23 / E4.1: which action view each side of each env decides in. Per side, not per env,
        // because a tournament puts an E4 agent and an E4.1 agent in the same game. The cached mask
        // is built for whoever decides next, and a step is applied in the view of whoever made it.
        // All zero is exactly E4.
        std::vector<uint8_t> merged_us;
        std::vector<uint8_t> merged_ussr;
        size_t num_envs;
        const size_t obs_width = ts::OBS_SIZE_V23;

        bool merged_for(size_t idx, ts::Player p) const {
            if (p == ts::Player::US) return merged_us[idx] != 0;
            if (p == ts::Player::USSR) return merged_ussr[idx] != 0;
            return false;
        }

        ts::Player decider(size_t idx) const {
            return (states[idx].ctx().decision_player != ts::Player::NONE)
                ? states[idx].ctx().decision_player : states[idx].phasing_player;
        }

        void set_merged_influence(const std::vector<bool>& us, const std::vector<bool>& ussr) {
            if (us.size() != num_envs || ussr.size() != num_envs) {
                throw std::invalid_argument("set_merged_influence: one flag per env for each side");
            }
            for (size_t i = 0; i < num_envs; ++i) {
                merged_us[i] = us[i] ? 1 : 0;
                merged_ussr[i] = ussr[i] ? 1 : 0;
            }
            refresh_all();  // cached masks were built in the previous view
        }

        void set_merged_influence_env(size_t idx, bool us, bool ussr) {
            if (idx >= num_envs) throw std::out_of_range("set_merged_influence_env: env index");
            merged_us[idx] = us ? 1 : 0;
            merged_ussr[idx] = ussr ? 1 : 0;
            refresh_single(idx);
        }

        VectorizedBatchRunner(size_t n, uint64_t base_seed)
            : num_envs(n) {
            states.resize(n);
            obs_buffer.resize(n * obs_width);
            mask_buffer.resize(n * ts::FLAT_ACTION_SPACE_SIZE);
            merged_us.assign(n, 0);
            merged_ussr.assign(n, 0);
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
            ts::Engine::get_flat_action_mask(states[idx], &mask_buffer[idx * ts::FLAT_ACTION_SPACE_SIZE],
                                             merged_for(idx, p));
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
                if (static_cast<size_t>(i) >= act_count) return;
                if (states[i].current_phase == ts::Phase::GAME_OVER ||
                    states[i].victory_points >= 20 ||
                    states[i].victory_points <= -20) {
                    results[i] = 2; // Terminal
                    return;
                }
                bool ok;
                if (merged_for(static_cast<size_t>(i), decider(static_cast<size_t>(i))) &&
                    ts::ActionMask::is_merged_influence_action(states[i], actions[i])) {
                    // A composed E4.1 action: both E4 steps, atomically. Chance nodes are drained
                    // below exactly as for any other action.
                    ok = ts::Engine::step_flat(states[i], actions[i], false, true);
                } else {
                    ts::MicroAction ma = ts::ActionMask::decode_flat_action_212(states[i], actions[i]);
                    ok = ts::StateMachine::step(states[i], ma);
                }
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
        .def("set_merged_influence", &VectorizedBatchRunner::set_merged_influence, nb::arg("us"), nb::arg("ussr"),
             "P23 / E4.1: per env and side, whether that side decides in the merged-influence view. "
             "Rebuilds the cached masks.")
        .def("set_merged_influence_env", &VectorizedBatchRunner::set_merged_influence_env,
             nb::arg("env_index"), nb::arg("us"), nb::arg("ussr"),
             "P23 / E4.1: the same for one env; rebuilds only that env's cached mask.")
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
            gomp_parallel_for(static_cast<int64_t>(n), [&](int64_t i) {
                // Strategic potential Phi(s) strictly defined from US perspective
                potentials[i] = ts::Scoring::compute_useful_actions_potential(self.states[i], ts::Player::US);
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
