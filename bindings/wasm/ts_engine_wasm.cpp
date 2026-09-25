// The engine for the browser workbench: a flat C API compiled to WebAssembly by Emscripten.
//
// The page drives one game at a time through this, plays models with onnxruntime-web, and keeps
// its own undo stack of raw GameState bytes (the state is trivially copyable). Everything it
// shows about a position comes from the same C++ as the Python bindings -- the display state and
// the save from state_json.cpp, the rules from the engine -- so the browser and the Python stack
// cannot disagree about a position, and a position saved in one opens in the other.
//
// Strings returned here live until the next call that returns one; the JS wrapper copies them.
#include <cstdint>
#include <cstring>
#include <string>

#include "selftest.hpp"
#include "state_json.hpp"
#include "ts/action_mask.hpp"
#include "ts/card_data.hpp"
#include "ts/engine.hpp"
#include "ts/map_data.hpp"
#include "ts/game_state.hpp"
#include "ts/observation.hpp"
#include "ts/prng.hpp"
#include "ts/state_machine.hpp"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define TS_API extern "C" EMSCRIPTEN_KEEPALIVE
#else
#define TS_API extern "C"
#endif

#ifndef TS_ENGINE_FINGERPRINT
#define TS_ENGINE_FINGERPRINT "unknown"
#endif

namespace {

namespace sj = ts::state_json;

ts::GameState g_state{};
uint8_t g_mask[ts::FLAT_ACTION_SPACE_SIZE];
float g_obs[ts::OBS_SIZE_V23];
int32_t g_decoded[4];
std::string g_text;
std::string g_error;

const char* keep(std::string s) {
    g_text = std::move(s);
    return g_text.c_str();
}

// Resolve the chance nodes the game is sitting on, as tools.lib.game_step.drain_chance does:
// `forced_die` 0 rolls from the state's RNG, 1..6 forces the face (the workbench's manual die).
bool drain(ts::GameState& s, int forced_die) {
    while (!ts::Engine::is_terminal(s) && s.ctx().decision_player == ts::Player::NONE &&
           s.ctx().decision_type == ts::DecisionType::ROLL_DIE) {
        const ts::MicroAction roll{ts::DecisionType::ROLL_DIE, static_cast<uint8_t>(forced_die), 0, 0};
        if (!ts::Engine::step(s, roll)) return false;
    }
    return true;
}

// All or nothing: a refused action, or a refused forced die, leaves the game where it was.
int apply(const ts::MicroAction& action, int forced_die) {
    if (ts::Engine::is_terminal(g_state)) { g_error = "the game is over"; return 0; }
    if (forced_die < 0 || forced_die > 6) { g_error = "forced die must be 0..6"; return 0; }
    const ts::GameState before = g_state;
    if (!ts::Engine::step(g_state, action)) {
        g_state = before;
        g_error = "the engine refused the action";
        return 0;
    }
    if (!drain(g_state, forced_die)) {
        g_state = before;
        g_error = "the engine refused the die roll";
        return 0;
    }
    return 1;
}

ts::Player player_from_int(int p) {
    return p > 0 ? ts::Player::US : (p < 0 ? ts::Player::USSR : ts::Player::NONE);
}

}  // namespace

// ---- identity ---------------------------------------------------------------------------------

// tools/lib/engine_fingerprint.py over the sources this build was made from: the page shows it,
// and compares it with the fingerprint a model was exported next to.
TS_API const char* ts_fingerprint() { return TS_ENGINE_FINGERPRINT; }
TS_API int ts_obs_size() { return static_cast<int>(ts::OBS_SIZE_V23); }
TS_API int ts_action_size() { return static_cast<int>(ts::FLAT_ACTION_SPACE_SIZE); }

// The flat action space's layout, so the page never keeps its own copy of these offsets.
TS_API const char* ts_action_layout_json() {
    namespace fs = ts::flat_slots;
    sj::Value v = sj::Value::object();
    v.set("size", static_cast<int>(fs::SIZE));
    sj::Value off = sj::Value::object();
    off.set("card", static_cast<int>(fs::CARD));
    off.set("play_mode", static_cast<int>(fs::RESOLUTION));
    off.set("op_mode", static_cast<int>(fs::OP_MODE));
    off.set("roll_die", static_cast<int>(fs::ROLL_DIE));
    off.set("node", static_cast<int>(fs::NODE));
    off.set("branch", static_cast<int>(fs::BRANCH));
    off.set("defcon_value", static_cast<int>(fs::DEFCON_VALUE));
    off.set("region", static_cast<int>(fs::REGION));
    v.set("offsets", std::move(off));
    v.set("confirm_done_index", static_cast<int>(fs::CONFIRM_DONE));
    v.set("ops_influence_index", static_cast<int>(fs::OPS_INFLUENCE));
    sj::Value dt = sj::Value::object();
    dt.set("SELECT_CARD", static_cast<int>(ts::DecisionType::SELECT_CARD));
    dt.set("SELECT_PLAY_MODE", static_cast<int>(ts::DecisionType::SELECT_PLAY_MODE));
    dt.set("CHOOSE_TIMING_BRANCH", static_cast<int>(ts::DecisionType::CHOOSE_TIMING_BRANCH));
    dt.set("SELECT_OP_MODE", static_cast<int>(ts::DecisionType::SELECT_OP_MODE));
    dt.set("POINT_NODE", static_cast<int>(ts::DecisionType::POINT_NODE));
    dt.set("CHOOSE_BRANCH", static_cast<int>(ts::DecisionType::CHOOSE_BRANCH));
    dt.set("ROLL_DIE", static_cast<int>(ts::DecisionType::ROLL_DIE));
    v.set("decision_types", std::move(dt));
    return keep(v.dump());
}

TS_API const char* ts_last_error() { return g_error.c_str(); }

// The engine's own names, for the action log -- the same text the Python session wrote.
TS_API const char* ts_card_name(int id) {
    return (id >= 1 && id <= 110) ? keep(std::string(ts::CardData::get_card_name(static_cast<uint8_t>(id)))) : "";
}
TS_API int ts_card_ops(int id) {
    return (id >= 1 && id <= 110) ? static_cast<int>(ts::CardData::get_card(static_cast<uint8_t>(id)).ops) : 0;
}
TS_API const char* ts_country_name(int id) {
    return (id >= 0 && id < 84) ? keep(std::string(ts::MapData::get_country_name(static_cast<uint8_t>(id)))) : "";
}

// ---- the position -----------------------------------------------------------------------------

TS_API void ts_new_game(uint32_t seed_lo, uint32_t seed_hi) {
    ts::Engine::init_game(g_state, (static_cast<uint64_t>(seed_hi) << 32) | seed_lo);
    drain(g_state, 0);
}

// Raw GameState bytes, for the page's undo stack: copy out ts_state_size() bytes from
// ts_state_ptr(), and write them back to restore. Trivially copyable by the engine's invariant.
TS_API int ts_state_size() { return static_cast<int>(sizeof(ts::GameState)); }
TS_API uint8_t* ts_state_ptr() { return reinterpret_cast<uint8_t*>(&g_state); }

TS_API const char* ts_display_json() { return keep(sj::display_state(g_state).dump()); }
TS_API const char* ts_save_json() { return keep(sj::save(g_state).dump()); }

// Open a saved position (a shared link). Returns 1, or 0 with ts_last_error() -- and a save
// that does not survive load -> save unchanged is refused rather than opened approximately.
TS_API int ts_load_save_json(const char* text) {
    sj::Value v;
    std::string error;
    if (!sj::Value::parse(text, v, &error)) { g_error = "not JSON: " + error; return 0; }
    ts::GameState s{};
    if (!sj::load_save(v, s, &error)) { g_error = "not a save: " + error; return 0; }
    if (sj::save(s).dump() != v.dump()) { g_error = "the save does not round-trip through the engine"; return 0; }
    if (!drain(s, 0)) { g_error = "the saved position's die roll was refused"; return 0; }
    g_state = s;
    return 1;
}

TS_API int ts_is_terminal() { return ts::Engine::is_terminal(g_state) ? 1 : 0; }
TS_API float ts_terminal_utility() { return ts::Engine::get_terminal_utility(g_state); }
TS_API const char* ts_ending_reason() { return sj::ending_reason(g_state); }
TS_API int ts_decision_player() { return static_cast<int>(g_state.ctx().decision_player); }
TS_API int ts_decision_type() { return static_cast<int>(g_state.ctx().decision_type); }

// ---- acting -----------------------------------------------------------------------------------

// A click: one MicroAction, then the chance nodes it lands on (forced_die 0 = roll).
TS_API int ts_step(int decision_type, int primary, int secondary, int flags, int forced_die) {
    const ts::MicroAction a{static_cast<ts::DecisionType>(decision_type), static_cast<uint8_t>(primary),
                            static_cast<uint8_t>(secondary), static_cast<uint8_t>(flags)};
    if (a.decision_type != g_state.ctx().decision_type) {
        g_error = "the action is for a different decision than the one open";
        return 0;
    }
    return apply(a, forced_die);
}

// The flat legal mask in either action view (E4, or E4.1's merged influence).
TS_API const uint8_t* ts_mask(int merged) {
    ts::Engine::get_flat_action_mask(g_state, g_mask, merged != 0);
    return g_mask;
}

// The MicroAction a flat index stands for at the open decision: [type, primary, secondary, flags].
TS_API const int32_t* ts_decode_flat(int idx) {
    const ts::MicroAction a = ts::ActionMask::decode_flat_action_212(g_state, static_cast<uint16_t>(idx));
    g_decoded[0] = static_cast<int32_t>(a.decision_type);
    g_decoded[1] = a.primary_id;
    g_decoded[2] = a.secondary_id;
    g_decoded[3] = a.flags;
    return g_decoded;
}

// Whether a flat index is an E4.1 composed action here ("Ops for influence, first point at X").
TS_API int ts_is_merged_influence_action(int idx) {
    return ts::ActionMask::is_merged_influence_action(g_state, static_cast<uint16_t>(idx)) ? 1 : 0;
}

// The observation from one side's perspective (1 = US, -1 = USSR), OBS_SIZE floats.
TS_API const float* ts_observation(int player) {
    ts::ObservationBufferV23 ob;
    ts::Observation::extract(g_state, player_from_int(player), &ob);
    std::memcpy(g_obs, reinterpret_cast<const float*>(&ob), sizeof(g_obs));
    return g_obs;
}

// ---- the workbench's debug tools --------------------------------------------------------------

TS_API int ts_set_country(int cid, int us, int ussr) {
    if (cid < 0 || cid >= 84 || us < 0 || us > 255 || ussr < 0 || ussr > 255) { g_error = "out of range"; return 0; }
    g_state.countries[cid].us_influence = static_cast<uint8_t>(us);
    g_state.countries[cid].ussr_influence = static_cast<uint8_t>(ussr);
    return 1;
}
TS_API int ts_set_defcon(int v) {
    if (v < 1 || v > 5) { g_error = "DEFCON must be 1..5"; return 0; }
    g_state.defcon = static_cast<uint8_t>(v);
    return 1;
}
TS_API int ts_set_vp(int v) {
    if (v < -20 || v > 20) { g_error = "VP must be -20..20"; return 0; }
    g_state.victory_points = static_cast<int8_t>(v);
    return 1;
}

// ---- parity -----------------------------------------------------------------------------------

// The shared whole-game digest (bindings/selftest.hpp): the Python module computes the same
// number natively, so tests/web/test_wasm_engine.py can compare the two builds directly.
TS_API uint32_t ts_selftest(int games, uint32_t* steps_out) { return ts::selftest::digest(games, steps_out); }
