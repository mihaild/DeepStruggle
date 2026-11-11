// The engine for the browser workbench: a flat C API compiled to WebAssembly by Emscripten.
//
// The page drives one game at a time through this, plays models with onnxruntime-web, and keeps
// its own undo stack of raw GameState bytes (the state is trivially copyable). Everything it
// shows about a position comes from the same C++ as the Python bindings -- the display state and
// the save from state_json.cpp, the rules from the engine -- so the browser and the Python stack
// cannot disagree about a position, and a position saved in one opens in the other.
//
// Strings returned here live until the next call that returns one; the JS wrapper copies them.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "belief.hpp"
#include "rollout_batch.hpp"
#include "selftest.hpp"
#include "state_json.hpp"
#include "ts/action_mask.hpp"
#include "ts/card_data.hpp"
#include "ts/constants.hpp"
#include "ts/engine.hpp"
#include "ts/map_data.hpp"
#include "ts/game_state.hpp"
#include "ts/observation.hpp"
#include "ts/prng.hpp"
#include "ts/scoring.hpp"
#include "ts/space_race.hpp"
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
// What the last click itself did: the state after its action, or after its last die roll if it
// rolled one -- before the chance nodes that follow without a roll. The turn's cleanup is one: an
// action round that ends the turn stops on it, and resolving it (final VP, the military-ops reset,
// DEFCON, the next deal) is an engine step, and every engine step clears the roll record -- so read
// from the end state, the realignment or coup that ended the turn had no dice in the page's log, and
// the turn's end read as part of it. The page describes the click from here and the rest apart.
// Display only: the game is g_state, stepped exactly as before.
ts::GameState g_settled{};
// Every die the last click rolled, in order, as the display state writes a roll ("die_roll"). One
// click can roll more than once -- a realignment with the opponent's war card, whose event fires in
// the same click and rolls too -- and each engine step keeps only its own roll.
std::vector<sj::Value> g_rolls;
// Every Event's change to VP and DEFCON over the same steps as g_rolls: the second headline commit
// resolves both headlines in one step, and the click's before/after shows only their sum.
std::vector<sj::Value> g_event_tracks;
uint8_t g_mask[ts::FLAT_ACTION_SPACE_SIZE];
float g_obs[ts::OBS_SIZE_V23 + ts::obs_features::extra_width(ts::obs_features::ALL)];
int32_t g_decoded[4];
std::string g_text;
std::string g_error;

const char* keep(std::string s) {
    g_text = std::move(s);
    return g_text.c_str();
}

// Resolve the chance nodes the game is sitting on, as tools.lib.game_step.drain_chance does:
// `forced_die` 0 rolls from the state's RNG, 1..6 forces the face (the workbench's manual die).
// The roll a state's last step made, as the display state writes it.
void keep_roll(const ts::GameState& s) {
    if (s.last_roll.type == ts::RollType::NONE) return;
    const sj::Value display = sj::display_state(s);   // held: find() points into it
    if (const sj::Value* r = display.find("die_roll")) g_rolls.push_back(*r);
}

// The Events a state's last step recorded, as the display state writes them.
void keep_event_tracks(const ts::GameState& s) {
    if (s.last_event_track_count == 0) return;
    const sj::Value display = sj::display_state(s);   // held: find() points into it
    if (const sj::Value* t = display.find("event_tracks")) {
        for (const sj::Value& e : t->items()) g_event_tracks.push_back(e);
    }
}

// `settled`, if given, takes the state after each step that rolled a die (see g_settled), and that
// step's roll joins g_rolls. `forced_opp_die` is the other die of an opposed roll (a realignment:
// the roller's die is `forced_die`, the opponent's this one), 0 to roll it.
bool drain(ts::GameState& s, int forced_die, ts::GameState* settled = nullptr, int forced_opp_die = 0) {
    while (!ts::Engine::is_terminal(s) && s.ctx().decision_player == ts::Player::NONE &&
           s.ctx().decision_type == ts::DecisionType::ROLL_DIE) {
        const ts::MicroAction roll{ts::DecisionType::ROLL_DIE, static_cast<uint8_t>(forced_die),
                                   static_cast<uint8_t>(forced_opp_die), 0};
        if (!ts::Engine::step(s, roll)) return false;
        if (settled && s.last_roll.type != ts::RollType::NONE) {
            *settled = s;
            keep_roll(s);
            keep_event_tracks(s);
        }
    }
    return true;
}

// All or nothing: a refused action, or a refused forced die, leaves the game where it was.
int apply(const ts::MicroAction& action, int forced_die, int forced_opp_die = 0) {
    if (ts::Engine::is_terminal(g_state)) { g_error = "the game is over"; return 0; }
    if (forced_die < 0 || forced_die > 6) { g_error = "forced die must be 0..6"; return 0; }
    if (forced_opp_die < 0 || forced_opp_die > 6) { g_error = "forced die must be 0..6"; return 0; }
    const ts::GameState before = g_state;
    if (!ts::Engine::step(g_state, action)) {
        g_state = before;
        g_error = "the engine refused the action";
        return 0;
    }
    g_settled = g_state;
    g_rolls.clear();
    g_event_tracks.clear();
    keep_roll(g_state);
    keep_event_tracks(g_state);
    if (!drain(g_state, forced_die, &g_settled, forced_opp_die)) {
        g_state = before;
        g_error = "the engine refused the die roll";
        return 0;
    }
    return 1;
}

ts::Player player_from_int(int p) {
    return p > 0 ? ts::Player::US : (p < 0 ? ts::Player::USSR : ts::Player::NONE);
}

// Numbers handed in by the page (a network's logits, the critic's values): a NaN fails every
// comparison, so a masked argmax over it silently picks the lowest legal index. Refused instead;
// -inf (a masked logit) is a number and passes.
bool any_nan(const float* v, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(v[i])) return true;
    }
    return false;
}

// rollout::Stop and rollout::Leaf by name, in enum order: the names of the search= grammar
// (web/ui/src/analysis/rollout_spec.ts) and of ai/search/rollout_root.py's STOPS / LEAF_SIDES.
constexpr const char* kStopNames[] = {"boundaries", "turn", "later", "earlier", "predeal", "earlier_predeal"};
static_assert(static_cast<size_t>(ts::rollout::Stop::kEarlierPredeal) + 1 == std::size(kStopNames));
constexpr const char* kLeafNames[] = {"mover", "decider", "own"};
static_assert(static_cast<size_t>(ts::rollout::Leaf::kOwn) + 1 == std::size(kLeafNames));

}  // namespace

// ---- identity ---------------------------------------------------------------------------------

// tools/lib/engine_fingerprint.py over the sources this build was made from: the page shows it,
// and compares it with the fingerprint a model was exported next to.
TS_API const char* ts_fingerprint() { return TS_ENGINE_FINGERPRINT; }
TS_API int ts_obs_size() { return static_cast<int>(ts::OBS_SIZE_V23); }
TS_API int ts_action_size() { return static_cast<int>(ts::FLAT_ACTION_SPACE_SIZE); }
// The base slots (E4 and E4.1); a base network's logits are this wide and padded to ts_action_size (P42).
TS_API int ts_base_action_size() { return static_cast<int>(ts::flat_slots::BASE_SIZE); }

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
    dt.set("NONE", static_cast<int>(ts::DecisionType::NONE));
    v.set("decision_types", std::move(dt));
    // Resolution (how a card is played) and OpMode, by name.
    sj::Value res = sj::Value::object();
    res.set("EVENT", static_cast<int>(ts::Resolution::EVENT));
    res.set("SPACE", static_cast<int>(ts::Resolution::SPACE));
    res.set("OPS_INFLUENCE", static_cast<int>(ts::Resolution::OPS_INFLUENCE));
    res.set("OPS_COUP", static_cast<int>(ts::Resolution::OPS_COUP));
    res.set("OPS_REALIGN", static_cast<int>(ts::Resolution::OPS_REALIGN));
    v.set("resolutions", std::move(res));
    sj::Value om = sj::Value::object();
    om.set("INFLUENCE", static_cast<int>(ts::OpMode::INFLUENCE));
    om.set("COUP", static_cast<int>(ts::OpMode::COUP));
    om.set("REALIGN", static_cast<int>(ts::OpMode::REALIGN));
    v.set("op_modes", std::move(om));
    // The Region enum: Chernobyl's branch and the region head's order.
    sj::Value rg = sj::Value::object();
    rg.set("EUROPE", static_cast<int>(ts::Region::EUROPE));
    rg.set("ASIA", static_cast<int>(ts::Region::ASIA));
    rg.set("MIDDLE_EAST", static_cast<int>(ts::Region::MIDDLE_EAST));
    rg.set("AFRICA", static_cast<int>(ts::Region::AFRICA));
    rg.set("CENTRAL_AMERICA", static_cast<int>(ts::Region::CENTRAL_AMERICA));
    rg.set("SOUTH_AMERICA", static_cast<int>(ts::Region::SOUTH_AMERICA));
    v.set("regions", std::move(rg));
    // The cards the page words on its own (the HUD's prompts, the log's event choices, the scoring
    // badges), by their ts::card_ids names -- never as literals of the page's.
    sj::Value ids = sj::Value::object();
#define TS_CARD_ID(n) ids.set(#n, static_cast<int>(ts::card_ids::n))
    TS_CARD_ID(ASIA_SCORING);
    TS_CARD_ID(EUROPE_SCORING);
    TS_CARD_ID(MIDDLE_EAST_SCORING);
    TS_CARD_ID(AFRICA_SCORING);
    TS_CARD_ID(CENTRAL_AMERICA_SCORING);
    TS_CARD_ID(SOUTH_AMERICA_SCORING);
    TS_CARD_ID(SE_ASIA_SCORING);
    TS_CARD_ID(FIVE_YEAR_PLAN);
    TS_CARD_ID(BLOCKADE);
    TS_CARD_ID(WARSAW_PACT_FORMED);
    TS_CARD_ID(OLYMPIC_GAMES);
    TS_CARD_ID(UN_INTERVENTION);
    TS_CARD_ID(SALT_NEGOTIATIONS);
    TS_CARD_ID(SUMMIT);
    TS_CARD_ID(HOW_I_LEARNED_TO_STOP_WORRYING);
    TS_CARD_ID(MISSILE_ENVY);
    TS_CARD_ID(GRAIN_SALES_TO_SOVIETS);
    TS_CARD_ID(SHUTTLE_DIPLOMACY);
    TS_CARD_ID(ASK_NOT_WHAT_YOUR_COUNTRY_CAN_DO_FOR_YOU);
    TS_CARD_ID(STAR_WARS);
    TS_CARD_ID(TERRORISM);
    TS_CARD_ID(CHERNOBYL);
    TS_CARD_ID(ALDRICH_AMES);
    TS_CARD_ID(WARGAMES);
    TS_CARD_ID(OUR_MAN_IN_TEHRAN);
    TS_CARD_ID(EAGLE_BEAR_LANDED_DISCARD);
#undef TS_CARD_ID
    v.set("card_ids", std::move(ids));
    // The rollout root's stop rules and leaf sides by name, in enum order: ts_rollout_reset takes
    // the index.
    sj::Value stops = sj::Value::array();
    for (const char* n : kStopNames) stops.push(n);
    v.set("rollout_stops", std::move(stops));
    sj::Value leaves = sj::Value::array();
    for (const char* n : kLeafNames) leaves.push(n);
    v.set("rollout_leaves", std::move(leaves));
    // The Space Race track as the engine plays it (SpaceRace::get_box_info): box 0, the start, to
    // box 8, the Space Station.
    sj::Value boxes = sj::Value::array();
    for (uint8_t b = 0; b <= 8; ++b) {
        const ts::SpaceBoxInfo& info = ts::SpaceRace::get_box_info(b);
        sj::Value box = sj::Value::object();
        box.set("min_ops", static_cast<int>(info.min_ops));
        box.set("max_roll", static_cast<int>(info.max_roll));
        box.set("vp_first", static_cast<int>(info.vp_first));
        box.set("vp_second", static_cast<int>(info.vp_second));
        boxes.push(std::move(box));
    }
    v.set("space_boxes", std::move(boxes));
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
// The display state of what the last click itself did, before the roll-less chance steps after it
// (g_settled): its die roll, and the board before the turn's end.
TS_API const char* ts_settled_display_json() { return keep(sj::display_state(g_settled).dump()); }
// Every die the last click rolled, in order (g_rolls): a JSON array of display-state rolls.
TS_API const char* ts_click_rolls_json() {
    sj::Value a = sj::Value::array();
    for (const sj::Value& r : g_rolls) a.push(r);
    return keep(a.dump());
}
TS_API const char* ts_click_event_tracks_json() {
    sj::Value a = sj::Value::array();
    for (const sj::Value& t : g_event_tracks) a.push(t);
    return keep(a.dump());
}
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

// ---- what each scoring card would score now ---------------------------------------------------
//
// The map's region badges and summaries show what Scoring::evaluate_region (and, for Southeast
// Asia, score_southeast_asia) make of a position -- the engine's own scoring, never the page's
// arithmetic, which had drifted (Shuttle Diplomacy). The position is any display state the page
// draws: the live game, a replay's snapshot or a player's view, so it is read from that state's
// countries and flags rather than from the game held. Scoring reads nothing else.

namespace {

// Effect flag name -> bit, read off the display writer itself (one display state per bit), so the
// names are exactly the ones a display state carries.
const std::vector<std::pair<std::string, uint64_t>>& flag_bits() {
    static const std::vector<std::pair<std::string, uint64_t>> bits = [] {
        std::vector<std::pair<std::string, uint64_t>> out;
        ts::GameState s{};
        ts::Engine::init_game(s, 1);
        for (int b = 0; b < 64; ++b) {
            s.persistent_effects = uint64_t{1} << b;
            const sj::Value d = sj::display_state(s);   // held: find() points into it
            if (const sj::Value* f = d.find("flags")) {
                for (const sj::Value& n : f->items()) out.emplace_back(n.as_string(), s.persistent_effects);
            }
        }
        return out;
    }();
    return bits;
}

const char* status_name(ts::RegionalStatus s) {
    switch (s) {
        case ts::RegionalStatus::PRESENCE: return "PRESENCE";
        case ts::RegionalStatus::DOMINATION: return "DOMINATION";
        case ts::RegionalStatus::CONTROL: return "CONTROL";
        default: return "NONE";
    }
}

template <typename F>
void each_country(const std::array<uint64_t, 2>& mask, F&& f) {
    for (size_t word = 0; word < 2; ++word) {
        for (uint64_t bits = mask[word]; bits != 0; bits &= bits - 1) {
            f(static_cast<uint8_t>(word * 64 + static_cast<size_t>(std::countr_zero(bits))));
        }
    }
}

sj::Value side_json(const char* status, int countries, int battlegrounds, int adjacent, int base_vp, bool base_known,
                    int vp, sj::Value controlled) {
    sj::Value v = sj::Value::object();
    v.set("status", status ? sj::Value(status) : sj::Value());
    v.set("countries", countries);
    v.set("battlegrounds", battlegrounds);
    v.set("adjacent", adjacent);
    v.set("base_vp", base_known ? sj::Value(base_vp) : sj::Value());
    v.set("vp", vp);
    v.set("controlled", std::move(controlled));
    return v;
}

// One of the six regions. The counts shown are those the scoring *counted*: Shuttle Diplomacy leaves
// one USSR battleground (its country, and in Asia the enemy-superpower adjacency) out, which the
// engine's summary reports in its *_scored fields.
sj::Value region_json(const ts::GameState& s, ts::Region r) {
    const ts::RegionScoreSummary sum = ts::Scoring::evaluate_region(s, r);
    const auto& mask = ts::MapData::get_region_mask(r);

    // Who controls what, each country marked a battleground when this scoring counts it as one (Formosan
    // Resolution makes Taiwan one): the scoring of that country alone.
    ts::Player control[84];
    for (uint8_t cid = 0; cid < 84; ++cid) control[cid] = ts::Player::NONE;
    sj::Value us_list = sj::Value::array(), ussr_list = sj::Value::array();
    int map_bgs = 0, bgs_held_on_map = 0;
    each_country(mask, [&](uint8_t cid) {
        const bool map_bg = ts::MapData::get_country(cid).battleground;
        map_bgs += map_bg ? 1 : 0;
        const ts::Player who = ts::Scoring::get_country_control(s, cid);
        if (who == ts::Player::NONE) return;
        bgs_held_on_map += map_bg ? 1 : 0;
        control[cid] = who;
        const ts::RegionScoreSummary alone = ts::Scoring::evaluate_region_with_control(s, r, control);
        control[cid] = ts::Player::NONE;
        sj::Value c = sj::Value::object();
        c.set("id", static_cast<int>(cid));
        c.set("battleground", (who == ts::Player::US ? alone.us_battlegrounds : alone.ussr_battlegrounds) > 0);
        (who == ts::Player::US ? us_list : ussr_list).push(std::move(c));
    });
    // The battlegrounds there are: the map's, and those a scoring counts besides (Taiwan).
    const int extra = sum.us_battlegrounds + sum.ussr_battlegrounds - bgs_held_on_map;

    // A scoring that ends the game (Control of Europe): what score_region does with it.
    ts::GameState probe = s;
    probe.victory_points = 0;
    probe.clear_flag(ts::effect_bits::EUROPE_CONTROL_WIN);
    ts::Scoring::score_region(probe, r);
    const bool wins = probe.has_flag(ts::effect_bits::EUROPE_CONTROL_WIN);

    sj::Value v = sj::Value::object();
    v.set("slot", static_cast<int>(r));
    v.set("battlegrounds", map_bgs + std::max(0, extra));
    v.set("net", static_cast<int>(sum.net_delta));
    v.set("wins", wins ? sj::Value(probe.victory_points > 0 ? "US" : "USSR") : sj::Value());
    v.set("us", side_json(status_name(sum.us_status), sum.us_countries, sum.us_battlegrounds,
                          sum.us_superpower_adjacent,
                          sum.us_score - sum.us_battlegrounds - sum.us_superpower_adjacent, true,
                          sum.us_score, std::move(us_list)));
    sj::Value ussr = side_json(status_name(sum.ussr_status), sum.ussr_countries_scored,
                               sum.ussr_battlegrounds_scored, sum.ussr_superpower_adjacent_scored,
                               sum.ussr_score - sum.ussr_battlegrounds_scored - sum.ussr_superpower_adjacent_scored,
                               true, sum.ussr_score, std::move(ussr_list));
    ussr.set("shuttle_diplomacy", sum.shuttle_diplomacy_applied);
    v.set("ussr", std::move(ussr));
    return v;
}

// Southeast Asia Scoring: score_southeast_asia gives only the net, so each side's points, and each
// country's, are what it scores with everything else in Southeast Asia emptied.
sj::Value southeast_asia_json(const ts::GameState& s) {
    const auto& mask = ts::MapData::get_southeast_asia_mask();
    auto points = [&](auto keep) {
        ts::GameState probe = s;
        probe.victory_points = 0;
        each_country(mask, [&](uint8_t cid) {
            if (!keep(cid)) probe.countries[cid].us_influence = probe.countries[cid].ussr_influence = 0;
        });
        ts::Scoring::score_southeast_asia(probe);
        return static_cast<int>(probe.victory_points);
    };
    sj::Value lists[2] = {sj::Value::array(), sj::Value::array()};
    int counts[2] = {0, 0};
    each_country(mask, [&](uint8_t cid) {
        const ts::Player who = ts::Scoring::get_country_control(s, cid);
        if (who == ts::Player::NONE) return;
        const int k = who == ts::Player::US ? 0 : 1;
        sj::Value c = sj::Value::object();
        c.set("id", static_cast<int>(cid));
        c.set("vp", std::abs(points([&](uint8_t x) { return x == cid; })));
        lists[k].push(std::move(c));
        ++counts[k];
    });
    const int us_vp = points([&](uint8_t x) { return ts::Scoring::get_country_control(s, x) == ts::Player::US; });
    const int ussr_vp = -points([&](uint8_t x) { return ts::Scoring::get_country_control(s, x) == ts::Player::USSR; });
    const int net = points([](uint8_t) { return true; });
    sj::Value v = sj::Value::object();
    v.set("slot", static_cast<int>(ts::ScoringTap::kSoutheastAsia));
    v.set("battlegrounds", 0);
    v.set("net", net);
    v.set("wins", sj::Value());
    v.set("us", side_json(nullptr, counts[0], 0, 0, 0, false, us_vp, std::move(lists[0])));
    sj::Value ussr = side_json(nullptr, counts[1], 0, 0, 0, false, ussr_vp, std::move(lists[1]));
    ussr.set("shuttle_diplomacy", false);
    v.set("ussr", std::move(ussr));
    return v;
}

}  // namespace

// What every scoring card would score in the position `state_json` describes -- {"countries":
// display-state countries, "flags": display-state flags} -- as a JSON array: the six regions in
// Region order, then Southeast Asia ("slot" 6, ScoringTap's). Per region: "battlegrounds" there,
// "net" VP (US positive), "wins" (the side a scoring would hand the game, else null), and per side
// "status", the scored "countries" / "battlegrounds" / "adjacent" (to the enemy superpower), the
// standing's "base_vp", "vp", and "controlled" [{id, battleground | vp}]; the USSR's
// "shuttle_diplomacy" says whether that card took a battleground out of its count. A flag this build does not know is listed under
// "ignored_flags". "" with ts_last_error() for a malformed state.
TS_API const char* ts_region_scores_json(const char* state_json) {
    sj::Value in;
    std::string error;
    if (!sj::Value::parse(state_json, in, &error)) { g_error = "not JSON: " + error; return ""; }
    const sj::Value* countries = in.is_object() ? in.find("countries") : nullptr;
    const sj::Value* flags = in.is_object() ? in.find("flags") : nullptr;
    if (!countries || !countries->is_object() || !flags || !flags->is_array()) {
        g_error = "a display state's countries and flags are needed";
        return "";
    }
    ts::GameState s{};
    for (const auto& [name, c] : countries->members()) {
        const sj::Value* id = c.is_object() ? c.find("id") : nullptr;
        const sj::Value* us = c.is_object() ? c.find("us_influence") : nullptr;
        const sj::Value* ussr = c.is_object() ? c.find("ussr_influence") : nullptr;
        int64_t cid = 0, a = 0, b = 0;
        uint64_t unused = 0;
        if (!id || !us || !ussr || !id->to_integral(0, 83, &cid, &unused) || !us->to_integral(0, 255, &a, &unused) ||
            !ussr->to_integral(0, 255, &b, &unused)) {
            g_error = "bad country " + name;
            return "";
        }
        s.countries[cid].us_influence = static_cast<uint8_t>(a);
        s.countries[cid].ussr_influence = static_cast<uint8_t>(b);
    }
    sj::Value ignored = sj::Value::array();
    for (const sj::Value& f : flags->items()) {
        if (f.kind() != sj::Value::Kind::String) { g_error = "a flag is not a name"; return ""; }
        bool known = false;
        for (const auto& [name, bit] : flag_bits()) {
            if (name == f.as_string()) { s.set_flag(bit); known = true; }
        }
        if (!known) ignored.push(f.as_string());
    }
    sj::Value regions = sj::Value::array();
    for (ts::Region r : {ts::Region::EUROPE, ts::Region::ASIA, ts::Region::MIDDLE_EAST, ts::Region::AFRICA,
                         ts::Region::CENTRAL_AMERICA, ts::Region::SOUTH_AMERICA}) {
        regions.push(region_json(s, r));
    }
    regions.push(southeast_asia_json(s));
    sj::Value out = sj::Value::object();
    out.set("regions", std::move(regions));
    out.set("ignored_flags", std::move(ignored));
    return keep(out.dump());
}

TS_API int ts_is_terminal() { return ts::Engine::is_terminal(g_state) ? 1 : 0; }
TS_API float ts_terminal_utility() { return ts::Engine::get_terminal_utility(g_state); }
TS_API const char* ts_ending_reason() { return sj::ending_reason(g_state); }
TS_API int ts_decision_player() { return static_cast<int>(g_state.ctx().decision_player); }
TS_API int ts_decision_type() { return static_cast<int>(g_state.ctx().decision_type); }

// ---- acting -----------------------------------------------------------------------------------

// Is `a` an answer to the open decision? Its type is the decision's, but for one action the engine
// itself excepts (StateMachine::step): UN Intervention played ON the card Grain Sales drew arrives
// card-shaped at a resolution node. Refusing it here made a move the mask offers impossible -- for
// the AI, a refused move and a game that stops. The engine's mask still has the final say.
static bool for_open_decision(const ts::MicroAction& a) {
    const auto& c = g_state.ctx();
    return a.decision_type == c.decision_type ||
           (c.decision_type == ts::DecisionType::SELECT_PLAY_MODE && c.resolving_card == ts::card_ids::GRAIN_SALES &&
            a.decision_type == ts::DecisionType::SELECT_CARD && a.primary_id == ts::card_ids::UN_INTERVENTION);
}

// A click: one MicroAction, then the chance nodes it lands on (forced_die 0 = roll).
TS_API int ts_step(int decision_type, int primary, int secondary, int flags, int forced_die) {
    const ts::MicroAction a{static_cast<ts::DecisionType>(decision_type), static_cast<uint8_t>(primary),
                            static_cast<uint8_t>(secondary), static_cast<uint8_t>(flags)};
    if (!for_open_decision(a)) {
        g_error = "the action is for a different decision than the one open";
        return 0;
    }
    return apply(a, forced_die);
}

// The same, with both dice of an opposed roll named: `forced_opp_die` is the opponent's die of a
// realignment (0 = roll it). Twilight Struggle /ai uses it to enter the dice of a game played
// elsewhere, where both players rolled real dice.
TS_API int ts_step_dice(int decision_type, int primary, int secondary, int flags, int forced_die,
                        int forced_opp_die) {
    const ts::MicroAction a{static_cast<ts::DecisionType>(decision_type), static_cast<uint8_t>(primary),
                            static_cast<uint8_t>(secondary), static_cast<uint8_t>(flags)};
    if (!for_open_decision(a)) {
        g_error = "the action is for a different decision than the one open";
        return 0;
    }
    return apply(a, forced_die, forced_opp_die);
}

// The first headline on its own, for the page's log. Steps a copy of the game with the action (and
// its dice, as ts_step_dice), stopping where the first headline is done and the second not begun
// (StateMachine::pause_between_headlines), and returns {"state": <display>, "rolls": n} -- n the
// dice rolled up to there -- or "" when the action does not get there: no headline resolving, or
// the game over with the first. The game itself is not touched. The real step's own dice record
// is cleared by the next ts_step, which is why this may run just before it.
TS_API const char* ts_first_headline_json(int decision_type, int primary, int secondary, int flags, int forced_die,
                                          int forced_opp_die) {
    const ts::MicroAction a{static_cast<ts::DecisionType>(decision_type), static_cast<uint8_t>(primary),
                            static_cast<uint8_t>(secondary), static_cast<uint8_t>(flags)};
    if (g_state.current_phase != ts::Phase::HEADLINE) return keep("");
    ts::GameState copy = g_state;
    g_rolls.clear();
    ts::StateMachine::pause_between_headlines = true;
    ts::StateMachine::paused_between_headlines = false;
    // As apply() and drain() step a click, but no further than the pause: a die node the first
    // headline left behind is not one to roll.
    bool ok = ts::Engine::step(copy, a);
    if (ok) keep_roll(copy);
    while (ok && !ts::StateMachine::paused_between_headlines && !ts::Engine::is_terminal(copy) &&
           copy.ctx().decision_player == ts::Player::NONE && copy.ctx().decision_type == ts::DecisionType::ROLL_DIE) {
        const ts::MicroAction roll{ts::DecisionType::ROLL_DIE, static_cast<uint8_t>(forced_die),
                                   static_cast<uint8_t>(forced_opp_die), 0};
        ok = ts::Engine::step(copy, roll);
        if (ok) keep_roll(copy);
    }
    const bool paused = ts::StateMachine::paused_between_headlines;
    ts::StateMachine::pause_between_headlines = false;
    ts::StateMachine::paused_between_headlines = false;
    if (!ok || !paused) return keep("");
    sj::Value out = sj::Value::object();
    out.set("state", sj::display_state(copy));
    out.set("rolls", static_cast<int>(g_rolls.size()));
    return keep(out.dump());
}

// The flat legal mask in an action view: 0 E4, 1 E4.1's merged influence, 2 P42's macro (until-control) view.
TS_API const uint8_t* ts_mask(int view) {
    if (view < 0 || view > 2) view = 0;
    ts::Engine::get_flat_action_mask(g_state, g_mask, static_cast<ts::ActionView>(view));
    return g_mask;
}

// The MicroAction a flat index stands for at the open decision: [type, primary, secondary, flags].
// Null, with ts_last_error(), for an index outside the flat action space.
TS_API const int32_t* ts_decode_flat(int idx) {
    if (idx < 0 || idx >= static_cast<int>(ts::FLAT_ACTION_SPACE_SIZE)) {
        g_error = "flat index outside the action space";
        return nullptr;
    }
    const ts::MicroAction a = ts::ActionMask::decode_flat_action_212(g_state, static_cast<uint16_t>(idx));
    g_decoded[0] = static_cast<int32_t>(a.decision_type);
    g_decoded[1] = a.primary_id;
    g_decoded[2] = a.secondary_id;
    g_decoded[3] = a.flags;
    return g_decoded;
}

// P42: how many influence points the until-control slot `idx` places here (0 where it is not offered). The page
// applies a macro move as that many one-point placements in its country, which is what the engine's expansion is.
TS_API int ts_macro_points(int idx) {
    if (!ts::ActionMask::is_macro_action(static_cast<uint16_t>(idx < 0 ? 0 : idx))) return 0;
    uint8_t mask[ts::FLAT_ACTION_SPACE_SIZE];
    ts::Engine::get_flat_action_mask(g_state, mask, ts::ActionView::MACRO);
    if (!mask[idx]) return 0;
    ts::GameState copy = g_state;
    return static_cast<int>(ts::ActionMask::apply_macro_expansion(
        copy, static_cast<uint8_t>(idx - static_cast<int>(ts::flat_slots::MACRO_NODE))));
}

// Whether a flat index is an E4.1 composed action here ("Ops for influence, first point at X").
TS_API int ts_is_merged_influence_action(int idx) {
    if (idx < 0 || idx >= static_cast<int>(ts::FLAT_ACTION_SPACE_SIZE)) return 0;
    return ts::ActionMask::is_merged_influence_action(g_state, static_cast<uint16_t>(idx)) ? 1 : 0;
}

// The observation from one side's perspective (1 = US, -1 = USSR), OBS_SIZE floats.
TS_API const float* ts_observation(int player) {
    ts::ObservationBufferV23 ob;
    ts::Observation::extract(g_state, player_from_int(player), &ob);
    std::memcpy(g_obs, reinterpret_cast<const float*>(&ob), sizeof(ob));
    return g_obs;
}

// The width of the observation a model reading the obs_features bits `features` takes, or 0 (and the
// reason in ts_last_error) for a bit this build does not know.
TS_API int ts_obs_size_for(uint32_t features) {
    if (features & ~ts::obs_features::ALL) { g_error = "unknown obs_features bit"; return 0; }
    return static_cast<int>(ts::OBS_SIZE_V23 + ts::obs_features::extra_width(features));
}

// Seat a model of obs_features set `features` at `player` (1 US, -1 USSR): that side plays the set's
// rule bits (GameState::seat_rules). Returns 0 and sets the error on an unknown bit or no player.
TS_API int ts_seat_obs_features(int player, uint32_t features) {
    if (features & ~ts::obs_features::ALL) { g_error = "unknown obs_features bit"; return 0; }
    const ts::Player p = player_from_int(player);
    if (p == ts::Player::NONE) { g_error = "a seat is US or USSR"; return 0; }
    ts::seat_obs_features(g_state, p, features);
    return 1;
}

// The rule bits `player` plays now (-1 and the error for no player).
TS_API int ts_seat_rules(int player) {
    const ts::Player p = player_from_int(player);
    if (p == ts::Player::NONE) { g_error = "a seat is US or USSR"; return -1; }
    return static_cast<int>(g_state.seat_rules[ts::seat_index(p)]);
}

// The current rules, which a person plays: every game rule bit, not a model's action order
// (obs_features::PERSON_RULES leaves out P43's CANONICAL_FIRST / CANONICAL_LAST).
TS_API uint32_t ts_obs_features_rules() { return ts::obs_features::PERSON_RULES; }

// The observation in a model's own view: the base layout as the bits `features` edit and extend it
// (extract_observation_features, the same call Python makes for that model), ts_obs_size_for(features)
// floats; null for an unknown bit. HEADLINE_REVEAL has no width, so only this tells its view apart.
TS_API const float* ts_observation_features(int player, uint32_t features) {
    if (features & ~ts::obs_features::ALL) { g_error = "unknown obs_features bit"; return nullptr; }
    ts::extract_observation_features(g_state, player_from_int(player), features, g_obs);
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

// ---- entering a game played elsewhere (Twilight Struggle /ai) ----------------------------------
//
// The analysis board mirrors a game played at a table or in another app, so what chance decided
// there has to be put on the board here: which cards were dealt, which card a random draw took.
// A deal is not a decision, and the engine draws from its RNG; these two calls are how the page
// corrects it. Moving a card keeps every count right only if the page swaps cards in pairs, which
// it does; the engine itself checks nothing here, as with the debug overrides above.

TS_API int ts_card_location(int card) {
    if (card < 1 || card > 110) { g_error = "card ids are 1..110"; return -1; }
    return static_cast<int>(g_state.card_locations[card]);
}
TS_API int ts_set_card_location(int card, int loc) {
    if (card < 1 || card > 110) { g_error = "card ids are 1..110"; return 0; }
    if (loc < 0 || loc > static_cast<int>(ts::CardLocation::HEADLINE_COMMITTED)) { g_error = "no such card location"; return 0; }
    g_state.card_locations[card] = static_cast<ts::CardLocation>(loc);
    return 1;
}
// The RNG state, so a random event can be re-run until it draws the card that was really drawn.
TS_API void ts_set_rng(uint32_t lo, uint32_t hi) {
    g_state.rng_state = (static_cast<uint64_t>(hi) << 32) | lo;
}

// ---- the rollout root's playouts (bindings/rollout_batch.hpp, the same C++ as Python's) --------
//
// One batch at a time. The page: ts_rollout_reset, ts_rollout_add on the position it holds (its
// candidates, its worlds), then until ts_rollout_all_done: ts_rollout_fill_live -> read the
// observation and mask buffers -> the network -> ts_rollout_step(logits); then
// ts_rollout_fill_leaves -> the critic -> ts_rollout_set_leaf_values; then ts_rollout_values and
// the row tables to average each candidate over its worlds.

namespace {
std::unique_ptr<ts::rollout::RolloutBatch> g_rollout;
std::unique_ptr<ts::belief::Tracker> g_belief;
std::vector<uint8_t> g_belief_hands;    // the worlds' hands ts_rollout_add_belief dealt
size_t g_belief_added = 0;              // of them, the ones ts_rollout_add_belief played (the rest: the race's)
std::vector<uint8_t> g_belief_held;     // a chunk's hands for ts_belief_rows
std::vector<float> g_ro_obs;
std::vector<uint8_t> g_ro_masks;
std::vector<float> g_ro_values;
std::vector<uint16_t> g_ro_cands;
}  // namespace

// `stop`: 0 = after `horizon` boundaries, 1 = at the end of the decision's turn, 2 = the later of the two,
// 3 = the earlier of the two, 4 = at the turn-end park before the deal, 5 = the earlier of the boundaries
// and that park (ts::rollout::Stop). `leaf`: 0 = read every leaf from the root mover's side,
// 1 = from the leaf's decider's, negated into the mover's frame, 2 = play on to the mover's next decision
// (ts::rollout::Leaf). `aligned`: count every candidate's horizon from the end of the mover's own
// action round, the headline one boundary (rollout::Config::aligned); `box4`: keep the opponent's headline
// revealed by Space Race box 4 in every world (rollout::Config::box4). `keyed`: 0 = the shared stream,
// 1 = keyed chance, 2 = keyed2's packing (rollout::Config::keyed / keyed2, bindings/keyed_chance.hpp).
TS_API int ts_rollout_reset(int horizon, int shared_dice, int guard, uint32_t features,
                            uint32_t seed_lo, uint32_t seed_hi, int stop, int leaf, int keyed, int aligned, int box4) {
    if (horizon < 0) { g_error = "horizon must be >= 0"; return 0; }
    if (stop < 0 || stop > 5) {
        g_error = "stop must be 0 (boundaries), 1 (turn), 2 (later), 3 (earlier), 4 (predeal) or 5 (earlier_predeal)";
        return 0;
    }
    if (leaf < 0 || leaf > 2) { g_error = "leaf must be 0 (mover), 1 (decider) or 2 (own)"; return 0; }
    if (keyed < 0 || keyed > 2) { g_error = "keyed must be 0 (off), 1 (keyed) or 2 (keyed2)"; return 0; }
    if (features & ~ts::obs_features::ALL) { g_error = "unknown obs_features bit"; return 0; }
    const uint64_t seed = (static_cast<uint64_t>(seed_hi) << 32) | seed_lo;
    g_rollout = std::make_unique<ts::rollout::RolloutBatch>(
        ts::rollout::Config{horizon, shared_dice != 0, guard != 0, features, static_cast<ts::rollout::Stop>(stop),
                            static_cast<ts::rollout::Leaf>(leaf), keyed != 0, false, aligned != 0, box4 != 0,
                            keyed == 2}, seed);
    return 1;
}

// The position held, with `k` flat candidates at `candidates`, in `worlds` worlds. Rows added.
TS_API int ts_rollout_add(const int32_t* candidates, int k, int worlds) {
    if (!g_rollout) { g_error = "ts_rollout_reset first"; return 0; }
    g_ro_cands.clear();
    for (int i = 0; i < k; ++i) {
        if (candidates[i] < 0 || candidates[i] >= static_cast<int32_t>(ts::FLAT_ACTION_SPACE_SIZE)) {
            g_error = "a candidate is outside the flat action space";
            return 0;
        }
        g_ro_cands.push_back(static_cast<uint16_t>(candidates[i]));
    }
    const size_t n = g_rollout->add(g_state, g_ro_cands.data(), g_ro_cands.size(),
                                    static_cast<size_t>(worlds < 0 ? 0 : worlds));
    if (n == 0 && g_rollout->last_error()[0]) g_error = g_rollout->last_error();
    return static_cast<int>(n);
}

// ---- the belief over the opponent's hand (bindings/belief.hpp, the same C++ as Python's) -------
//
// One tracker, one game (game 0). The page: ts_belief_reset at a new game; ts_belief_observe before
// every move is applied (the position held is the one the move is made in); at the AI's decision
// ts_belief_prepare, then for each recorded decision and each chunk of hands ts_belief_rows -> the
// network -> the log-probability of ts_belief_action(d) at each of ts_belief_temperature(t) ->
// ts_belief_set_log_likelihoods; then ts_rollout_add_belief deals the root's worlds from it.

// `reveal`: every hand holds the opponent's headline revealed by Space Race box 4 (the root's box4).
TS_API void ts_belief_reset(uint32_t particles, int all_decisions, uint32_t seed_lo, uint32_t seed_hi, int reveal) {
    g_belief = std::make_unique<ts::belief::Tracker>(particles, all_decisions != 0,
                                                     (static_cast<uint64_t>(seed_hi) << 32) | seed_lo, reveal != 0);
}
TS_API int ts_belief_temperatures() { return static_cast<int>(ts::belief::kTemps); }
TS_API float ts_belief_temperature(int t) {
    return (t >= 0 && static_cast<size_t>(t) < ts::belief::kTemps) ? ts::belief::kTemperatures[t] : 0.0f;
}
TS_API int ts_belief_particles() { return g_belief ? static_cast<int>(g_belief->particles()) : 0; }

// The move [type, primary, secondary, flags] about to be applied at the position held: recorded as the
// flat action that decodes to it. 0 when there is no tracker, no decider, or no flat action is the move
// (then nothing is recorded -- a move the network could not have made carries no likelihood).
TS_API int ts_belief_observe(int decision_type, int primary, int secondary, int flags) {
    if (!g_belief) return 0;
    const ts::Player decider = g_state.ctx().decision_player;
    if (decider != ts::Player::US && decider != ts::Player::USSR) return 0;
    uint8_t mask[ts::FLAT_ACTION_SPACE_SIZE];
    ts::ActionMask::generate_flat_mask_212(g_state, mask);
    for (size_t a = 0; a < ts::FLAT_ACTION_SPACE_SIZE; ++a) {
        if (!mask[a]) continue;
        const ts::MicroAction m = ts::ActionMask::decode_flat_action_212(g_state, static_cast<uint16_t>(a));
        if (static_cast<int>(m.decision_type) == decision_type && m.primary_id == primary && m.secondary_id == secondary &&
            m.flags == flags) {
            g_belief->observe(0, g_state, decider, static_cast<uint16_t>(a));
            return 1;
        }
    }
    return 0;
}

// The belief for the position held: 1 if it has one (then ts_belief_decisions() rows blocks to
// evaluate, 0 when its weights are reused), 0 where there is nothing to believe.
TS_API int ts_belief_prepare() {
    if (!g_belief) return 0;
    const std::vector<size_t> games{0};
    const std::vector<const ts::GameState*> states{&g_state};
    return g_belief->prepare(games, states)[0] ? 1 : 0;
}
TS_API int ts_belief_decisions() {
    return (g_belief && !g_belief->jobs().empty()) ? static_cast<int>(g_belief->jobs()[0].decisions.size()) : 0;
}
TS_API int ts_belief_action(int d) {
    if (!g_belief || g_belief->jobs().empty() || d < 0 || static_cast<size_t>(d) >= g_belief->jobs()[0].decisions.size())
        return -1;
    return g_belief->decision(0, static_cast<size_t>(d)).action;
}
// Recorded decision d under hands lo .. lo + n - 1: n observation rows (from the opponent's side, in
// feature set `features`) and masks into the rollout buffers (ts_rollout_buf_obs / _masks). Rows written.
TS_API int ts_belief_rows(int d, int lo, int n, uint32_t features) {
    if (!g_belief || g_belief->jobs().empty()) { g_error = "ts_belief_prepare made no job"; return 0; }
    const auto& job = g_belief->jobs()[0];
    const size_t N = g_belief->particles();
    if (d < 0 || static_cast<size_t>(d) >= job.decisions.size() || lo < 0 || n <= 0 || static_cast<size_t>(lo + n) > N) {
        g_error = "no such decision or hands";
        return 0;
    }
    const auto& left = job.lefts[static_cast<size_t>(d)];
    g_belief_held.assign(static_cast<size_t>(n) * 110, 0);
    for (int k = 0; k < n; ++k)
        for (size_t c = 0; c < 110; ++c)
            g_belief_held[static_cast<size_t>(k) * 110 + c] =
                static_cast<uint8_t>(job.hands[static_cast<size_t>(lo + k) * 110 + c] | left[c]);
    const size_t width = ts::OBS_SIZE_V23 + ts::obs_features::extra_width(features);
    g_ro_obs.assign(static_cast<size_t>(n) * width, 0.0f);
    g_ro_masks.assign(static_cast<size_t>(n) * ts::FLAT_ACTION_SPACE_SIZE, 0);
    ts::rollout::hypothesis_rows(g_belief->decision(0, static_cast<size_t>(d)).state, job.me, g_belief_held.data(),
                                 static_cast<size_t>(n), features, g_ro_obs.data(), g_ro_masks.data());
    return n;
}
// lp: temperatures x decisions x particles floats, the log-probability of each recorded decision's move.
TS_API int ts_belief_set_log_likelihoods(const float* lp) {
    if (!g_belief || g_belief->jobs().empty()) { g_error = "ts_belief_prepare made no job"; return 0; }
    if (any_nan(lp, ts::belief::kTemps * g_belief->jobs()[0].decisions.size() * g_belief->particles())) {
        g_error = "a log-likelihood is NaN";
        return 0;
    }
    g_belief->set_log_likelihoods(lp, g_belief->jobs()[0].decisions.size());
    return 1;
}
// The temperature weights of the position prepared (ts_belief_temperatures() floats into `out`).
TS_API int ts_belief_temperature_weights(float* out) {
    if (!g_belief) return 0;
    const auto w = g_belief->temperature_weights(0);
    std::copy(w.begin(), w.end(), out);
    return 1;
}
// `n` hands resampled from the prepared belief (n x 110 bytes, card c at c - 1); null where it has none.
TS_API const uint8_t* ts_belief_worlds(int n) {
    if (!g_belief || n <= 0) return nullptr;
    g_belief_hands.assign(static_cast<size_t>(n) * 110, 0);
    return g_belief->worlds(0, static_cast<size_t>(n), g_belief_hands.data()) ? g_belief_hands.data() : nullptr;
}

// ts_rollout_add with the worlds' opposing hands resampled from the prepared belief. Rows added; 0 with
// ts_last_error() when the position has no belief (the caller then uses ts_rollout_add). `deal` hands are
// dealt in one draw, as Python's root deals a race's (ai/search/rollout_root.py), and the first `worlds`
// played; ts_rollout_race(..., belief) plays the rest. deal <= worlds deals only those.
TS_API int ts_rollout_add_belief(const int32_t* candidates, int k, int worlds, int deal) {
    if (!g_rollout) { g_error = "ts_rollout_reset first"; return 0; }
    if (!g_belief || worlds <= 0) { g_error = "no belief"; return 0; }
    const size_t dealt = static_cast<size_t>(std::max(worlds, deal));
    g_belief_hands.assign(dealt * 110, 0);
    if (!g_belief->worlds(0, dealt, g_belief_hands.data())) { g_error = "no belief here"; return 0; }
    g_belief_added = static_cast<size_t>(worlds);
    g_ro_cands.clear();
    for (int i = 0; i < k; ++i) {
        if (candidates[i] < 0 || candidates[i] >= static_cast<int32_t>(ts::FLAT_ACTION_SPACE_SIZE)) {
            g_error = "a candidate is outside the flat action space";
            return 0;
        }
        g_ro_cands.push_back(static_cast<uint16_t>(candidates[i]));
    }
    const size_t n = g_rollout->add(g_state, g_ro_cands.data(), g_ro_cands.size(), static_cast<size_t>(worlds),
                                    g_belief_hands.data());
    if (n == 0 && g_rollout->last_error()[0]) g_error = g_rollout->last_error();
    return static_cast<int>(n);
}

// RolloutBatch::race at `position` once its rows are valued: the challengers trailing the network's move
// leave, the rest play `worlds` more worlds -- with `belief`, the hands ts_rollout_add_belief dealt past
// the ones it played. Rows added (0 with no challenger left); -1 with ts_last_error() on a refusal.
TS_API int ts_rollout_race(int position, int worlds, int belief) {
    if (!g_rollout) { g_error = "ts_rollout_reset first"; return -1; }
    if (position < 0 || worlds <= 0) { g_error = "race a position by its index, for at least one world"; return -1; }
    const uint8_t* hands = nullptr;
    if (belief) {
        if (g_belief_hands.size() < (g_belief_added + static_cast<size_t>(worlds)) * 110) {
            g_error = "ts_rollout_add_belief dealt no hands for the race";
            return -1;
        }
        hands = g_belief_hands.data() + g_belief_added * 110;
    }
    const size_t n = g_rollout->race(static_cast<size_t>(position), static_cast<size_t>(worlds), hands);
    if (n == 0 && g_rollout->last_error()[0]) { g_error = g_rollout->last_error(); return -1; }
    return static_cast<int>(n);
}
TS_API int ts_rollout_in_race(int position, int c) {
    return g_rollout && position >= 0 && c >= 0 && g_rollout->in_race(static_cast<size_t>(position), static_cast<size_t>(c)) ? 1 : 0;
}

TS_API int ts_rollout_rows() { return g_rollout ? static_cast<int>(g_rollout->rows()) : 0; }
TS_API int ts_rollout_live() { return g_rollout ? static_cast<int>(g_rollout->live()) : 0; }
TS_API int ts_rollout_all_done() { return !g_rollout || g_rollout->all_done() ? 1 : 0; }
TS_API int ts_rollout_obs_width() { return g_rollout ? static_cast<int>(g_rollout->obs_width()) : 0; }
TS_API int ts_rollout_refused() { return g_rollout ? static_cast<int>(g_rollout->refused()) : 0; }
TS_API int ts_rollout_worlds_used(int position) {
    if (!g_rollout || position < 0 || static_cast<size_t>(position) >= g_rollout->positions()) return 0;
    return static_cast<int>(g_rollout->worlds_used(static_cast<size_t>(position)));
}

// The live rows' observations and masks into the buffers below; returns the count.
TS_API int ts_rollout_fill_live() {
    if (!g_rollout) return 0;
    const size_t n = g_rollout->live();
    g_ro_obs.assign(std::max<size_t>(1, n * g_rollout->obs_width()), 0.0f);
    g_ro_masks.assign(std::max<size_t>(1, n * ts::FLAT_ACTION_SPACE_SIZE), 0);
    return static_cast<int>(g_rollout->fill_live(g_ro_obs.data(), g_ro_masks.data()));
}
TS_API const float* ts_rollout_buf_obs() { return g_ro_obs.data(); }
TS_API const uint8_t* ts_rollout_buf_masks() { return g_ro_masks.data(); }

// `logits`: count x FLAT_ACTION_SPACE_SIZE floats for the rows the last fill_live listed.
TS_API int ts_rollout_step(const float* logits) {
    if (!g_rollout) { g_error = "ts_rollout_reset first"; return 0; }
    if (any_nan(logits, g_rollout->listed() * ts::FLAT_ACTION_SPACE_SIZE)) { g_error = "a logit is NaN"; return 0; }
    g_rollout->step(logits);
    return 1;
}

// The rows at their horizon and not yet valued, from the mover's side, into the same buffers.
TS_API int ts_rollout_fill_leaves() {
    if (!g_rollout) return 0;
    const size_t n = g_rollout->leaves_pending();
    g_ro_obs.assign(std::max<size_t>(1, n * g_rollout->obs_width()), 0.0f);
    g_ro_masks.assign(std::max<size_t>(1, n * ts::FLAT_ACTION_SPACE_SIZE), 0);
    return static_cast<int>(g_rollout->fill_leaves(g_ro_obs.data(), g_ro_masks.data()));
}
// What the last ts_rollout_fill_leaves listed, read from each leaf's root mover's side, for a page
// that weighs the critic's value with what it knows the critic misjudges -- Europe, above all, where
// a player who controls it when the scoring card comes wins at once. kLeafFeatures floats a leaf, in
// the order fill_leaves wrote them: the mover's and the opponent's Europe status (0 none, 1 presence,
// 2 domination, 3 control), Europe battlegrounds the mover controls less the opponent's, where
// Europe Scoring is in that world (0 the mover's hand, 1 the draw deck, 2 the opponent's hand, 3
// anywhere else: played, discarded), and the turn.
namespace {
constexpr int kLeafFeatures = 5;
std::vector<float> g_ro_leaf_features;
}  // namespace
TS_API int ts_rollout_leaf_feature_count() { return kLeafFeatures; }
TS_API const float* ts_rollout_leaf_features() {
    const size_t n = g_rollout ? g_rollout->leaves_listed() : 0;
    g_ro_leaf_features.assign(std::max<size_t>(1, n * kLeafFeatures), 0.0f);
    for (size_t j = 0; j < n; ++j) {
        const auto& row = g_rollout->row(g_rollout->leaf_row(j));
        const ts::Player me = g_rollout->mover(row.position);
        const auto e = ts::Scoring::evaluate_region(row.state, ts::Region::EUROPE);
        const bool us = me == ts::Player::US;
        float* f = g_ro_leaf_features.data() + j * kLeafFeatures;
        f[0] = static_cast<float>(us ? e.us_status : e.ussr_status);
        f[1] = static_cast<float>(us ? e.ussr_status : e.us_status);
        f[2] = static_cast<float>(us ? e.us_battlegrounds - e.ussr_battlegrounds : e.ussr_battlegrounds - e.us_battlegrounds);
        const auto loc = row.state.card_locations[ts::card_ids::EUROPE_SCORING];
        f[3] = ts::in_hand_of(loc, me) ? 0.0f : loc == ts::CardLocation::DRAW_DECK ? 1.0f
             : ts::in_hand_of(loc, ts::get_opponent(me)) ? 2.0f : 3.0f;
        f[4] = static_cast<float>(row.state.turn);
    }
    return g_ro_leaf_features.data();
}

TS_API int ts_rollout_set_leaf_values(const float* values) {
    if (!g_rollout) { g_error = "ts_rollout_reset first"; return 0; }
    if (any_nan(values, g_rollout->leaves_listed())) { g_error = "a leaf value is NaN"; return 0; }
    g_rollout->set_leaf_values(values);
    return 1;
}

// Every row's value from its mover's side (NaN while live or unvalued), rows() floats.
TS_API const float* ts_rollout_values() {
    g_ro_values.assign(std::max<size_t>(1, g_rollout ? g_rollout->rows() : 0), 0.0f);
    if (g_rollout) g_rollout->values(g_ro_values.data());
    return g_ro_values.data();
}
TS_API int ts_rollout_row_position(int r) {
    return (g_rollout && r >= 0 && static_cast<size_t>(r) < g_rollout->rows()) ? static_cast<int>(g_rollout->row(r).position) : -1;
}
TS_API int ts_rollout_row_world(int r) {
    return (g_rollout && r >= 0 && static_cast<size_t>(r) < g_rollout->rows()) ? g_rollout->row(r).world : -1;
}
TS_API int ts_rollout_row_candidate(int r) {
    return (g_rollout && r >= 0 && static_cast<size_t>(r) < g_rollout->rows()) ? g_rollout->row(r).candidate : -1;
}
TS_API int ts_rollout_row_done(int r) {
    return (g_rollout && r >= 0 && static_cast<size_t>(r) < g_rollout->rows()) ? (g_rollout->row(r).done ? 1 : 0) : -1;
}
TS_API int ts_rollout_row_terminal(int r) {
    return (g_rollout && r >= 0 && static_cast<size_t>(r) < g_rollout->rows()) ? (g_rollout->row(r).terminal ? 1 : 0) : -1;
}

// The `k` most probable legal moves by the root's logits (FLAT_ACTION_SPACE_SIZE floats) under
// `mask`, best first, into `out` (k int32s); returns how many, or -1 with ts_last_error() for a NaN
// logit.
TS_API int ts_rollout_candidates(const float* logits, const uint8_t* mask, int k, int32_t* out) {
    if (any_nan(logits, ts::FLAT_ACTION_SPACE_SIZE)) { g_error = "a logit is NaN"; return -1; }
    if (k <= 0) return 0;
    std::vector<uint16_t> c(static_cast<size_t>(k));
    const size_t n = ts::rollout::top_candidates(logits, mask, static_cast<size_t>(k), c.data());
    for (size_t i = 0; i < n; ++i) out[i] = c[i];
    return static_cast<int>(n);
}

// One position's decision from the valued batch, as JSON: {"pick", "worlds", "q", "lead",
// "lead_se"}; "" with ts_last_error() for an unknown rule or position.
TS_API const char* ts_rollout_decide(int position, const int32_t* candidates, int k, const float* logits,
                                     const uint8_t* mask, const char* rule_text) {
    if (!g_rollout || position < 0 || static_cast<size_t>(position) >= g_rollout->positions()) {
        g_error = "no such position in the rollout batch";
        return "";
    }
    ts::rollout::Rule rule;
    if (!ts::rollout::parse_rule(rule_text, rule)) { g_error = "unknown rule: argmax, z<x> or kl<t>"; return ""; }
    if (any_nan(logits, ts::FLAT_ACTION_SPACE_SIZE)) { g_error = "a logit is NaN"; return ""; }
    std::vector<uint16_t> c;
    for (int i = 0; i < k; ++i) {
        if (candidates[i] < 0 || candidates[i] >= static_cast<int32_t>(ts::FLAT_ACTION_SPACE_SIZE)) {
            g_error = "a candidate is outside the flat action space";
            return "";
        }
        c.push_back(static_cast<uint16_t>(candidates[i]));
    }
    const ts::rollout::Decision d = ts::rollout::decide(*g_rollout, static_cast<size_t>(position), c.data(), c.size(),
                                                        logits, mask, rule);
    auto list = [](const std::vector<float>& v) {
        std::string s = "[";
        char buf[32];
        for (size_t i = 0; i < v.size(); ++i) {
            std::snprintf(buf, sizeof buf, "%.9g", static_cast<double>(v[i]));
            s += (i ? "," : "");
            s += buf;
        }
        return s + "]";
    };
    std::string out = "{\"pick\":" + std::to_string(d.pick) + ",\"worlds\":" + std::to_string(d.worlds)
        + ",\"q\":" + list(d.q) + ",\"lead\":" + list(d.lead) + ",\"lead_se\":" + list(d.lead_se) + "}";
    return keep(out);
}

// ---- parity -----------------------------------------------------------------------------------

// The shared whole-game digest (bindings/selftest.hpp): the Python module computes the same
// number natively, so tests/web/test_wasm_engine.py can compare the two builds directly.
TS_API uint32_t ts_selftest(int games, uint32_t* steps_out) { return ts::selftest::digest(games, steps_out); }
