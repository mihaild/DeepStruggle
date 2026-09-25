#include "state_json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <type_traits>

#include "ts/card_data.hpp"
#include "ts/engine.hpp"
#include "ts/map_data.hpp"

namespace ts::state_json {

// ---- Value ----------------------------------------------------------------------------------

Value& Value::set(std::string key, Value v) {
    for (auto& m : members_) {
        if (m.first == key) {
            m.second = std::move(v);
            return m.second;
        }
    }
    members_.emplace_back(std::move(key), std::move(v));
    return members_.back().second;
}

const Value* Value::find(std::string_view key) const {
    for (const auto& m : members_) {
        if (m.first == key) return &m.second;
    }
    return nullptr;
}

bool Value::to_integral(int64_t lo, uint64_t hi, int64_t* out_signed, uint64_t* out_unsigned) const {
    if (kind_ == Kind::Int) {
        if (i_ < lo) return false;
        if (i_ >= 0 && static_cast<uint64_t>(i_) > hi) return false;
        if (out_signed) *out_signed = i_;
        if (out_unsigned) *out_unsigned = static_cast<uint64_t>(i_);
        return true;
    }
    if (kind_ == Kind::UInt) {
        if (u_ > hi) return false;
        if (out_signed) *out_signed = static_cast<int64_t>(u_);
        if (out_unsigned) *out_unsigned = u_;
        return true;
    }
    return false;   // bools, floats, strings, containers: not an integer, as for nanobind
}

namespace {

void dump_string(std::string& out, const std::string& s) {
    // As Python's json.dumps with ensure_ascii=True: non-ASCII becomes \uXXXX (UTF-16).
    out.push_back('"');
    size_t i = 0;
    auto hex4 = [&out](unsigned v) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "\\u%04x", v);
        out += buf;
    };
    while (i < s.size()) {
        const auto c = static_cast<unsigned char>(s[i]);
        if (c == '"') { out += "\\\""; ++i; continue; }
        if (c == '\\') { out += "\\\\"; ++i; continue; }
        if (c == '\n') { out += "\\n"; ++i; continue; }
        if (c == '\r') { out += "\\r"; ++i; continue; }
        if (c == '\t') { out += "\\t"; ++i; continue; }
        if (c == '\b') { out += "\\b"; ++i; continue; }
        if (c == '\f') { out += "\\f"; ++i; continue; }
        if (c < 0x20) { hex4(c); ++i; continue; }
        if (c < 0x80) { out.push_back(static_cast<char>(c)); ++i; continue; }
        // Decode one UTF-8 sequence; a malformed byte is emitted as U+FFFD.
        uint32_t cp = 0xFFFD;
        size_t len = 1;
        if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) { cp = ((c & 0x1Fu) << 6) | (s[i + 1] & 0x3F); len = 2; }
        else if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) {
            cp = ((c & 0x0Fu) << 12) | ((s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F); len = 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < s.size()) {
            cp = ((c & 0x07u) << 18) | ((s[i + 1] & 0x3F) << 12) | ((s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F);
            len = 4;
        }
        if (cp >= 0x10000) {
            cp -= 0x10000;
            hex4(0xD800 + (cp >> 10));
            hex4(0xDC00 + (cp & 0x3FF));
        } else {
            hex4(cp);
        }
        i += len;
    }
    out.push_back('"');
}

}  // namespace

void Value::dump_to(std::string& out) const {
    switch (kind_) {
        case Kind::Null: out += "null"; return;
        case Kind::Bool: out += b_ ? "true" : "false"; return;
        case Kind::Int: out += std::to_string(i_); return;
        case Kind::UInt: out += std::to_string(u_); return;
        case Kind::Double: {
            if (!std::isfinite(d_)) { out += "null"; return; }
            // Shortest text that parses back to the same double, and always visibly a float.
            char buf[40];
            for (int prec = 1; prec <= 17; ++prec) {
                std::snprintf(buf, sizeof(buf), "%.*g", prec, d_);
                if (std::strtod(buf, nullptr) == d_) break;
            }
            std::string s(buf);
            if (s.find_first_of(".eE") == std::string::npos) s += ".0";
            out += s;
            return;
        }
        case Kind::String: dump_string(out, s_); return;
        case Kind::Array: {
            out.push_back('[');
            for (size_t k = 0; k < items_.size(); ++k) {
                if (k) out.push_back(',');
                items_[k].dump_to(out);
            }
            out.push_back(']');
            return;
        }
        case Kind::Object: {
            std::vector<const Member*> sorted;
            sorted.reserve(members_.size());
            for (const auto& m : members_) sorted.push_back(&m);
            std::sort(sorted.begin(), sorted.end(),
                      [](const Member* a, const Member* b) { return a->first < b->first; });
            out.push_back('{');
            for (size_t k = 0; k < sorted.size(); ++k) {
                if (k) out.push_back(',');
                dump_string(out, sorted[k]->first);
                out.push_back(':');
                sorted[k]->second.dump_to(out);
            }
            out.push_back('}');
            return;
        }
    }
}

std::string Value::dump() const {
    std::string out;
    dump_to(out);
    return out;
}

namespace {

class Parser {
public:
    explicit Parser(std::string_view t) : t_(t) {}

    bool document(Value& out, std::string* error) {
        ws();
        if (!value(out, 0)) return fail(error);
        ws();
        if (p_ != t_.size()) { err_ = "trailing characters"; return fail(error); }
        return true;
    }

private:
    bool fail(std::string* error) {
        if (error) *error = err_ + " at offset " + std::to_string(p_);
        return false;
    }
    void ws() {
        while (p_ < t_.size() && (t_[p_] == ' ' || t_[p_] == '\n' || t_[p_] == '\r' || t_[p_] == '\t')) ++p_;
    }
    bool lit(std::string_view w) {
        if (t_.substr(p_, w.size()) != w) { err_ = "invalid literal"; return false; }
        p_ += w.size();
        return true;
    }
    bool value(Value& out, int depth) {
        if (depth > 64) { err_ = "nesting too deep"; return false; }
        if (p_ >= t_.size()) { err_ = "unexpected end"; return false; }
        const char c = t_[p_];
        if (c == '{') return object(out, depth);
        if (c == '[') return array(out, depth);
        if (c == '"') { std::string s; if (!string(s)) return false; out = Value(std::move(s)); return true; }
        if (c == 't') { if (!lit("true")) return false; out = Value(true); return true; }
        if (c == 'f') { if (!lit("false")) return false; out = Value(false); return true; }
        if (c == 'n') { if (!lit("null")) return false; out = Value(); return true; }
        return number(out);
    }
    bool object(Value& out, int depth) {
        ++p_;
        out = Value::object();
        ws();
        if (p_ < t_.size() && t_[p_] == '}') { ++p_; return true; }
        while (true) {
            ws();
            if (p_ >= t_.size() || t_[p_] != '"') { err_ = "expected a key"; return false; }
            std::string key;
            if (!string(key)) return false;
            ws();
            if (p_ >= t_.size() || t_[p_] != ':') { err_ = "expected ':'"; return false; }
            ++p_;
            ws();
            Value v;
            if (!value(v, depth + 1)) return false;
            out.set(std::move(key), std::move(v));
            ws();
            if (p_ < t_.size() && t_[p_] == ',') { ++p_; continue; }
            if (p_ < t_.size() && t_[p_] == '}') { ++p_; return true; }
            err_ = "expected ',' or '}'";
            return false;
        }
    }
    bool array(Value& out, int depth) {
        ++p_;
        out = Value::array();
        ws();
        if (p_ < t_.size() && t_[p_] == ']') { ++p_; return true; }
        while (true) {
            ws();
            Value v;
            if (!value(v, depth + 1)) return false;
            out.push(std::move(v));
            ws();
            if (p_ < t_.size() && t_[p_] == ',') { ++p_; continue; }
            if (p_ < t_.size() && t_[p_] == ']') { ++p_; return true; }
            err_ = "expected ',' or ']'";
            return false;
        }
    }
    static void utf8(std::string& s, uint32_t cp) {
        if (cp < 0x80) { s.push_back(static_cast<char>(cp)); }
        else if (cp < 0x800) { s.push_back(static_cast<char>(0xC0 | (cp >> 6))); s.push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
        else if (cp < 0x10000) {
            s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    bool hex4(uint32_t& v) {
        if (p_ + 4 > t_.size()) { err_ = "short \\u escape"; return false; }
        v = 0;
        for (int k = 0; k < 4; ++k) {
            const char h = t_[p_++];
            v <<= 4;
            if (h >= '0' && h <= '9') v |= static_cast<uint32_t>(h - '0');
            else if (h >= 'a' && h <= 'f') v |= static_cast<uint32_t>(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') v |= static_cast<uint32_t>(h - 'A' + 10);
            else { err_ = "bad \\u escape"; return false; }
        }
        return true;
    }
    bool string(std::string& s) {
        ++p_;   // opening quote
        while (p_ < t_.size()) {
            const char c = t_[p_++];
            if (c == '"') return true;
            if (static_cast<unsigned char>(c) < 0x20) { err_ = "control character in string"; return false; }
            if (c != '\\') { s.push_back(c); continue; }
            if (p_ >= t_.size()) break;
            const char e = t_[p_++];
            switch (e) {
                case '"': s.push_back('"'); break;
                case '\\': s.push_back('\\'); break;
                case '/': s.push_back('/'); break;
                case 'b': s.push_back('\b'); break;
                case 'f': s.push_back('\f'); break;
                case 'n': s.push_back('\n'); break;
                case 'r': s.push_back('\r'); break;
                case 't': s.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!hex4(cp)) return false;
                    if (cp >= 0xD800 && cp < 0xDC00 && p_ + 6 <= t_.size() && t_[p_] == '\\' && t_[p_ + 1] == 'u') {
                        p_ += 2;
                        uint32_t lo = 0;
                        if (!hex4(lo)) return false;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    utf8(s, cp);
                    break;
                }
                default: err_ = "bad escape"; return false;
            }
        }
        err_ = "unterminated string";
        return false;
    }
    bool number(Value& out) {
        const size_t start = p_;
        bool neg = false;
        if (t_[p_] == '-') { neg = true; ++p_; }
        if (p_ >= t_.size() || !(t_[p_] >= '0' && t_[p_] <= '9')) { err_ = "invalid value"; return false; }
        if (t_[p_] == '0') { ++p_; } else { while (p_ < t_.size() && t_[p_] >= '0' && t_[p_] <= '9') ++p_; }
        bool integral = true;
        if (p_ < t_.size() && t_[p_] == '.') {
            integral = false;
            ++p_;
            if (p_ >= t_.size() || !(t_[p_] >= '0' && t_[p_] <= '9')) { err_ = "bad fraction"; return false; }
            while (p_ < t_.size() && t_[p_] >= '0' && t_[p_] <= '9') ++p_;
        }
        if (p_ < t_.size() && (t_[p_] == 'e' || t_[p_] == 'E')) {
            integral = false;
            ++p_;
            if (p_ < t_.size() && (t_[p_] == '+' || t_[p_] == '-')) ++p_;
            if (p_ >= t_.size() || !(t_[p_] >= '0' && t_[p_] <= '9')) { err_ = "bad exponent"; return false; }
            while (p_ < t_.size() && t_[p_] >= '0' && t_[p_] <= '9') ++p_;
        }
        const std::string text(t_.substr(start, p_ - start));
        if (!integral) { out = Value(std::strtod(text.c_str(), nullptr)); return true; }
        // Exact integers, the full uint64 range included (the RNG state and the effect bits).
        const std::string digits = neg ? text.substr(1) : text;
        uint64_t mag = 0;
        bool overflow = false;
        for (char d : digits) {
            const uint64_t dv = static_cast<uint64_t>(d - '0');
            if (mag > (std::numeric_limits<uint64_t>::max() - dv) / 10) { overflow = true; break; }
            mag = mag * 10 + dv;
        }
        if (overflow) { out = Value(std::strtod(text.c_str(), nullptr)); return true; }  // as a float, like nothing we store
        if (!neg) {
            if (mag <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) out = Value(static_cast<int64_t>(mag));
            else out = Value(mag);
            return true;
        }
        if (mag <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1) {
            out = Value(mag == static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1
                            ? std::numeric_limits<int64_t>::min()
                            : -static_cast<int64_t>(mag));
            return true;
        }
        out = Value(std::strtod(text.c_str(), nullptr));
        return true;
    }

    std::string_view t_;
    size_t p_ = 0;
    std::string err_ = "invalid JSON";
};

}  // namespace

bool Value::parse(std::string_view text, Value& out, std::string* error) {
    return Parser(text).document(out, error);
}

// ---- names ----------------------------------------------------------------------------------

namespace {

Value player_name(Player p) {
    return p == Player::US ? "US" : (p == Player::USSR ? "USSR" : "NONE");
}

const char* decision_type_name(DecisionType dt) {
    switch (dt) {
        case DecisionType::NONE: return "NONE";
        case DecisionType::SELECT_CARD: return "SELECT_CARD";
        case DecisionType::SELECT_PLAY_MODE: return "SELECT_PLAY_MODE";
        case DecisionType::CHOOSE_TIMING_BRANCH: return "CHOOSE_TIMING_BRANCH";
        case DecisionType::SELECT_OP_MODE: return "SELECT_OP_MODE";
        case DecisionType::POINT_NODE: return "POINT_NODE";
        case DecisionType::CHOOSE_BRANCH: return "CHOOSE_BRANCH";
        case DecisionType::ROLL_DIE: return "ROLL_DIE";
        default: return "UNKNOWN";
    }
}

const char* roll_type_name(RollType t) {
    switch (t) {
        case RollType::COUP: return "COUP";
        case RollType::REALIGNMENT: return "REALIGNMENT";
        case RollType::SPACE_RACE: return "SPACE_RACE";
        case RollType::WAR_EVENT: return "WAR_EVENT";
        case RollType::OLYMPIC_GAMES: return "OLYMPIC_GAMES";
        case RollType::SUMMIT: return "SUMMIT";
        case RollType::TRAP_ESCAPE: return "TRAP_ESCAPE";
        case RollType::TURN_CLEANUP: return "TURN_CLEANUP";
        default: return "NONE";
    }
}

const char* phase_name(Phase phase) {
    switch (phase) {
        case Phase::SETUP: return "SETUP";
        case Phase::HEADLINE: return "HEADLINE";
        case Phase::ACTION_ROUND: return "ACTION_ROUND";
        case Phase::INTERRUPT: return "INTERRUPT";
        case Phase::DISCARD: return "DISCARD";
        case Phase::END_TURN: return "END_TURN";
        case Phase::GAME_OVER: return "GAME_OVER";
        default: return "UNKNOWN";
    }
}

const char* play_mode_name(uint8_t mode) {
    // P17: a SELECT_PLAY_MODE node carries a Resolution, so this names those.
    switch (mode) {
        case static_cast<uint8_t>(Resolution::EVENT): return "EVENT";
        case static_cast<uint8_t>(Resolution::SPACE): return "SPACE";
        case static_cast<uint8_t>(Resolution::OPS_INFLUENCE): return "OPS_INFLUENCE";
        case static_cast<uint8_t>(Resolution::OPS_COUP): return "OPS_COUP";
        case static_cast<uint8_t>(Resolution::OPS_REALIGN): return "OPS_REALIGN";
        default: return "UNKNOWN";
    }
}

const char* op_mode_name(uint8_t mode) {
    switch (mode) {
        case static_cast<uint8_t>(OpMode::INFLUENCE): return "INFLUENCE";
        case static_cast<uint8_t>(OpMode::COUP): return "COUP";
        case static_cast<uint8_t>(OpMode::REALIGN): return "REALIGN";
        default: return "UNKNOWN";
    }
}

const char* timing_branch_name(uint8_t branch) {
    switch (branch) {
        case static_cast<uint8_t>(TimingBranch::OPS_FIRST): return "OPS_FIRST";
        case static_cast<uint8_t>(TimingBranch::EVENT_FIRST): return "EVENT_FIRST";
        default: return "UNKNOWN";
    }
}

const char* card_location_name(CardLocation loc) {
    switch (loc) {
        case CardLocation::HEADLINE_COMMITTED: return "HEADLINE_COMMITTED";
        case CardLocation::UNAVAILABLE: return "UNAVAILABLE";
        case CardLocation::DRAW_DECK: return "DRAW_DECK";
        case CardLocation::HAND_US_UNKNOWN: return "HAND_US_UNKNOWN";
        case CardLocation::HAND_US_KNOWN: return "HAND_US_KNOWN";
        case CardLocation::HAND_USSR_UNKNOWN: return "HAND_USSR_UNKNOWN";
        case CardLocation::HAND_USSR_KNOWN: return "HAND_USSR_KNOWN";
        case CardLocation::DISCARD_PILE: return "DISCARD_PILE";
        case CardLocation::REMOVED_FROM_GAME: return "REMOVED_FROM_GAME";
        case CardLocation::ONGOING_EVENT: return "ONGOING_EVENT";
        default: return "UNAVAILABLE";
    }
}

Value I(int v) { return Value(v); }

}  // namespace

// ---- the display state ----------------------------------------------------------------------

Value display_state(const GameState& state) {
    Value d = Value::object();

    // 1. Global tracks
    d.set("victory_points", I(state.victory_points));
    d.set("defcon", I(state.defcon));
    Value mil_ops = Value::object();
    mil_ops.set("US", I(state.us_mil_ops));
    mil_ops.set("USSR", I(state.ussr_mil_ops));
    d.set("mil_ops", std::move(mil_ops));
    Value space = Value::object();
    space.set("US", I(state.us_space_track));
    space.set("USSR", I(state.ussr_space_track));
    d.set("space", std::move(space));
    d.set("turn", I(state.turn));
    d.set("action_round", I(state.action_round));

    // 2. Phasing and priority
    d.set("phasing_player", player_name(state.phasing_player));
    d.set("current_phase", I(static_cast<int>(state.current_phase)));
    d.set("current_phase_name", phase_name(state.current_phase));
    d.set("phase_name", phase_name(state.current_phase));
    d.set("headline_us_card", I(state.headline_us_card));
    d.set("headline_ussr_card", I(state.headline_ussr_card));
    d.set("headline_first_card", I(state.headline_first_card));
    d.set("headline_second_card", I(state.headline_second_card));
    d.set("headline_stage", I(state.headline_stage));
    d.set("forced_card_player", player_name(state.forced_card_player));
    d.set("forced_card_id", I(state.forced_card_id));
    d.set("last_die_roll", I(state.last_die_roll));
    d.set("last_opp_die_roll", I(state.last_opp_die_roll));

    const auto& r = state.last_roll;
    Value roll = Value::object();
    roll.set("type", roll_type_name(r.type));
    roll.set("type_id", I(static_cast<uint8_t>(r.type)));
    roll.set("roller", player_name(r.roller));
    roll.set("card_id", I(r.card_id));
    roll.set("card_name", (r.card_id >= 1 && r.card_id <= 110) ? std::string(CardData::get_card(r.card_id).name) : std::string());
    roll.set("country_id", I(r.country_id));
    roll.set("country_name", (r.country_id < 84) ? std::string(MapData::get_country(r.country_id).name) : std::string());
    roll.set("roll1", I(r.roll1));
    roll.set("mod1", I(r.mod1));
    roll.set("total1", I(r.roll1 + r.mod1));
    roll.set("roll2", I(r.roll2));
    roll.set("mod2", I(r.mod2));
    roll.set("total2", I(r.roll2 + r.mod2));
    roll.set("success", Value(static_cast<bool>(r.success)));
    roll.set("net_delta", I(r.net_delta));
    d.set("die_roll", std::move(roll));

    // 3. Space turns used
    Value space_turns = Value::object();
    space_turns.set("US", I(state.get_space_turns_used(Player::US)));
    space_turns.set("USSR", I(state.get_space_turns_used(Player::USSR)));
    d.set("space_turns_used", std::move(space_turns));

    // 4. The China Card
    Value china = Value::object();
    china.set("holder", state.china_card_holder == Player::US ? "US" : "USSR");
    china.set("playable", Value(state.china_card_playable != 0));
    d.set("china_card", std::move(china));

    // 5. Persistent flags
    Value flags = Value::array();
#define CHECK_FLAG(f) if (state.has_flag(effect_bits::f)) flags.push(#f)
    CHECK_FLAG(NATO_ACTIVE);
    CHECK_FLAG(NATO_CANCELED_FRANCE);
    CHECK_FLAG(NATO_CANCELED_WEST_GERMANY);
    CHECK_FLAG(MARSHALL_PLAN_PLAYED);
    CHECK_FLAG(WARSAW_PACT_PLAYED);
    CHECK_FLAG(US_JAPAN_PACT_ACTIVE);
    CHECK_FLAG(CONTAINMENT_ACTIVE);
    CHECK_FLAG(PURGE_US_ACTIVE);
    CHECK_FLAG(PURGE_USSR_ACTIVE);
    CHECK_FLAG(VIETNAM_REVOLTS_ACTIVE);
    CHECK_FLAG(FORMOSAN_RESOLUTION_ACTIVE);
    CHECK_FLAG(CMC_ACTIVE_US);
    CHECK_FLAG(CMC_ACTIVE_USSR);
    CHECK_FLAG(NUCLEAR_SUBS_ACTIVE);
    CHECK_FLAG(QUAGMIRE_ACTIVE);
    CHECK_FLAG(BEAR_TRAP_ACTIVE);
    CHECK_FLAG(SALT_ACTIVE);
    CHECK_FLAG(WE_WILL_BURY_YOU_PENDING);
    CHECK_FLAG(BREZHNEV_DOCTRINE_ACTIVE);
    CHECK_FLAG(FLOWER_POWER_ACTIVE);
    CHECK_FLAG(U2_INCIDENT_ACTIVE);
    CHECK_FLAG(SHUTTLE_DIPLOMACY_ACTIVE);
    CHECK_FLAG(DEATH_SQUADS_US);
    CHECK_FLAG(DEATH_SQUADS_USSR);
    CHECK_FLAG(CAMP_DAVID_PLAYED);
    CHECK_FLAG(IRON_LADY_PLAYED);
    CHECK_FLAG(NORTH_SEA_OIL_PLAYED);
    CHECK_FLAG(NORTH_SEA_OIL_ACTIVE);
    CHECK_FLAG(THE_REFORMER_PLAYED);
    CHECK_FLAG(IRAN_CONTRA_ACTIVE);
    CHECK_FLAG(EVIL_EMPIRE_PLAYED);
    CHECK_FLAG(ALDRICH_AMES_ACTIVE);
    CHECK_FLAG(JOHN_PAUL_II_PLAYED);
    CHECK_FLAG(NORAD_ACTIVE);
    CHECK_FLAG(YURI_AND_SAMANTHA_ACTIVE);
    CHECK_FLAG(AWACS_PLAYED);
    CHECK_FLAG(IRANIAN_HOSTAGE_CRISIS_PLAY);
    CHECK_FLAG(WILLY_BRANDT_PLAYED);
    CHECK_FLAG(TEAR_DOWN_THIS_WALL_PLAYED);
    CHECK_FLAG(CHERNOBYL_ACTIVE);
    CHECK_FLAG(SPACE_US_ATTEMPT_1);
    CHECK_FLAG(SPACE_US_ATTEMPT_2);
    CHECK_FLAG(SPACE_USSR_ATTEMPT_1);
    CHECK_FLAG(SPACE_USSR_ATTEMPT_2);
    CHECK_FLAG(DEFCON_SUICIDE_PROVOKED);
    CHECK_FLAG(CMC_SUICIDE_LOSS);
    CHECK_FLAG(EUROPE_CONTROL_WIN);
#undef CHECK_FLAG
    d.set("flags", std::move(flags));
    // uint64: exact in Python; a JS reader must use `flags`, not this, past 2^53.
    d.set("persistent_effects", Value(static_cast<uint64_t>(state.persistent_effects)));

    // 6. Countries, by name
    Value countries = Value::object();
    for (uint8_t i = 0; i < 84; ++i) {
        const auto& info = MapData::get_country(i);
        const uint8_t us_inf = state.countries[i].us_influence;
        const uint8_t ussr_inf = state.countries[i].ussr_influence;
        const uint8_t stab = info.stability;
        Value c = Value::object();
        c.set("id", I(i));
        c.set("name", std::string(info.name));
        c.set("stability", I(info.stability));
        c.set("battleground", Value(static_cast<bool>(info.battleground)));
        c.set("region", I(static_cast<int>(info.region)));
        c.set("us_influence", I(us_inf));
        c.set("ussr_influence", I(ussr_inf));
        if (us_inf >= stab && us_inf >= ussr_inf + stab) c.set("controlled_by", "US");
        else if (ussr_inf >= stab && ussr_inf >= us_inf + stab) c.set("controlled_by", "USSR");
        else c.set("controlled_by", "NONE");
        countries.set(std::string(info.name), std::move(c));
    }
    d.set("countries", std::move(countries));

    // 7. Cards and their locations
    Value us_hand = Value::array(), ussr_hand = Value::array();
    Value us_cards = Value::array(), ussr_cards = Value::array();
    Value discard = Value::array(), removed = Value::array(), unavailable = Value::array();
    int draw_deck_count = 0;
    for (uint8_t i = 1; i <= 110; ++i) {
        switch (state.card_locations[i]) {
            case CardLocation::HAND_US_UNKNOWN:
            case CardLocation::HAND_US_KNOWN:
            case CardLocation::HAND_USSR_UNKNOWN:
            case CardLocation::HAND_USSR_KNOWN: {
                const bool us = in_hand_of(state.card_locations[i], Player::US);
                Value ci = Value::object();
                ci.set("id", I(i));
                ci.set("name", std::string(CardData::get_card_name(i)));
                ci.set("ops", I(CardData::get_card(i).ops));
                (us ? us_hand : ussr_hand).push(I(i));
                (us ? us_cards : ussr_cards).push(std::move(ci));
                break;
            }
            case CardLocation::DISCARD_PILE: discard.push(I(i)); break;
            case CardLocation::REMOVED_FROM_GAME: removed.push(I(i)); break;
            case CardLocation::DRAW_DECK: ++draw_deck_count; break;
            case CardLocation::UNAVAILABLE: unavailable.push(I(i)); break;
            default: break;
        }
    }
    Value hands = Value::object();
    hands.set("US", std::move(us_hand));
    hands.set("USSR", std::move(ussr_hand));
    hands.set("US_cards", std::move(us_cards));
    hands.set("USSR_cards", std::move(ussr_cards));
    d.set("hands", std::move(hands));
    d.set("discard_pile", std::move(discard));
    d.set("removed_pile", std::move(removed));
    d.set("unavailable_cards", std::move(unavailable));
    d.set("draw_deck_count", I(draw_deck_count));

    Value all_locs = Value::object();
    for (uint8_t i = 1; i <= 110; ++i) {
        all_locs.set(std::to_string(i), card_location_name(state.card_locations[i]));
    }
    d.set("card_locations", std::move(all_locs));

    // 8. The open decision
    const auto& ctx = state.ctx();
    Value c = Value::object();
    c.set("decision_player", player_name(ctx.decision_player));
    c.set("decision_type", I(static_cast<int>(ctx.decision_type)));
    c.set("decision_type_name", decision_type_name(ctx.decision_type));
    c.set("op_mode", I(static_cast<int>(ctx.op_mode)));
    c.set("op_mode_name", op_mode_name(static_cast<uint8_t>(ctx.op_mode)));
    c.set("pending_op_card", I(ctx.pending_op_card));
    c.set("pending_op_card_name", ctx.pending_op_card > 0 ? std::string(CardData::get_card_name(ctx.pending_op_card)) : std::string());
    c.set("pending_ops_value", I(ctx.pending_ops_value));
    c.set("remaining_steps", I(ctx.remaining_steps));
    c.set("max_per_country", I(ctx.max_per_country));
    c.set("allow_early_stop", Value(ctx.allow_early_stop != 0));
    c.set("resolving_card", I(ctx.resolving_card));
    c.set("resolving_card_name", ctx.resolving_card > 0 ? std::string(CardData::get_card_name(ctx.resolving_card)) : std::string());
    c.set("stack_depth", I(state.ctx_stack_depth));
    d.set("decision_context", std::move(c));

    // 9. The legal actions of that decision, with readable labels
    uint8_t mask[128];
    size_t mask_size = 0;
    Engine::get_legal_action_mask(state, mask, &mask_size);
    Value legal = Value::array();
    Value labels = Value::object();
    for (size_t i = 0; i < mask_size; ++i) {
        if (!mask[i]) continue;
        const int id = static_cast<int>(i);
        legal.push(I(id));
        std::string label;
        switch (ctx.decision_type) {
            case DecisionType::POINT_NODE:
                if (i < 84) label = std::string(MapData::get_country_name(static_cast<uint8_t>(i)));
                else if (i == 84) label = "Done / Pass";
                break;
            case DecisionType::SELECT_CARD:
                if (i >= 1 && i <= 110) label = std::string(CardData::get_card_name(static_cast<uint8_t>(i)));
                break;
            case DecisionType::SELECT_PLAY_MODE: label = play_mode_name(static_cast<uint8_t>(i)); break;
            case DecisionType::SELECT_OP_MODE: label = op_mode_name(static_cast<uint8_t>(i)); break;
            case DecisionType::CHOOSE_TIMING_BRANCH: label = timing_branch_name(static_cast<uint8_t>(i)); break;
            default: label = "Option " + std::to_string(i); break;
        }
        if (!label.empty()) labels.set(std::to_string(i), std::move(label));
    }
    Value legal_actions = Value::object();
    legal_actions.set("decision_type", I(static_cast<int>(ctx.decision_type)));
    legal_actions.set("decision_type_name", decision_type_name(ctx.decision_type));
    legal_actions.set("decision_player", player_name(ctx.decision_player));
    legal_actions.set("valid_ids", std::move(legal));
    legal_actions.set("valid_action_labels", std::move(labels));
    legal_actions.set("allow_early_stop", Value(ctx.allow_early_stop != 0));
    d.set("legal_actions", std::move(legal_actions));

    const bool terminal = Engine::is_terminal(state);
    d.set("is_terminal", Value(terminal));
    d.set("terminal_utility", Value(terminal ? static_cast<double>(Engine::get_terminal_utility(state)) : 0.0));
    return d;
}

// ---- the save ---------------------------------------------------------------------------------

Value save(const GameState& state) {
    Value d = Value::object();
    d.set("format", "ts_save_v2");   // v1 dropped the decision stack; readers must know which

    d.set("victory_points", I(state.victory_points));
    d.set("defcon", I(state.defcon));
    d.set("turn", I(state.turn));
    d.set("action_round", I(state.action_round));
    d.set("us_mil_ops", I(state.us_mil_ops));
    d.set("ussr_mil_ops", I(state.ussr_mil_ops));
    d.set("us_space_track", I(state.us_space_track));
    d.set("ussr_space_track", I(state.ussr_space_track));
    d.set("phasing_player", I(static_cast<int>(state.phasing_player)));
    d.set("current_phase", I(static_cast<int>(state.current_phase)));
    d.set("headline_us_card", I(state.headline_us_card));
    d.set("headline_ussr_card", I(state.headline_ussr_card));
    d.set("headline_first_card", I(state.headline_first_card));
    d.set("headline_second_card", I(state.headline_second_card));
    d.set("headline_stage", I(state.headline_stage));
    d.set("forced_card_player", I(static_cast<int>(state.forced_card_player)));
    d.set("forced_card_id", I(state.forced_card_id));
    d.set("defcon_dropped_to_2", I(state.defcon_dropped_to_2));
    d.set("china_card_holder", I(static_cast<int>(state.china_card_holder)));
    d.set("china_card_playable", I(state.china_card_playable));
    d.set("persistent_effects", Value(static_cast<uint64_t>(state.persistent_effects)));
    d.set("rng_state", Value(static_cast<uint64_t>(state.rng_state)));

    d.set("headline_first_owner", I(static_cast<int>(state.headline_first_owner)));
    d.set("headline_second_owner", I(static_cast<int>(state.headline_second_owner)));
    d.set("last_die_roll", I(state.last_die_roll));
    d.set("last_opp_die_roll", I(state.last_opp_die_roll));

    const auto& r = state.last_roll;
    Value roll = Value::object();
    roll.set("type", I(static_cast<int>(r.type)));
    roll.set("roller", I(static_cast<int>(r.roller)));
    roll.set("card_id", I(r.card_id));
    roll.set("country_id", I(r.country_id));
    roll.set("roll1", I(r.roll1));
    roll.set("mod1", I(r.mod1));
    roll.set("roll2", I(r.roll2));
    roll.set("mod2", I(r.mod2));
    roll.set("success", Value(static_cast<bool>(r.success)));
    roll.set("net_delta", I(r.net_delta));
    d.set("last_roll", std::move(roll));

    // The decision state machine. Without it the restored state points at a different decision
    // than the saved one, and every legal action and every observation differs.
    Value frames = Value::array();
    for (size_t i = 0; i < state.ctx_stack.size(); ++i) {
        const DecisionContext& c = state.ctx_stack[i];
        Value f = Value::object();
        f.set("decision_player", I(static_cast<int>(c.decision_player)));
        f.set("decision_type", I(static_cast<int>(c.decision_type)));
        f.set("op_mode", I(static_cast<int>(c.op_mode)));
        f.set("pending_op_card", I(c.pending_op_card));
        f.set("pending_ops_value", I(c.pending_ops_value));
        f.set("remaining_steps", I(c.remaining_steps));
        f.set("max_per_country", I(c.max_per_country));
        f.set("allow_early_stop", I(c.allow_early_stop));
        f.set("resolving_card", I(c.resolving_card));
        f.set("timing_branch", I(c.timing_branch));
        f.set("suppress_op_card_event", I(c.suppress_op_card_event));
        f.set("event_granted_ops", I(c.event_granted_ops));
        f.set("pending_roll", I(static_cast<int>(c.pending_roll)));
        f.set("roll_target", I(c.roll_target));
        f.set("roll_actor", I(static_cast<int>(c.roll_actor)));
        f.set("event_stage", I(c.event_stage));
        Value si = Value::array(), vn = Value::array(), nc = Value::array();
        for (uint64_t v : c.start_influence_nodes) si.push(Value(v));
        for (uint64_t v : c.visited_nodes) vn.push(Value(v));
        for (uint64_t v : c.node_count_bits) nc.push(Value(v));
        f.set("start_influence_nodes", std::move(si));
        f.set("visited_nodes", std::move(vn));
        f.set("node_count_bits", std::move(nc));
        frames.push(std::move(f));
    }
    d.set("ctx_stack", std::move(frames));
    d.set("ctx_stack_depth", I(state.ctx_stack_depth));

    Value locs = Value::array();
    for (int i = 0; i <= 110; ++i) locs.push(I(static_cast<int>(state.card_locations[i])));
    d.set("card_locations", std::move(locs));

    Value us_inf = Value::array(), ussr_inf = Value::array();
    for (int i = 0; i < 84; ++i) {
        us_inf.push(I(state.countries[i].us_influence));
        ussr_inf.push(I(state.countries[i].ussr_influence));
    }
    d.set("us_influence", std::move(us_inf));
    d.set("ussr_influence", std::move(ussr_inf));
    return d;
}

// ---- loading a save ---------------------------------------------------------------------------

namespace {

// A scalar field: missing -> fallback, and so is a value of the wrong type or out of range --
// exactly what the nanobind loader did with its try/catch around nb::cast.
template <typename T>
T get(const Value& obj, const char* key, T fallback) {
    const Value* v = obj.find(key);
    if (!v) return fallback;
    if constexpr (std::is_same_v<T, bool>) {
        return v->kind() == Value::Kind::Bool ? v->as_bool() : fallback;
    } else {
        int64_t s = 0;
        uint64_t u = 0;
        const int64_t lo = std::is_signed_v<T> ? static_cast<int64_t>(std::numeric_limits<T>::min()) : 0;
        const uint64_t hi = static_cast<uint64_t>(std::numeric_limits<T>::max());
        if (!v->to_integral(lo, hi, &s, &u)) return fallback;
        return std::is_signed_v<T> ? static_cast<T>(s) : static_cast<T>(u);
    }
}

// An element of a list field: unlike a scalar there is no fallback -- the nanobind loader cast
// these without a guard, so a bad element failed the whole load.
template <typename T>
bool element(const Value& v, T* out) {
    int64_t s = 0;
    uint64_t u = 0;
    const int64_t lo = std::is_signed_v<T> ? static_cast<int64_t>(std::numeric_limits<T>::min()) : 0;
    const uint64_t hi = static_cast<uint64_t>(std::numeric_limits<T>::max());
    if (!v.to_integral(lo, hi, &s, &u)) return false;
    *out = std::is_signed_v<T> ? static_cast<T>(s) : static_cast<T>(u);
    return true;
}

bool list_of(const Value& obj, const char* key, const Value** out, std::string* error) {
    const Value* v = obj.find(key);
    *out = nullptr;
    if (!v) return true;
    if (!v->is_array()) {
        if (error) *error = std::string("'") + key + "' is not a list";
        return false;
    }
    *out = v;
    return true;
}

}  // namespace

bool load_save(const Value& d, GameState& out, std::string* error) {
    if (!d.is_object()) {
        if (error) *error = "a save is a JSON object";
        return false;
    }
    GameState s{};
    Engine::init_game(s, 1);   // a valid baseline; every field below overwrites it

    s.victory_points = get<int8_t>(d, "victory_points", s.victory_points);
    s.defcon = get<uint8_t>(d, "defcon", s.defcon);
    s.turn = get<uint8_t>(d, "turn", s.turn);
    s.action_round = get<uint8_t>(d, "action_round", s.action_round);
    s.us_mil_ops = get<uint8_t>(d, "us_mil_ops", s.us_mil_ops);
    s.ussr_mil_ops = get<uint8_t>(d, "ussr_mil_ops", s.ussr_mil_ops);
    s.us_space_track = get<uint8_t>(d, "us_space_track", s.us_space_track);
    s.ussr_space_track = get<uint8_t>(d, "ussr_space_track", s.ussr_space_track);
    s.phasing_player = static_cast<Player>(get<int>(d, "phasing_player", static_cast<int>(s.phasing_player)));
    s.current_phase = static_cast<Phase>(get<int>(d, "current_phase", static_cast<int>(s.current_phase)));
    s.headline_us_card = get<uint8_t>(d, "headline_us_card", s.headline_us_card);
    s.headline_ussr_card = get<uint8_t>(d, "headline_ussr_card", s.headline_ussr_card);
    s.headline_first_card = get<uint8_t>(d, "headline_first_card", s.headline_first_card);
    s.headline_second_card = get<uint8_t>(d, "headline_second_card", s.headline_second_card);
    s.headline_stage = get<uint8_t>(d, "headline_stage", s.headline_stage);
    s.forced_card_player = static_cast<Player>(get<int>(d, "forced_card_player", static_cast<int>(s.forced_card_player)));
    s.forced_card_id = get<uint8_t>(d, "forced_card_id", s.forced_card_id);
    s.defcon_dropped_to_2 = get<uint8_t>(d, "defcon_dropped_to_2", s.defcon_dropped_to_2);
    s.china_card_holder = static_cast<Player>(get<int>(d, "china_card_holder", static_cast<int>(s.china_card_holder)));
    s.china_card_playable = get<uint8_t>(d, "china_card_playable", s.china_card_playable);
    s.persistent_effects = get<uint64_t>(d, "persistent_effects", s.persistent_effects);
    s.rng_state = get<uint64_t>(d, "rng_state", s.rng_state);

    s.headline_first_owner = static_cast<Player>(get<int>(d, "headline_first_owner", static_cast<int>(s.headline_first_owner)));
    s.headline_second_owner = static_cast<Player>(get<int>(d, "headline_second_owner", static_cast<int>(s.headline_second_owner)));
    s.last_die_roll = get<uint8_t>(d, "last_die_roll", s.last_die_roll);
    s.last_opp_die_roll = get<uint8_t>(d, "last_opp_die_roll", s.last_opp_die_roll);

    if (const Value* roll = d.find("last_roll")) {
        if (!roll->is_object()) {
            if (error) *error = "'last_roll' is not an object";
            return false;
        }
        auto& r = s.last_roll;
        r.type = static_cast<RollType>(get<int>(*roll, "type", static_cast<int>(r.type)));
        r.roller = static_cast<Player>(get<int>(*roll, "roller", static_cast<int>(r.roller)));
        r.card_id = get<uint8_t>(*roll, "card_id", r.card_id);
        r.country_id = get<uint8_t>(*roll, "country_id", r.country_id);
        r.roll1 = get<uint8_t>(*roll, "roll1", r.roll1);
        r.mod1 = get<int8_t>(*roll, "mod1", r.mod1);
        r.roll2 = get<uint8_t>(*roll, "roll2", r.roll2);
        r.mod2 = get<int8_t>(*roll, "mod2", r.mod2);
        r.success = get<bool>(*roll, "success", r.success);
        r.net_delta = get<int8_t>(*roll, "net_delta", r.net_delta);
    }

    const Value* frames = nullptr;
    if (!list_of(d, "ctx_stack", &frames, error)) return false;
    if (frames) {
        const auto& items = frames->items();
        for (size_t i = 0; i < items.size() && i < s.ctx_stack.size(); ++i) {
            const Value& f = items[i];
            if (!f.is_object()) {
                if (error) *error = "ctx_stack[" + std::to_string(i) + "] is not an object";
                return false;
            }
            DecisionContext& c = s.ctx_stack[i];
            c = DecisionContext{};
            c.decision_player = static_cast<Player>(get<int>(f, "decision_player", 0));
            c.decision_type = static_cast<DecisionType>(get<int>(f, "decision_type", 0));
            c.op_mode = static_cast<OpMode>(get<int>(f, "op_mode", 0));
            c.pending_op_card = get<uint8_t>(f, "pending_op_card", 0);
            c.pending_ops_value = get<uint8_t>(f, "pending_ops_value", 0);
            c.remaining_steps = get<uint8_t>(f, "remaining_steps", 0);
            c.max_per_country = get<uint8_t>(f, "max_per_country", 0);
            c.allow_early_stop = get<uint8_t>(f, "allow_early_stop", 0);
            c.resolving_card = get<uint8_t>(f, "resolving_card", 0);
            c.timing_branch = get<uint8_t>(f, "timing_branch", 0);
            c.suppress_op_card_event = get<uint8_t>(f, "suppress_op_card_event", 0);
            c.event_granted_ops = get<uint8_t>(f, "event_granted_ops", 0);
            c.pending_roll = static_cast<RollType>(get<int>(f, "pending_roll", 0));
            c.roll_target = get<uint8_t>(f, "roll_target", 0);
            c.roll_actor = static_cast<Player>(get<int>(f, "roll_actor", 0));
            c.event_stage = get<uint8_t>(f, "event_stage", 0);
            const std::pair<const char*, uint64_t*> bitsets[] = {
                {"start_influence_nodes", c.start_influence_nodes.data()},
                {"visited_nodes", c.visited_nodes.data()},
                {"node_count_bits", c.node_count_bits.data()},
            };
            const size_t sizes[] = {c.start_influence_nodes.size(), c.visited_nodes.size(), c.node_count_bits.size()};
            for (size_t b = 0; b < 3; ++b) {
                const Value* list = nullptr;
                if (!list_of(f, bitsets[b].first, &list, error)) return false;
                if (!list) continue;
                for (size_t k = 0; k < list->items().size() && k < sizes[b]; ++k) {
                    if (!element<uint64_t>(list->items()[k], &bitsets[b].second[k])) {
                        if (error) *error = std::string("bad element in ") + bitsets[b].first;
                        return false;
                    }
                }
            }
        }
    }
    s.ctx_stack_depth = get<uint8_t>(d, "ctx_stack_depth", s.ctx_stack_depth);

    const Value* locs = nullptr;
    if (!list_of(d, "card_locations", &locs, error)) return false;
    if (locs) {
        for (size_t i = 0; i < locs->items().size() && i <= 110; ++i) {
            int v = 0;
            if (!element<int>(locs->items()[i], &v)) {
                if (error) *error = "bad element in card_locations";
                return false;
            }
            s.card_locations[i] = static_cast<CardLocation>(v);
        }
    }
    const Value* us_inf = nullptr;
    const Value* ussr_inf = nullptr;
    if (!list_of(d, "us_influence", &us_inf, error)) return false;
    if (!list_of(d, "ussr_influence", &ussr_inf, error)) return false;
    if (us_inf && ussr_inf) {
        for (size_t i = 0; i < us_inf->items().size() && i < 84; ++i) {
            if (!element<uint8_t>(us_inf->items()[i], &s.countries[i].us_influence)) {
                if (error) *error = "bad element in us_influence";
                return false;
            }
        }
        for (size_t i = 0; i < ussr_inf->items().size() && i < 84; ++i) {
            if (!element<uint8_t>(ussr_inf->items()[i], &s.countries[i].ussr_influence)) {
                if (error) *error = "bad element in ussr_influence";
                return false;
            }
        }
    }
    out = s;
    return true;
}

// ---- why a game ended ---------------------------------------------------------------------------

const char* ending_reason(const GameState& state) {
    // 0. Cuban Missile Crisis suicide: couping while CMC is active without the influence to
    // cancel it. The engine ends the game at +/-20 VP and deliberately leaves DEFCON alone, so
    // without this check the loss is indistinguishable from a legitimate 20 VP win -- and it is
    // a self-inflicted loss, not a win by anyone's play.
    if (state.has_flag(effect_bits::CMC_SUICIDE_LOSS)) return "DEFCON 1 (own decision)";

    // 1. DEFCON 1 takes precedence over VP.
    if (state.defcon <= 1) {
        return state.has_flag(effect_bits::DEFCON_SUICIDE_PROVOKED) ? "DEFCON 1 (opponent decision)"
                                                                     : "DEFCON 1 (own decision)";
    }

    const int vp = state.victory_points;
    const int abs_vp = vp < 0 ? -vp : vp;

    // 2. Wargames (#100): over before final scoring, without 20 VP. `<= 10`, not `< 10`: a game
    // that goes the distance terminates holding turn 11 (finish_end_turn increments the turn and
    // only then tests `turn <= 10`), so a Wargames in turn 10 is still a Wargames.
    if (state.turn <= 10 && abs_vp < 20 && state.current_phase == Phase::GAME_OVER) return "wargames";

    // 3. Europe Control ends the game at +/-20 VP, indistinguishable from any other 20 VP win
    // without the flag the engine sets.
    if (state.has_flag(effect_bits::EUROPE_CONTROL_WIN)) return "Europe Control";

    // 4. 20 VP, or a held scoring card.
    if (abs_vp >= 20) return "20 VP";

    // 5. Final scoring.
    if (state.turn >= 10) return "final scoring";

    if (state.current_phase == Phase::GAME_OVER) return "wargames";
    return "20 VP";
}

}  // namespace ts::state_json
