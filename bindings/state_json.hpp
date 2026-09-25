// The engine state as JSON, for every consumer outside C++: the Python bindings and the
// browser (WebAssembly) build.
//
// Three views of a GameState used to be written directly as nanobind dicts in ts_bindings.cpp:
// the workbench's display state, and the named-field save a position is stored and shared in.
// The browser engine needs exactly the same three, and two hand-kept copies would drift -- the
// same display state rendering differently depending on which runtime produced it, and a
// shared-position link that one of them cannot open. So they are written once, here, as a small
// JSON value tree: the bindings convert the tree to Python objects directly, the WebAssembly
// build prints it.
//
// No dependencies beyond the standard library; allocation is fine here -- this is presentation,
// not the zero-allocation simulation core.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ts/game_state.hpp"

namespace ts::state_json {

class Value {
public:
    enum class Kind : uint8_t { Null, Bool, Int, UInt, Double, String, Array, Object };
    using Member = std::pair<std::string, Value>;

    Value() = default;
    Value(bool b) : kind_(Kind::Bool), b_(b) {}                       // NOLINT: implicit on purpose
    Value(int v) : kind_(Kind::Int), i_(v) {}                         // NOLINT
    Value(int64_t v) : kind_(Kind::Int), i_(v) {}                     // NOLINT
    Value(uint64_t v) : kind_(Kind::UInt), u_(v) {}                   // NOLINT
    Value(double v) : kind_(Kind::Double), d_(v) {}                   // NOLINT
    Value(const char* s) : kind_(Kind::String), s_(s) {}              // NOLINT
    Value(std::string s) : kind_(Kind::String), s_(std::move(s)) {}   // NOLINT

    static Value array() { Value v; v.kind_ = Kind::Array; return v; }
    static Value object() { Value v; v.kind_ = Kind::Object; return v; }

    Kind kind() const { return kind_; }
    bool is_object() const { return kind_ == Kind::Object; }
    bool is_array() const { return kind_ == Kind::Array; }

    // Object: set (replacing an existing key) and look up. Members keep insertion order; dump()
    // sorts them, so the text is canonical whatever order they were set in.
    Value& set(std::string key, Value v);
    const Value* find(std::string_view key) const;
    const std::vector<Member>& members() const { return members_; }

    // Array.
    void push(Value v) { items_.push_back(std::move(v)); }
    const std::vector<Value>& items() const { return items_; }

    bool as_bool() const { return b_; }
    int64_t as_int() const { return i_; }
    uint64_t as_uint() const { return u_; }
    double as_double() const { return d_; }
    const std::string& as_string() const { return s_; }

    // Exact integral conversion: false when the value is not an integer or does not fit in
    // [lo, hi]. Mirrors a nanobind integer cast, which refuses floats, bools and overflow.
    bool to_integral(int64_t lo, uint64_t hi, int64_t* out_signed, uint64_t* out_unsigned) const;

    // Compact, keys sorted -- byte for byte what Python's
    // json.dumps(obj, sort_keys=True, separators=(",", ":")) prints for the same value, so a
    // save written here and one written by Python are the same text.
    std::string dump() const;

    // Strict RFC 8259 parse of a whole document. On failure returns false and says where.
    static bool parse(std::string_view text, Value& out, std::string* error);

private:
    void dump_to(std::string& out) const;

    Kind kind_ = Kind::Null;
    bool b_ = false;
    int64_t i_ = 0;
    uint64_t u_ = 0;
    double d_ = 0.0;
    std::string s_;
    std::vector<Value> items_;
    std::vector<Member> members_;
};

// The workbench's display state: tracks, countries by name with control, hands with names,
// piles, card locations, the decision context, the legal actions of the open decision with
// labels, and the terminal flag/utility. What `GameState.to_dict()` returns in Python.
Value display_state(const GameState& state);

// The named-field save of a position, mid-game included (format "ts_save_v2"): scalars, RNG,
// card locations, influence, headline owners, the die-roll record and the decision-context
// stack. Round-trips through load_save exactly (the observation and the legal mask included).
// Does not carry action_history or turn_aggregates, which are diagnostics no rule reads.
Value save(const GameState& state);

// Rebuild a position from a save. Missing keys take their default and unknown keys are ignored,
// so a save from an older or newer build still opens minus what it cannot use; a scalar of the
// wrong type or out of range also falls back to its default. A malformed list or frame is an
// error: returns false and fills `error`.
bool load_save(const Value& save, GameState& out, std::string* error);

}  // namespace ts::state_json
