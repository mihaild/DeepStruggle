// The exact outcomes of a pending die roll, compound rolls included (owner, 2026-11-22).
//
// The observation does not show which roll is pending (a coup in Italy and one in Iran are the same input before
// the die), so the critic is never read or trained at a state waiting on a die: it is replaced by the probability-
// weighted average of the critic over the roll's outcomes. Two rolls can follow each other with no decision between
// them -- an Ops-first coup or realignment on Korean War or Arab-Israeli War whose owed Event then rolls at once, two
// rolling headlines back to back -- so an outcome of the first roll may itself wait on a die. Here such an outcome is
// replaced by the second roll's outcomes, recursively, probabilities multiplied, until a decision, a chance node that
// rolls no die (the turn-end park), or the end of the game: one compound chance node, whose outcomes are the states
// after its LAST roll.
//
// "Waits on a die": the state is at a die roll, or (forced settle mode, the runner's default) reaches one through
// single-choice decisions only -- the steps the runner takes by itself. A chain longer than kMaxRollChain dice or a
// node with more than kMaxCompoundOutcomes distinct outcomes is an error the caller raises, never a truncation.
//
// Each face is stepped on a copy with the forced die ROLL_DIE carries (primary_id the roller's, secondary_id the
// opponent's). A realignment, the Summit and the Olympic Games depend on the two dice only through their difference:
// 11 differences weighted (6 - |d|) / 36. An Olympics tie (re-rolled when rolled, given to the opponent when forced)
// is no outcome, the rest renormalised. Randomness the same engine step draws after the die comes from the state's
// own stream, as after the real roll. Used by the batch runner's afterstate capture (ts_bindings.cpp) and the
// standalone ts_engine.roll_outcomes; no engine change.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ts/engine.hpp"
#include "ts/game_state.hpp"
#include "ts/micro_action.hpp"
#include "ts/state_machine.hpp"

namespace ts::chance {

//: The most dice one compound chance node may chain (by the rules two: an Ops-first coup or realignment and the war it
//: owes, or two rolling headlines).
constexpr int kMaxRollChain = 4;
//: The most distinct outcomes one compound chance node may have (a realignment then a war: 11 x 6 = 66).
constexpr size_t kMaxCompoundOutcomes = 512;
//: The most single-choice steps followed between two rolls (afterstate_critic.MAX_WALK).
constexpr size_t kMaxForcedSteps = 64;

enum class Status : uint8_t { NONE = 0, OK = 1, CHAIN_TOO_LONG = 2, TOO_MANY_OUTCOMES = 3, NO_OUTCOME = 4,
                              FORCED_TOO_LONG = 5 };

inline const char* status_message(Status s) {
    switch (s) {
        case Status::CHAIN_TOO_LONG: return "a compound chance node chains more dice than kMaxRollChain";
        case Status::TOO_MANY_OUTCOMES: return "a compound chance node has more distinct outcomes than kMaxCompoundOutcomes";
        case Status::NO_OUTCOME: return "a die roll has no outcome the engine accepts";
        case Status::FORCED_TOO_LONG: return "more than kMaxForcedSteps single-choice steps after a die roll";
        default: return "no error";
    }
}

inline bool at_chance(const GameState& g) {
    return g.ctx().decision_player == Player::NONE && g.ctx().decision_type == DecisionType::ROLL_DIE;
}

//: A chance node that rolls a die (the turn-end park draws cards and rolls nothing).
inline bool rolls_a_die(const GameState& g) {
    return at_chance(g) && g.ctx().pending_roll != RollType::NONE && g.ctx().pending_roll != RollType::TURN_CLEANUP &&
           !Engine::is_terminal(g);
}

//: Whether `g` waits on a die: it is at one, or (forced) reaches one through single-choice decisions alone. `at_die`
//: receives the state at that die. FORCED_TOO_LONG in `err` past kMaxForcedSteps steps.
inline bool waits_on_die(const GameState& g, bool forced, GameState& at_die, Status& err) {
    if (rolls_a_die(g)) { at_die = g; return true; }
    if (!forced || Engine::is_terminal(g) || at_chance(g)) return false;
    at_die = g;
    for (size_t k = 0; k < kMaxForcedSteps; ++k) {
        if (Engine::auto_advance_step(at_die, 1) == 0) return false;     // a decision with a choice
        if (Engine::is_terminal(at_die)) return false;
        if (at_chance(at_die)) return rolls_a_die(at_die);
    }
    err = Status::FORCED_TOO_LONG;
    return false;
}

inline void clear_die_record(GameState& c) {
    c.last_roll = DieRollRecord{};
    c.last_die_roll = 0;
    c.last_opp_die_roll = 0;
}

//: One outcome: the state after the node's last roll with its die record cleared (the observation does not read it,
//: and outcomes equal but for it are one), and its probability.
struct Outcome {
    GameState state;
    float prob;
};

namespace detail {

inline void add_outcome(std::vector<Outcome>& out, const GameState& c, float p) {
    for (Outcome& o : out) {
        if (std::memcmp(&o.state, &c, sizeof(GameState)) == 0) { o.prob += p; return; }
    }
    out.push_back(Outcome{c, p});
}

inline Status expand(const GameState& g, bool forced, bool compound, float weight, int depth, std::vector<Outcome>& out,
                     int& dice) {
    if (depth >= kMaxRollChain) return Status::CHAIN_TOO_LONG;
    dice = std::max(dice, depth + 1);
    const RollType rt = g.ctx().pending_roll;
    const bool two = rt == RollType::REALIGNMENT || rt == RollType::SUMMIT || rt == RollType::OLYMPIC_GAMES;
    // The faces' states, raw (a later roll's step may read the die record) and deduplicated on the raw state.
    thread_local std::vector<GameState> level_states[kMaxRollChain];
    thread_local std::vector<float> level_weights[kMaxRollChain];
    std::vector<GameState>& states = level_states[depth];
    std::vector<float>& weights = level_weights[depth];
    states.clear(); weights.clear();
    float kept = 0.0f;
    for (int k = two ? -5 : 1; k <= (two ? 5 : 6); ++k) {
        const uint8_t a = two ? static_cast<uint8_t>(std::max(1, 1 + k)) : static_cast<uint8_t>(k);
        const uint8_t b = two ? static_cast<uint8_t>(a - k) : 0;
        const float w = two ? static_cast<float>(6 - std::abs(k)) / 36.0f : 1.0f / 6.0f;
        // Two dice by their difference: each side's modifiers are added apart, and the higher total removes the
        // difference or wins. The Olympic Games re-roll a tie of the sponsor's die + 2 against the opponent's only
        // when both dice are rolled: a forced tie goes to the opponent without drawing (card_dispatcher.cpp), so a tie
        // pair is not an outcome and the rest are renormalised.
        if (rt == RollType::OLYMPIC_GAMES && a + 2 == b) continue;
        GameState c = g;
        // Randomness the same engine step draws after the die (rare: a coup or realignment whose step goes on into a
        // random draw) comes from the state's own stream, as it would after the real roll: unbiased in average.
        const MicroAction ma{DecisionType::ROLL_DIE, a, b, 0};
        if (!StateMachine::step(c, ma)) continue;
        kept += w;
        bool known = false;
        for (size_t j = 0; j < states.size(); ++j) {
            if (std::memcmp(&states[j], &c, sizeof(GameState)) == 0) { weights[j] += w; known = true; break; }
        }
        if (!known) { states.push_back(c); weights.push_back(w); }
    }
    if (kept <= 0.0f) return Status::NO_OUTCOME;
    // Copied out: a deeper level reuses its own buffers, never this one's, but the vector may be needed after.
    const size_t n = states.size();
    for (size_t j = 0; j < n; ++j) {
        const float p = weight * (level_weights[depth][j] / kept);
        GameState at_die;
        Status err = Status::OK;
        if (compound && waits_on_die(level_states[depth][j], forced, at_die, err)) {
            const Status s = expand(at_die, forced, compound, p, depth + 1, out, dice);
            if (s != Status::OK) return s;
        } else {
            if (err != Status::OK) return err;
            GameState c = level_states[depth][j];
            clear_die_record(c);
            add_outcome(out, c, p);
        }
        if (out.size() > kMaxCompoundOutcomes) return Status::TOO_MANY_OUTCOMES;
    }
    return Status::OK;
}

}  // namespace detail

//: The distinct outcomes of the die pending in `g` (a die roll: rolls_a_die), in order of first appearance, their
//: probabilities summing to 1. `compound`: an outcome that waits on another die is that die's outcomes, recursively.
//: NONE where `g` rolls no die; otherwise OK or the error that stopped the enumeration (`out` then incomplete).
//: `dice`, if given, receives the longest chain of dice met (1 for a plain roll).
inline Status compound_outcomes(const GameState& g, bool forced, bool compound, std::vector<Outcome>& out,
                                int* dice = nullptr) {
    out.clear();
    int d = 0;
    if (dice) *dice = 0;
    if (!rolls_a_die(g)) return Status::NONE;
    const Status s = detail::expand(g, forced, compound, 1.0f, 0, out, d);
    if (dice) *dice = d;
    return s;
}

}  // namespace ts::chance
