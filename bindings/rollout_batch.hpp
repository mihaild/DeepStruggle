// The rollout root's playouts, written once for both bindings (ai/search/rollout_root.py and the
// browser workbench): the worlds of a decision, their greedy playouts to a horizon, and the leaves
// the critic reads. The network stays outside -- the caller runs it once per ply over the live rows
// and hands the logits back -- so the same C++ drives torch from Python and onnxruntime from the page.
//
// One decision: the mover's `k` candidate moves, each played in `worlds` sampled worlds. A world
// redeals the cards the mover cannot see (`determinize`, the honest searcher's) and, with
// `shared_dice`, draws one dice stream that every candidate of the world then plays under, so two
// candidates' values in a world differ only by the move. A world in which any candidate is illegal
// (The Cambridge Five can change the legal set) is dropped for every candidate. Each row then plays
// greedily -- both sides the network's argmax -- through `horizon` action-round boundaries (a change
// of turn, action round, phasing side or phase; 0 reads the position the move leads to; with
// `Config::aligned`, counted from the end of the mover's own action round for every candidate) -- or, by
// `Config::stop`, to the end of the decision's turn, or to the later of the two -- with the
// DEFCON-2 suicide guard when `guard` is set: at a play mode, an event branch or a coup's target,
// a greedy move that ends the game against its own player on the spot is replaced by the next-best
// of the top four that does not. A finished game is the row's value from the mover's side; a row
// at its horizon is a leaf, whose observation from the mover's side the caller values with the
// critic. Several decisions can share one batch, so a tournament's hundred positions make one
// network call per ply.
//
// Racing (`race`): a decision can be played in two stages -- every candidate on the first worlds, then,
// once those are valued, more worlds for the network's own move and only the challengers that do not
// trail it. `decide` reads the candidates still in the race over the worlds they were all valued in.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ts/game_state.hpp"

namespace ts::rollout {

// A clone of `s` with the cards `me` cannot see -- the opponent's unknown hand and the draw deck --
// reshuffled between those two places, every count kept and every card `me` has seen left where
// it is; an opponent's chosen but unrevealed headline is forgotten, so the world asks it again
// from the hand it dealt. `rng` is a SplitMix64 state, advanced. The port of
// ai/search/dmcts.determinize; the Python one draws from random.Random, so the two draw different
// worlds from one seed, by design -- this one is what the rollout root uses in both builds.
// `keep_hand` (the oracle's world, Config::oracle) leaves the opponent's hand as it truly is and
// reshuffles only the draw deck: the hand is known, the deck order and the forgotten headline are not.
// `keep_revealed` (Config::box4) keeps an opposing headline `me` has seen (revealed_headline): the card
// stays chosen and in the opponent's hand, and only the rest of the hand is redealt.
void determinize(GameState& s, Player me, uint64_t& rng, bool keep_hand = false, bool keep_revealed = false) noexcept;

// The opposing headline `me` has seen, else 0. With Man in Earth Orbit (Space Race box 4) held by `me` alone
// the opponent chooses its headline first and reveals it (the engine asks it first, StateMachine::
// start_turn), so at `me`'s own headline choice the card is known -- though no observation slot shows it.
uint8_t revealed_headline(const GameState& s, Player me) noexcept;

// `me`'s hidden pool -- the opponent's unknown hand and the draw deck -- split as `held` says: card
// `cid` in the opponent's unknown hand where held[cid - 1] is set, in the deck otherwise. Nothing else
// changes. With determinize's forgetting (forced deals, an unrevealed opposing headline) this is a
// belief's world (`add`'s `hands`); without it, a past position rebuilt under a hypothesised hand
// (`hypothesis_rows`).
void place_hand(GameState& s, Player me, const uint8_t* held) noexcept;

// `n` copies of `s`, copy j with `me`'s hidden pool split by held[j * 110 .. j * 110 + 109]
// (place_hand), each copy's observation from the opponent's side in feature set `features` and its
// legal mask, row-major into caller buffers of n * obs_size_for(features) floats and
// n * FLAT_ACTION_SPACE_SIZE bytes: the network's view of a past decision of the opponent's under each
// hypothesised hand (ai/search/belief.py).
void hypothesis_rows(const GameState& s, Player me, const uint8_t* held, size_t n, uint32_t features,
                     float* obs, uint8_t* masks);
// hypothesis_rows' masks alone: from the opponent's side the hypotheses differ in the observation
// only in each hidden card's MY_HAND / DECK_OR_HIDDEN slot (the counts are fixed by the hand size), so
// a caller can build one row and set those slots itself; the legal masks it cannot.
void hypothesis_masks(const GameState& s, Player me, const uint8_t* held, size_t n, uint8_t* masks);

// The side to act at `s`: the decision's player, else the phasing player (ai/search/pimcts.acting_player).
Player acting_player(const GameState& s) noexcept;

// The turn-end park: after a turn's last action the engine waits at a chance node
// (StateMachine::begin_turn_cleanup) before the military-ops check, the China card, turn + 1 and the
// deal. Nobody decides there and only the deal that follows is random -- the "pre-deal" position P33's
// critic is trained on (the runner's capture) and searched to (Stop::kPredeal).
inline bool at_turn_cleanup(const GameState& s) noexcept {
    return s.current_phase != Phase::GAME_OVER && s.victory_points < 20 && s.victory_points > -20 &&
           s.ctx().decision_player == Player::NONE && s.ctx().decision_type == DecisionType::ROLL_DIE &&
           s.ctx().pending_roll == RollType::TURN_CLEANUP;
}

// Where a playout stops. kBoundaries: after `Config::horizon` action-round boundaries (the rollout
// root as measured). kTurnEnd: at the first position whose turn differs from the decision's -- the next
// turn's start, however many action rounds the turn turns out to have (North Sea Oil, the space race's
// eighth); `horizon` is not read. kLater: at the later of the two, so at least `horizon` boundaries and
// never inside the decision's turn. kEarlier: at the earlier of the two -- the turn's end when it comes
// within `horizon` boundaries, else after `horizon` boundaries. kPredeal: at the turn-end park of the
// decision's turn (at_turn_cleanup: after the last action, before the deal), nobody's decision, read from
// the root mover's side; `horizon` is not read. kEarlierPredeal: the earlier of `horizon` boundaries and
// that park.
enum class Stop : uint8_t { kBoundaries = 0, kTurnEnd = 1, kLater = 2, kEarlier = 3, kPredeal = 4,
                            kEarlierPredeal = 5 };

// Whose side a leaf's critic is read from. kMover: the root mover's, at every leaf (the root as measured).
// kDecider: the leaf's own decider's -- the side the critic is trained on -- with the value negated into
// the mover's frame where that is the opponent. The trainer only ever reads a decision from its decider's
// side, so a kMover leaf where the opponent decides is outside the critic's training distribution
// (P33: 15-55% of leaves by stop rule; the off-side reading is off by ~0.10). kDecider gives up the
// mover's knowledge of its own hand at those leaves. kOwn: a playout that reaches its stop plays on to the
// root mover's next decision and is read there, from the mover's side -- on-distribution and with the
// mover's information (the owner's design, P33).
enum class Leaf : uint8_t { kMover = 0, kDecider = 1, kOwn = 2 };

struct Config {
    int horizon = 4;            // action-round boundaries to play through; 0 = none
    bool shared_dice = true;    // one dice stream per world, every candidate
    bool guard = true;          // the DEFCON-2 suicide guard in the playouts
    uint32_t features = 0;      // obs_features bits the network reads (0: the base layout)
    Stop stop = Stop::kBoundaries;
    Leaf leaf = Leaf::kMover;
    // Keyed chance (keyed_chance.hpp): every draw tied to the situation it is made in, so a world's
    // candidates draw the same dice and cards wherever they reach the same situation. Implies one
    // key per world for every candidate (shared_dice is then not read).
    bool keyed = false;
    // The oracle's worlds: the opponent's true hand kept, only the draw deck reshuffled
    // (determinize's keep_hand). Not a deployable searcher -- the ceiling on what any belief over the
    // opponent's hand could give the root (P35 item 3).
    bool oracle = false;
    // Count every candidate's horizon from the end of the mover's own action round. Without it a
    // candidate that leaves choices inside the round (Ops points, an event's choices) counts the
    // round's end as its first boundary, while one that ends the round inside its own settling
    // (space, a decision-free event) does not -- so the two are read an action round apart, at
    // different points of the turn (the owner's Decolonization replay, 2026-10-19). Off by default:
    // every rollout player measured before it played the unaligned count.
    bool aligned = false;
    // Space Race box 4: at the headline choice of the side holding Man in Earth Orbit alone, the opponent's
    // headline -- chosen first and revealed -- stays in every world (revealed_headline), and the
    // opponent's hand is redealt around it. Off: the world forgets it and asks it again, as every
    // rollout player measured before it did.
    bool box4 = false;
    // With keyed: the situation packed with its player fields narrowed to a byte (keyed_chance.hpp's
    // keyed2). Off: the original packing, which every keyed player measured before it played.
    bool keyed2 = false;
};

struct Row {
    GameState state;
    uint32_t position;          // which decision of the batch
    uint16_t world;             // index of the world within the position (dropped worlds excluded)
    uint16_t candidate;         // index into that decision's candidate list
    int32_t crossed;            // action-round boundaries crossed so far (aligned: -1 while still in the mover's round;
                                //   the whole headline one boundary)
    uint8_t turn, action_round; // the row's current key, with phasing and phase below
    uint8_t start_turn;         // the decision's turn (Stop::kTurnEnd / kLater / kEarlier read it)
    bool parked = false;        // stopped at the turn-end park (Stop::kPredeal / kEarlierPredeal)
    Player phasing;
    Phase phase;
    uint64_t world_key = 0;     // keyed chance: the world's key
    bool done, terminal, leaf_valued, refused;
    float value;                // from the mover's side, once terminal or valued
};

class RolloutBatch {
public:
    explicit RolloutBatch(const Config& cfg, uint64_t seed) noexcept;

    // Add one decision: `base` at a decision of `mover()`, `k` flat candidate actions, `worlds`
    // worlds. Returns the rows added (k times the worlds kept). Every candidate must be legal at
    // `base`; a caller passing one that is not gets 0 rows and `last_error()`.
    // `hands` (worlds x 110 bytes), when given, deals each world's opposing unknown hand from its row
    // (place_hand) instead of redealing it at random -- a belief's worlds; the deck is the rest of the
    // hidden pool either way.
    size_t add(const GameState& base, const uint16_t* candidates, size_t k, size_t worlds,
               const uint8_t* hands = nullptr);

    // Race one decision's second stage, once its rows are valued (after the first stage's playouts and
    // leaves): every challenger whose paired lead over candidate 0 -- over the worlds every candidate
    // was valued in, as `decide` reads it -- is below 0 leaves the race, and `worlds` fresh worlds
    // (drawn on from the batch's stream, `hands` as `add`'s) are added for candidate 0 and the
    // challengers left. Nothing is added when none is left. Returns the rows added; 0 with
    // `last_error()` for an unknown position, one already raced, or a row not yet valued.
    size_t race(size_t position, size_t worlds, const uint8_t* hands = nullptr);
    // Whether candidate `c` of `position` is still in the race (every candidate is, until `race`).
    bool in_race(size_t position, size_t c) const noexcept {
        return position < in_race_.size() && c < in_race_[position].size() && in_race_[position][c];
    }
    bool raced(size_t position) const noexcept { return position < raced_.size() && raced_[position]; }

    size_t positions() const noexcept { return movers_.size(); }
    Player mover(size_t position) const noexcept { return movers_[position]; }
    size_t worlds_used(size_t position) const noexcept { return worlds_used_[position]; }
    size_t rows() const noexcept { return rows_.size(); }
    const Row& row(size_t r) const noexcept { return rows_[r]; }
    size_t obs_width() const noexcept { return obs_width_; }
    size_t live() const noexcept;           // rows not yet done
    bool all_done() const noexcept { return live() == 0; }
    size_t listed() const noexcept { return live_order_.size(); }        // rows the last fill_live wrote
    size_t leaves_listed() const noexcept { return leaf_order_.size(); } // rows the last fill_leaves wrote
    // Rows at their horizon still waiting for a leaf value: what the next fill_leaves writes.
    size_t leaves_pending() const noexcept {
        size_t n = 0;
        for (const auto& r : rows_) n += (r.done && !r.terminal && !r.refused && !r.leaf_valued) ? 1 : 0;
        return n;
    }
    size_t refused() const noexcept;        // rows the engine refused a step on (an error)
    const char* last_error() const noexcept { return error_; }

    // Write the live rows' observations (from the row's acting side, in the batch's feature set)
    // and legal masks, row-major, into caller buffers of live() * obs_width() floats and
    // live() * FLAT_ACTION_SPACE_SIZE bytes; returns the count and remembers the order for step().
    size_t fill_live(float* obs, uint8_t* masks);
    // Advance every row fill_live listed by the masked argmax of its logits row
    // (count x FLAT_ACTION_SPACE_SIZE), the guard applied, then settle what the player does not
    // control; count boundaries; mark terminal rows and rows at their horizon done.
    void step(const float* logits);

    // Rows done, not terminal and not yet valued: their observations from the MOVER's side and
    // legal masks into caller buffers; returns the count and remembers the order.
    size_t fill_leaves(float* obs, uint8_t* masks);
    // The critic's values for those rows, in that order, from the mover's side.
    void set_leaf_values(const float* values);

    // Every row's value from the mover's side, NaN while a row is live or an unvalued leaf.
    void values(float* out) const noexcept;

private:
    Config cfg_;
    uint64_t rng_;
    size_t obs_width_;
    std::vector<Row> rows_;
    std::vector<Player> movers_;
    std::vector<size_t> worlds_used_;
    std::vector<GameState> bases_;                  // each decision's position, for race's worlds
    std::vector<std::vector<uint16_t>> cands_;      // and its candidates
    std::vector<std::vector<uint8_t>> in_race_;     // and which are still in the race
    std::vector<uint8_t> raced_;
    std::vector<uint32_t> live_order_;
    std::vector<uint32_t> leaf_order_;
    const char* error_ = "";

    // `worlds` worlds of decision `position` for the candidates in_race_ marks, rows carrying each
    // candidate's index in the decision's list; a world in which one of them is illegal is dropped.
    size_t add_worlds(uint32_t position, size_t worlds, const uint8_t* hands);
    void settle_and_mark(Row& r) noexcept;
    // Auto-advance; under a pre-deal stop it halts at the turn-end park and marks the row parked.
    void advance(Row& r) const noexcept;
    bool at_stop(const Row& r) const noexcept;   // a row at its stop under cfg_.horizon and cfg_.stop
    uint16_t guarded_choice(const GameState& s, const float* logits, const uint8_t* mask,
                            uint16_t argmax) const noexcept;
};

// ---- the root's choice, shared by ai/search/rollout_root.py and the page -----------------------

// `argmax`: the best mean value. `z<x>`: the best mean value only if its paired lead over the
// network's own move (candidate 0) is at least x standard errors, else the network's move.
// `kl<t>`: argmax of log pi(a) + Q(a) / t over the candidates, one mirror-descent step.
struct Rule {
    enum Kind : uint8_t { ARGMAX, Z, KL };
    Kind kind = Z;
    float param = 2.0f;
};

// "argmax", "z", "z2", "z1.5", "kl", "kl0.05" -> the rule; false for anything else.
bool parse_rule(const char* text, Rule& out) noexcept;

// The `k` most probable legal moves by `logits` (FLAT_ACTION_SPACE_SIZE floats) under `mask`,
// best first, into `out`; returns how many (fewer than k where fewer are legal).
size_t top_candidates(const float* logits, const uint8_t* mask, size_t k, uint16_t* out) noexcept;

struct Decision {
    uint16_t pick = 0;          // the flat action to play
    size_t worlds = 0;          // worlds every candidate still in the race was valued in
    std::vector<float> q;       // each candidate's mean value, from the mover's side
    std::vector<float> lead;    // each candidate's paired lead over candidate 0 (0 for itself)
    std::vector<float> lead_se; // its standard error (0 with fewer than two worlds)
};

// One position's decision from a batch whose rows are all valued: the candidates as added (the
// network's own move first), the root's `logits` under `mask` (for the KL rule), the rule. A
// position with no valued world picks candidate 0. The rule chooses among the candidates still in
// the race (RolloutBatch::race), their statistics over the worlds they were all valued in; a
// candidate out of the race keeps its statistics over those of the worlds it was valued in too
// (the first stage's), and is never picked. A batch that was not raced reads every candidate.
Decision decide(const RolloutBatch& batch, size_t position, const uint16_t* candidates, size_t k,
                const float* logits, const uint8_t* mask, const Rule& rule);

}  // namespace ts::rollout
