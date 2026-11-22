// The policy-likelihood belief over the opponent's unseen hand (P35 3b), written once for both builds:
// the Python rollout root (ai/search/belief.py) and the browser. The network stays outside, as with
// RolloutBatch: the caller evaluates the rows a request lists and hands back the log-likelihoods.
//
// A tracker sees every decision of every game it is given (`observe`). At a root decision it lists
// the opponent's decisions this turn that fit the hand (the record: a hypothesis for the hand now fixes
// the hand at each earlier decision of the turn -- the hypothesis plus the cards that have left the
// decider's hidden pool since, for anywhere but the decider's own hand), leaving out an opposing headline
// the decider cannot have seen yet, draws `particles` hands of the opponent's unseen size from the
// hidden pool, CONSISTENT WITH EVERY DECISION ON THE RECORD (below), and asks for the network's
// log-probability of each recorded decision under each hand at each temperature of kTemperatures. From
// those it weighs the hands, and the temperatures by how well each explains this opponent's choices so
// far in the game (Bayesian model averaging, uniform prior: each finished turn's evidence, summed, plus
// the current turn's), and deals the root's worlds by resampling the mixture.
//
// Consistency (audit S-1, 2026-11-22). A hand is consistent with a recorded decision when, with it placed
// into the decision's position (place_hand), the engine's own legal mask (generate_flat_mask_212) allows
// the move made, and -- where the decision would not have been asked otherwise -- offers at least two
// moves: always when the stream is auto-advanced (`auto_advanced`: the engine plays every single-choice
// decision itself, so every recorded one had a choice), and in any stream at the three questions the
// engine asks only when the hand allows a choice (Blockade's and Latin American Debt Crisis's discard,
// Missile Envy's tie) unless a seat plays ALWAYS_ASK_DISCARD. The opponent's real hand is always
// consistent: its moves were legal and its questions were asked. A hand inconsistent with any decision has
// probability zero and is never a particle. The particles are drawn from the uniform prior restricted to
// consistent hands: by rejection (exact) while that finds them, and otherwise by a stratified proposal
// corrected by importance weights (`Job::log_weight`; belief.cpp `draw`). Only if no consistent hand is
// found at all -- which a correct record never produces -- does the position fall back to the uniform
// prior, with a diagnostic (`exhausted()`) naming the position and the decisions, so the root still plays;
// the bindings' `strict` makes it an error instead (tests, offline measurement).
//
// P35's measurements: on the network's own games the averaged belief scores the opponent's hand 8.1%
// better than the uniform prior (all its weight near T = 1), on the human corpus 0.8% better (T 2-3)
// where T = 1 alone is 11.7% worse -- it finds the opponent unasked (research/plans/P35_*.md).
//
// Every random draw is keyed by the tracker's seed and the position (its dice-stream state and the
// game's decision count), so a game's worlds do not depend on which other games share the batch: the
// browser playing one game deals what a tournament would.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "ts/game_state.hpp"

namespace ts::belief {

constexpr size_t kTemps = 6;
constexpr float kTemperatures[kTemps] = {1.0f, 1.5f, 2.0f, 3.0f, 5.0f, 8.0f};

struct Decision {
    GameState state;            // the position before the decision
    Player decider;
    uint16_t action;
    std::array<uint8_t, 111> locs;   // every card's location then (index = card id)
};

// One opposing decision a hand must be consistent with: the decision, the cards departed from the hidden
// pool since (a hypothesis for the hand now plus these is the hand then), and whether the move must also
// have been a choice among at least two (see the header comment).
struct Constraint {
    uint32_t index;                    // into the game's log
    std::array<uint8_t, 110> left;
    bool choice;
};

// One position's belief, between `prepare` and `worlds`.
struct Job {
    size_t position;            // index into the prepare() call's positions
    size_t game;
    Player me;
    uint64_t key;               // the position's draw key
    std::vector<uint8_t> hands; // particles x 110 (card id c at c - 1), every one consistent
    std::vector<uint32_t> decisions;              // the likelihood record: indices into the game's log
    std::vector<std::array<uint8_t, 110>> lefts;  // per recorded decision: cards departed since
    // Per recorded decision: 1 when its evidence is not yet in the finished turns' sum (`Side::done`). The
    // hands are weighed on every recorded decision; the temperatures' evidence sums only these.
    std::vector<uint8_t> unspent;
    // Per particle: the log importance weight, normalized (logsumexp = 0). All equal when every particle
    // came from rejection sampling the prior; otherwise prior / proposal (balance heuristic), divided by a
    // duplicate's multiplicity.
    std::vector<double> log_weight;
    double log_mass = 0.0;      // log of the estimated prior probability that a hand is consistent
    std::array<double, kTemps> evidence{};        // log E_prior prod pi_T over the unspent decisions
    std::vector<float> mixture;                   // particles, the averaged weights (sum 1)
};

// Counters across the tracker's life (reports, measurement).
struct Stats {
    uint64_t positions = 0;       // positions given a belief
    uint64_t sampled = 0;         // of them, hands drawn against at least one recorded decision
    uint64_t draws = 0;           // candidate hands drawn
    uint64_t prior_short = 0;     // sampled positions where rejection from the prior did not fill the particles
    uint64_t first_round_empty = 0;  // sampled positions where the first `particles` prior draws held no consistent
                                     // hand: what the uniform sampler before 2026-11-22 weighed at -1e9 each
    uint64_t guided = 0;          // positions that used the stratified proposal
    uint64_t exhausted = 0;       // positions with no consistent hand found: the uniform fallback
};

class Tracker {
public:
    // `reveal` (the rollout root's box4): at the headline choice of the side holding Man in Earth Orbit
    // alone, every hand holds the opponent's revealed headline (rollout::revealed_headline) and the rest
    // of it is drawn around that card.
    Tracker(uint32_t particles, bool all_decisions, uint64_t seed, bool reveal = false) noexcept
        : particles_(particles), all_decisions_(all_decisions), reveal_(reveal), seed_(seed) {}

    uint32_t particles() const noexcept { return particles_; }
    void reseed(uint64_t seed) noexcept { seed_ = seed; }
    void new_games() noexcept { games_.clear(); }
    // Whether the decisions observed come from an auto-advanced stream (Engine::auto_advance_step after
    // every move: no single-choice decision is ever asked). Off (the default) assumes nothing about it,
    // which is sound for any stream; on is sound only for such a stream, and is stronger.
    void set_auto_advanced(bool on) noexcept { auto_advanced_ = on; }
    bool auto_advanced() const noexcept { return auto_advanced_; }
    // Whether the bindings make a fallback (`exhausted()`) an error. Read by the bindings only.
    void set_strict(bool on) noexcept { strict_ = on; }
    bool strict() const noexcept { return strict_; }

    // The decision about to be applied in `game`: the position before it, who decides, the move.
    void observe(size_t game, const GameState& s, Player decider, uint16_t action);

    // Positions the caller is about to search (each its game's current state, `me` to move; no null
    // states). Clears the previous call's jobs. A tracker of 0 particles believes nothing. Returns, per position, whether it has a belief: false where there is
    // nothing to believe (no hidden opposing hand, a transient location) -- the root redeals at random.
    // Positions whose game has not seen the opponent move since the last call reuse its weights and
    // make no job; the rest make one, listed by jobs() -- except a fallback (`exhausted()`), which is
    // weighed uniformly at once and makes none.
    std::vector<bool> prepare(const std::vector<size_t>& games, const std::vector<const GameState*>& states);
    const std::vector<Job>& jobs() const noexcept { return jobs_; }
    // The recorded decision `d` of job `j`: its position and the opponent who decided.
    const Decision& decision(size_t j, size_t d) const noexcept { return games_.at(jobs_[j].game).log[jobs_[j].decisions[d]]; }
    // The last prepare() call's fallbacks: (position, what could not be satisfied).
    const std::vector<std::pair<size_t, std::string>>& exhausted() const noexcept { return exhausted_; }
    const Stats& stats() const noexcept { return stats_; }

    // The caller's log-probabilities of every job's recorded decisions under every hand at every
    // temperature: lp[((t * R) + r) * particles + k] for row block r (jobs in order, their decisions in
    // order; R blocks in all). Weighs the hands and the temperatures and updates each game's evidence.
    // Every particle is consistent, so every value must be finite: returns an error message (nothing
    // changed) on a wrong block count or a non-finite value, else nullptr.
    const char* set_log_likelihoods(const float* lp, size_t blocks);

    // `n` hands (n x 110) resampled from position `p`'s belief into `out`; false where it has none --
    // including a job prepare() listed that set_log_likelihoods has not weighed (no stale hands).
    bool worlds(size_t p, size_t n, uint8_t* out) const;

    // The temperature weights of position p's game after the last set_log_likelihoods (for reports).
    std::array<float, kTemps> temperature_weights(size_t p) const;

    // Position p's current belief: its hands (particles x 110) and their weights; empty where it has none.
    const std::vector<uint8_t>& hands(size_t p) const;
    const std::vector<float>& mixture(size_t p) const;
    // Whether `hand` (110 bytes: the opponent's unseen hand now) is consistent with every opposing
    // decision position p's belief conditions on (the header's rule); false where p has no belief.
    bool consistent(size_t p, const uint8_t* hand) const;

private:
    // One side's belief about its opponent in one game.
    struct Side {
        std::array<double, kTemps> done{};   // finished turns' evidence, summed
        std::array<double, kTemps> current{};// this turn's latest evidence
        bool has_current = false;
        std::vector<uint32_t> current_decisions;  // the log indices `current` weighs (the unspent ones)
        // Decisions of this turn whose evidence is already in `done`: a record that lost a decision `current`
        // weighs (a card entered the pool, an event's draw) folds `current` into `done` and spends its
        // decisions. A spent decision still on the record still weighs the hands; it only leaves the
        // temperature evidence, so nothing is lost from either and nothing is counted twice.
        std::vector<bool> spent;
        // the last belief: valid while the record and the board are unchanged
        size_t cached_record = SIZE_MAX;
        std::array<uint8_t, 111> cached_locs{};
        std::vector<uint8_t> hands;
        std::vector<float> mixture;
        std::vector<Constraint> constraints;   // what `hands` is consistent with
    };
    struct Game {
        std::vector<Decision> log;          // this turn's decisions
        int turn = -1;
        uint32_t seen = 0;                  // decisions observed in the whole game (draw keys)
        std::array<Side, 2> side;           // [0] the US's belief, [1] the USSR's
    };
    static size_t side_index(Player me) noexcept { return me == Player::US ? 0 : 1; }
    const Side* side_of(size_t p) const;
    uint32_t particles_;
    bool all_decisions_;
    bool reveal_;
    bool auto_advanced_ = false;
    bool strict_ = false;
    uint64_t seed_;
    std::unordered_map<size_t, Game> games_;
    std::vector<Job> jobs_;
    std::vector<size_t> position_game_;     // per prepare() position: its game, or SIZE_MAX (no belief)
    std::vector<uint64_t> position_key_;
    std::vector<Player> position_me_;
    std::vector<std::pair<size_t, std::string>> exhausted_;
    Stats stats_;
};

// The cards that were in `me`'s hidden pool at `d` and have left it since, for anywhere but `me`'s own
// hand, given every card's location now; false when `d` is not on this hand's record.
bool left_since(const Decision& d, Player me, const std::array<uint8_t, 111>& now, std::array<uint8_t, 110>& left);

// Whether `hand` (the opponent's unseen hand now, 110 bytes) is consistent with constraint `c` on `log`
// (the header's rule), `me` the believer.
bool consistent_with(const std::vector<Decision>& log, const Constraint& c, Player me, const uint8_t* hand);

}  // namespace ts::belief
