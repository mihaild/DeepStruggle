// The policy-likelihood belief over the opponent's unseen hand (P35 3b), written once for both builds:
// the Python rollout root (ai/search/belief.py) and the browser. The network stays outside, as with
// RolloutBatch: the caller evaluates the rows a request lists and hands back the log-likelihoods.
//
// A tracker sees every decision of every game it is given (`observe`). At a root decision it lists
// the opponent's decisions this turn that fit the hand (`record`: a hypothesis for the hand now fixes the
// hand at each earlier decision of the turn -- the hypothesis plus the cards that have left the
// decider's hidden pool since, for anywhere but the decider's own hand), leaving out an opposing headline
// the decider cannot have seen yet, draws `particles` hands of the
// opponent's unseen size uniformly from the hidden pool, and asks for the network's log-probability of
// each recorded decision under each hand at each temperature of kTemperatures. From those it weighs the
// hands, and the temperatures by how well each explains this opponent's choices so far in the game
// (Bayesian model averaging, uniform prior: each finished turn's evidence log mean_k prod pi_T(a | hand
// k), summed, plus the current turn's), and deals the root's worlds by resampling the mixture.
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

// One position's belief, between `prepare` and `worlds`.
struct Job {
    size_t position;            // index into the prepare() call's positions
    size_t game;
    Player me;
    uint64_t key;               // the position's draw key
    std::vector<uint8_t> hands; // particles x 110 (card id c at c - 1)
    std::vector<uint32_t> decisions;              // indices into the game's log
    std::vector<std::array<uint8_t, 110>> lefts;  // per recorded decision: cards departed since
    std::array<float, kTemps> evidence{};         // log mean_k prod pi_T, this turn so far
    std::vector<float> mixture;                   // particles, the averaged weights (sum 1)
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

    // The decision about to be applied in `game`: the position before it, who decides, the move.
    void observe(size_t game, const GameState& s, Player decider, uint16_t action);

    // Positions the caller is about to search (each its game's current state, `me` to move). Clears
    // the previous call's jobs. Returns, per position, whether it has a belief: false where there is
    // nothing to believe (no hidden opposing hand, a transient location) -- the root redeals at random.
    // Positions whose game has not seen the opponent move since the last call reuse its weights and
    // make no job; the rest make one, listed by jobs().
    std::vector<bool> prepare(const std::vector<size_t>& games, const std::vector<const GameState*>& states);
    const std::vector<Job>& jobs() const noexcept { return jobs_; }
    // The recorded decision `d` of job `j`: its position and the opponent who decided.
    const Decision& decision(size_t j, size_t d) const noexcept { return games_.at(jobs_[j].game).log[jobs_[j].decisions[d]]; }

    // The caller's log-probabilities of every job's recorded decisions under every hand at every
    // temperature: lp[((t * R) + r) * particles + k] for row block r (jobs in order, their decisions in
    // order; R blocks in all). Weighs the hands and the temperatures and updates each game's evidence.
    void set_log_likelihoods(const float* lp, size_t blocks);

    // `n` hands (n x 110) resampled from position `p`'s belief into `out`; false where it has none.
    bool worlds(size_t p, size_t n, uint8_t* out) const;

    // The temperature weights of position p's game after the last set_log_likelihoods (for reports).
    std::array<float, kTemps> temperature_weights(size_t p) const;

private:
    // One side's belief about its opponent in one game.
    struct Side {
        std::array<float, kTemps> done{};   // finished turns' evidence, summed
        std::array<float, kTemps> current{};// this turn's latest evidence
        bool has_current = false;
        // the last belief: valid while the record and the board are unchanged
        size_t cached_record = SIZE_MAX;
        std::array<uint8_t, 111> cached_locs{};
        std::vector<uint8_t> hands;
        std::vector<float> mixture;
    };
    struct Game {
        std::vector<Decision> log;          // this turn's decisions
        int turn = -1;
        uint32_t seen = 0;                  // decisions observed in the whole game (draw keys)
        std::array<Side, 2> side;           // [0] the US's belief, [1] the USSR's
    };
    static size_t side_index(Player me) noexcept { return me == Player::US ? 0 : 1; }
    uint32_t particles_;
    bool all_decisions_;
    bool reveal_;
    uint64_t seed_;
    std::unordered_map<size_t, Game> games_;
    std::vector<Job> jobs_;
    std::vector<size_t> position_game_;     // per prepare() position: its game, or SIZE_MAX (no belief)
    std::vector<uint64_t> position_key_;
    std::vector<Player> position_me_;
};

// The cards that were in `me`'s hidden pool at `d` and have left it since, for anywhere but `me`'s own
// hand, given every card's location now; false when `d` is not on this hand's record.
bool left_since(const Decision& d, Player me, const std::array<uint8_t, 111>& now, std::array<uint8_t, 110>& left);

}  // namespace ts::belief
