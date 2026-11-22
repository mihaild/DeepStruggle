// P34's playout labeller: positions labelled by greedy playouts to the game's end.
//
// Each queued position is dealt into `worlds` worlds from its mover's view (rollout::determinize),
// each with fresh dice, and every world is played to the end by the caller's network on both sides
// -- fill_live() -> logits -> step() -- through the safe-playout rule of
// ai/eval/paired_playouts.play_safe: while the position's own action round lasts and its mover
// decides, at DEFCON 2 or below, no option that loses on the spot is taken while another exists.
// What each world records, from the mover's side: the score (1 / 0.5 / 0), the VP track at the end,
// the VP track and DEFCON as the position's turn ends (at the end, if the game ends first), and
// each region's swing at its next scoring (ts::ScoringTap; masked where it is never scored).
//
// A position queued with candidates (the sibling slice) plays each candidate in the same worlds,
// on the same dice, and also keeps each row's leaf: the first position after the candidate at
// which the mover decides again (or, with leaf_horizon > 0, the first after that many action-round
// boundaries or the turn's end, whichever is sooner -- the rollout root's stop=earlier, leaf=own).
//
// Batching is continuous: a finished world frees its row at once, and the next queued position
// fills the free rows, so the network always sees a full batch. A position's worlds are drawn from
// a stream seeded by (seed, id), so its labels do not depend on what else shared the batch.
// Native only: the browser has no use for it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "ts/action_mask.hpp"
#include "ts/game_state.hpp"
#include "ts/scoring.hpp"

namespace ts::labels {

constexpr size_t kRegions = ScoringTap::kSlots;   // Europe .. South America, then Southeast Asia
constexpr size_t kMaxCandidates = 8;

// Observation floats as one-byte codes: the observation takes few distinct values (about a hundred),
// so a code table of 256 holds them all and a code decodes exactly. Thread-safe: a value's code is
// looked up without a lock (stored before its key is published) and added under one. Codes are
// assigned in the order values first appear, which depends on thread timing -- the decoding never does.
class ObsCodebook {
public:
    ObsCodebook();
    // The code of a float's bits; sets full() and returns 0 when a 257th value appears.
    uint8_t code_of(uint32_t bits) noexcept;
    // Encode one observation row.
    void encode(const float* row, size_t width, uint8_t* out) noexcept;
    size_t values(float* out) const noexcept;   // code -> value, code 0 = 0.0f
    size_t size() const noexcept { return size_.load(std::memory_order_acquire); }
    bool full() const noexcept { return full_.load(std::memory_order_relaxed); }

private:
    static constexpr size_t kSlots = 2048;
    static constexpr uint32_t kEmpty = 0xFFFFFFFFu;   // a NaN pattern no observation holds
    std::unique_ptr<std::atomic<uint32_t>[]> keys_;
    std::unique_ptr<uint8_t[]> codes_;
    float values_[256] = {};
    std::atomic<uint32_t> size_{0};
    std::atomic<bool> full_{false};
    std::mutex lock_;
};

// Encode `n` raw states' observations, each from its acting side, into codes (n x width), across the
// OpenMP team. Returns false if the codebook overflowed.
bool encode_observations(const GameState* states, size_t n, uint32_t features, ObsCodebook& book,
                         uint8_t* codes);

struct Config {
    uint32_t features = 0;      // obs_features of the network that plays
    uint16_t worlds = 32;
    bool safe = true;           // the safe-playout rule in the position's own action round
    int leaf_horizon = 0;       // sibling leaves: boundaries before the mover's own decision is the leaf
    uint32_t max_plies = 8000;  // a world still running after this many decisions is a failure
    bool keyed = false;         // keyed chance (keyed_chance.hpp): a world's candidates draw alike
    bool keyed2 = false;        // with keyed: the situation's player fields narrowed (keyed_chance.hpp)
    bool keyedsym = false;      // with keyed: both sides' player fields as all ones (keyed_chance.hpp); not with keyed2
};

// One world's labels, from the mover's side.
struct Outcome {
    float score = 0.0f;         // 1 win, 0.5 draw, 0 loss
    int8_t final_vp = 0;        // the VP track at the end
    int8_t turn_vp = 0;         // the VP track as the position's turn ended
    uint8_t turn_defcon = 0;    // DEFCON then
    uint8_t scored = 0;         // bit per region: scored before the end
    int16_t swing[kRegions] = {};
};

struct Row {
    GameState state;
    uint32_t entry = 0;
    uint16_t world = 0;
    uint16_t candidate = 0;
    uint32_t plies = 0;
    uint64_t world_key = 0;     // keyed chance: the world's key
    uint8_t crossed = 0;        // action-round boundaries since the candidate (leaf horizon)
    uint8_t key_turn = 0, key_round = 0;
    Player key_phasing = Player::NONE;
    Phase key_phase = Phase::SETUP;
    bool occupied = false;
    bool guarding = false;      // the safe rule still applies
    bool turn_seen = false;
    bool leaf_taken = false;
    bool done = false;
    bool failed = false;        // refused by the engine, or past max_plies
    uint32_t forced = 0;        // decisions with one legal option, played without the network
    uint32_t chosen = 0;        // decisions the network chose
    ScoringTap tap;
    Outcome out;
    uint8_t mask[FLAT_ACTION_SPACE_SIZE];   // the legal mask at the row's current decision
};

struct Entry {
    int64_t id = 0;
    Player mover = Player::NONE;
    uint8_t start_turn = 0;
    uint8_t g_turn = 0, g_round = 0;   // the safe rule's key: the position's own action round
    Player g_phasing = Player::NONE;
    Phase g_phase = Phase::SETUP;
    uint8_t k = 0;                     // candidates; 0 = a main-label position
    uint32_t rows_left = 0;
    // Main labels, summed over the worlds that finished.
    uint32_t n = 0, failed = 0;
    double score = 0, score_sq = 0, final_vp = 0, turn_vp = 0, turn_defcon = 0;
    double swing[kRegions] = {};
    uint32_t swing_n[kRegions] = {};
};

struct Queued {
    GameState state;
    int64_t id;
    uint8_t k;
    uint16_t candidates[kMaxCandidates];
};

// A finished main-label position.
struct PositionLabel {
    int64_t id;
    uint32_t n, failed;
    float score, score_sq, final_vp, turn_vp, turn_defcon;   // sums over the n worlds
    float swing[kRegions];
    uint32_t swing_n[kRegions];
};

// A finished sibling row.
struct SiblingLabel {
    int64_t id;
    uint16_t candidate, world;
    bool failed, has_leaf;
    Outcome out;
};

class PlayoutLabels {
public:
    PlayoutLabels(const Config& cfg, size_t capacity, uint64_t seed);

    // Queue positions; `candidates` is n x k (k = 0 for main labels). Returns "" or what was wrong
    // (a terminal position, a candidate that is not legal), having queued nothing then.
    std::string enqueue(const GameState* states, const int64_t* ids, size_t n,
                        const uint16_t* candidates, size_t k);

    size_t queued() const noexcept { return queue_.size(); }
    size_t occupied() const noexcept { return capacity_ - free_.size(); }
    size_t capacity() const noexcept { return capacity_; }
    size_t obs_width() const noexcept { return obs_width_; }
    size_t listed() const noexcept { return live_.size(); }
    bool idle() const noexcept { return queue_.empty() && occupied() == 0; }
    // Decisions of finished rows: played without the network (one legal option), and chosen by it.
    uint64_t forced() const noexcept { return forced_; }
    uint64_t chosen() const noexcept { return chosen_; }

    // Refill free rows from the queue, then write each live row's observation (from its acting
    // side) to obs (capacity x obs_width). Returns the rows written, in step()'s order.
    size_t fill_live(float* obs);
    // The same, each float written as its one-byte code in codebook() -- the observation's values
    // are few (under a hundred seen), so the upload is a quarter of the bytes and decodes exactly.
    // Returns SIZE_MAX if a 257th distinct value appeared (nothing usable was written then).
    size_t fill_live_codes(uint8_t* codes);
    // The values the codes stand for, code 0 = 0.0f; grows as new values appear, never reorders.
    size_t codebook(float* out) const noexcept;
    // Each listed row takes the masked argmax of its logits row (under the safe rule), and settles.
    // `forced`, if given, is one action per listed row (-1: the argmax): a stronger player's choice
    // for that row (P34's D3 follow-up), taken unless the safe rule refuses it.
    void step(const float* logits, const int32_t* forced = nullptr);
    // What each listed row is deciding, for a caller choosing where to play something else:
    // out is listed() x 4 = decision type, 1 if a headline card choice, the row's turn, its position's turn.
    void listed_info(int32_t* out) const noexcept;
    const GameState& listed_state(size_t j) const noexcept { return rows_[live_[j]].state; }

    // World w of position `id` as refill deals it (tests and diagnostics).
    GameState world(const GameState& base, int64_t id, uint16_t w) const noexcept;

    std::vector<PositionLabel> take_positions();
    std::vector<SiblingLabel> take_siblings(std::vector<GameState>& leaves);

private:
    void refill();
    void start_row(uint32_t slot, const Entry& e, const GameState& world, uint32_t entry, uint16_t w, uint16_t c,
                   const uint16_t* candidate);
    bool play(uint32_t slot, const Entry& e, uint16_t action);
    void advance(Row& r) const noexcept;
    void settle(uint32_t slot, const Entry& e);
    void after_step(uint32_t slot, const Entry& e);
    void finish(Row& r, const Entry& e);
    void harvest();
    void update_guard(Row& r, const Entry& e) const noexcept;
    uint16_t choose(Row& r, const float* logits, int32_t forced = -1) const;

    Config cfg_;
    size_t capacity_;
    uint64_t seed_;
    size_t obs_width_;
    std::vector<Row> rows_;
    std::vector<GameState> leaves_;     // per row slot
    std::vector<uint32_t> free_;
    std::vector<uint32_t> live_;
    std::vector<Entry> entries_;
    std::vector<uint32_t> free_entries_;
    std::deque<Queued> queue_;
    std::vector<PositionLabel> positions_out_;
    std::vector<SiblingLabel> siblings_out_;
    std::vector<GameState> leaves_out_;
    uint64_t forced_ = 0, chosen_ = 0;
    ObsCodebook book_;
};

}  // namespace ts::labels
