#include "playout_labels.hpp"

#include "gomp_parallel.hpp"   // the loops share torch's OpenMP pool, as the batch runner's do

#include <algorithm>
#include <atomic>
#include <cstring>

#include "keyed_chance.hpp"
#include "rollout_batch.hpp"   // determinize, acting_player
#include "ts/action_mask.hpp"
#include "ts/engine.hpp"
#include "ts/observation.hpp"
#include "ts/prng.hpp"

namespace ts::labels {

namespace {

constexpr size_t A = FLAT_ACTION_SPACE_SIZE;
static_assert(obs_features::extra_width(obs_features::ALL) <= 64, "fill_live_codes' row buffer");

void drain_chance(GameState& s) noexcept {
    while (!Engine::is_terminal(s) && s.ctx().decision_player == Player::NONE &&
           s.ctx().decision_type == DecisionType::ROLL_DIE) {
        if (!Engine::step(s, MicroAction{DecisionType::ROLL_DIE, 0, 0, 0})) return;
    }
}

// Does `a` end the game against `who` on the spot? (ai/eval/paired_playouts.loses_now)
bool loses_now(const GameState& s, uint16_t a, Player who) noexcept {
    GameState probe = s;
    if (!Engine::step_flat(probe, a)) return false;
    drain_chance(probe);
    if (!Engine::is_terminal(probe)) return false;
    const float u = Engine::get_terminal_utility(probe);
    return who == Player::US ? (u < 0.0f) : (u > 0.0f);
}

// The row's tap is live only around the row's real steps -- never around a safety probe.
struct Tapped {
    explicit Tapped(ScoringTap& t) noexcept { scoring_tap = &t; }
    ~Tapped() { scoring_tap = nullptr; }
};

int8_t mover_vp(int v, Player mover) noexcept {
    v = std::clamp(mover == Player::US ? v : -v, -127, 127);
    return static_cast<int8_t>(v);
}

// States are copied byte for byte, padding included: a leaf's raw bytes go into the dataset, and a
// member-wise copy leaves padding as it found it, so the same leaf would hash differently.
void copy_state(GameState& to, const GameState& from) noexcept {
    std::memcpy(static_cast<void*>(&to), static_cast<const void*>(&from), sizeof(GameState));
}

uint64_t entry_seed(uint64_t seed, int64_t id) noexcept {
    uint64_t z = seed ^ (static_cast<uint64_t>(id) * 0x9E3779B97F4A7C15ULL);
    return Prng::next_u64(z);
}

constexpr int64_t kParallelRows = 32;
constexpr int64_t kBlock = 32;

// The rows across the OpenMP team (torch's pool, gomp_parallel.hpp), handed out in blocks from a
// shared counter rather than in equal static chunks: this machine's cores are not equal (a 12900KF's
// efficiency cores run a row at about half a performance core's speed), and with static chunks the
// fast cores wait for the slow ones. Each row is independent and writes only its own slots, so the
// results do not depend on which thread ran it.
template <class F>
void for_rows(int64_t n, F body) {
    if (n < kParallelRows) {
        for (int64_t j = 0; j < n; ++j) body(j);
        return;
    }
    struct Job {
        int64_t n;
        std::atomic<int64_t> next{0};
        F* body;
    };
    Job job;
    job.n = n;
    job.body = &body;
    GOMP_parallel(
        [](void* p) {
            Job* j = static_cast<Job*>(p);
            for (;;) {
                const int64_t lo = j->next.fetch_add(kBlock, std::memory_order_relaxed);
                if (lo >= j->n) return;
                const int64_t hi = std::min(j->n, lo + kBlock);
                for (int64_t i = lo; i < hi; ++i) (*j->body)(i);
            }
        },
        &job, /*num_threads=*/0, /*flags=*/0);
}

}  // namespace

PlayoutLabels::PlayoutLabels(const Config& cfg, size_t capacity, uint64_t seed)
    : cfg_(cfg), capacity_(capacity), seed_(seed),
      obs_width_(OBS_SIZE_V23 + obs_features::extra_width(cfg.features)),
      rows_(capacity), leaves_(capacity) {
    free_.reserve(capacity);
    for (size_t i = capacity; i-- > 0;) free_.push_back(static_cast<uint32_t>(i));
    live_.reserve(capacity);
}

std::string PlayoutLabels::enqueue(const GameState* states, const int64_t* ids, size_t n,
                                   const uint16_t* candidates, size_t k) {
    if (k > kMaxCandidates) return "at most 8 candidates";
    if (cfg_.worlds == 0) return "worlds must be positive";
    const size_t need = (k ? k : 1) * cfg_.worlds;
    if (need > capacity_) return "a position's rows (worlds x candidates) exceed the capacity";
    for (size_t i = 0; i < n; ++i) {
        if (Engine::is_terminal(states[i])) return "position " + std::to_string(ids[i]) + " is terminal";
        if (k) {
            uint8_t mask[A];
            ActionMask::generate_flat_mask_212(states[i], mask);
            for (size_t c = 0; c < k; ++c) {
                const uint16_t a = candidates[i * k + c];
                if (a >= A || !mask[a])
                    return "a candidate of position " + std::to_string(ids[i]) + " is not legal there";
            }
        }
    }
    for (size_t i = 0; i < n; ++i) {
        Queued q{};
        copy_state(q.state, states[i]);
        q.id = ids[i];
        q.k = static_cast<uint8_t>(k);
        for (size_t c = 0; c < k; ++c) q.candidates[c] = candidates[i * k + c];
        queue_.push_back(q);
    }
    return "";
}

void PlayoutLabels::start_row(uint32_t slot, const Entry& e, const GameState& world, uint32_t entry, uint16_t w,
                              uint16_t c, const uint16_t* candidate) {
    Row& r = rows_[slot];
    r = Row{};
    copy_state(r.state, world);
    r.entry = entry;
    r.world = w;
    r.candidate = c;
    r.occupied = true;
    r.guarding = cfg_.safe;
    r.world_key = world.rng_state;
    if (candidate != nullptr) {
        // The candidate is forced, not chosen: the safe rule applies from the next decision on, as
        // play_safe does after paired_playouts.apply_move.
        Tapped tap(r.tap);
        if (cfg_.keyed) keyed::key(r.state, r.world_key, keyed::mode_of(cfg_.keyed2, cfg_.keyedsym));
        if (!Engine::step_flat(r.state, *candidate)) {
            r.failed = r.done = true;
            return;
        }
        advance(r);
    }
    r.key_turn = r.state.turn;
    r.key_round = r.state.action_round;
    r.key_phasing = r.state.phasing_player;
    r.key_phase = r.state.current_phase;
    after_step(slot, e);
    settle(slot, e);
}

void PlayoutLabels::refill() {
    while (!queue_.empty()) {
        const Queued& q = queue_.front();
        const size_t need = (q.k ? q.k : 1) * cfg_.worlds;
        if (free_.size() < need) break;
        uint32_t ei;
        if (free_entries_.empty()) {
            ei = static_cast<uint32_t>(entries_.size());
            entries_.emplace_back();
        } else {
            ei = free_entries_.back();
            free_entries_.pop_back();
        }
        Entry& e = entries_[ei];
        e = Entry{};
        e.id = q.id;
        e.mover = rollout::acting_player(q.state);
        e.start_turn = q.state.turn;
        e.g_turn = q.state.turn;
        e.g_round = q.state.action_round;
        e.g_phasing = q.state.phasing_player;
        e.g_phase = q.state.current_phase;
        e.k = q.k;
        e.rows_left = static_cast<uint32_t>(need);
        uint64_t rng = entry_seed(seed_, q.id);
        for (uint16_t w = 0; w < cfg_.worlds; ++w) {
            GameState world{};
            copy_state(world, q.state);
            rollout::determinize(world, e.mover, rng);
            world.rng_state = Prng::next_u64(rng);   // fresh dice per world, shared by its candidates
            const size_t rows_here = q.k ? q.k : 1;
            for (size_t c = 0; c < rows_here; ++c) {
                const uint32_t slot = free_.back();
                free_.pop_back();
                start_row(slot, e, world, ei, w, static_cast<uint16_t>(c), q.k ? &q.candidates[c] : nullptr);
            }
        }
        queue_.pop_front();
    }
    harvest();   // rows a candidate ended at once
}

size_t PlayoutLabels::fill_live(float* obs) {
    refill();
    live_.clear();
    for (uint32_t i = 0; i < rows_.size(); ++i) {
        if (rows_[i].occupied && !rows_[i].done) live_.push_back(i);
    }
    const int64_t n = static_cast<int64_t>(live_.size());
    const size_t width = obs_width_;
    for_rows(n, [&](int64_t j) {
        const Row& r = rows_[live_[j]];
        extract_observation_features(r.state, rollout::acting_player(r.state), cfg_.features, obs + j * width);
    });
    return static_cast<size_t>(n);
}

// play_safe's bookkeeping, at every decision the row reaches (forced ones too): the rule lapses for
// good once the position's own action round is over or someone else decides.
void PlayoutLabels::update_guard(Row& r, const Entry& e) const noexcept {
    if (!r.guarding) return;
    if (r.state.turn != e.g_turn || r.state.action_round != e.g_round || r.state.phasing_player != e.g_phasing ||
        r.state.current_phase != e.g_phase || rollout::acting_player(r.state) != e.mover) {
        r.guarding = false;
    }
}

size_t PlayoutLabels::fill_live_codes(uint8_t* codes) {
    refill();
    live_.clear();
    for (uint32_t i = 0; i < rows_.size(); ++i) {
        if (rows_[i].occupied && !rows_[i].done) live_.push_back(i);
    }
    const int64_t n = static_cast<int64_t>(live_.size());
    const size_t width = obs_width_;
    for_rows(n, [&](int64_t j) {
        float row[OBS_SIZE_V23 + 64];
        const Row& r = rows_[live_[j]];
        extract_observation_features(r.state, rollout::acting_player(r.state), cfg_.features, row);
        book_.encode(row, width, codes + j * width);
    });
    return book_.full() ? SIZE_MAX : static_cast<size_t>(n);
}

size_t PlayoutLabels::codebook(float* out) const noexcept { return book_.values(out); }

ObsCodebook::ObsCodebook() : keys_(new std::atomic<uint32_t>[kSlots]), codes_(new uint8_t[kSlots]()) {
    for (size_t i = 0; i < kSlots; ++i) keys_[i].store(kEmpty, std::memory_order_relaxed);
    code_of(0u);   // code 0 is 0.0f
}

uint8_t ObsCodebook::code_of(uint32_t bits) noexcept {
    size_t h = (bits * 0x9E3779B1u) >> 21;   // 11 bits: kSlots = 2048
    for (;;) {
        const uint32_t k = keys_[h].load(std::memory_order_acquire);
        if (k == bits) return codes_[h];
        if (k == kEmpty) break;
        h = (h + 1) & (kSlots - 1);
    }
    std::lock_guard<std::mutex> g(lock_);
    h = (bits * 0x9E3779B1u) >> 21;
    for (;;) {
        const uint32_t k = keys_[h].load(std::memory_order_relaxed);
        if (k == bits) return codes_[h];   // another thread added it meanwhile
        if (k == kEmpty) break;
        h = (h + 1) & (kSlots - 1);
    }
    const uint32_t code = size_.load(std::memory_order_relaxed);
    if (code >= 256) {
        full_.store(true, std::memory_order_relaxed);
        return 0;
    }
    std::memcpy(&values_[code], &bits, sizeof(float));
    codes_[h] = static_cast<uint8_t>(code);
    keys_[h].store(bits, std::memory_order_release);
    size_.store(code + 1, std::memory_order_release);
    return static_cast<uint8_t>(code);
}

void ObsCodebook::encode(const float* row, size_t width, uint8_t* out) noexcept {
    for (size_t i = 0; i < width; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &row[i], sizeof(bits));
        out[i] = bits == 0 ? 0 : code_of(bits);
    }
}

size_t ObsCodebook::values(float* out) const noexcept {
    const size_t n = size();
    std::memcpy(out, values_, n * sizeof(float));
    return n;
}

bool encode_observations(const GameState* states, size_t n, uint32_t features, ObsCodebook& book,
                         uint8_t* codes) {
    const size_t width = OBS_SIZE_V23 + obs_features::extra_width(features);
    for_rows(static_cast<int64_t>(n), [&](int64_t j) {
        float row[OBS_SIZE_V23 + 64];
        const GameState& s = states[j];
        extract_observation_features(s, rollout::acting_player(s), features, row);
        book.encode(row, width, codes + j * width);
    });
    return !book.full();
}

uint16_t PlayoutLabels::choose(Row& r, const float* lg, int32_t forced) const {
    uint8_t mask[A];
    std::memcpy(mask, r.mask, A);
    int best = -1;
    for (size_t a = 0; a < A; ++a) {
        if (mask[a] && (best < 0 || lg[a] > lg[best])) best = static_cast<int>(a);
    }
    if (best < 0) return UINT16_MAX;
    const bool guarded = r.guarding && r.state.defcon <= 2;
    const Player who = rollout::acting_player(r.state);
    if (forced >= 0 && static_cast<size_t>(forced) < A && mask[forced] &&
        !(guarded && loses_now(r.state, static_cast<uint16_t>(forced), who))) {
        return static_cast<uint16_t>(forced);
    }
    if (!guarded) return static_cast<uint16_t>(best);
    // The masked argmax over the options that do not lose on the spot, when there are any: walk the
    // legal options best first and take the first safe one.
    if (!loses_now(r.state, static_cast<uint16_t>(best), who)) return static_cast<uint16_t>(best);
    mask[best] = 0;
    for (;;) {
        int next = -1;
        for (size_t a = 0; a < A; ++a) {
            if (mask[a] && (next < 0 || lg[a] > lg[next])) next = static_cast<int>(a);
        }
        if (next < 0) return static_cast<uint16_t>(best);   // every option loses: the argmax
        if (!loses_now(r.state, static_cast<uint16_t>(next), who)) return static_cast<uint16_t>(next);
        mask[next] = 0;
    }
}

void PlayoutLabels::after_step(uint32_t slot, const Entry& e) {
    Row& r = rows_[slot];
    const GameState& s = r.state;
    const bool terminal = Engine::is_terminal(s);
    if (!r.turn_seen && (terminal || s.turn != e.start_turn)) {
        r.turn_seen = true;
        // Final Scoring runs as the tenth turn ends; the turn's VP is the track before it.
        const int vp = (s.turn != e.start_turn && r.tap.final_scoring) ? r.tap.vp_before_final : s.victory_points;
        r.out.turn_vp = mover_vp(vp, e.mover);
        r.out.turn_defcon = s.defcon;
    }
    if (e.k && !r.leaf_taken && !terminal) {
        if (r.key_turn != s.turn || r.key_round != s.action_round || r.key_phasing != s.phasing_player ||
            r.key_phase != s.current_phase) {
            r.crossed = static_cast<uint8_t>(std::min(255, r.crossed + 1));
            r.key_turn = s.turn;
            r.key_round = s.action_round;
            r.key_phasing = s.phasing_player;
            r.key_phase = s.current_phase;
        }
        const bool far = r.crossed >= cfg_.leaf_horizon || s.turn != e.start_turn;
        if (far && rollout::acting_player(s) == e.mover) {
            r.leaf_taken = true;
            copy_state(leaves_[slot], s);
        }
    }
    if (terminal) finish(r, e);
}

void PlayoutLabels::finish(Row& r, const Entry& e) {
    const GameState& s = r.state;
    const float u = Engine::get_terminal_utility(s);
    const float mu = e.mover == Player::US ? u : -u;
    r.out.score = mu > 0 ? 1.0f : (mu < 0 ? 0.0f : 0.5f);
    r.out.final_vp = mover_vp(s.victory_points, e.mover);
    r.out.scored = r.tap.scored;
    const int sign = e.mover == Player::US ? 1 : -1;
    for (size_t g = 0; g < kRegions; ++g) r.out.swing[g] = static_cast<int16_t>(sign * r.tap.swing[g]);
    r.done = true;
}

void PlayoutLabels::advance(Row& r) const noexcept {
    if (cfg_.keyed) keyed::advance(r.state, r.world_key, keyed::mode_of(cfg_.keyed2, cfg_.keyedsym));
    else Engine::auto_advance_step(r.state);
}

bool PlayoutLabels::play(uint32_t slot, const Entry& e, uint16_t a) {
    Row& r = rows_[slot];
    bool ok = a != UINT16_MAX;
    if (ok) {
        Tapped tap(r.tap);
        if (cfg_.keyed) keyed::key(r.state, r.world_key, keyed::mode_of(cfg_.keyed2, cfg_.keyedsym));
        ok = Engine::step_flat(r.state, a);
        if (ok) advance(r);
    }
    if (!ok || ++r.plies > cfg_.max_plies) {
        r.failed = r.done = true;
        return false;
    }
    after_step(slot, e);
    return true;
}

// Bring a row to its next decision with a choice: a decision with one legal option is played here,
// without the network (its masked argmax is that option, under the safe rule too); the row's mask is
// left cached for choose().
void PlayoutLabels::settle(uint32_t slot, const Entry& e) {
    Row& r = rows_[slot];
    while (!r.done) {
        ActionMask::generate_flat_mask_212(r.state, r.mask);
        update_guard(r, e);
        int only = -1, legal = 0;
        for (size_t a = 0; a < A && legal < 2; ++a) {
            if (r.mask[a]) { only = static_cast<int>(a); ++legal; }
        }
        if (legal != 1) return;
        if (!play(slot, e, static_cast<uint16_t>(only))) return;
        ++r.forced;
    }
}

void PlayoutLabels::listed_info(int32_t* out) const noexcept {
    for (size_t j = 0; j < live_.size(); ++j) {
        const Row& r = rows_[live_[j]];
        const auto& c = r.state.ctx();
        out[j * 4 + 0] = static_cast<int32_t>(c.decision_type);
        out[j * 4 + 1] = (c.decision_type == DecisionType::SELECT_CARD && r.state.current_phase == Phase::HEADLINE) ? 1 : 0;
        out[j * 4 + 2] = r.state.turn;
        out[j * 4 + 3] = entries_[r.entry].start_turn;
    }
}

void PlayoutLabels::step(const float* logits, const int32_t* forced) {
    for_rows(static_cast<int64_t>(live_.size()), [&](int64_t j) {
        const uint32_t slot = live_[j];
        Row& r = rows_[slot];
        const Entry& e = entries_[r.entry];
        if (cfg_.keyed) keyed::key(r.state, r.world_key, keyed::mode_of(cfg_.keyed2, cfg_.keyedsym));   // the safe rule's probes see the same draws
        if (!play(slot, e, choose(r, logits + j * A, forced ? forced[j] : -1))) return;
        ++r.chosen;
        settle(slot, e);
    });
    live_.clear();
    harvest();
}

void PlayoutLabels::harvest() {
    for (uint32_t slot = 0; slot < rows_.size(); ++slot) {
        Row& r = rows_[slot];
        if (!r.occupied || !r.done) continue;
        Entry& e = entries_[r.entry];
        forced_ += r.forced;
        chosen_ += r.chosen;
        if (e.k) {
            SiblingLabel l;
            l.id = e.id;
            l.candidate = r.candidate;
            l.world = r.world;
            l.failed = r.failed;
            l.has_leaf = r.leaf_taken && !r.failed;
            l.out = r.out;
            siblings_out_.push_back(l);
            leaves_out_.emplace_back();
            if (l.has_leaf) copy_state(leaves_out_.back(), leaves_[slot]);
            else std::memset(static_cast<void*>(&leaves_out_.back()), 0, sizeof(GameState));
        } else if (r.failed) {
            ++e.failed;
        } else {
            ++e.n;
            e.score += r.out.score;
            e.score_sq += static_cast<double>(r.out.score) * r.out.score;
            e.final_vp += r.out.final_vp;
            e.turn_vp += r.out.turn_vp;
            e.turn_defcon += r.out.turn_defcon;
            for (size_t g = 0; g < kRegions; ++g) {
                if (r.out.scored & (1u << g)) {
                    e.swing[g] += r.out.swing[g];
                    ++e.swing_n[g];
                }
            }
        }
        r.occupied = false;
        free_.push_back(slot);
        if (--e.rows_left == 0) {
            if (!e.k) {
                PositionLabel p;
                p.id = e.id;
                p.n = e.n;
                p.failed = e.failed;
                p.score = static_cast<float>(e.score);
                p.score_sq = static_cast<float>(e.score_sq);
                p.final_vp = static_cast<float>(e.final_vp);
                p.turn_vp = static_cast<float>(e.turn_vp);
                p.turn_defcon = static_cast<float>(e.turn_defcon);
                for (size_t g = 0; g < kRegions; ++g) {
                    p.swing[g] = static_cast<float>(e.swing[g]);
                    p.swing_n[g] = e.swing_n[g];
                }
                positions_out_.push_back(p);
            }
            free_entries_.push_back(r.entry);
        }
    }
}

GameState PlayoutLabels::world(const GameState& base, int64_t id, uint16_t w) const noexcept {
    const Player mover = rollout::acting_player(base);
    uint64_t rng = entry_seed(seed_, id);
    GameState out{};
    for (uint16_t i = 0; i <= w; ++i) {
        copy_state(out, base);
        rollout::determinize(out, mover, rng);
        out.rng_state = Prng::next_u64(rng);
    }
    return out;
}

std::vector<PositionLabel> PlayoutLabels::take_positions() {
    std::vector<PositionLabel> out;
    out.swap(positions_out_);
    return out;
}

std::vector<SiblingLabel> PlayoutLabels::take_siblings(std::vector<GameState>& leaves) {
    std::vector<SiblingLabel> out;
    out.swap(siblings_out_);
    leaves.clear();
    leaves.swap(leaves_out_);
    return out;
}

}  // namespace ts::labels
