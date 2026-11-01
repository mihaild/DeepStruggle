#include "belief.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "rollout_batch.hpp"
#include "ts/prng.hpp"

namespace ts::belief {

namespace {

constexpr uint8_t kDeck = static_cast<uint8_t>(CardLocation::DRAW_DECK);

uint8_t unknown_of(Player p) noexcept {
    return static_cast<uint8_t>(p == Player::US ? CardLocation::HAND_US_UNKNOWN : CardLocation::HAND_USSR_UNKNOWN);
}
uint8_t known_of(Player p) noexcept {
    return static_cast<uint8_t>(p == Player::US ? CardLocation::HAND_US_KNOWN : CardLocation::HAND_USSR_KNOWN);
}
Player other(Player p) noexcept { return p == Player::US ? Player::USSR : Player::US; }

std::array<uint8_t, 111> locations(const GameState& s) noexcept {
    std::array<uint8_t, 111> out{};
    for (uint8_t c = 1; c <= 110; ++c) out[c] = static_cast<uint8_t>(s.card_locations[c]);
    return out;
}

bool transient(const std::array<uint8_t, 111>& l) noexcept {
    for (uint8_t c = 1; c <= 110; ++c) {
        if (l[c] == static_cast<uint8_t>(CardLocation::PEEKED_TEMP) ||
            l[c] == static_cast<uint8_t>(CardLocation::HEADLINE_COMMITTED)) return true;
    }
    return false;
}

double uniform01(uint64_t& rng) noexcept {
    return static_cast<double>(Prng::next_u64(rng) >> 11) * (1.0 / 9007199254740992.0);
}

}  // namespace

bool left_since(const Decision& d, Player me, const std::array<uint8_t, 111>& now, std::array<uint8_t, 110>& left) {
    const Player opp = other(me);
    const uint8_t ou = unknown_of(opp), mu = unknown_of(me), mk = known_of(me);
    size_t n_left = 0, hand_now = 0, unknown_then = 0;
    for (uint8_t c = 1; c <= 110; ++c) {
        const bool pool_then = d.locs[c] == kDeck || d.locs[c] == ou;
        const bool pool_now = now[c] == kDeck || now[c] == ou;
        if (pool_now && !pool_then) return false;          // a card entered the pool since
        const bool mine = now[c] == mu || now[c] == mk;
        left[c - 1] = (pool_then && !pool_now && !mine) ? 1 : 0;
        n_left += left[c - 1];
        hand_now += now[c] == ou;
        unknown_then += d.locs[c] == ou;
    }
    return unknown_then == n_left + hand_now;               // else the hand changed some other way
}

void Tracker::observe(size_t game, const GameState& s, Player decider, uint16_t action) {
    Game& g = games_[game];
    const int turn = static_cast<int>(s.turn);
    if (turn != g.turn) {
        for (Side& sd : g.side) {
            if (sd.has_current) {
                for (size_t t = 0; t < kTemps; ++t) sd.done[t] += sd.current[t];
            }
            sd.has_current = false;
            sd.cached_record = SIZE_MAX;
        }
        g.log.clear();
        g.turn = turn;
    }
    g.log.push_back(Decision{s, decider, action, locations(s)});
    ++g.seen;
}

std::vector<bool> Tracker::prepare(const std::vector<size_t>& games, const std::vector<const GameState*>& states) {
    jobs_.clear();
    position_game_.assign(games.size(), SIZE_MAX);
    position_key_.assign(games.size(), 0);
    position_me_.assign(games.size(), Player::NONE);
    std::vector<bool> has(games.size(), false);
    for (size_t p = 0; p < games.size(); ++p) {
        const GameState& s = *states[p];
        const Player me = s.ctx().decision_player;
        if (me != Player::US && me != Player::USSR) continue;
        const auto now = locations(s);
        if (transient(now)) continue;
        const uint8_t ou = unknown_of(other(me));
        const uint8_t revealed = reveal_ ? rollout::revealed_headline(s, me) : 0;   // in every hand
        std::vector<uint8_t> pool;
        size_t hand = 0;
        for (uint8_t c = 1; c <= 110; ++c) {
            if (c == revealed) continue;
            if (now[c] == kDeck || now[c] == ou) pool.push_back(c);
            hand += now[c] == ou;
        }
        if ((hand == 0 && !revealed) || hand > pool.size()) continue;
        Game& g = games_[games[p]];
        uint64_t key = seed_ ^ s.rng_state ^ (static_cast<uint64_t>(g.seen) << 20) ^ 0x9E3779B97F4A7C15ull;
        Prng::next_u64(key);
        position_game_[p] = games[p];
        position_key_[p] = key;
        position_me_[p] = me;
        has[p] = true;
        Side& sd = g.side[side_index(me)];

        Job job;
        job.position = p;
        job.game = games[p];
        job.me = me;
        job.key = key;
        // The opponent's headline, chosen but not revealed: the card stays in its hand until both sides
        // have chosen, so conditioning on the choice would deal it into every world.
        const bool headline_hidden = s.current_phase == Phase::HEADLINE && s.headline_stage == 0 && !revealed;
        for (uint32_t i = 0; i < g.log.size(); ++i) {
            const Decision& d = g.log[i];
            if (d.decider != other(me)) continue;
            const DecisionType t = d.state.ctx().decision_type;
            if (headline_hidden && d.state.current_phase == Phase::HEADLINE && d.state.headline_stage == 0 &&
                t == DecisionType::SELECT_CARD && d.state.ctx().resolving_card == 0) continue;
            if (!all_decisions_ && t != DecisionType::SELECT_CARD && t != DecisionType::SELECT_PLAY_MODE) continue;
            std::array<uint8_t, 110> left{};
            if (!left_since(d, me, now, left)) continue;
            job.decisions.push_back(i);
            job.lefts.push_back(left);
        }
        if (sd.cached_record == job.decisions.size() && sd.cached_locs == now && !sd.mixture.empty()) continue;

        // particles_ hands of the unseen size, uniform over the hidden pool (a partial Fisher-Yates each).
        uint64_t rng = key;
        job.hands.assign(static_cast<size_t>(particles_) * 110, 0);
        std::vector<uint8_t> perm(pool);
        for (uint32_t k = 0; k < particles_; ++k) {
            if (revealed) job.hands[static_cast<size_t>(k) * 110 + revealed - 1] = 1;
            for (size_t i = 0; i < hand; ++i) {
                const size_t j = i + Prng::random_index(rng, static_cast<uint32_t>(perm.size() - i));
                std::swap(perm[i], perm[j]);
                job.hands[static_cast<size_t>(k) * 110 + perm[i] - 1] = 1;
            }
        }
        if (job.decisions.empty()) {
            // Nothing to condition on: the uniform prior, weighed now with no rows to ask for.
            sd.hands = std::move(job.hands);
            sd.mixture.assign(particles_, 1.0f / static_cast<float>(particles_));
            sd.cached_record = 0;
            sd.cached_locs = now;
            continue;
        }
        sd.cached_record = SIZE_MAX;   // invalid until set_log_likelihoods weighs this job
        sd.cached_locs = now;
        jobs_.push_back(std::move(job));
    }
    return has;
}

void Tracker::set_log_likelihoods(const float* lp, size_t blocks) {
    size_t total = 0;
    for (const Job& j : jobs_) total += j.decisions.size();
    if (total != blocks) return;
    const size_t N = particles_;
    size_t r0 = 0;
    std::vector<double> logw(kTemps * N);
    for (Job& j : jobs_) {
        const size_t D = j.decisions.size();
        std::fill(logw.begin(), logw.end(), 0.0);
        for (size_t t = 0; t < kTemps; ++t) {
            for (size_t d = 0; d < D; ++d) {
                const float* row = lp + ((t * blocks) + r0 + d) * N;
                for (size_t k = 0; k < N; ++k) logw[t * N + k] += row[k];
            }
        }
        Side& g = games_[j.game].side[side_index(j.me)];
        std::array<double, kTemps> total_ev{};
        std::array<double, kTemps> mx{};
        for (size_t t = 0; t < kTemps; ++t) {
            double m = -std::numeric_limits<double>::infinity();
            for (size_t k = 0; k < N; ++k) m = std::max(m, logw[t * N + k]);
            double s = 0.0;
            for (size_t k = 0; k < N; ++k) s += std::exp(logw[t * N + k] - m);
            mx[t] = m;
            const double ev = m + std::log(s / static_cast<double>(N));
            j.evidence[t] = static_cast<float>(ev);
            total_ev[t] = g.done[t] + ev;
        }
        const double top = *std::max_element(total_ev.begin(), total_ev.end());
        std::array<double, kTemps> wt{};
        double wsum = 0.0;
        for (size_t t = 0; t < kTemps; ++t) wsum += (wt[t] = std::exp(total_ev[t] - top));
        j.mixture.assign(N, 0.0f);
        for (size_t t = 0; t < kTemps; ++t) {
            double s = 0.0;
            for (size_t k = 0; k < N; ++k) s += std::exp(logw[t * N + k] - mx[t]);
            const double scale = wt[t] / wsum / s;
            for (size_t k = 0; k < N; ++k)
                j.mixture[k] += static_cast<float>(scale * std::exp(logw[t * N + k] - mx[t]));
        }
        g.current = j.evidence;
        g.has_current = true;
        g.cached_record = D;
        g.hands = j.hands;
        g.mixture = j.mixture;
        r0 += D;
    }
}

bool Tracker::worlds(size_t p, size_t n, uint8_t* out) const {
    if (p >= position_game_.size() || position_game_[p] == SIZE_MAX) return false;
    const auto it = games_.find(position_game_[p]);
    if (it == games_.end()) return false;
    const Side& g = it->second.side[side_index(position_me_[p])];
    if (g.mixture.empty()) return false;
    const size_t N = g.mixture.size();
    std::vector<double> cdf(N);
    double acc = 0.0;
    for (size_t k = 0; k < N; ++k) cdf[k] = (acc += g.mixture[k]);
    uint64_t rng = position_key_[p] ^ 0xD1B54A32D192ED03ull;
    for (size_t i = 0; i < n; ++i) {
        const double u = uniform01(rng) * acc;
        const size_t k = std::min<size_t>(N - 1, static_cast<size_t>(std::upper_bound(cdf.begin(), cdf.end(), u) - cdf.begin()));
        std::copy_n(g.hands.begin() + static_cast<std::ptrdiff_t>(k * 110), 110, out + i * 110);
    }
    return true;
}

std::array<float, kTemps> Tracker::temperature_weights(size_t p) const {
    std::array<float, kTemps> w{};
    if (p >= position_game_.size() || position_game_[p] == SIZE_MAX) return w;
    const auto it = games_.find(position_game_[p]);
    if (it == games_.end()) return w;
    const Side& g = it->second.side[side_index(position_me_[p])];
    std::array<double, kTemps> e{};
    for (size_t t = 0; t < kTemps; ++t) e[t] = g.done[t] + (g.has_current ? g.current[t] : 0.0f);
    const double top = *std::max_element(e.begin(), e.end());
    double s = 0.0;
    for (size_t t = 0; t < kTemps; ++t) s += std::exp(e[t] - top);
    for (size_t t = 0; t < kTemps; ++t) w[t] = static_cast<float>(std::exp(e[t] - top) / s);
    return w;
}

}  // namespace ts::belief
