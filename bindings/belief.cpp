#include "belief.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>

#include "rollout_batch.hpp"
#include "ts/action_mask.hpp"
#include "ts/card_data.hpp"
#include "ts/constants.hpp"
#include "ts/ops.hpp"
#include "ts/prng.hpp"
#ifndef __EMSCRIPTEN__
#include "gomp_parallel.hpp"
#endif

namespace ts::belief {

namespace {

constexpr uint8_t kDeck = static_cast<uint8_t>(CardLocation::DRAW_DECK);
constexpr size_t A = FLAT_ACTION_SPACE_SIZE;

// The sampler's budgets, in rounds of `particles` candidate hands each (see `draw`).
constexpr int kPriorRounds = 8;    // rejection from the prior
constexpr int kGuidedRounds = 8;   // the stratified proposal
constexpr int kReps = 4;           // hands tried per stratum to estimate its consistent fraction
constexpr double kDefensive = 0.05;  // every stratum's floor in the proposal: full support over the hands
constexpr size_t kClasses = 5;     // strata of a card: scoring, then 1, 2, 3, 4+ effective Ops

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

// The three questions the engine asks only when the hand gives the asked a choice (unless a seat plays
// ALWAYS_ASK_DISCARD): Blockade's and Latin American Debt Crisis's 3-Ops discard (else resolved at once) and
// Missile Envy's tie (else exchanged at once). Asked, the hand then allowed two answers or more.
bool asked_only_with_choice(const GameState& s) noexcept {
    const auto& c = s.ctx();
    if (c.decision_type != DecisionType::SELECT_CARD) return false;
    if (c.resolving_card != card_ids::BLOCKADE && c.resolving_card != card_ids::LATIN_AMERICAN_DEBT_CRISIS &&
        c.resolving_card != card_ids::MISSILE_ENVY) return false;
    return !either_seat_plays_rule(s, obs_features::ALWAYS_ASK_DISCARD);
}

template <class F>
void for_each_index(int64_t n, F body) {
#ifndef __EMSCRIPTEN__
    if (n >= 32) {
        gomp_parallel_for(n, std::move(body));
        return;
    }
#endif
    for (int64_t j = 0; j < n; ++j) body(j);
}

double log_choose(size_t n, size_t k) noexcept {
    if (k > n) return -std::numeric_limits<double>::infinity();
    return std::lgamma(static_cast<double>(n) + 1.0) - std::lgamma(static_cast<double>(k) + 1.0) -
           std::lgamma(static_cast<double>(n - k) + 1.0);
}

double logsumexp(const std::vector<double>& v) noexcept {
    double m = -std::numeric_limits<double>::infinity();
    for (double x : v) m = std::max(m, x);
    if (!std::isfinite(m)) return m;
    double s = 0.0;
    for (double x : v) s += std::exp(x - m);
    return m + std::log(s);
}

const char* side_name(Player p) noexcept { return p == Player::US ? "US" : "USSR"; }

// The particles one position's belief starts from.
struct Draw {
    std::vector<uint8_t> hands;       // N x 110
    std::vector<double> log_weight;   // N, normalized
    double log_mass = 0.0;
    bool exhausted = false;
    std::string why;
};

// The hand sampler (audit S-1). The target is the uniform prior over hands of the unseen size from the pool,
// restricted to hands consistent with every constraint -- the posterior of the hand under the moves'
// legality alone, which the network's likelihood then reweighs.
//
// 1. Rejection from the prior, `particles` candidates a round for up to kPriorRounds rounds, each candidate
//    checked against the constraints in the engine (consistent_with). Accepted candidates are independent
//    draws from the target, exactly, so when they fill the particles every weight is equal. The first round
//    is the uniform sampler's own draws (same stream), so existing beliefs are unchanged wherever every one of
//    them was consistent. Once the acceptance rate so far cannot fill the particles in the rounds left, it
//    stops early.
// 2. Otherwise a stratified proposal with importance weights. The pool's cards are split into strata by what
//    the hand-dependent rules read -- scoring or not, and the effective Ops the opponent would have at the
//    constraint that rejected the most candidates (a trap takes 2+ Ops, Missile Envy takes the highest,
//    held scoring counts scoring cards) -- and a hand's stratum is its count vector over them. The proposal
//    picks a count vector kappa with probability proportional to its prior mass m(kappa) (the product of
//    binomials, so within a stratum it is the prior) times f(kappa) + kDefensive, f being the fraction of
//    kReps hands of that count vector found consistent, and then cards uniformly within each stratum. Every
//    hand keeps a positive probability, so the support covers every consistent hand whether or not the strata
//    capture the rule. Every candidate drawn by either step is weighed by the balance heuristic,
//    prior / ((n1 prior + n2 proposal) / (n1 + n2)) = (n1 + n2) / (n1 + n2 rho(kappa)) with
//    rho = (f + kDefensive) / (prior mean of f + kDefensive): a consistent estimator of any expectation under
//    the target (self-normalized importance sampling, the same O(1/N) bias the likelihood weighting has).
// 3. Fewer consistent hands than particles: the slots repeat them, each copy's weight divided by its
//    multiplicity, which leaves the estimator that of the distinct hands.
// 4. None at all -- which a correct record never produces, the opponent's real hand being consistent -- is a
//    fallback: the first round's prior hands, equally weighed, with `why` naming the position and the
//    decisions no drawn hand satisfies, so the caller can log it and still play.
Draw draw(const std::vector<Decision>& log, const std::vector<Constraint>& cons, Player me, const GameState& now_state,
          const std::vector<uint8_t>& pool, size_t hand, uint8_t revealed, uint32_t N, uint64_t key, Stats& st) {
    Draw out;
    const size_t C = cons.size();
    uint64_t rng = key;
    std::vector<uint8_t> perm(pool);
    auto prior_hand = [&](uint8_t* h) {
        std::fill(h, h + 110, 0);
        if (revealed) h[revealed - 1] = 1;
        for (size_t i = 0; i < hand; ++i) {
            const size_t j = i + Prng::random_index(rng, static_cast<uint32_t>(perm.size() - i));
            std::swap(perm[i], perm[j]);
            h[perm[i] - 1] = 1;
        }
    };
    // Constraints tried in order of how often they reject, so a candidate fails as early as it can.
    std::vector<size_t> order(C);
    std::iota(order.begin(), order.end(), 0);
    std::vector<uint64_t> fails(C, 0);
    const size_t B = std::max<size_t>(N, kReps);   // the candidate buffer: a round, or one stratum's tries
    std::vector<uint8_t> cand(B * 110);
    std::vector<uint8_t> ok(B);
    std::vector<int32_t> fail_at(B);
    auto check = [&](size_t m) {
        for_each_index(static_cast<int64_t>(m), [&](int64_t k) {
            ok[k] = 1;
            fail_at[k] = -1;
            for (size_t i : order) {
                if (!consistent_with(log, cons[i], me, cand.data() + k * 110)) {
                    ok[k] = 0;
                    fail_at[k] = static_cast<int32_t>(i);
                    break;
                }
            }
        });
        for (size_t k = 0; k < m; ++k)
            if (fail_at[k] >= 0) ++fails[static_cast<size_t>(fail_at[k])];
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return fails[a] > fails[b]; });
    };

    std::vector<uint8_t> acc;          // accepted hands, M x 110
    std::vector<uint8_t> first_round;  // the fallback's hands
    size_t M = 0;
    uint64_t n1 = 0, n2 = 0;
    for (int round = 0; round < kPriorRounds && M < N; ++round) {
        for (uint32_t k = 0; k < N; ++k) prior_hand(cand.data() + static_cast<size_t>(k) * 110);
        if (round == 0) first_round.assign(cand.begin(), cand.begin() + static_cast<std::ptrdiff_t>(N) * 110);
        if (C == 0) std::fill(ok.begin(), ok.begin() + N, 1);
        else check(N);
        size_t used = N;
        for (uint32_t k = 0; k < N; ++k) {
            if (!ok[k]) continue;
            acc.insert(acc.end(), cand.begin() + static_cast<std::ptrdiff_t>(k) * 110,
                       cand.begin() + static_cast<std::ptrdiff_t>(k + 1) * 110);
            if (++M == N) { used = k + 1; break; }
        }
        n1 += used;
        if (round == 0 && M == 0) ++st.first_round_empty;
        const double rate = static_cast<double>(M) / static_cast<double>(n1);
        const double left = static_cast<double>(kPriorRounds - round - 1) * N;
        if (M < N && rate * left < static_cast<double>(N - M)) break;
    }
    st.draws += n1;

    // The strata, when the prior did not fill the particles.
    std::array<std::vector<uint8_t>, kClasses> cls;
    uint8_t class_of[111] = {};
    std::unordered_map<uint32_t, double> f_of;   // count vector (4 bits a stratum) -> consistent fraction
    double f_bar = 0.0;
    auto kappa_key = [&](const uint8_t* h) {
        uint32_t k = 0;
        for (uint8_t c = 1; c <= 110; ++c)
            if (h[c - 1] && c != revealed) k += 1u << (4 * class_of[c]);
        return k;
    };
    if (M < N) {
        ++st.prior_short;
        ++st.guided;
        const size_t worst = static_cast<size_t>(std::max_element(fails.begin(), fails.end()) - fails.begin());
        const GameState& at = log[cons[worst].index].state;
        const Player opp = other(me);
        for (uint8_t c : pool) {
            const uint8_t ops = CardData::is_scoring_card(c) ? 0 : Operations::get_effective_ops(at, c, opp);
            const uint8_t k = CardData::is_scoring_card(c) ? 0 : static_cast<uint8_t>(std::clamp<int>(ops, 1, 4));
            class_of[c] = k;
            cls[k].push_back(c);
        }
        // Every count vector of `hand` cards over the strata.
        std::vector<std::array<uint8_t, kClasses>> kappas;
        std::array<uint8_t, kClasses> cur{};
        auto rec = [&](auto&& self, size_t i, size_t rest) -> void {
            if (i + 1 == kClasses) {
                if (rest <= cls[i].size()) { cur[i] = static_cast<uint8_t>(rest); kappas.push_back(cur); }
                return;
            }
            for (size_t c = 0; c <= std::min(rest, cls[i].size()); ++c) {
                cur[i] = static_cast<uint8_t>(c);
                self(self, i + 1, rest - c);
            }
        };
        rec(rec, 0, hand);
        // A hand within count vector kappa, uniformly within each stratum.
        std::array<std::vector<uint8_t>, kClasses> cperm = cls;
        auto stratum_hand = [&](const std::array<uint8_t, kClasses>& kv, uint8_t* h) {
            std::fill(h, h + 110, 0);
            if (revealed) h[revealed - 1] = 1;
            for (size_t i = 0; i < kClasses; ++i) {
                auto& p = cperm[i];
                for (size_t a = 0; a < kv[i]; ++a) {
                    const size_t b = a + Prng::random_index(rng, static_cast<uint32_t>(p.size() - a));
                    std::swap(p[a], p[b]);
                    h[p[a] - 1] = 1;
                }
            }
        };
        // f(kappa): kReps hands of each count vector, checked together.
        const size_t K = kappas.size();
        std::vector<double> logm(K), f(K, 0.0);
        for (size_t q = 0; q < K; ++q) {
            logm[q] = 0.0;
            for (size_t i = 0; i < kClasses; ++i) logm[q] += log_choose(cls[i].size(), kappas[q][i]);
        }
        const size_t per = std::max<size_t>(1, B / kReps);
        for (size_t lo = 0; lo < K; lo += per) {
            const size_t hi = std::min(K, lo + per);
            size_t m = 0;
            for (size_t q = lo; q < hi; ++q)
                for (int r = 0; r < kReps; ++r) stratum_hand(kappas[q], cand.data() + (m++) * 110);
            check(m);
            m = 0;
            for (size_t q = lo; q < hi; ++q)
                for (int r = 0; r < kReps; ++r) f[q] += ok[m++] ? 1.0 / kReps : 0.0;
        }
        const double total = log_choose(pool.size(), hand);
        std::vector<double> cdf(K);
        const double top = *std::max_element(logm.begin(), logm.end());
        double acc_r = 0.0;
        for (size_t q = 0; q < K; ++q) {
            f_bar += std::exp(logm[q] - total) * f[q];
            cdf[q] = (acc_r += std::exp(logm[q] - top) * (f[q] + kDefensive));
            uint32_t key_q = 0;
            for (size_t i = 0; i < kClasses; ++i) key_q += static_cast<uint32_t>(kappas[q][i]) << (4 * i);
            f_of[key_q] = f[q];
        }
        for (int round = 0; round < kGuidedRounds && M < N; ++round) {
            for (uint32_t k = 0; k < N; ++k) {
                const double u = uniform01(rng) * acc_r;
                const size_t q = std::min<size_t>(K - 1, static_cast<size_t>(std::upper_bound(cdf.begin(), cdf.end(), u) - cdf.begin()));
                stratum_hand(kappas[q], cand.data() + static_cast<size_t>(k) * 110);
            }
            check(N);
            size_t used = N;
            for (uint32_t k = 0; k < N; ++k) {
                if (!ok[k]) continue;
                acc.insert(acc.end(), cand.begin() + static_cast<std::ptrdiff_t>(k) * 110,
                           cand.begin() + static_cast<std::ptrdiff_t>(k + 1) * 110);
                if (++M == N) { used = k + 1; break; }
            }
            n2 += used;
        }
        st.draws += n2;
    }

    if (M == 0) {
        ++st.exhausted;
        out.exhausted = true;
        out.hands = std::move(first_round);
        out.log_weight.assign(N, -std::log(static_cast<double>(N)));
        // Which decisions no hand of the first round satisfies on its own.
        std::string never;
        std::vector<uint8_t> alone(N);
        for (size_t i = 0; i < C; ++i) {
            for_each_index(static_cast<int64_t>(N), [&](int64_t k) {
                alone[k] = consistent_with(log, cons[i], me, out.hands.data() + k * 110) ? 1 : 0;
            });
            if (std::find(alone.begin(), alone.end(), 1) != alone.end()) continue;
            const Decision& d = log[cons[i].index];
            char buf[200];
            std::snprintf(buf, sizeof buf, "%s[turn %d AR %d, decision type %d, action %d, resolving card %d%s]",
                          never.empty() ? "" : ", ", static_cast<int>(d.state.turn), static_cast<int>(d.state.action_round),
                          static_cast<int>(d.state.ctx().decision_type), static_cast<int>(d.action),
                          static_cast<int>(d.state.ctx().resolving_card), cons[i].choice ? ", asked with a choice" : "");
            never += buf;
        }
        char head[300];
        std::snprintf(head, sizeof head,
                      "belief: no %s hand of %zu cards (pool %zu) consistent with its %zu recorded decisions at turn %d AR %d "
                      "(%s to decide) in %llu draws; ",
                      side_name(other(me)), hand, pool.size(), C, static_cast<int>(now_state.turn),
                      static_cast<int>(now_state.action_round), side_name(me),
                      static_cast<unsigned long long>(n1 + n2));
        out.why = std::string(head) + (never.empty() ? "each decision is satisfiable alone, never all together"
                                                     : "decisions no drawn hand satisfies: " + never) +
                  "; the uniform prior is used instead";
        return out;
    }

    // Balance-heuristic weights of the consistent hands, then the slots.
    std::vector<double> lw(M);
    for (size_t j = 0; j < M; ++j) {
        double rho = 0.0;
        if (n2) {
            const auto it = f_of.find(kappa_key(acc.data() + j * 110));
            rho = ((it == f_of.end() ? 0.0 : it->second) + kDefensive) / (f_bar + kDefensive);
        }
        lw[j] = -std::log(static_cast<double>(n1) + static_cast<double>(n2) * rho);
    }
    out.log_mass = logsumexp(lw);
    out.hands.resize(static_cast<size_t>(N) * 110);
    out.log_weight.resize(N);
    for (uint32_t k = 0; k < N; ++k) {
        const size_t j = k % M;
        const size_t mult = N / M + (j < N % M ? 1 : 0);
        std::copy_n(acc.begin() + static_cast<std::ptrdiff_t>(j) * 110, 110,
                    out.hands.begin() + static_cast<std::ptrdiff_t>(k) * 110);
        out.log_weight[k] = lw[j] - out.log_mass - std::log(static_cast<double>(mult));
    }
    return out;
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
        // A card the opponent held at d that `me` holds now (a Missile Envy gift, a Grain Sales draw kept):
        // `me` saw where it came from. Without this the count test below reduces to "cards gone opponent
        // -> me == cards gone deck -> elsewhere + cards gone deck -> opponent" and passes whenever the two
        // sides happen to match (a gift and an Our Man in Tehran discard in one turn), with the deck's
        // discards dealt into the hand at d and the gift left out of it.
        if (mine && d.locs[c] == ou) return false;
        // A card the opponent held at d that is in the deck now went through the discard pile and a reshuffle
        // (Our Man in Tehran reshuffles the discard pile into the deck): the count test passes when it matches
        // a deck card discarded since, and the hand at d would be dealt that card instead of this one.
        if (now[c] == kDeck && d.locs[c] == ou) return false;
        left[c - 1] = (pool_then && !pool_now && !mine) ? 1 : 0;
        n_left += left[c - 1];
        hand_now += now[c] == ou;
        unknown_then += d.locs[c] == ou;
    }
    // With nothing gone opponent -> me or opponent -> deck, exact: unknown_then = hand_now - (deck -> opponent)
    // + n_left - (deck -> elsewhere), so equality holds only when neither happened (an event's draw or discard).
    return unknown_then == n_left + hand_now;
}

bool consistent_with(const std::vector<Decision>& log, const Constraint& c, Player me, const uint8_t* hand) {
    const Decision& d = log[c.index];
    uint8_t held[110];
    for (size_t i = 0; i < 110; ++i) held[i] = static_cast<uint8_t>(hand[i] | c.left[i]);
    GameState w = d.state;
    rollout::place_hand(w, me, held);
    uint8_t mask[A];
    ActionMask::generate_flat_mask_212(w, mask);
    if (!mask[d.action]) return false;
    if (!c.choice) return true;
    size_t n = 0;
    for (size_t a = 0; a < A && n < 2; ++a) n += mask[a] != 0;
    return n >= 2;
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
            sd.current_decisions.clear();
            sd.spent.clear();
            sd.cached_record = SIZE_MAX;
            sd.constraints.clear();
        }
        g.log.clear();
        g.turn = turn;
    }
    g.log.push_back(Decision{s, decider, action, locations(s)});
    ++g.seen;
}

std::vector<bool> Tracker::prepare(const std::vector<size_t>& games, const std::vector<const GameState*>& states) {
    jobs_.clear();
    exhausted_.clear();
    position_game_.assign(games.size(), SIZE_MAX);
    position_key_.assign(games.size(), 0);
    position_me_.assign(games.size(), Player::NONE);
    std::vector<bool> has(games.size(), false);
    if (particles_ == 0) return has;
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
        ++stats_.positions;
        Side& sd = g.side[side_index(me)];

        Job job;
        job.position = p;
        job.game = games[p];
        job.me = me;
        job.key = key;
        std::vector<Constraint> cons;
        // The opponent's headline, chosen but not revealed: the card stays in its hand until both sides
        // have chosen, so conditioning on the choice would deal it into every world.
        const bool headline_hidden = s.current_phase == Phase::HEADLINE && s.headline_stage == 0 && !revealed;
        for (uint32_t i = 0; i < g.log.size(); ++i) {
            const Decision& d = g.log[i];
            if (d.decider != other(me)) continue;
            const DecisionType t = d.state.ctx().decision_type;
            if (headline_hidden && d.state.current_phase == Phase::HEADLINE && d.state.headline_stage == 0 &&
                t == DecisionType::SELECT_CARD && d.state.ctx().resolving_card == 0) continue;
            std::array<uint8_t, 110> left{};
            if (!left_since(d, me, now, left)) continue;
            // Legality binds on every decision on the record; the likelihood only on the kinds asked for.
            cons.push_back(Constraint{i, left, auto_advanced_ || asked_only_with_choice(d.state)});
            if (!all_decisions_ && t != DecisionType::SELECT_CARD && t != DecisionType::SELECT_PLAY_MODE) continue;
            job.decisions.push_back(i);
            job.lefts.push_back(left);
        }
        // The record lost a decision the current evidence weighs (a reshuffle, an event's draw since): that
        // evidence is final. Fold it into done and spend its decisions; those still on the record keep
        // weighing the hands but leave the temperature evidence, so it does not count them twice.
        if (sd.has_current && !std::includes(job.decisions.begin(), job.decisions.end(),
                                             sd.current_decisions.begin(), sd.current_decisions.end())) {
            for (size_t t = 0; t < kTemps; ++t) sd.done[t] += sd.current[t];
            sd.has_current = false;
            for (uint32_t i : sd.current_decisions) {
                if (sd.spent.size() <= i) sd.spent.resize(i + 1, false);
                sd.spent[i] = true;
            }
            sd.current_decisions.clear();
            sd.cached_record = SIZE_MAX;
        }
        job.unspent.resize(job.decisions.size());
        for (size_t r = 0; r < job.decisions.size(); ++r) {
            const uint32_t i = job.decisions[r];
            job.unspent[r] = (i < sd.spent.size() && sd.spent[i]) ? 0 : 1;
        }
        if (sd.cached_record == job.decisions.size() && sd.constraints.size() == cons.size() && sd.cached_locs == now &&
            !sd.mixture.empty()) continue;

        if (!cons.empty()) ++stats_.sampled;
        Draw dr = draw(g.log, cons, me, s, pool, hand, revealed, particles_, key, stats_);
        sd.constraints = cons;
        sd.cached_locs = now;
        if (dr.exhausted) {
            exhausted_.emplace_back(p, std::move(dr.why));
            sd.hands = std::move(dr.hands);
            sd.mixture.assign(particles_, 1.0f / static_cast<float>(particles_));
            sd.cached_record = job.decisions.size();
            continue;
        }
        if (job.decisions.empty()) {
            // Nothing to weigh by likelihood: the consistent hands by their importance weights, now.
            sd.hands = std::move(dr.hands);
            sd.mixture.resize(particles_);
            for (uint32_t k = 0; k < particles_; ++k) sd.mixture[k] = static_cast<float>(std::exp(dr.log_weight[k]));
            sd.cached_record = 0;
            continue;
        }
        job.hands = std::move(dr.hands);
        job.log_weight = std::move(dr.log_weight);
        job.log_mass = dr.log_mass;
        sd.cached_record = SIZE_MAX;   // invalid until set_log_likelihoods weighs this job
        sd.hands.clear();              // and no world is dealt from the last belief meanwhile
        sd.mixture.clear();
        jobs_.push_back(std::move(job));
    }
    return has;
}

const char* Tracker::set_log_likelihoods(const float* lp, size_t blocks) {
    size_t total = 0;
    for (const Job& j : jobs_) total += j.decisions.size();
    if (total != blocks) return "set_log_likelihoods: one block per recorded decision of every job";
    const size_t N = particles_;
    // Every particle is consistent with every recorded move, so the move has positive probability under each:
    // a non-finite value is a caller's bug (a NaN, an infinite logit, a mask other than the engine's), and it
    // would otherwise make NaN weights or tie hands at a sentinel.
    for (size_t i = 0; i < kTemps * blocks * N; ++i) {
        if (!std::isfinite(lp[i])) return "set_log_likelihoods: a log-likelihood is not finite";
    }
    size_t r0 = 0;
    std::vector<double> la(N), lu(N);
    for (Job& j : jobs_) {
        const size_t D = j.decisions.size();
        Side& g = games_[j.game].side[side_index(j.me)];
        std::array<double, kTemps> total_ev{};
        std::vector<std::vector<double>> post(kTemps);
        for (size_t t = 0; t < kTemps; ++t) {
            for (size_t k = 0; k < N; ++k) la[k] = lu[k] = j.log_weight[k];
            for (size_t d = 0; d < D; ++d) {
                const float* row = lp + ((t * blocks) + r0 + d) * N;
                for (size_t k = 0; k < N; ++k) {
                    la[k] += row[k];
                    if (j.unspent[d]) lu[k] += row[k];
                }
            }
            // The evidence, log E_prior prod_unspent pi_T: the weighted mean over the consistent hands times the
            // prior mass of consistent hands (constant across T, so the temperature weights do not see it).
            j.evidence[t] = logsumexp(lu) + j.log_mass;
            total_ev[t] = g.done[t] + j.evidence[t];
            const double z = logsumexp(la);
            post[t].resize(N);
            for (size_t k = 0; k < N; ++k) post[t][k] = std::exp(la[k] - z);
        }
        const double top = *std::max_element(total_ev.begin(), total_ev.end());
        std::array<double, kTemps> wt{};
        double wsum = 0.0;
        for (size_t t = 0; t < kTemps; ++t) wsum += (wt[t] = std::exp(total_ev[t] - top));
        j.mixture.assign(N, 0.0f);
        for (size_t k = 0; k < N; ++k) {
            double m = 0.0;
            for (size_t t = 0; t < kTemps; ++t) m += wt[t] / wsum * post[t][k];
            j.mixture[k] = static_cast<float>(m);
        }
        g.current = j.evidence;
        g.current_decisions.clear();
        for (size_t d = 0; d < D; ++d)
            if (j.unspent[d]) g.current_decisions.push_back(j.decisions[d]);
        g.has_current = !g.current_decisions.empty();
        g.cached_record = D;
        g.hands = j.hands;
        g.mixture = j.mixture;
        r0 += D;
    }
    return nullptr;
}

const Tracker::Side* Tracker::side_of(size_t p) const {
    if (p >= position_game_.size() || position_game_[p] == SIZE_MAX) return nullptr;
    const auto it = games_.find(position_game_[p]);
    if (it == games_.end()) return nullptr;
    return &it->second.side[side_index(position_me_[p])];
}

bool Tracker::worlds(size_t p, size_t n, uint8_t* out) const {
    const Side* sd = side_of(p);
    if (sd == nullptr || sd->mixture.empty()) return false;
    const Side& g = *sd;
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
    const Side* sd = side_of(p);
    if (sd == nullptr) return w;
    const Side& g = *sd;
    std::array<double, kTemps> e{};
    for (size_t t = 0; t < kTemps; ++t) e[t] = g.done[t] + (g.has_current ? g.current[t] : 0.0);
    const double top = *std::max_element(e.begin(), e.end());
    double s = 0.0;
    for (size_t t = 0; t < kTemps; ++t) s += std::exp(e[t] - top);
    for (size_t t = 0; t < kTemps; ++t) w[t] = static_cast<float>(std::exp(e[t] - top) / s);
    return w;
}

const std::vector<uint8_t>& Tracker::hands(size_t p) const {
    static const std::vector<uint8_t> none;
    const Side* sd = side_of(p);
    return (sd == nullptr || sd->mixture.empty()) ? none : sd->hands;
}

const std::vector<float>& Tracker::mixture(size_t p) const {
    static const std::vector<float> none;
    const Side* sd = side_of(p);
    return sd == nullptr ? none : sd->mixture;
}

bool Tracker::consistent(size_t p, const uint8_t* hand) const {
    const Side* sd = side_of(p);
    if (sd == nullptr) return false;
    const auto& log = games_.at(position_game_[p]).log;
    for (const Constraint& c : sd->constraints)
        if (!consistent_with(log, c, position_me_[p], hand)) return false;
    return true;
}

}  // namespace ts::belief
