// obs_features::UNCAPPED_DEFICIT (owner, 2026-11-26): board slots 24 and 25, the influence each side lacks to
// control a country / 5, written without the base's cap of 2.0 (10 points). Clear, the observation is the
// capped base, bit for bit; set, only slots 24 and 25 of the countries past the cap move.
#include "test_framework.hpp"
#include "ts/action_mask.hpp"
#include "ts/card_data.hpp"
#include "ts/engine.hpp"
#include "ts/map_data.hpp"
#include "ts/observation.hpp"

#include <algorithm>
#include <vector>

namespace {

namespace of = ts::obs_features;

std::vector<float> obs(const ts::GameState& s, ts::Player p, uint32_t features) {
    std::vector<float> v(ts::OBS_SIZE_V23 + of::extra_width(features));
    ts::extract_observation_features(s, p, features, v.data());
    return v;
}

float slot(const std::vector<float>& v, uint8_t country, size_t k) {
    return v[static_cast<size_t>(country) * ts::V23_BOARD_FEATURES + k];
}

}  // namespace

TEST(ObservationFeatures, UncappedDeficitWritesTheMarginPastTen) {
    ts::GameState s{};
    ts::Engine::init_game(s, 11);
    const uint8_t wg = ts::MapData::get_country_by_name("West Germany");     // stability 4
    const uint8_t fr = ts::MapData::get_country_by_name("France");           // stability 3
    s.countries[wg].us_influence = 0;
    s.countries[wg].ussr_influence = 40;
    s.countries[fr].us_influence = 5;
    s.countries[fr].ussr_influence = 0;
    for (uint32_t extra : {0u, of::OPS_BUDGET | of::HEADLINE_REVEAL}) {
        const auto ussr_cap = obs(s, ts::Player::USSR, extra);
        const auto ussr = obs(s, ts::Player::USSR, extra | of::UNCAPPED_DEFICIT);
        const auto us_cap = obs(s, ts::Player::US, extra);
        const auto us = obs(s, ts::Player::US, extra | of::UNCAPPED_DEFICIT);
        ASSERT_EQ(ussr.size(), ussr_cap.size());                                  // no width
        // WG, USSR 40 against US 0: the US lacks 40 + 4 = 44 points, 8.8; the base caps it at 2.0.
        ASSERT_EQ(slot(ussr_cap, wg, of::BOARD_OPP_DEFICIT), 2.0f);
        ASSERT_EQ(slot(ussr, wg, of::BOARD_OPP_DEFICIT), 44.0f / 5.0f);
        ASSERT_EQ(slot(ussr, wg, of::BOARD_MY_DEFICIT), 0.0f);                    // controlled: still 0
        ASSERT_EQ(slot(us_cap, wg, of::BOARD_MY_DEFICIT), 2.0f);
        ASSERT_EQ(slot(us, wg, of::BOARD_MY_DEFICIT), 44.0f / 5.0f);
        ASSERT_EQ(slot(us, wg, of::BOARD_OPP_DEFICIT), 0.0f);
        // France, US 5 against 0: the USSR lacks 5 + 3 = 8, under the cap, so both views agree.
        ASSERT_EQ(slot(us, fr, of::BOARD_OPP_DEFICIT), 8.0f / 5.0f);
        ASSERT_EQ(slot(us_cap, fr, of::BOARD_OPP_DEFICIT), 8.0f / 5.0f);
        // Nothing else moves: every other float of both views is the capped one's.
        for (const auto* pair : {&ussr, &us}) {
            const auto& with = *pair;
            const auto& plain = pair == &ussr ? ussr_cap : us_cap;
            for (size_t k = 0; k < with.size(); ++k) {
                if (k == static_cast<size_t>(wg) * ts::V23_BOARD_FEATURES + of::BOARD_MY_DEFICIT ||
                    k == static_cast<size_t>(wg) * ts::V23_BOARD_FEATURES + of::BOARD_OPP_DEFICIT) continue;
                ASSERT_EQ(with[k], plain[k]);
            }
        }
    }
}

TEST(ObservationFeatures, UncappedDeficitClearIsTheBaseBitForBit) {
    // Without the bit the observation is Observation::extract's, capped; with it, on a played-out board, slots 24
    // and 25 equal the base wherever the base is under its cap and exceed it only where the base reads the cap.
    size_t past_cap = 0;
    for (uint64_t seed = 1; seed <= 20; ++seed) {
        ts::GameState s{};
        ts::Engine::init_game(s, seed);
        uint64_t lcg = seed * 2654435761ULL;
        uint8_t mask[ts::FLAT_ACTION_SPACE_SIZE];
        for (int step = 0; step < 4000 && !ts::Engine::is_terminal(s); ++step) {
            ts::Engine::auto_advance_step(s);
            if (ts::Engine::is_terminal(s)) break;
            ts::ObservationBufferV23 base{};
            ts::Observation::extract(s, ts::Player::NONE, &base);
            const auto plain = obs(s, ts::Player::NONE, 0);
            ASSERT_TRUE(std::equal(plain.begin(), plain.begin() + 84 * ts::V23_BOARD_FEATURES, base.board_features));
            const auto with = obs(s, ts::Player::NONE, of::UNCAPPED_DEFICIT);
            for (size_t j = 0; j < with.size(); ++j) {
                const size_t sl = j % ts::V23_BOARD_FEATURES;
                const bool deficit = j < 84 * ts::V23_BOARD_FEATURES &&
                                     (sl == of::BOARD_MY_DEFICIT || sl == of::BOARD_OPP_DEFICIT);
                if (!deficit || plain[j] < of::BOARD_DEFICIT_CAP) {
                    ASSERT_EQ(with[j], plain[j]);
                } else {
                    ASSERT_TRUE(with[j] >= of::BOARD_DEFICIT_CAP);
                    past_cap += with[j] > of::BOARD_DEFICIT_CAP;
                }
            }
            ts::Engine::get_flat_action_mask(s, mask, ts::ActionView::E4);
            int legal[ts::FLAT_ACTION_SPACE_SIZE];
            int n = 0;
            for (int i = 0; i < ts::flat_slots::BASE_SIZE; ++i)
                if (mask[i]) legal[n++] = i;
            ASSERT_TRUE(n > 0);
            lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
            ASSERT_TRUE(ts::Engine::step_flat(s, static_cast<uint16_t>(legal[(lcg >> 33) % n]), false));
        }
    }
    ASSERT_TRUE(past_cap > 0);              // the random games did stack past 10 somewhere
}

namespace {

float glob(const std::vector<float>& v, size_t k) {
    return v[84 * ts::V23_BOARD_FEATURES + 110 * ts::card_slots::V23_FEATURES + k];
}

// The decision-context slots HONEST_CONTEXT rewrites: OP_MODE's three, ALLOW_EARLY_STOP, TIMING_OPS_FIRST.
bool honest_slot(size_t j) {
    const size_t g0 = 84 * ts::V23_BOARD_FEATURES + 110 * ts::card_slots::V23_FEATURES;
    if (j < g0) return false;
    const size_t k = j - g0;
    return (k >= ts::ctx_slots::OP_MODE && k < ts::ctx_slots::OP_MODE + 3) || k == ts::ctx_slots::ALLOW_EARLY_STOP ||
           k == ts::ctx_slots::TIMING_OPS_FIRST;
}

// One random game's states, handed to `visit` with the action about to be taken.
template <typename F>
void random_game(uint64_t seed, F visit) {
    ts::GameState s{};
    ts::Engine::init_game(s, seed);
    uint64_t lcg = seed * 2654435761ULL + 17;
    uint8_t mask[ts::FLAT_ACTION_SPACE_SIZE];
    // Every decision is visited and stepped here, forced ones and chance nodes too (auto-advance would take them
    // unseen), so a tracker sees every choice.
    for (int step = 0; step < 8000 && !ts::Engine::is_terminal(s); ++step) {
        ts::Engine::get_flat_action_mask(s, mask, ts::ActionView::E4);
        int legal[ts::FLAT_ACTION_SPACE_SIZE];
        int n = 0;
        for (int i = 0; i < ts::flat_slots::BASE_SIZE; ++i)
            if (mask[i]) legal[n++] = i;
        ASSERT_TRUE(n > 0);
        lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
        const uint16_t a = static_cast<uint16_t>(legal[(lcg >> 33) % n]);
        visit(s, mask, a);
        ASSERT_TRUE(ts::Engine::step_flat(s, a, false));
    }
}

}  // namespace

TEST(ObservationFeatures, HonestContextSaysWhatTheStateSays) {
    // Over random games, at every decision type: OP_MODE is the chosen op mode exactly while Ops are being spent (a
    // tracker records the op mode each frame chose, from the actions taken) and 0 before; TIMING_OPS_FIRST is 1
    // exactly while the Ops of an opponent's card played ops first are spent and its Event is still owed (from the
    // actions too); ALLOW_EARLY_STOP is the mask's CONFIRM_DONE. Nothing else in the observation moves.
    size_t seen[8][2] = {};          // [decision type][the base's TIMING_OPS_FIRST differs from the honest one]
    size_t ops_first = 0, chosen_modes[3] = {}, stop_differs = 0, mode_differs = 0;
    for (uint64_t seed = 1; seed <= 60; ++seed) {
        // The op mode a frame chose, keyed by its depth: (card, mode); and the frame whose ops-first Event is owed.
        int chosen_card[8], chosen_mode[8], owed_card[8];
        for (int d = 0; d < 8; ++d) chosen_card[d] = chosen_mode[d] = owed_card[d] = -1;
        random_game(seed, [&](const ts::GameState& s, const uint8_t* mask, uint16_t a) {
            const auto& ctx = s.ctx();
            const int d = s.ctx_stack_depth;
            for (int k = d + 1; k < 8; ++k) chosen_card[k] = chosen_mode[k] = owed_card[k] = -1;   // popped frames
            if (ctx.timing_branch != static_cast<uint8_t>(ts::TimingBranch::OPS_FIRST)) owed_card[d] = -1;  // fired
            const ts::Player p = ctx.decision_player;
            for (ts::Player view : {ts::Player::US, ts::Player::USSR}) {
                const auto base = obs(s, view, 0);
                const auto honest = obs(s, view, of::HONEST_CONTEXT);
                for (size_t j = 0; j < base.size(); ++j) {
                    if (!honest_slot(j)) ASSERT_EQ(honest[j], base[j]);
                }
                // Past this frame's choices of card, use and op mode, with no Event resolving in it.
                const bool spending = ctx.pending_op_card != 0 && ctx.resolving_card == 0 &&
                                      ctx.decision_type != ts::DecisionType::NONE &&
                                      ctx.decision_type != ts::DecisionType::SELECT_CARD &&
                                      ctx.decision_type != ts::DecisionType::SELECT_PLAY_MODE &&
                                      ctx.decision_type != ts::DecisionType::SELECT_OP_MODE;
                const bool tracked = chosen_card[d] == ctx.pending_op_card && spending;
                for (int m = 0; m < 3; ++m) {
                    const float want = (tracked && chosen_mode[d] == m) ? 1.0f : 0.0f;
                    ASSERT_EQ(glob(honest, ts::ctx_slots::OP_MODE + m), want);
                }
                const bool owed = spending && owed_card[d] == ctx.pending_op_card;
                ASSERT_EQ(glob(honest, ts::ctx_slots::TIMING_OPS_FIRST), owed ? 1.0f : 0.0f);
                ASSERT_EQ(glob(honest, ts::ctx_slots::ALLOW_EARLY_STOP), mask[ts::flat_slots::CONFIRM_DONE] ? 1.0f : 0.0f);
                if (view == ts::Player::US) {
                    const size_t dt = static_cast<size_t>(ctx.decision_type);
                    seen[dt][glob(base, ts::ctx_slots::TIMING_OPS_FIRST) != glob(honest, ts::ctx_slots::TIMING_OPS_FIRST)]++;
                    ops_first += owed;
                    stop_differs += glob(base, ts::ctx_slots::ALLOW_EARLY_STOP) != glob(honest, ts::ctx_slots::ALLOW_EARLY_STOP);
                    for (int m = 0; m < 3; ++m) {
                        mode_differs += glob(base, ts::ctx_slots::OP_MODE + m) != glob(honest, ts::ctx_slots::OP_MODE + m);
                        chosen_modes[m] += glob(honest, ts::ctx_slots::OP_MODE + m) > 0.0f;
                    }
                }
            }
            // Record what this action chooses: a use of the card for Ops (the play mode's OPS_*), or an op mode.
            if (ctx.decision_type == ts::DecisionType::SELECT_PLAY_MODE && a >= ts::flat_slots::OPS_INFLUENCE &&
                a < ts::flat_slots::OPS_INFLUENCE + 3) {
                chosen_card[d] = ctx.pending_op_card;
                chosen_mode[d] = a - ts::flat_slots::OPS_INFLUENCE;
                owed_card[d] = ts::CardData::is_opponent_card(ctx.pending_op_card, p) ? ctx.pending_op_card : -1;
            } else if (ctx.decision_type == ts::DecisionType::SELECT_OP_MODE && a >= ts::flat_slots::OP_MODE &&
                       a < ts::flat_slots::OP_MODE + 3) {
                chosen_card[d] = ctx.pending_op_card;
                chosen_mode[d] = a - ts::flat_slots::OP_MODE;
                owed_card[d] = -1;    // event first, a lent card, or an event's own Ops: nothing owed after them
            }
        });
    }
    // The base said "ops first" at card selection and play-mode choices, where nothing was chosen yet.
    ASSERT_TRUE(seen[static_cast<size_t>(ts::DecisionType::SELECT_CARD)][1] > 0);
    ASSERT_TRUE(seen[static_cast<size_t>(ts::DecisionType::SELECT_PLAY_MODE)][1] > 0);
    ASSERT_TRUE(ops_first > 0 && stop_differs > 0 && mode_differs > 0);
    for (size_t m = 0; m < 3; ++m) ASSERT_TRUE(chosen_modes[m] > 0);
}

TEST(ObservationFeatures, RelativeEffectsAreTheObserversOwn) {
    // A US observer's view is unchanged. A USSR observer's mirrored pairs are swapped, and the two space-attempt slots
    // read its own attempts (bits 45/46) instead of the US's. Nothing else moves; unset, the base bit for bit.
    ts::GameState s{};
    ts::Engine::init_game(s, 5);
    s.persistent_effects |= ts::effect_bits::CONTAINMENT_ACTIVE | ts::effect_bits::PURGE_USSR_ACTIVE |
                            ts::effect_bits::CMC_ACTIVE_US | ts::effect_bits::BEAR_TRAP_ACTIVE |
                            ts::effect_bits::DEATH_SQUADS_US | ts::effect_bits::SPACE_US_ATTEMPT_1 |
                            ts::effect_bits::SPACE_USSR_ATTEMPT_1 | ts::effect_bits::SPACE_USSR_ATTEMPT_2 |
                            ts::effect_bits::NATO_ACTIVE;
    const size_t E = of::GLOBAL_EFFECT_BITS;
    ASSERT_TRUE(obs(s, ts::Player::US, of::RELATIVE_EFFECTS) == obs(s, ts::Player::US, 0));
    const auto base = obs(s, ts::Player::USSR, 0);
    const auto rel = obs(s, ts::Player::USSR, of::RELATIVE_EFFECTS);
    // Base, absolute: Containment (US's +1) at 6, the USSR's Purge at 8, US's CMC at 11, Bear Trap at 15, US's Death
    // Squads at 22, the US's first attempt at 43.
    ASSERT_EQ(glob(base, E + 6), 1.0f);
    ASSERT_EQ(glob(base, E + 18), 0.0f);
    ASSERT_EQ(glob(base, E + 43), 1.0f);
    ASSERT_EQ(glob(base, E + 44), 0.0f);
    // Relative, the USSR's own in the US's slot: Brezhnev's (none) at 6, the opponent's Containment at 18; my Purge
    // at 7; my CMC (none) at 11, the opponent's at 12; my Bear Trap at 14; the opponent's Death Squads at 23; my
    // attempts, both used, at 43 and 44.
    ASSERT_EQ(glob(rel, E + 6), 0.0f);
    ASSERT_EQ(glob(rel, E + 18), 1.0f);
    ASSERT_EQ(glob(rel, E + 7), 1.0f);
    ASSERT_EQ(glob(rel, E + 8), 0.0f);
    ASSERT_EQ(glob(rel, E + 11), 0.0f);
    ASSERT_EQ(glob(rel, E + 12), 1.0f);
    ASSERT_EQ(glob(rel, E + 14), 1.0f);
    ASSERT_EQ(glob(rel, E + 15), 0.0f);
    ASSERT_EQ(glob(rel, E + 22), 0.0f);
    ASSERT_EQ(glob(rel, E + 23), 1.0f);
    ASSERT_EQ(glob(rel, E + 43), 1.0f);
    ASSERT_EQ(glob(rel, E + 44), 1.0f);
    ASSERT_EQ(glob(rel, E + 0), 1.0f);                                        // NATO: no side, unchanged
    std::vector<size_t> moved;
    for (const auto& [a, b] : of::RELATIVE_EFFECT_PAIRS) { moved.push_back(E + a); moved.push_back(E + b); }
    moved.push_back(E + 43);
    moved.push_back(E + 44);
    const size_t g0 = 84 * ts::V23_BOARD_FEATURES + 110 * ts::card_slots::V23_FEATURES;
    for (size_t j = 0; j < base.size(); ++j) {
        if (j >= g0 && std::find(moved.begin(), moved.end(), j - g0) != moved.end()) continue;
        ASSERT_EQ(rel[j], base[j]);
    }
    // Over random games: the US view never moves; the USSR's is the base with its pairs swapped and its own attempts.
    for (uint64_t seed = 1; seed <= 20; ++seed) {
        random_game(seed, [&](const ts::GameState& t, const uint8_t*, uint16_t) {
            ASSERT_TRUE(obs(t, ts::Player::US, of::RELATIVE_EFFECTS) == obs(t, ts::Player::US, 0));
            auto want = obs(t, ts::Player::USSR, 0);
            float* g = want.data() + g0 + E;
            for (const auto& [a, b] : of::RELATIVE_EFFECT_PAIRS) std::swap(g[a], g[b]);
            g[43] = t.has_flag(ts::effect_bits::SPACE_USSR_ATTEMPT_1) ? 1.0f : 0.0f;
            g[44] = t.has_flag(ts::effect_bits::SPACE_USSR_ATTEMPT_2) ? 1.0f : 0.0f;
            ASSERT_TRUE(obs(t, ts::Player::USSR, of::RELATIVE_EFFECTS) == want);
        });
    }
}
