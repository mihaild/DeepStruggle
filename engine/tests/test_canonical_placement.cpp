// P43: canonical placement order (ts/canonical.hpp). A seat playing obs_features::CANONICAL_FIRST or
// CANONICAL_LAST places each placement run -- the setup's three, an Ops influence play, the verified event
// runs -- in non-decreasing rank of its order; a run that would strand points releases its floor for that
// choice. The boards a run can reach are exactly free order's, and a seat without either bit plays exactly
// as before.
#include "test_framework.hpp"
#include "ts/action_mask.hpp"
#include "ts/canonical.hpp"
#include "ts/constants.hpp"
#include "ts/engine.hpp"
#include "ts/map_data.hpp"
#include "ts/observation.hpp"

#include <array>
#include <cstring>
#include <set>
#include <vector>

namespace {

using ts::ActionView;
namespace fs = ts::flat_slots;
namespace of = ts::obs_features;
constexpr uint32_t ORDERS[] = {of::CANONICAL_FIRST, of::CANONICAL_LAST};

// A random permutation of 0..83 from `lcg`: a V3 run's given priority.
std::array<uint8_t, 84> random_priority(uint64_t& lcg) {
    std::array<uint8_t, 84> order{};
    for (uint8_t i = 0; i < 84; ++i) order[i] = i;
    for (int i = 83; i > 0; --i) {
        lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
        std::swap(order[i], order[(lcg >> 33) % static_cast<uint64_t>(i + 1)]);
    }
    std::array<uint8_t, 84> rank{};
    for (uint8_t r = 0; r < 84; ++r) rank[order[r]] = r;
    return rank;
}

ts::GameState to_ussr_setup(uint64_t seed, uint32_t ussr_features, uint32_t us_features) {
    ts::GameState s{};
    ts::Engine::init_game(s, seed);
    ts::seat_obs_features(s, ts::Player::USSR, ussr_features);
    ts::seat_obs_features(s, ts::Player::US, us_features);
    ts::Engine::auto_advance_step(s);
    ASSERT_TRUE(s.current_phase == ts::Phase::SETUP);
    ASSERT_TRUE(s.ctx().decision_player == ts::Player::USSR);
    return s;
}

bool node_legal(const ts::GameState& s, uint8_t c) {
    uint8_t mask[ts::FLAT_ACTION_SPACE_SIZE];
    ts::Engine::get_flat_action_mask(s, mask, ActionView::E4);
    return mask[fs::NODE + c] != 0;
}

uint8_t id(const char* name) { return ts::MapData::get_country_by_name(name); }

// The run pending at `s`: its tag and frame depth.
struct Run {
    uint16_t tag;
    uint8_t depth;
};

bool in_run(const ts::GameState& s, const Run& r) {
    return !ts::Engine::is_terminal(s) && s.ctx().decision_type == ts::DecisionType::POINT_NODE &&
           s.ctx_stack_depth == r.depth && ts::canonical::run_tag(s) == r.tag;
}

std::vector<uint8_t> outcome_key(const ts::GameState& s) {
    std::vector<uint8_t> k;
    for (uint8_t i = 0; i < 84; ++i) {
        k.push_back(s.countries[i].us_influence);
        k.push_back(s.countries[i].ussr_influence);
    }
    k.push_back(static_cast<uint8_t>(s.victory_points + 100));
    k.push_back(s.defcon);
    k.push_back(static_cast<uint8_t>(s.current_phase));
    k.push_back(static_cast<uint8_t>(s.ctx().decision_type));
    k.push_back(static_cast<uint8_t>(s.ctx().decision_player));
    k.push_back(s.ctx_stack_depth);
    return k;
}

std::vector<uint8_t> node_key(const ts::GameState& s) {
    std::vector<uint8_t> k = outcome_key(s);
    k.push_back(s.ctx().remaining_steps);
    k.push_back(s.ctx().pending_ops_value);
    k.push_back(s.ctx().canon_floor);
    k.push_back(s.ctx().canon_left);
    return k;
}

// Every outcome of the run pending at `start`, depth-first over its legal placements (and CONFIRM_DONE),
// memoised on the position. False if the search outgrew `cap` nodes. Every position on the way must offer a
// legal action.
bool outcomes(const ts::GameState& start, std::set<std::vector<uint8_t>>& out, size_t cap) {
    const Run run{ts::canonical::run_tag(start), start.ctx_stack_depth};
    std::set<std::vector<uint8_t>> seen;
    std::vector<ts::GameState> stack{start};
    while (!stack.empty()) {
        ts::GameState s = stack.back();
        stack.pop_back();
        if (!seen.insert(node_key(s)).second) continue;
        if (seen.size() > cap) return false;
        uint8_t mask[ts::FLAT_ACTION_SPACE_SIZE];
        ts::Engine::get_flat_action_mask(s, mask, ActionView::E4);
        bool any = false;
        for (int a = 0; a < fs::BASE_SIZE; ++a) {
            if (!mask[a]) continue;
            any = true;
            ts::GameState t = s;
            ASSERT_TRUE(ts::Engine::step_flat(t, static_cast<uint16_t>(a), false));
            if (in_run(t, run)) stack.push_back(t);
            else out.insert(outcome_key(t));
        }
        ASSERT_TRUE(any);
    }
    return true;
}

uint64_t multisets(uint64_t n, uint64_t k) {   // C(n + k - 1, k)
    uint64_t r = 1;
    for (uint64_t i = 1; i <= k; ++i) r = r * (n + k - i) / i;
    return r;
}

// Every canonical sequence of the current setup run: the number of sequences and the boards they reach.
void enumerate_setup(const ts::GameState& s, ts::Player side, size_t& leaves, std::set<std::vector<uint8_t>>& boards,
                     uint8_t stage) {
    const bool same_run = s.current_phase == ts::Phase::SETUP && s.ctx().decision_player == side &&
                          s.ctx().pending_ops_value == stage;
    if (!same_run) {
        ++leaves;
        std::vector<uint8_t> b(84);
        for (uint8_t i = 0; i < 84; ++i) b[i] = s.countries[i].get_influence(side);
        boards.insert(b);
        return;
    }
    uint8_t mask[ts::FLAT_ACTION_SPACE_SIZE];
    ts::Engine::get_flat_action_mask(s, mask, ActionView::E4);
    for (uint8_t c = 0; c < 84; ++c) {
        if (!mask[fs::NODE + c]) continue;
        ts::GameState next = s;
        ASSERT_TRUE(ts::Engine::step_flat(next, static_cast<uint16_t>(fs::NODE + c), false));
        enumerate_setup(next, side, leaves, boards, stage);
    }
}

}  // namespace

TEST(CanonicalPlacement, RankIsAPermutationMostImportantLast) {
    std::array<int, 84> seen{};
    for (uint8_t i = 0; i < 84; ++i) seen[ts::MapData::placement_rank(i)]++;
    for (int k : seen) ASSERT_EQ(k, 1);
    ASSERT_EQ(ts::MapData::placement_rank(id("Poland")), 83);
    ASSERT_EQ(ts::MapData::placement_rank(id("East Germany")), 82);
    ASSERT_EQ(ts::MapData::placement_rank(id("West Germany")), 81);
    ASSERT_EQ(ts::MapData::placement_rank(id("France")), 80);
    ASSERT_EQ(ts::MapData::placement_rank(id("Italy")), 79);
    ASSERT_EQ(ts::canonical::rank_in(ts::GameState{}, of::CANONICAL_FIRST, id("Poland")), 0);
    for (uint8_t a = 0; a < 84; ++a) {
        for (uint8_t b = 0; b < 84; ++b) {
            const auto& ca = ts::MapData::get_country(a);
            const auto& cb = ts::MapData::get_country(b);
            if (ca.region == ts::Region::MIDDLE_EAST && cb.region == ts::Region::EUROPE)
                ASSERT_TRUE(ts::MapData::placement_rank(a) < ts::MapData::placement_rank(b));
            if (ca.region == cb.region && !ca.battleground && cb.battleground)
                ASSERT_TRUE(ts::MapData::placement_rank(a) < ts::MapData::placement_rank(b));
        }
    }
}

TEST(CanonicalPlacement, WithoutABitNothingChanges) {
    // The USSR's seat plays no canonical bit; the US seat's must not touch the USSR's masks, and the run's
    // floor is never written.
    ts::GameState s = to_ussr_setup(7, 0, of::CANONICAL_FIRST);
    ASSERT_TRUE(ts::Engine::step_flat(s, static_cast<uint16_t>(fs::NODE + id("Poland")), true));
    ASSERT_EQ(s.ctx().canon_floor, 0);
    ASSERT_TRUE(node_legal(s, id("Austria")));
    ASSERT_TRUE(ts::Engine::step_flat(s, static_cast<uint16_t>(fs::NODE + id("Austria")), true));
    ASSERT_EQ(s.ctx().canon_floor, 0);
}

TEST(CanonicalPlacement, ASetupRunNeverGoesBackDownItsOrder) {
    for (uint32_t order : ORDERS) {
        ts::GameState s = to_ussr_setup(7, order, order);
        const uint8_t eg = id("East Germany");
        ASSERT_TRUE(ts::Engine::step_flat(s, static_cast<uint16_t>(fs::NODE + eg), true));
        for (uint8_t c = 0; c < 84; ++c) {
            if (!ts::MapData::get_country(c).in_eastern_europe) continue;
            const bool above = ts::canonical::rank_in(s, order, c) >= ts::canonical::rank_in(s, order, eg);
            ASSERT_EQ(node_legal(s, c), above);
            if (!above) {
                ts::GameState t = s;
                ASSERT_TRUE(!ts::Engine::step_flat(t, static_cast<uint16_t>(fs::NODE + c), false));
            }
        }
    }
}

TEST(CanonicalPlacement, EverySetupBoardHasExactlyOneSequence) {
    for (uint32_t order : ORDERS) {
        ts::GameState s = to_ussr_setup(11, order, order);
        uint64_t n = 0, m = 0;
        for (uint8_t c = 0; c < 84; ++c) {
            n += ts::MapData::get_country(c).in_eastern_europe ? 1 : 0;
            m += ts::MapData::get_country(c).in_western_europe ? 1 : 0;
        }
        size_t leaves = 0;
        std::set<std::vector<uint8_t>> boards;
        enumerate_setup(s, ts::Player::USSR, leaves, boards, 0);
        ASSERT_EQ(boards.size(), static_cast<size_t>(multisets(n, 6)));
        ASSERT_EQ(leaves, boards.size());

        for (int i = 0; i < 6; ++i) {   // the USSR's six in Poland (some may be forced, played by auto-advance)
            if (s.ctx().decision_player != ts::Player::USSR) break;
            ASSERT_TRUE(ts::Engine::step_flat(s, static_cast<uint16_t>(fs::NODE + id("Poland")), true));
        }
        ASSERT_TRUE(s.ctx().decision_player == ts::Player::US);
        leaves = 0;
        boards.clear();
        enumerate_setup(s, ts::Player::US, leaves, boards, 0);
        ASSERT_EQ(boards.size(), static_cast<size_t>(multisets(m, 7)));
        ASSERT_EQ(leaves, boards.size());
    }
}

// The heart of it: on real Ops influence plays and covered event runs from random games, the outcomes a
// canonical seat can reach (with the dead-end release) are exactly free order's, for both orders.
TEST(CanonicalPlacement, EveryRunReachesExactlyFreeOrdersBoards) {
    size_t checked = 0, too_large = 0, ops = 0, events = 0;
    for (uint64_t seed = 1; seed <= 120; ++seed) {
        ts::GameState s{};
        ts::Engine::init_game(s, 500 + seed);
        uint64_t lcg = seed * 2654435761ULL;
        uint8_t mask[ts::FLAT_ACTION_SPACE_SIZE];
        uint16_t last_tag = 0;
        for (int step = 0; step < 4000 && !ts::Engine::is_terminal(s); ++step) {
            ts::Engine::auto_advance_step(s);
            if (ts::Engine::is_terminal(s)) break;
            const auto& c = s.ctx();
            const bool covered_kind = c.decision_type == ts::DecisionType::POINT_NODE &&
                                      s.current_phase != ts::Phase::SETUP && c.remaining_steps >= 2 &&
                                      (c.is_ops_influence_play() ||
                                       (c.resolving_card != 0 && ts::canonical::event_covered(c.resolving_card)));
            const uint16_t tag = c.decision_type == ts::DecisionType::POINT_NODE ? ts::canonical::run_tag(s) : 0;
            if (covered_kind && tag != last_tag) {
                std::set<std::vector<uint8_t>> free_out;
                ts::GameState f = s;
                ts::seat_both(f, 0);
                if (!outcomes(f, free_out, 60000)) {
                    ++too_large;
                } else {
                    for (uint32_t order : ORDERS) {
                        ts::GameState k = s;
                        ts::seat_both(k, order);
                        std::set<std::vector<uint8_t>> canon_out;
                        ASSERT_TRUE(outcomes(k, canon_out, 600000));
                        ASSERT_TRUE(canon_out == free_out);
                    }
                    ++checked;
                    (c.is_ops_influence_play() ? ops : events)++;
                }
            }
            last_tag = tag;
            ts::Engine::get_flat_action_mask(s, mask, ActionView::E4);
            int legal[ts::FLAT_ACTION_SPACE_SIZE];
            int n = 0;
            for (int i = 0; i < fs::BASE_SIZE; ++i)
                if (mask[i]) legal[n++] = i;
            ASSERT_TRUE(n > 0);
            lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
            ASSERT_TRUE(ts::Engine::step_flat(s, static_cast<uint16_t>(legal[(lcg >> 33) % n]), false));
        }
    }
    ASSERT_TRUE(ops >= 200);
    ASSERT_TRUE(events >= 30);
    ASSERT_TRUE(too_large * 20 < checked + too_large);
}

TEST(CanonicalPlacement, LearnedOrderIsTheGivenPriority) {
    ts::GameState s = to_ussr_setup(7, of::CANONICAL_LEARNED, of::CANONICAL_LEARNED);
    ASSERT_TRUE(ts::canonical::priority_wanted(s));
    // Without its priority, a choice with more than one target is refused.
    {
        ts::GameState t = s;
        ASSERT_TRUE(!ts::Engine::step_flat(t, static_cast<uint16_t>(fs::NODE + id("Poland")), false));
    }
    uint64_t lcg = 99;
    const auto prio = random_priority(lcg);
    ASSERT_TRUE(ts::canonical::set_run_priority(s, prio.data()));
    ASSERT_TRUE(!ts::canonical::priority_wanted(s));
    const uint8_t eg = id("East Germany");
    ASSERT_TRUE(ts::Engine::step_flat(s, static_cast<uint16_t>(fs::NODE + eg), true));
    for (uint8_t c = 0; c < 84; ++c) {
        if (!ts::MapData::get_country(c).in_eastern_europe) continue;
        ASSERT_EQ(node_legal(s, c), prio[c] >= prio[eg] || s.ctx().decision_player != ts::Player::USSR);
    }
    // The priority ends with its run: the US's setup run wants its own.
    while (s.current_phase == ts::Phase::SETUP && s.ctx().decision_player == ts::Player::USSR) {
        uint8_t mask[ts::FLAT_ACTION_SPACE_SIZE];
        ts::Engine::get_flat_action_mask(s, mask, ActionView::E4);
        int a = -1;
        for (int i = 0; i < 84 && a < 0; ++i) if (mask[fs::NODE + i]) a = fs::NODE + i;
        ASSERT_TRUE(ts::Engine::step_flat(s, static_cast<uint16_t>(a), true));
    }
    ASSERT_TRUE(s.ctx().decision_player == ts::Player::US);
    ASSERT_TRUE(ts::canonical::priority_wanted(s));
    ASSERT_EQ(s.canon_prio_tag, 0);
}

TEST(CanonicalPlacement, LearnedRunsReachExactlyFreeOrdersBoards) {
    size_t checked = 0;
    for (uint64_t seed = 1; seed <= 60; ++seed) {
        ts::GameState s{};
        ts::Engine::init_game(s, 900 + seed);
        uint64_t lcg = seed * 2654435761ULL;
        uint8_t mask[ts::FLAT_ACTION_SPACE_SIZE];
        uint16_t last_tag = 0;
        for (int step = 0; step < 4000 && !ts::Engine::is_terminal(s); ++step) {
            ts::Engine::auto_advance_step(s);
            if (ts::Engine::is_terminal(s)) break;
            const auto& c = s.ctx();
            const bool covered_kind = c.decision_type == ts::DecisionType::POINT_NODE && c.remaining_steps >= 2 &&
                                      (s.current_phase == ts::Phase::SETUP || c.is_ops_influence_play() ||
                                       (c.resolving_card != 0 && ts::canonical::event_covered(c.resolving_card)));
            const uint16_t tag = c.decision_type == ts::DecisionType::POINT_NODE ? ts::canonical::run_tag(s) : 0;
            if (covered_kind && tag != last_tag) {
                std::set<std::vector<uint8_t>> free_out, canon_out;
                ts::GameState f = s;
                ts::seat_both(f, 0);
                if (outcomes(f, free_out, 60000)) {
                    ts::GameState k = s;
                    ts::seat_both(k, of::CANONICAL_LEARNED);
                    const auto prio = random_priority(lcg);
                    ASSERT_TRUE(ts::canonical::set_run_priority(k, prio.data()));
                    ASSERT_TRUE(outcomes(k, canon_out, 600000));
                    ASSERT_TRUE(canon_out == free_out);
                    ++checked;
                }
            }
            last_tag = tag;
            ts::Engine::get_flat_action_mask(s, mask, ActionView::E4);
            int legal[ts::FLAT_ACTION_SPACE_SIZE];
            int n = 0;
            for (int i = 0; i < fs::BASE_SIZE; ++i)
                if (mask[i]) legal[n++] = i;
            ASSERT_TRUE(n > 0);
            lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
            ASSERT_TRUE(ts::Engine::step_flat(s, static_cast<uint16_t>(legal[(lcg >> 33) % n]), false));
        }
    }
    ASSERT_TRUE(checked >= 100);
}

TEST(CanonicalPlacement, FullGamesPlayWithALearnedOrderOnBothSeats) {
    for (uint64_t seed = 1; seed <= 40; ++seed) {
        ts::GameState s{};
        ts::Engine::init_game(s, seed);
        ts::seat_both(s, of::CANONICAL_LEARNED);
        uint64_t lcg = seed * 2654435761ULL;
        uint8_t mask[ts::FLAT_ACTION_SPACE_SIZE];
        for (int step = 0; step < 4000 && !ts::Engine::is_terminal(s); ++step) {
            if (ts::canonical::priority_wanted(s)) {
                const auto prio = random_priority(lcg);
                ASSERT_TRUE(ts::canonical::set_run_priority(s, prio.data()));
            }
            ts::Engine::get_flat_action_mask(s, mask, ActionView::E4);
            int legal[ts::FLAT_ACTION_SPACE_SIZE];
            int n = 0;
            for (int i = 0; i < fs::BASE_SIZE; ++i)
                if (mask[i]) legal[n++] = i;
            ASSERT_TRUE(n > 0);
            lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
            ASSERT_TRUE(ts::Engine::step_flat(s, static_cast<uint16_t>(legal[(lcg >> 33) % n]), false));
            // Forced choices are played here one at a time; a run that wants its priority is answered first.
        }
        ASSERT_TRUE(ts::Engine::is_terminal(s));
    }
}

TEST(CanonicalPlacement, PlacementLegalShowsTheMaskInTheDecidersPlaceableSlot) {
    // USSR seated with CANONICAL_LAST; after East Germany, the countries below it in the order are not legal, and
    // a model reading PLACEMENT_LEGAL sees exactly that in slot 19, while the base observation and the
    // opponent's view keep bare eligibility.
    ts::GameState s = to_ussr_setup(7, of::CANONICAL_LAST, of::CANONICAL_LAST);
    ASSERT_TRUE(ts::Engine::step_flat(s, static_cast<uint16_t>(fs::NODE + id("East Germany")), true));
    std::vector<float> plain(ts::OBS_SIZE_V23), with(ts::OBS_SIZE_V23), opp(ts::OBS_SIZE_V23);
    ts::extract_observation_features(s, ts::Player::USSR, 0, plain.data());
    ts::extract_observation_features(s, ts::Player::USSR, of::PLACEMENT_LEGAL, with.data());
    ts::extract_observation_features(s, ts::Player::US, of::PLACEMENT_LEGAL, opp.data());
    std::vector<float> opp_plain(ts::OBS_SIZE_V23);
    ts::extract_observation_features(s, ts::Player::US, 0, opp_plain.data());
    uint8_t mask[ts::FLAT_ACTION_SPACE_SIZE];
    ts::Engine::get_flat_action_mask(s, mask, ActionView::E4);
    size_t differ = 0;
    for (uint8_t c = 0; c < 84; ++c) {
        const size_t k = static_cast<size_t>(c) * ts::V23_BOARD_FEATURES + of::BOARD_MY_PLACEABLE;
        ASSERT_EQ(with[k], mask[fs::NODE + c] ? 1.0f : 0.0f);
        differ += with[k] != plain[k] ? 1 : 0;
    }
    ASSERT_TRUE(differ > 0);                                  // Austria, Finland, ... left behind
    for (size_t k = 0; k < with.size(); ++k) {
        if (k % ts::V23_BOARD_FEATURES == of::BOARD_MY_PLACEABLE && k < 84 * ts::V23_BOARD_FEATURES) continue;
        ASSERT_EQ(with[k], plain[k]);                         // nothing else moves
    }
    ASSERT_TRUE(opp == opp_plain);                            // not the decider: unchanged
}

TEST(CanonicalPlacement, FullGamesPlayWithEitherOrderOnBothSeats) {
    for (uint32_t order : ORDERS) {
        for (uint64_t seed = 1; seed <= 40; ++seed) {
            ts::GameState s{};
            ts::Engine::init_game(s, seed);
            ts::seat_both(s, order);
            uint64_t lcg = seed * 2654435761ULL;
            uint8_t mask[ts::FLAT_ACTION_SPACE_SIZE];
            for (int step = 0; step < 4000 && !ts::Engine::is_terminal(s); ++step) {
                ts::Engine::auto_advance_step(s);
                if (ts::Engine::is_terminal(s)) break;
                ts::Engine::get_flat_action_mask(s, mask, ActionView::E4);
                int legal[ts::FLAT_ACTION_SPACE_SIZE];
                int n = 0;
                for (int i = 0; i < fs::BASE_SIZE; ++i)
                    if (mask[i]) legal[n++] = i;
                ASSERT_TRUE(n > 0);
                lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
                ASSERT_TRUE(ts::Engine::step_flat(s, static_cast<uint16_t>(legal[(lcg >> 33) % n]), false));
            }
            ASSERT_TRUE(ts::Engine::is_terminal(s));
        }
    }
}
