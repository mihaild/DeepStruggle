// Fuzz test for Influence placement (rule 4.1): random boards, random cards, random
// countries, always spending an Ops allotment fully rather than declining early. Checks,
// independently of the engine's own bookkeeping wherever practical:
//
//   1. Per-country cost: 2 Ops per point while the opponent still controls the country,
//      1 Op per point once that control is broken (Operations::get_influence_cost's
//      contract), verified against an independent reimplementation of the TS control
//      rule (influence advantage >= stability), not by calling get_influence_cost itself.
//   2. The China Card's Asia bonus (base 4 -> 5) and Vietnam Revolts' Southeast Asia
//      bonus (base -> +1) apply exactly when ALL Influence added under that Op stayed
//      inside the qualifying region, and are forfeited the moment it doesn't -- checked
//      via a same-starting-state A/B: one run confined to the region, one left free.
//   3. Legality: every country touched must be adjacent to the acting superpower, have
//      had that player's influence before the operation started, or be adjacent to a
//      country that did -- evaluated against a snapshot taken before the first
//      placement, so a country reached only via influence gained mid-operation does not
//      count (no chaining credit within a single Op).
//
// If this test fails, it means the engine's cost/legality bookkeeping disagrees with an
// independent reimplementation of the printed rule for some reachable board state --
// report the failing seed rather than loosening the check.

#include "test_framework.hpp"
#include "ts/engine.hpp"
#include "ts/state_machine.hpp"
#include "ts/action_mask.hpp"
#include "ts/ops.hpp"
#include "ts/map_data.hpp"
#include "ts/card_data.hpp"
#include "ts/constants.hpp"
#include "ts/prng.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using namespace ts;

uint64_t g_fuzz_rng;

uint32_t frand(uint32_t bound) noexcept {
    return Prng::random_index(g_fuzz_rng, bound);
}

// Independent reimplementation of TS rule 1.5 ("controls... Influence exceeds that of
// his opponent by at least the country's Stability number"), used to predict placement
// cost without calling Operations::get_influence_cost.
bool opponent_controls(const GameState& state, Player p, uint8_t cid) noexcept {
    Player opp = get_opponent(p);
    const auto& c = state.countries[cid];
    int16_t my_inf = static_cast<int16_t>(c.get_influence(p));
    int16_t opp_inf = static_cast<int16_t>(c.get_influence(opp));
    int16_t stab = MapData::get_country(cid).stability;
    return (opp_inf - my_inf) >= stab;
}

struct InfluenceSnapshot {
    std::array<bool, 84> had_influence{};
};

InfluenceSnapshot snapshot_influence(const GameState& state, Player p) {
    InfluenceSnapshot snap;
    for (uint8_t i = 0; i < 84; ++i) {
        snap.had_influence[i] = state.countries[i].get_influence(p) > 0;
    }
    return snap;
}

const char* player_name(Player p) { return p == Player::US ? "US" : "USSR"; }

// Checks the legality claim the user specified, using ONLY the pre-operation snapshot:
// no credit for influence gained earlier in the SAME operation (no chaining).
void assert_adjacency_legal(const InfluenceSnapshot& snap, uint8_t cid, Player p,
                             int iter, const char* scenario) {
    const auto& c = MapData::get_country(cid);
    bool superpower_adjacent = (c.superpower_adjacent == p);
    bool had_influence = snap.had_influence[cid];
    bool adjacent_to_pre_existing = false;
    for (uint8_t n = 0; n < c.num_neighbors; ++n) {
        if (snap.had_influence[c.neighbors[n]]) { adjacent_to_pre_existing = true; break; }
    }
    bool legal = superpower_adjacent || had_influence || adjacent_to_pre_existing;
    if (!legal) {
        std::fprintf(stderr,
            "[InfluenceFuzz] iter %d [%s]: engine allowed placing in %s (id %d) for %s -- "
            "not superpower-adjacent, no pre-operation influence there, and not adjacent "
            "to any country that had pre-operation influence.\n",
            iter, scenario, std::string(c.name).c_str(), static_cast<int>(cid), player_name(p));
    }
    ASSERT_TRUE(legal);
}

enum class RegionFilter { NONE, ASIA_ONLY, SOUTHEAST_ASIA_ONLY };

struct DriveResult {
    uint32_t predicted_total_cost = 0;
    uint32_t placements = 0;
    bool left_asia = false;
    bool left_southeast_asia = false;
    bool ran_out_of_targets = false;
};

// Drives one Influence Op to completion (or until no legal target remains), always
// spending -- never declining early ("use the card fully"). Picks uniformly among
// currently legal targets, optionally restricted to a qualifying region (stopping
// rather than leaving it, for the "confined" half of an A/B pair).
DriveResult drive_influence_operation(GameState& state, Player p, RegionFilter filter,
                                       int iter, const char* scenario) {
    DriveResult result;
    InfluenceSnapshot snap = snapshot_influence(state, p);

    while (state.ctx().decision_type == DecisionType::POINT_NODE &&
           state.ctx().op_mode == OpMode::INFLUENCE &&
           state.ctx().resolving_card == 0 &&
           state.ctx().remaining_steps > 0) {

        uint8_t mask_212[ts::FLAT_ACTION_SPACE_SIZE];
        ActionMask::generate_flat_mask_212(state, mask_212);

        std::vector<uint8_t> candidates;
        for (uint8_t cid = 0; cid < 84; ++cid) {
            if (!mask_212[ts::flat_slots::NODE + cid]) continue;
            const auto& c = MapData::get_country(cid);
            if (filter == RegionFilter::ASIA_ONLY && c.region != Region::ASIA) continue;
            if (filter == RegionFilter::SOUTHEAST_ASIA_ONLY && !c.in_southeast_asia) continue;
            candidates.push_back(cid);
        }
        if (candidates.empty()) {
            result.ran_out_of_targets = true;
            break;
        }

        uint8_t cid = candidates[frand(static_cast<uint32_t>(candidates.size()))];
        assert_adjacency_legal(snap, cid, p, iter, scenario);

        uint8_t predicted_cost = opponent_controls(state, p, cid) ? 2 : 1;
        const auto& c_info = MapData::get_country(cid);
        if (c_info.region != Region::ASIA) result.left_asia = true;
        if (!c_info.in_southeast_asia) result.left_southeast_asia = true;

        bool ok = StateMachine::step(state, MicroAction{DecisionType::POINT_NODE, cid, 0, 0});
        if (!ok) {
            std::fprintf(stderr,
                "[InfluenceFuzz] iter %d [%s]: engine's own mask offered %s (id %d) for %s "
                "as legal but StateMachine::step rejected it.\n",
                iter, scenario, std::string(c_info.name).c_str(), static_cast<int>(cid), player_name(p));
        }
        ASSERT_TRUE(ok);

        result.predicted_total_cost += predicted_cost;
        result.placements++;
    }
    return result;
}

// Randomizes a plausible mid-game board: each country independently gets a small
// chance of some pre-existing influence for either side, producing a mix of neutral,
// single-controlled, and contested countries.
void randomize_board(GameState& state) {
    for (uint8_t cid = 0; cid < 84; ++cid) {
        uint8_t us = (frand(100) < 25) ? static_cast<uint8_t>(1 + frand(3)) : 0;
        uint8_t ussr = (frand(100) < 25) ? static_cast<uint8_t>(1 + frand(3)) : 0;
        state.countries[cid].us_influence = us;
        state.countries[cid].ussr_influence = ussr;
    }
}

// A random real card with a nonzero Ops value, excluding the China Card (which is
// handled as its own scenario since its bonus is unconditional on side).
uint8_t random_ops_card() {
    uint8_t card_id;
    do {
        card_id = static_cast<uint8_t>(1 + frand(110));
    } while (CardData::get_card(card_id).ops == 0 || card_id == card_ids::THE_CHINA_CARD);
    return card_id;
}

// Enters SELECT_OP_MODE -> INFLUENCE directly (skipping SELECT_CARD/SELECT_PLAY_MODE,
// which are covered by other tests) so the fuzz driver controls exactly which card and
// player are in play. Mirrors state_machine.cpp's own SELECT_PLAY_MODE->OPS and
// SELECT_OP_MODE->INFLUENCE transitions.
void enter_influence_op(GameState& state, Player p, uint8_t card_id) {
    state.current_phase = Phase::ACTION_ROUND;
    state.phasing_player = p;
    state.ctx() = DecisionContext{};
    state.ctx().decision_player = p;
    state.ctx().decision_type = DecisionType::SELECT_OP_MODE;
    state.ctx().pending_op_card = card_id;
    state.ctx().pending_ops_value = Operations::grant_ops_for_card(state, card_id, p);

    bool ok = StateMachine::step(state, MicroAction{DecisionType::SELECT_OP_MODE,
                                                     static_cast<uint8_t>(OpMode::INFLUENCE), 0, 0});
    ASSERT_TRUE(ok);
    ASSERT_EQ(static_cast<int>(state.ctx().decision_type), static_cast<int>(DecisionType::POINT_NODE));
    ASSERT_EQ(static_cast<int>(state.ctx().op_mode), static_cast<int>(OpMode::INFLUENCE));
}

} // namespace

// The general cost/legality invariants, across many random boards, players, and cards,
// with no card-specific region bonus in play.
TEST(InfluencePlacementFuzzTest, BaselineCostAndAdjacencyHoldAcrossRandomContexts) {
    g_fuzz_rng = 0x494E464C55454E43ULL; // fixed per-test seed; independent of test execution order
    constexpr int kIterations = 600;
    int total_placements = 0;
    int exhausted_naturally = 0;

    for (int iter = 0; iter < kIterations; ++iter) {
        GameState state{};
        Engine::init_game(state, Prng::next_u64(g_fuzz_rng));
        randomize_board(state);

        Player p = (frand(2) == 0) ? Player::US : Player::USSR;
        uint8_t card_id = random_ops_card();
        uint8_t plain = Operations::get_effective_ops(state, card_id, p, Region::NONE_REGION);

        enter_influence_op(state, p, card_id);
        ASSERT_EQ(static_cast<int>(state.ctx().pending_ops_value), static_cast<int>(plain));

        DriveResult res = drive_influence_operation(state, p, RegionFilter::NONE, iter, "baseline");
        total_placements += static_cast<int>(res.placements);

        if (!res.ran_out_of_targets) {
            exhausted_naturally++;
            if (res.predicted_total_cost != plain) {
                std::fprintf(stderr,
                    "[InfluenceFuzz] iter %d [baseline]: %s spent %u predicted Ops but was "
                    "granted %u for card #%d (ops %d), and never ran out of legal targets.\n",
                    iter, player_name(p), res.predicted_total_cost, plain,
                    static_cast<int>(card_id), static_cast<int>(CardData::get_card(card_id).ops));
            }
            ASSERT_EQ(res.predicted_total_cost, plain);
        } else {
            ASSERT_TRUE(res.predicted_total_cost <= plain);
        }
    }

    std::printf("[InfluenceFuzz] baseline: %d iterations, %d placements, %d exhausted naturally\n",
                kIterations, total_placements, exhausted_naturally);
}

// The China Card: +1 Ops (4 -> 5) when, and only when, every Influence Point placed
// under it lands in Asia. A/B pair from an identical starting board.
TEST(InfluencePlacementFuzzTest, ChinaCardAsiaBonusAppliesOnlyWhenFullyConfined) {
    g_fuzz_rng = 0x4348494E41415349ULL; // fixed per-test seed
    constexpr int kIterations = 300;
    int confirmed_five = 0;
    int confirmed_forfeit = 0;

    for (int iter = 0; iter < kIterations; ++iter) {
        GameState base{};
        Engine::init_game(base, Prng::next_u64(g_fuzz_rng));
        randomize_board(base);
        Player p = (frand(2) == 0) ? Player::US : Player::USSR;
        uint8_t plain = Operations::get_effective_ops(base, card_ids::THE_CHINA_CARD, p, Region::NONE_REGION);
        ASSERT_EQ(static_cast<int>(plain), 4); // printed Ops value; see engine/src/card_data.cpp

        GameState confined = base;
        enter_influence_op(confined, p, card_ids::THE_CHINA_CARD);
        DriveResult r_confined = drive_influence_operation(confined, p, RegionFilter::ASIA_ONLY, iter, "china-confined");
        ASSERT_FALSE(r_confined.left_asia);

        GameState leaked = base;
        enter_influence_op(leaked, p, card_ids::THE_CHINA_CARD);
        DriveResult r_leaked = drive_influence_operation(leaked, p, RegionFilter::NONE, iter, "china-leaked");

        if (!r_confined.ran_out_of_targets) {
            if (r_confined.predicted_total_cost != 5) {
                std::fprintf(stderr,
                    "[InfluenceFuzz] iter %d [china-confined]: all placements stayed in Asia "
                    "but total cost was %u, not 5 (4 base + 1 Asia bonus).\n",
                    iter, r_confined.predicted_total_cost);
            }
            ASSERT_EQ(r_confined.predicted_total_cost, 5u);
            confirmed_five++;
        }

        if (r_leaked.left_asia && !r_leaked.ran_out_of_targets) {
            if (r_leaked.predicted_total_cost != plain) {
                std::fprintf(stderr,
                    "[InfluenceFuzz] iter %d [china-leaked]: left Asia but total cost was %u, "
                    "not the unbonused %u.\n",
                    iter, r_leaked.predicted_total_cost, plain);
            }
            ASSERT_EQ(r_leaked.predicted_total_cost, plain);
            confirmed_forfeit++;
        }
    }

    std::printf("[InfluenceFuzz] china: %d iterations, %d confirmed =5 confined, %d confirmed forfeit\n",
                kIterations, confirmed_five, confirmed_forfeit);
    ASSERT_GT(confirmed_five, 0);
    ASSERT_GT(confirmed_forfeit, 0);
}

// Vietnam Revolts: USSR gets +1 Ops when, and only when, every Influence Point placed
// under that card lands in Southeast Asia. Same A/B design as the China Card case.
TEST(InfluencePlacementFuzzTest, VietnamRevoltsSoutheastAsiaBonusAppliesOnlyWhenFullyConfined) {
    g_fuzz_rng = 0x5669746E616D5265ULL; // fixed per-test seed
    constexpr int kIterations = 300;
    int confirmed_bonus = 0;
    int confirmed_forfeit = 0;

    for (int iter = 0; iter < kIterations; ++iter) {
        GameState base{};
        Engine::init_game(base, Prng::next_u64(g_fuzz_rng));
        randomize_board(base);
        base.set_flag(effect_bits::VIETNAM_REVOLTS_ACTIVE);
        Player p = Player::USSR; // Vietnam Revolts only benefits the USSR
        uint8_t card_id = random_ops_card();
        uint8_t plain = Operations::get_effective_ops(base, card_id, p, Region::NONE_REGION);

        GameState confined = base;
        enter_influence_op(confined, p, card_id);
        DriveResult r_confined = drive_influence_operation(confined, p, RegionFilter::SOUTHEAST_ASIA_ONLY, iter, "vietnam-confined");
        ASSERT_FALSE(r_confined.left_southeast_asia);

        GameState leaked = base;
        enter_influence_op(leaked, p, card_id);
        DriveResult r_leaked = drive_influence_operation(leaked, p, RegionFilter::NONE, iter, "vietnam-leaked");

        if (!r_confined.ran_out_of_targets) {
            uint8_t expected = static_cast<uint8_t>(plain + 1);
            if (r_confined.predicted_total_cost != expected) {
                std::fprintf(stderr,
                    "[InfluenceFuzz] iter %d [vietnam-confined]: all placements stayed in "
                    "Southeast Asia but total cost was %u, not %u (card #%d base %u + 1).\n",
                    iter, r_confined.predicted_total_cost, expected,
                    static_cast<int>(card_id), plain);
            }
            ASSERT_EQ(r_confined.predicted_total_cost, expected);
            confirmed_bonus++;
        }

        if (r_leaked.left_southeast_asia && !r_leaked.ran_out_of_targets) {
            if (r_leaked.predicted_total_cost != plain) {
                std::fprintf(stderr,
                    "[InfluenceFuzz] iter %d [vietnam-leaked]: left Southeast Asia but total "
                    "cost was %u, not the unbonused %u.\n",
                    iter, r_leaked.predicted_total_cost, plain);
            }
            ASSERT_EQ(r_leaked.predicted_total_cost, plain);
            confirmed_forfeit++;
        }
    }

    std::printf("[InfluenceFuzz] vietnam revolts: %d iterations, %d confirmed bonus, %d confirmed forfeit\n",
                kIterations, confirmed_bonus, confirmed_forfeit);
    ASSERT_GT(confirmed_bonus, 0);
    ASSERT_GT(confirmed_forfeit, 0);
}

// Both bonuses stacked: China Card, USSR, Vietnam Revolts active, confined to Southeast
// Asia -- 4 base + 1 (all Ops in Asia) + 1 (all Ops in Southeast Asia) = 6. This is the
// exact case get_effective_ops's own comment calls out as previously under-clamped to 5.
TEST(InfluencePlacementFuzzTest, ChinaCardAndVietnamRevoltsStackToSix) {
    g_fuzz_rng = 0x5374616B53697853ULL; // fixed per-test seed
    constexpr int kIterations = 150;
    int confirmed_six = 0;

    for (int iter = 0; iter < kIterations; ++iter) {
        GameState state{};
        Engine::init_game(state, Prng::next_u64(g_fuzz_rng));
        randomize_board(state);
        state.set_flag(effect_bits::VIETNAM_REVOLTS_ACTIVE);
        Player p = Player::USSR;

        enter_influence_op(state, p, card_ids::THE_CHINA_CARD);
        DriveResult res = drive_influence_operation(state, p, RegionFilter::SOUTHEAST_ASIA_ONLY, iter, "china+vietnam");
        ASSERT_FALSE(res.left_southeast_asia);

        if (!res.ran_out_of_targets) {
            if (res.predicted_total_cost != 6) {
                std::fprintf(stderr,
                    "[InfluenceFuzz] iter %d [china+vietnam]: confined to Southeast Asia but "
                    "total cost was %u, not 6 (4 base + 1 Asia + 1 Southeast Asia).\n",
                    iter, res.predicted_total_cost);
            }
            ASSERT_EQ(res.predicted_total_cost, 6u);
            confirmed_six++;
        }
    }

    std::printf("[InfluenceFuzz] china+vietnam: %d iterations, %d confirmed =6\n", kIterations, confirmed_six);
    ASSERT_GT(confirmed_six, 0);
}
