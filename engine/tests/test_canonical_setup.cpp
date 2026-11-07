// P43 phase A: obs_features::CANONICAL_SETUP. A seat playing the rule bit places each setup run (USSR 6;
// US 7; US 2) in non-decreasing MapData::placement_rank, so every setup board has exactly one action
// sequence. The bit constrains the order of the clicks, never the boards a setup can reach; a seat
// without it plays exactly as before.
#include "test_framework.hpp"
#include "ts/action_mask.hpp"
#include "ts/constants.hpp"
#include "ts/engine.hpp"
#include "ts/map_data.hpp"

#include <array>
#include <cstring>
#include <set>
#include <vector>

namespace {

using ts::ActionView;
namespace fs = ts::flat_slots;
namespace of = ts::obs_features;

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

void place(ts::GameState& s, uint8_t c) {
    ASSERT_TRUE(ts::Engine::step_flat(s, static_cast<uint16_t>(fs::NODE + c), false));
    ts::Engine::auto_advance_step(s);
}

// Every canonical sequence of the current setup run, depth-first: the number of sequences and the set of
// boards (the placing side's influence per country) they reach.
void enumerate(const ts::GameState& s, ts::Player side, std::vector<uint8_t>& path, size_t& leaves,
               std::set<std::vector<uint8_t>>& boards, const ts::GameState& start) {
    const bool same_run = s.current_phase == ts::Phase::SETUP && s.ctx().decision_player == side &&
                          s.ctx().pending_ops_value == start.ctx().pending_ops_value;
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
        path.push_back(c);
        enumerate(next, side, path, leaves, boards, start);
        path.pop_back();
    }
}

uint64_t multisets(uint64_t n, uint64_t k) {   // C(n + k - 1, k)
    uint64_t r = 1;
    for (uint64_t i = 1; i <= k; ++i) r = r * (n + k - i) / i;
    return r;
}

}  // namespace

TEST(CanonicalSetup, RankIsAPermutationMostImportantLast) {
    std::array<int, 84> seen{};
    for (uint8_t i = 0; i < 84; ++i) seen[ts::MapData::placement_rank(i)]++;
    for (int k : seen) ASSERT_EQ(k, 1);
    auto rank = [](const char* n) { return ts::MapData::placement_rank(ts::MapData::get_country_by_name(n)); };
    ASSERT_EQ(rank("Poland"), 83);
    ASSERT_EQ(rank("East Germany"), 82);
    ASSERT_EQ(rank("West Germany"), 81);
    ASSERT_EQ(rank("France"), 80);
    ASSERT_EQ(rank("Italy"), 79);
    // Every European country after every Middle Eastern one, and every non-battleground of a region before
    // its battlegrounds.
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

TEST(CanonicalSetup, WithoutTheBitNothingChanges) {
    // The USSR plays the bit-less seat; the US seat's bit must not touch the USSR's masks, and the
    // context's floor is never written.
    ts::GameState s = to_ussr_setup(7, 0, of::CANONICAL_SETUP);
    const uint8_t poland = ts::MapData::get_country_by_name("Poland");
    const uint8_t austria = ts::MapData::get_country_by_name("Austria");
    place(s, poland);
    ASSERT_EQ(s.ctx().canon_floor, 0);
    ASSERT_TRUE(node_legal(s, austria));
    place(s, austria);
    ASSERT_EQ(s.ctx().canon_floor, 0);
}

TEST(CanonicalSetup, ARunNeverGoesBackDownTheOrder) {
    ts::GameState s = to_ussr_setup(7, of::CANONICAL_SETUP, of::CANONICAL_SETUP);
    const uint8_t poland = ts::MapData::get_country_by_name("Poland");
    const uint8_t east_germany = ts::MapData::get_country_by_name("East Germany");
    const uint8_t austria = ts::MapData::get_country_by_name("Austria");
    ASSERT_TRUE(node_legal(s, austria));
    place(s, east_germany);
    ASSERT_TRUE(!node_legal(s, austria));
    ASSERT_TRUE(node_legal(s, east_germany));   // the last country placed stays legal
    ASSERT_TRUE(node_legal(s, poland));
    // Poland is last in the order: once there, it is the only legal country, and the step refuses
    // anything below it.
    {
        ts::GameState t = s;
        ASSERT_TRUE(ts::Engine::step_flat(t, static_cast<uint16_t>(fs::NODE + poland), false));
        ASSERT_TRUE(!node_legal(t, east_germany));
        ASSERT_TRUE(!ts::Engine::step_flat(t, static_cast<uint16_t>(fs::NODE + east_germany), false));
    }
    // Auto-advance plays the forced rest (four more in Poland); the US run then starts from the bottom
    // of the order.
    const uint8_t before = s.countries[poland].ussr_influence;
    place(s, poland);
    ASSERT_TRUE(s.ctx().decision_player == ts::Player::US);
    ASSERT_EQ(s.countries[poland].ussr_influence, before + 5);
    ASSERT_EQ(s.ctx().canon_floor, 0);
    ASSERT_TRUE(node_legal(s, austria));
}

TEST(CanonicalSetup, EveryUssrSetupHasExactlyOneSequence) {
    ts::GameState s = to_ussr_setup(11, of::CANONICAL_SETUP, of::CANONICAL_SETUP);
    uint64_t n = 0;
    for (uint8_t c = 0; c < 84; ++c) n += ts::MapData::get_country(c).in_eastern_europe ? 1 : 0;
    std::vector<uint8_t> path;
    size_t leaves = 0;
    std::set<std::vector<uint8_t>> boards;
    enumerate(s, ts::Player::USSR, path, leaves, boards, s);
    // Free order reaches every multiset of 6 over the Eastern European countries; canonical order reaches
    // each of them once.
    ASSERT_EQ(boards.size(), static_cast<size_t>(multisets(n, 6)));
    ASSERT_EQ(leaves, boards.size());
}

TEST(CanonicalSetup, EveryUsSetupHasExactlyOneSequence) {
    ts::GameState s = to_ussr_setup(11, of::CANONICAL_SETUP, of::CANONICAL_SETUP);
    const uint8_t poland = ts::MapData::get_country_by_name("Poland");
    place(s, poland);   // last in the order: auto-advance places the other five there
    ASSERT_TRUE(s.ctx().decision_player == ts::Player::US);
    uint64_t m = 0;
    for (uint8_t c = 0; c < 84; ++c) m += ts::MapData::get_country(c).in_western_europe ? 1 : 0;
    std::vector<uint8_t> path;
    size_t leaves = 0;
    std::set<std::vector<uint8_t>> boards;
    enumerate(s, ts::Player::US, path, leaves, boards, s);
    ASSERT_EQ(boards.size(), static_cast<size_t>(multisets(m, 7)));
    ASSERT_EQ(leaves, boards.size());

    // The bonus run: from one stage-0 setup, every board of the two bonus points once.
    const uint8_t france = ts::MapData::get_country_by_name("France");
    for (int i = 0; i < 7; ++i) place(s, france);   // West Germany ranks above France, so none is forced
    ASSERT_TRUE(s.ctx().pending_ops_value == 1);
    ASSERT_EQ(s.ctx().canon_floor, 0);
    uint64_t k = 0;
    for (uint8_t c = 0; c < 84; ++c) k += s.countries[c].us_influence > 0 ? 1 : 0;
    path.clear();
    leaves = 0;
    boards.clear();
    enumerate(s, ts::Player::US, path, leaves, boards, s);
    ASSERT_EQ(boards.size(), static_cast<size_t>(multisets(k, 2)));
    ASSERT_EQ(leaves, boards.size());
}

TEST(CanonicalSetup, FullGamesPlayWithTheBitOnBothSeats) {
    for (uint64_t seed = 1; seed <= 50; ++seed) {
        ts::GameState s{};
        ts::Engine::init_game(s, seed);
        ts::seat_both(s, of::CANONICAL_SETUP);
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
            if (s.current_phase != ts::Phase::SETUP) ASSERT_EQ(s.ctx().canon_floor, 0);
        }
        ASSERT_TRUE(ts::Engine::is_terminal(s));
    }
}
