// obs_features::UNCAPPED_DEFICIT (owner, 2026-11-26): board slots 24 and 25, the influence each side lacks to
// control a country / 5, written without the base's cap of 2.0 (10 points). Clear, the observation is the
// capped base, bit for bit; set, only slots 24 and 25 of the countries past the cap move.
#include "test_framework.hpp"
#include "ts/action_mask.hpp"
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
