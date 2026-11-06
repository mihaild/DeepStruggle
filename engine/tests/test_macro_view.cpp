// P42: the until-control macro view. Slot MACRO_NODE + c places in country c until the deciding side
// controls it or its points there run out, as the one-point steps that define it. The base views never
// offer it; the macro view offers it only where it would place at least two points.
#include "test_framework.hpp"
#include "ts/action_mask.hpp"
#include "ts/constants.hpp"
#include "ts/engine.hpp"
#include "ts/scoring.hpp"

#include <cstring>

namespace {

using ts::ActionView;
namespace fs = ts::flat_slots;

int first_legal(const uint8_t* mask, int lo, int hi) {
    for (int i = lo; i < hi; ++i)
        if (mask[i]) return i;
    return -1;
}

// Advance a fresh game, taking the first legal base action, until the US is to place in the setup.
ts::GameState to_us_setup(uint64_t seed) {
    ts::GameState s{};
    ts::Engine::init_game(s, seed);
    uint8_t mask[ts::FLAT_ACTION_SPACE_SIZE];
    for (int guard = 0; guard < 200; ++guard) {
        ts::Engine::auto_advance_step(s);
        const auto& ctx = s.ctx();
        if (s.current_phase == ts::Phase::SETUP && ctx.decision_player == ts::Player::US &&
            ctx.decision_type == ts::DecisionType::POINT_NODE)
            return s;
        ts::Engine::get_flat_action_mask(s, mask, ActionView::E4);
        const int a = first_legal(mask, 0, fs::BASE_SIZE);
        ASSERT_TRUE(a >= 0);
        ASSERT_TRUE(ts::Engine::step_flat(s, static_cast<uint16_t>(a), false));
    }
    ASSERT_TRUE(false);
    return s;
}

}  // namespace

TEST(MacroView, BaseViewsNeverOfferTheMacroSlots) {
    ts::GameState s{};
    ts::Engine::init_game(s, 4242);
    uint8_t e4[ts::FLAT_ACTION_SPACE_SIZE], merged[ts::FLAT_ACTION_SPACE_SIZE];
    uint64_t lcg = 12345;
    for (int step = 0; step < 3000 && !ts::Engine::is_terminal(s); ++step) {
        ts::Engine::auto_advance_step(s);
        if (ts::Engine::is_terminal(s)) break;
        ts::Engine::get_flat_action_mask(s, e4, ActionView::E4);
        ts::Engine::get_flat_action_mask(s, merged, ActionView::MERGED);
        for (int i = fs::BASE_SIZE; i < static_cast<int>(ts::FLAT_ACTION_SPACE_SIZE); ++i) {
            ASSERT_EQ(e4[i], 0);
            ASSERT_EQ(merged[i], 0);
        }
        int legal[ts::FLAT_ACTION_SPACE_SIZE];
        int n = 0;
        for (int i = 0; i < fs::BASE_SIZE; ++i)
            if (e4[i]) legal[n++] = i;
        ASSERT_TRUE(n > 0);
        lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
        ASSERT_TRUE(ts::Engine::step_flat(s, static_cast<uint16_t>(legal[(lcg >> 33) % n]), false));
    }
}

TEST(MacroView, WestGermanyUntilControlInTheUsSetupIsWestGermanyFour) {
    ts::GameState s = to_us_setup(7);
    const uint8_t wg = ts::countries::WEST_GERMANY;
    ASSERT_EQ(s.countries[wg].us_influence, 0);
    uint8_t mask[ts::FLAT_ACTION_SPACE_SIZE];
    ts::Engine::get_flat_action_mask(s, mask, ActionView::MACRO);
    ASSERT_EQ(mask[fs::MACRO_NODE + wg], 1);
    // The base part of the macro mask is the base mask.
    uint8_t e4[ts::FLAT_ACTION_SPACE_SIZE];
    ts::Engine::get_flat_action_mask(s, e4, ActionView::E4);
    ASSERT_EQ(std::memcmp(mask, e4, fs::BASE_SIZE), 0);
    ASSERT_TRUE(ts::Engine::step_flat(s, static_cast<uint16_t>(fs::MACRO_NODE + wg), false, ActionView::MACRO));
    ASSERT_EQ(s.countries[wg].us_influence, 4);
    ASSERT_TRUE(ts::Scoring::get_country_control(s, wg) == ts::Player::US);
}

TEST(MacroView, NotOfferedWhereOnePointGivesControlOrIsAlreadyControlled) {
    ts::GameState s = to_us_setup(11);
    uint8_t mask[ts::FLAT_ACTION_SPACE_SIZE];
    ts::Engine::get_flat_action_mask(s, mask, ActionView::MACRO);
    for (uint8_t c = 0; c < fs::NODE_COUNT; ++c) {
        if (!mask[fs::NODE + c]) {
            ASSERT_EQ(mask[fs::MACRO_NODE + c], 0);
            continue;
        }
        ts::GameState one = s;
        ASSERT_TRUE(ts::Engine::step_flat(one, static_cast<uint16_t>(fs::NODE + c), false));
        const bool one_point_controls = ts::Scoring::get_country_control(one, c) == ts::Player::US;
        if (one_point_controls) ASSERT_EQ(mask[fs::MACRO_NODE + c], 0);
    }
}

TEST(MacroView, AnUnofferedMacroSlotIsRefusedAndTheStateKept) {
    ts::GameState s = to_us_setup(13);
    uint8_t mask[ts::FLAT_ACTION_SPACE_SIZE];
    ts::Engine::get_flat_action_mask(s, mask, ActionView::MACRO);
    int bad = -1;
    for (uint8_t c = 0; c < fs::NODE_COUNT; ++c)
        if (!mask[fs::MACRO_NODE + c]) { bad = fs::MACRO_NODE + c; break; }
    ASSERT_TRUE(bad >= 0);
    const ts::GameState before = s;
    ASSERT_TRUE(!ts::Engine::step_flat(s, static_cast<uint16_t>(bad), false, ActionView::MACRO));
    ASSERT_EQ(std::memcmp(&before, &s, sizeof(ts::GameState)), 0);
    // A macro slot in a base view is refused too.
    ASSERT_TRUE(!ts::Engine::step_flat(s, static_cast<uint16_t>(fs::MACRO_NODE + ts::countries::WEST_GERMANY), false));
    ASSERT_EQ(std::memcmp(&before, &s, sizeof(ts::GameState)), 0);
}

TEST(MacroView, EveryMacroStepIsTheOnePointStepsThatDefineIt) {
    // Random games in the macro view, preferring macro slots: each one applied through step_flat must
    // equal applying the base one-point slot the same number of times, byte for byte.
    uint8_t mask[ts::FLAT_ACTION_SPACE_SIZE];
    uint64_t lcg = 99;
    int macros = 0;
    for (uint64_t seed = 1; seed <= 12; ++seed) {
        ts::GameState s{};
        ts::Engine::init_game(s, seed);
        for (int step = 0; step < 4000 && !ts::Engine::is_terminal(s); ++step) {
            ts::Engine::auto_advance_step(s);
            if (ts::Engine::is_terminal(s)) break;
            ts::Engine::get_flat_action_mask(s, mask, ActionView::MACRO);
            int legal[ts::FLAT_ACTION_SPACE_SIZE];
            int n = 0, nm = 0;
            int macro_legal[fs::NODE_COUNT];
            for (int i = 0; i < static_cast<int>(ts::FLAT_ACTION_SPACE_SIZE); ++i) {
                if (!mask[i]) continue;
                legal[n++] = i;
                if (ts::ActionMask::is_macro_action(static_cast<uint16_t>(i))) macro_legal[nm++] = i;
            }
            ASSERT_TRUE(n > 0);
            lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
            const bool take_macro = nm > 0 && ((lcg >> 40) & 1);
            const int a = take_macro ? macro_legal[(lcg >> 33) % nm] : legal[(lcg >> 33) % n];
            if (ts::ActionMask::is_macro_action(static_cast<uint16_t>(a))) {
                const uint8_t c = static_cast<uint8_t>(a - fs::MACRO_NODE);
                ts::GameState by_hand = s;
                const int placed = ts::ActionMask::apply_macro_expansion(by_hand, c);
                ASSERT_TRUE(placed >= 2);
                ts::GameState by_slot = s;
                ASSERT_TRUE(ts::Engine::step_flat(by_slot, static_cast<uint16_t>(a), false, ActionView::MACRO));
                ASSERT_EQ(std::memcmp(&by_hand, &by_slot, sizeof(ts::GameState)), 0);
                // ... and the same number of base one-point steps reach the same state.
                ts::GameState base = s;
                for (int k = 0; k < placed; ++k)
                    ASSERT_TRUE(ts::Engine::step_flat(base, static_cast<uint16_t>(fs::NODE + c), false));
                ASSERT_EQ(std::memcmp(&base, &by_slot, sizeof(ts::GameState)), 0);
                ++macros;
            }
            ASSERT_TRUE(ts::Engine::step_flat(s, static_cast<uint16_t>(a), false, ActionView::MACRO));
        }
    }
    ASSERT_TRUE(macros > 20);
}
