#include "test_framework.hpp"
#include "ts/map_data.hpp"

TEST(MapTest, TotalCounts) {
    uint8_t total_bg = 0;
    for (uint8_t i = 0; i < 84; ++i) {
        if (ts::MapData::get_country(i).battleground) total_bg++;
    }
    ASSERT_EQ(total_bg, 29); // 29 battlegrounds in Twilight Struggle
}

TEST(MapTest, AdjacencySymmetry) {
    for (uint8_t i = 0; i < 84; ++i) {
        const auto& c = ts::MapData::get_country(i);
        for (uint8_t n = 0; n < c.num_neighbors; ++n) {
            uint8_t neighbor = c.neighbors[n];
            ASSERT_TRUE(neighbor < 84);
            ASSERT_TRUE(ts::MapData::is_adjacent(neighbor, i));
        }
    }
}

TEST(MapTest, SuperpowerAdjacency) {
    // US: Canada(0), Japan(45), Mexico(64), Cuba(71)
    ASSERT_TRUE(ts::MapData::is_adjacent_to_superpower(ts::countries::CANADA, ts::Player::US));
    ASSERT_TRUE(ts::MapData::is_adjacent_to_superpower(ts::countries::JAPAN, ts::Player::US));
    ASSERT_TRUE(ts::MapData::is_adjacent_to_superpower(ts::countries::MEXICO, ts::Player::US));
    ASSERT_TRUE(ts::MapData::is_adjacent_to_superpower(ts::countries::CUBA, ts::Player::US));

    // USSR: Finland(5), Poland(15), Romania(19), Turkey(12), Afghanistan(31), North Korea(43)
    ASSERT_TRUE(ts::MapData::is_adjacent_to_superpower(ts::countries::FINLAND, ts::Player::USSR));
    ASSERT_TRUE(ts::MapData::is_adjacent_to_superpower(ts::countries::POLAND, ts::Player::USSR));
    ASSERT_TRUE(ts::MapData::is_adjacent_to_superpower(ts::countries::ROMANIA, ts::Player::USSR));
    ASSERT_TRUE(ts::MapData::is_adjacent_to_superpower(ts::countries::TURKEY, ts::Player::USSR));
    ASSERT_TRUE(ts::MapData::is_adjacent_to_superpower(ts::countries::AFGHANISTAN, ts::Player::USSR));
    ASSERT_TRUE(ts::MapData::is_adjacent_to_superpower(ts::countries::NORTH_KOREA, ts::Player::USSR));

    // Bulgaria and Iran must NOT be adjacent to USSR
    ASSERT_FALSE(ts::MapData::is_adjacent_to_superpower(ts::countries::BULGARIA, ts::Player::USSR));
    ASSERT_FALSE(ts::MapData::is_adjacent_to_superpower(ts::countries::IRAN, ts::Player::USSR));

    // Gulf States must NOT be adjacent to Iran
    ASSERT_FALSE(ts::MapData::is_adjacent(ts::countries::GULF_STATES, ts::countries::IRAN));
    ASSERT_FALSE(ts::MapData::is_adjacent(ts::countries::IRAN, ts::countries::GULF_STATES));
}
