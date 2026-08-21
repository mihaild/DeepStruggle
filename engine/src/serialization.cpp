#include "ts/serialization.hpp"
#include "ts/map_data.hpp"
#include "ts/card_data.hpp"
#include <cstring>
#include <sstream>
#include <iomanip>

namespace ts {

void Serializer::serialize_binary(const GameState& state, uint8_t* out_bytes, size_t max_bytes) noexcept {
    if (!out_bytes || max_bytes < sizeof(GameState)) return;
    std::memcpy(out_bytes, &state, sizeof(GameState));
}

void Serializer::deserialize_binary(GameState& state, const uint8_t* in_bytes, size_t in_size) noexcept {
    if (!in_bytes || in_size < sizeof(GameState)) return;
    std::memcpy(&state, in_bytes, sizeof(GameState));
}

std::string Serializer::to_json(const GameState& state) {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"victory_points\": " << static_cast<int>(state.victory_points) << ",\n";
    ss << "  \"defcon\": " << static_cast<int>(state.defcon) << ",\n";
    ss << "  \"us_mil_ops\": " << static_cast<int>(state.us_mil_ops) << ",\n";
    ss << "  \"ussr_mil_ops\": " << static_cast<int>(state.ussr_mil_ops) << ",\n";
    ss << "  \"us_space_track\": " << static_cast<int>(state.us_space_track) << ",\n";
    ss << "  \"ussr_space_track\": " << static_cast<int>(state.ussr_space_track) << ",\n";
    ss << "  \"turn\": " << static_cast<int>(state.turn) << ",\n";
    ss << "  \"action_round\": " << static_cast<int>(state.action_round) << ",\n";
    ss << "  \"phasing_player\": " << (state.phasing_player == Player::US ? "\"US\"" : (state.phasing_player == Player::USSR ? "\"USSR\"" : "\"NONE\"")) << ",\n";
    ss << "  \"current_phase\": " << static_cast<int>(state.current_phase) << ",\n";
    ss << "  \"china_card_holder\": " << (state.china_card_holder == Player::US ? "\"US\"" : "\"USSR\"") << ",\n";
    ss << "  \"china_card_playable\": " << (state.china_card_playable ? "true" : "false") << ",\n";
    ss << "  \"persistent_effects\": \"0x" << std::hex << state.persistent_effects << std::dec << "\",\n";

    ss << "  \"countries\": {\n";
    for (uint8_t i = 0; i < 84; ++i) {
        ss << "    \"" << MapData::get_country_name(i) << "\": {\"US\": "
           << static_cast<int>(state.countries[i].us_influence) << ", \"USSR\": "
           << static_cast<int>(state.countries[i].ussr_influence) << "}"
           << (i == 83 ? "\n" : ",\n");
    }
    ss << "  }\n";
    ss << "}\n";
    return ss.str();
}

} // namespace ts
