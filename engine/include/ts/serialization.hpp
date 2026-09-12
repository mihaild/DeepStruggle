#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include "game_state.hpp"

namespace ts {

class Serializer {
public:
    static void serialize_binary(const GameState& state, uint8_t* out_bytes, size_t max_bytes) noexcept;
    static void deserialize_binary(GameState& state, const uint8_t* in_bytes, size_t in_size) noexcept;

    static std::string to_json(const GameState& state);
};

} // namespace ts
