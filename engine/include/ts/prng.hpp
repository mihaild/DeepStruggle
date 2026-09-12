#pragma once
#include <cstdint>

namespace ts {

// Fast, deterministic SplitMix64 PRNG
class Prng {
public:
    static constexpr uint64_t next_u64(uint64_t& state) noexcept {
        uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    static constexpr uint8_t roll_d6(uint64_t& state) noexcept {
        // Fast unbiased or simple modulo for 1..6
        uint64_t val = next_u64(state);
        return static_cast<uint8_t>(1 + (val % 6));
    }

    static constexpr uint32_t random_index(uint64_t& state, uint32_t size) noexcept {
        if (size <= 1) return 0;
        uint64_t val = next_u64(state);
        return static_cast<uint32_t>(val % size);
    }

    static constexpr uint32_t random_range(uint64_t& state, uint32_t min_v, uint32_t max_v) noexcept {
        if (min_v >= max_v) return min_v;
        uint32_t range = max_v - min_v + 1;
        return min_v + random_index(state, range);
    }
};

} // namespace ts
