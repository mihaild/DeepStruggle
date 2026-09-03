cmake_minimum_required(VERSION 3.20)
project(ts_engine_core CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization Flags
if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
else()
    add_compile_options(-O3 -march=native -Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# Core Library
set(TS_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(TS_SOURCES
    src/map_data.cpp
    src/card_data.cpp
    src/scoring.cpp
    src/ops.cpp
    src/space_race.cpp
    src/events/early_war.cpp
    src/events/mid_war.cpp
    src/events/late_war.cpp
    src/card_dispatcher.cpp
    src/action_mask.cpp
    src/state_machine.cpp
    src/observation.cpp
    src/serialization.cpp
    src/invariant.cpp
    src/engine.cpp
)

add_library(ts_engine_core STATIC ${TS_SOURCES})
target_include_directories(ts_engine_core PUBLIC ${TS_INCLUDE_DIRS})
set_target_properties(ts_engine_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Alias for backward compatibility
add_library(ts_engine::core ALIAS ts_engine_core)

# Unit & Integration Tests Executable
set(TEST_SOURCES
    tests/test_main.cpp
    tests/test_map.cpp
    tests/test_scoring.cpp
    tests/test_bugs_regression.cpp
    tests/test_ops.cpp
    tests/test_cards_early.cpp
    tests/test_cards_mid.cpp
    tests/test_cards_late.cpp
    tests/test_defcon_suicide.cpp
    tests/test_reentrancy.cpp
    tests/test_card_interactions.cpp
    tests/test_state_lifecycle.cpp
    tests/test_states.cpp
    tests/test_card_edge_cases.cpp
    tests/test_full_game.cpp
)

add_executable(ts_tests ${TEST_SOURCES})
target_link_libraries(ts_tests PRIVATE ts_engine_core)

# Invariant Fuzzer
add_executable(ts_fuzz tests/test_fuzz.cpp)
target_link_libraries(ts_fuzz PRIVATE ts_engine_core)

# Event-Biased Fuzzer
add_executable(ts_fuzz_events tests/test_fuzz_events.cpp)
target_link_libraries(ts_fuzz_events PRIVATE ts_engine_core)

# Performance Throughput Benchmark
add_executable(ts_benchmark tests/test_benchmark.cpp)
target_link_libraries(ts_benchmark PRIVATE ts_engine_core)
