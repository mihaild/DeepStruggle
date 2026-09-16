cmake_minimum_required(VERSION 3.20)
project(ts_ai CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization Flags
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /we4834)  # 4834 = discarding a [[nodiscard]] value
else()
    add_compile_options(-O3 -march=native -Wall -Wextra -Wpedantic -Wno-unused-parameter
        # A discarded step() verdict is how a refused action becomes a silent no-op or an
        # unbounded retry loop. [[nodiscard]] only warns unless this promotes it.
        -Werror=unused-result)
endif()

find_package(OpenMP)

add_subdirectory(engine)
add_subdirectory(bindings)
