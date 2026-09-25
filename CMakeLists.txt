cmake_minimum_required(VERSION 3.20)

# The engine is built with clang. Measured on the same sources and flags, clang 21 against
# GCC 15: +20-24% on raw stepping and on the batch runner's step + observation path, with the two
# builds playing bit-identical games. So clang is chosen here unless a compiler is named
# explicitly, and anything else is refused below rather than built slower without a word.
# ~/.local/bin is searched for the wrappers tools/scripts/install_clang_userspace.sh writes on a
# machine without root.
# A toolchain file (emcmake's, for the browser build) names its own compiler.
if(NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX} AND NOT CMAKE_TOOLCHAIN_FILE)
    find_program(TS_CLANGXX
        NAMES clang++ clang++-22 clang++-21 clang++-20 clang++-19 clang++-18
        HINTS "$ENV{HOME}/.local/bin")
    if(TS_CLANGXX)
        set(CMAKE_CXX_COMPILER "${TS_CLANGXX}")
    endif()
endif()

project(ts_ai CXX)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR
        "ts_ai builds the engine with clang, but this build directory uses "
        "${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER}).\n"
        "  - install clang: apt-get install clang   (no root: tools/scripts/install_clang_userspace.sh)\n"
        "  - then configure afresh -- the compiler is fixed when a build directory is created:\n"
        "      cmake --fresh -B build/release -S . -DPython_EXECUTABLE=$(pwd)/.venv/bin/python3\n"
        "    (tools/scripts/check_engine_fresh.sh does this by itself for an existing build dir)")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization Flags
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /we4834)  # 4834 = discarding a [[nodiscard]] value
elseif(EMSCRIPTEN)
    # The browser build (tools/scripts/build_web.sh): WebAssembly has no -march to tune for.
    add_compile_options(-O3 -Wall -Wextra -Wpedantic -Wno-unused-parameter -Werror=unused-result)
else()
    add_compile_options(-O3 -march=native -Wall -Wextra -Wpedantic -Wno-unused-parameter
        # A discarded step() verdict is how a refused action becomes a silent no-op or an
        # unbounded retry loop. [[nodiscard]] only warns unless this promotes it.
        -Werror=unused-result)
endif()

add_subdirectory(engine)
add_subdirectory(bindings)
