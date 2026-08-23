cmake_minimum_required(VERSION 3.20)
project(ts_ai CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization Flags
if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
else()
    add_compile_options(-O3 -march=native -Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

add_subdirectory(engine)

# Python & nanobind configuration
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

find_package(nanobind CONFIG QUIET)
if(NOT nanobind_FOUND)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NANOBIND_CMAKE_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR}")
        find_package(nanobind CONFIG REQUIRED)
    endif()
endif()

if(nanobind_FOUND)
    nanobind_add_module(ts_engine bindings/ts_bindings.cpp)
    target_include_directories(ts_engine PRIVATE engine/include)
    target_link_libraries(ts_engine PRIVATE ts_engine_core)
    set_target_properties(ts_engine PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()
