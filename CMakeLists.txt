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

find_package(OpenMP)

add_subdirectory(engine)
add_subdirectory(bindings)
