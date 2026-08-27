# Python & nanobind configuration for Twilight Struggle simulation engine
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
    nanobind_add_module(ts_engine ts_bindings.cpp)
    target_include_directories(ts_engine PRIVATE ${CMAKE_SOURCE_DIR}/engine/include)
    target_link_libraries(ts_engine PRIVATE ts_engine_core)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(ts_engine PRIVATE OpenMP::OpenMP_CXX)
    endif()
    # Output to both binary root and bindings dir for python import convenience
    set_target_properties(ts_engine PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endif()
