# Language bindings for the engine: Python (nanobind) and, under Emscripten, the browser.

# The state as JSON -- the display state and the named-field save -- written once for both
# bindings, so the Python workbench and the browser one render and save positions identically.
add_library(ts_state_json STATIC state_json.cpp)
target_include_directories(ts_state_json PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/engine/include)
target_link_libraries(ts_state_json PUBLIC ts_engine_core)
set_target_properties(ts_state_json PROPERTIES POSITION_INDEPENDENT_CODE ON)

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
    target_link_libraries(ts_engine PRIVATE ts_engine_core ts_state_json)
    # GCC's OpenMP runtime, linked by exact file rather than `-fopenmp`: the batch runner calls
    # GOMP_parallel itself (see the comment above gomp_parallel_for in ts_bindings.cpp), so under
    # any compiler the process shares torch's libgomp and its one thread pool. The runtime file
    # `libgomp.so.1` is named, not `-lgomp`: on a machine with LLVM's libomp-dev installed,
    # clang's library path holds a `libgomp.so` that is an alias for libomp -- exactly the second
    # runtime this exists to avoid -- and it would be linked without a word.
    find_library(TS_LIBGOMP NAMES libgomp.so.1
                 HINTS ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES}
                 PATHS /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/lib64 /usr/lib)
    if(NOT TS_LIBGOMP)
        message(FATAL_ERROR "ts_engine needs GCC's OpenMP runtime, libgomp.so.1 "
                            "(Debian/Ubuntu: apt-get install libgomp1).")
    endif()
    get_filename_component(TS_LIBGOMP_REAL "${TS_LIBGOMP}" REALPATH)
    get_filename_component(TS_LIBGOMP_NAME "${TS_LIBGOMP_REAL}" NAME)
    if(NOT TS_LIBGOMP_NAME MATCHES "^libgomp")
        message(FATAL_ERROR "${TS_LIBGOMP} resolves to ${TS_LIBGOMP_REAL}, which is not GCC's "
                            "libgomp (LLVM's libomp installs a libgomp alias). The engine must "
                            "share torch's libgomp; point TS_LIBGOMP at the real libgomp.so.1.")
    endif()
    message(STATUS "ts_engine OpenMP runtime: ${TS_LIBGOMP_REAL}")
    target_link_libraries(ts_engine PRIVATE "${TS_LIBGOMP}")
    # Output to both binary root and bindings dir for python import convenience
    set_target_properties(ts_engine PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )

    # The type stubs are generated from the built module, not maintained by hand.
    #
    # A hand-written stub drifts from the bindings silently, and a stub that lies is worse than
    # no stub: it is the thing type checking trusts. These are produced by nanobind's stubgen by
    # introspecting the module that was just built, so classes, enums, overloads and read-only
    # fields follow the C++ automatically. The pattern file supplies only what introspection
    # cannot recover -- the shape of values returned as an untyped nb::dict.
    #
    # Generated recursively, as a stub *package* (`bindings/ts_engine/__init__.pyi` plus one
    # file per submodule), rather than a single ts_engine.pyi. EffectBits is a nanobind
    # submodule, and a flat stub renders it as `from ts_engine import EffectBits`, which has
    # nothing to resolve to and is an error in its own right.
    #
    # `bindings` is already a pyrefly search-path, so the package is picked up as the stubs for
    # top-level `ts_engine`. It cannot shadow the real module at runtime: `bindings` is not on
    # sys.path itself, and the directory holds no .py files.
    #
    # Committed, so editors and pyrefly work without building first;
    # tests/bindings/test_stub_is_generated.py fails if the committed copy no longer matches
    # what the current bindings produce.
    if (nanobind_FOUND)
        set(TS_STUB_DIR "${CMAKE_SOURCE_DIR}/bindings/ts_engine")
        add_custom_command(
            OUTPUT "${TS_STUB_DIR}/__init__.pyi"
            COMMAND "${Python_EXECUTABLE}" -m nanobind.stubgen
                    -m ts_engine
                    -i "$<TARGET_FILE_DIR:ts_engine>"
                    -r
                    -O "${CMAKE_SOURCE_DIR}/bindings"
                    -p "${CMAKE_SOURCE_DIR}/bindings/ts_engine.pyi.pattern"
                    -q
            DEPENDS ts_engine "${CMAKE_SOURCE_DIR}/bindings/ts_engine.pyi.pattern"
            COMMENT "Generating type stubs for ts_engine"
            VERBATIM
        )
        add_custom_target(ts_engine_stub ALL DEPENDS "${TS_STUB_DIR}/__init__.pyi")
    endif()
endif()
