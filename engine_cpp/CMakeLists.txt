cmake_minimum_required(VERSION 3.21)
project(hana_engine_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------
# Dependencies via FetchContent
# ---------------------------------------------------------
include(FetchContent)

# pybind11 — prefer the venv-installed copy if available,
# fall back to FetchContent
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found via find_package; fetching via FetchContent")
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v3.0.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# ---------------------------------------------------------
# Core C++ library (static, no Python dependency)
# ---------------------------------------------------------
add_library(hana_core STATIC
    src/stub.cpp          # placeholder — replaced in Phase 1+
)
target_include_directories(hana_core PUBLIC include)
target_compile_options(hana_core PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-g -O0>
)

# ---------------------------------------------------------
# pybind11 extension module
# ---------------------------------------------------------
pybind11_add_module(hana_engine_cpp bindings/bindings.cpp)
target_link_libraries(hana_engine_cpp PRIVATE hana_core)
target_compile_options(hana_engine_cpp PRIVATE -Wall -Wextra)

# Install the .so at the repo root (one level up from engine_cpp/)
# scikit-build-core handles this via SKBUILD_PLATLIB_DIR
install(TARGETS hana_engine_cpp
        LIBRARY DESTINATION .)
