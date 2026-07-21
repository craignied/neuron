cmake_minimum_required(VERSION 3.16)
project(neuron VERSION 3.0.0 LANGUAGES CXX)

# Default to Release: a bare `cmake -B build` otherwise builds with NO
# optimization at all, and the engine runs ~15x slower -- discovered 2026-07-21
# when a locally reconfigured build dir had an empty CMAKE_BUILD_TYPE and every
# timing measured against it was wrong. CI builds Release explicitly; asserts
# vanish under -DNDEBUG by design (handlers validate; Matrix bounds checks are
# unconditional throws, not asserts). Single-config generators only -- MSVC
# picks its config at build time (--config Release, as CI does).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(GSL REQUIRED)
find_package(Threads REQUIRED)

# Embed the GUI page (src/gui_page.html) into the binary as a raw string
file(READ src/gui_page.html GUI_PAGE_CONTENT)
configure_file(src/gui_page.h.in generated/gui_page.h @ONLY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS src/gui_page.html)

file(GLOB ENGINE_SOURCES src/*.cpp)
add_executable(neuron ${ENGINE_SOURCES})
target_include_directories(neuron PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/generated third_party)
target_link_libraries(neuron PRIVATE GSL::gsl GSL::gslcblas Threads::Threads)
if(WIN32)
    target_link_libraries(neuron PRIVATE ws2_32)
endif()

# Unit test: engine's binormal (Wickens) statistical ROC vs theory.
# Links only the statistics core — no GSL needed.
enable_testing()
add_executable(check_az
    tests/binormal/check_az.cpp
    src/twoset.cpp src/stats.cpp src/matrix.cpp src/vector_ops.cpp src/utility.cpp)
target_include_directories(check_az PRIVATE src)
add_test(NAME binormal_az COMMAND check_az)

# Literature acceptance test: the engine's binormal ROC against Wickens' own
# published worked example (Table 5.1, analysed by hand through chapter 5).
# Self-verifying against the literature, like the Hosmer-Lemeshow log likelihood.
add_executable(check_wickens
    tests/binormal/check_wickens.cpp
    src/twoset.cpp src/stats.cpp src/matrix.cpp src/vector_ops.cpp src/utility.cpp)
target_include_directories(check_wickens PRIVATE src)
add_test(NAME binormal_wickens COMMAND check_wickens)

# Unit test: the Matrix row-gather primitive (includerows -- repeats and
# arbitrary order allowed, bounds throw) and includecols' output dimension.
add_executable(check_matrix
    tests/matrix/check_matrix.cpp src/matrix.cpp src/utility.cpp)
target_include_directories(check_matrix PRIVATE src)
add_test(NAME matrix_gather COMMAND check_matrix)

# Regression test: Hosmer-Lemeshow on datasets past 10,000 rows (fixed-size
# stack arrays segfaulted there until 2026-07-16). No GSL needed.
add_executable(check_hl
    tests/twoset/check_hl.cpp
    src/twoset.cpp src/stats.cpp src/matrix.cpp src/vector_ops.cpp src/utility.cpp)
target_include_directories(check_hl PRIVATE src)
add_test(NAME hosmer_lemeshow_large COMMAND check_hl)

# Unit test: the plateau early-stop detector (ROADMAP 2 Phase 3). Header-only
# and engine-free -- five synthetic error traces. No GSL needed.
add_executable(check_plateau
    tests/plateau/check_plateau.cpp)
target_include_directories(check_plateau PRIVATE src)
add_test(NAME plateau_detector COMMAND check_plateau)

# Unit test: engine output is capturable via util::set_screen() (the
# GUI/server contract). No GSL needed.
add_executable(check_capture
    tests/capture/check_capture.cpp
    src/dataset.cpp src/twoset.cpp src/stats.cpp src/matrix.cpp
    src/vector_ops.cpp src/utility.cpp)
target_include_directories(check_capture PRIVATE src)
add_test(NAME screen_capture COMMAND check_capture)

# Unit test: the OBD hidden-layer resizing primitives AND the grow-then-prune
# sizing driver (ROADMAP 2 Phase 4). Trains real SimpleProp networks and runs
# the driver (which clones via netclone and probes via autoalgo), so it links
# the training engine and those helpers (and GSL, which network.cpp references).
add_executable(check_obd
    tests/obd/check_obd.cpp
    src/obd.cpp src/netclone.cpp src/autoalgo.cpp
    src/simpleprop.cpp src/bareprop.cpp src/backprop.cpp src/logistic.cpp
    src/network.cpp src/iterative.cpp src/model.cpp src/dataset.cpp
    src/twoset.cpp src/stats.cpp src/matrix.cpp src/vector_ops.cpp src/utility.cpp)
target_include_directories(check_obd PRIVATE src)
target_link_libraries(check_obd PRIVATE GSL::gsl GSL::gslcblas)
add_test(NAME obd_sizing COMMAND check_obd)
