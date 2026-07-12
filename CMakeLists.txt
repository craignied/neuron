cmake_minimum_required(VERSION 3.16)
project(neuron VERSION 3.0.0 LANGUAGES CXX)

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

# Unit test: engine output is capturable via util::set_screen() (the
# GUI/server contract). No GSL needed.
add_executable(check_capture
    tests/capture/check_capture.cpp
    src/dataset.cpp src/twoset.cpp src/stats.cpp src/matrix.cpp
    src/vector_ops.cpp src/utility.cpp)
target_include_directories(check_capture PRIVATE src)
add_test(NAME screen_capture COMMAND check_capture)
