cmake_minimum_required(VERSION 3.16)
project(neuron VERSION 3.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(GSL REQUIRED)

file(GLOB ENGINE_SOURCES src/*.cpp)
add_executable(neuron ${ENGINE_SOURCES})
target_link_libraries(neuron PRIVATE GSL::gsl GSL::gslcblas)

# Unit test: engine's binormal (Wickens) statistical ROC vs theory.
# Links only the statistics core — no GSL needed.
enable_testing()
add_executable(check_az
    tests/binormal/check_az.cpp
    src/twoset.cpp src/stats.cpp src/matrix.cpp src/vector_ops.cpp src/utility.cpp)
target_include_directories(check_az PRIVATE src)
add_test(NAME binormal_az COMMAND check_az)
