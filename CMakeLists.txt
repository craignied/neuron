cmake_minimum_required(VERSION 3.16)
project(neuron VERSION 3.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(GSL REQUIRED)

file(GLOB ENGINE_SOURCES engine/src/*.cpp)
add_executable(neuron ${ENGINE_SOURCES})
target_link_libraries(neuron PRIVATE GSL::gsl GSL::gslcblas)
