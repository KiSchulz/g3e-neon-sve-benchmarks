cmake_minimum_required(VERSION 3.22)
project(Neon_SVE_Bench)

set(CMAKE_CXX_STANDARD 23)
if (NOT CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_BUILD_TYPE Release)
    add_definitions(-DNDEBUG)
endif ()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wshadow -Wpedantic -march=armv8-a+sve")
set(CMAKE_CXX_FLAGS_DEBUG "-O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
set(BENCHMARK_USE_BUNDLED_GTEST OFF)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
set(BENCHMARK_ENABLE_TESTIN OFF)
set(BENCHMARK_ENABLE_LIBPFM ON)
add_subdirectory(benchmark)

add_subdirectory(FirstGBenchmarkTest)
