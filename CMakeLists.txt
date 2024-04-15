cmake_minimum_required(VERSION 3.23)
project(Neon_SVE_Bench)

set(CMAKE_CXX_STANDARD 23)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fno-omit-frame-pointer -Wall -Wextra -Wshadow -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BENCHMARK_USE_BUNDLED_GTEST OFF)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
set(BENCHMARK_ENABLE_TESTIN OFF)
set(BENCHMARK_ENABLE_LIBPFM ON)
add_subdirectory(benchmark)

add_subdirectory(FirstGBenchmarkTest)
