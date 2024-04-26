#include <benchmark/benchmark.h>
#include <chrono>
#include <iostream>

#include "hello_benchmark.h"
#include "memcmp_benchmark.h"

int main(int argc, char **argv) {
  const auto start = std::chrono::high_resolution_clock::now();

  if (argc == 1) {
    std::vector<std::string> defaultArgv = {
        argv[0], "--benchmark_counters_tabular=true",
        "--benchmark_perf_counters=CYCLES,INSTRUCTIONS,L1-DCACHE-LOAD-MISSES", // CACHE-MISSES,BRANCH-MISSES dont work
        "--benchmark_display_aggregates_only=true" /*, "--benchmark_repetitions=10"*/};

    // to avoid some undefined behaviour when using string literals in vector initialization
    std::vector<char *> defaultArgv_cStr{};
    defaultArgv_cStr.reserve(defaultArgv.size());
    for (auto &arg : defaultArgv) {
      defaultArgv_cStr.push_back(&arg.front());
    }

    int defaultArgc = (int)defaultArgv_cStr.size();
    benchmark::Initialize(&defaultArgc, defaultArgv_cStr.data());
  } else {
    benchmark::Initialize(&argc, argv);
  }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  std::cout
      << "Benchmark duration: "
      << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count()
      << "s\n";
}
