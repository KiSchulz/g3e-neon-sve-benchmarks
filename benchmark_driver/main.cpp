#include <benchmark/benchmark.h>
#include <chrono>
#include <cstring>
#include <iostream>

#include "hello_benchmark.h"
#include "memcmp_benchmark.h"
#include "nBody_step_benchmark.h"
#include "intersectP_benchmark.h"

int main(int argc, char **argv) {
  const auto start = std::chrono::high_resolution_clock::now();

  if (argc == 1 || (argc >= 2 && std::strcmp(argv[1], "-def") == 0)) {
    std::vector<std::string> customArgv = {
        argv[0], "--benchmark_counters_tabular=true",
        //"--benchmark_perf_counters=CYCLES,INSTRUCTIONS,BRANCH-MISSES",
        "--benchmark_perf_counters=CYCLES,INSTRUCTIONS,STALLED-CYCLES-BACKEND",
        //"--benchmark_perf_counters=CYCLES,INSTRUCTIONS,STALLED-CYCLES-FRONTEND",
        "--benchmark_display_aggregates_only=true" /*, "--benchmark_repetitions=10"*/};

    // to avoid some undefined behaviour when using string literals in vector initialization
    std::vector<char *> customArgv_cStr{};
    customArgv_cStr.reserve(customArgv.size());
    for (auto &arg : customArgv) {
      customArgv_cStr.push_back(&arg.front());
    }
    for (int i = 1; i < argc; i++) {
      customArgv_cStr.push_back(argv[i]);
    }

    int customArgc = (int)customArgv_cStr.size();
    benchmark::Initialize(&customArgc, customArgv_cStr.data());
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
