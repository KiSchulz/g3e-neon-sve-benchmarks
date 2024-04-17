#include <benchmark/benchmark.h>
#include <cstdint>
#include <iostream>
#include <unistd.h>

#ifndef __ARM_FEATURE_SVE
#define __ARM_FREATURE_SVE
#endif

#include <arm_sve.h>

// template<class T = float>
// void print_svfloat32_t(std::string_view name, svbool_t pred, svfloat32_t m) {
//     const uint64_t n = svcntp_b32(pred, pred);
//
// }
//
// int main () {
//     std::cout << svcntb() << "\n";
// }

uint64_t someCoolFunc() {
  uint64_t n = 0;
  for (int i = 0; i < 100; i++) {
    n += getpid();
  }
  return n;
}

static void bm_func(benchmark::State &state) {
  for ([[maybe_unused]] auto _ : state) {
    someCoolFunc();
  }
}

BENCHMARK(bm_func)->Repetitions(10)->ReportAggregatesOnly(true);

int main(int argc, char **argv) {
  if (argc == 1) {
    std::vector<char *> defaultArgv = {argv[0], "--benchmark_counters_tabular=true",
                                       "--benchmark_perf_counters=CYCLES,INSTRUCTIONS"};
    int defaultArgc = (int)defaultArgv.size();
    benchmark::Initialize(&defaultArgc, &defaultArgv[0]);
  } else {
    benchmark::Initialize(&argc, argv);
  }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
}
