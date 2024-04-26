#ifndef NEON_SVE_BENCH_HELLO_BENCHMARK_H
#define NEON_SVE_BENCH_HELLO_BENCHMARK_H

#include "benchmark_common.h"

template <class... Args> void BM_hello(benchmark::State &state, Args &&...args) {
  const auto args_tuple = std::make_tuple(std::move(args)...);
  const auto func = std::get<0>(args_tuple);
  for (auto _ : state) {
    benchmark::DoNotOptimize(func());
  }
}

BENCHMARK_CAPTURE(BM_hello, Neon, &neon::helloNeon)->Iterations(5);
BENCHMARK_CAPTURE(BM_hello, SVE, &sve::helloSVE)->Iterations(5);

#endif // NEON_SVE_BENCH_HELLO_BENCHMARK_H
