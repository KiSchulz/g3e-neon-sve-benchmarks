#ifndef NEON_SVE_BENCH_HELLO_KERNEL_H
#define NEON_SVE_BENCH_HELLO_KERNEL_H

#include "benchmark_common.h"

static void bm_helloNeon(benchmark::State &state) {
  for (auto _ : state) {
    neon::helloNeon();
  }
}

static void bm_helloSVE(benchmark::State &state) {
  for (auto _ : state) {
    sve::helloSVE();
  }
}

BENCHMARK(bm_helloNeon);
BENCHMARK(bm_helloSVE);

#endif // NEON_SVE_BENCH_HELLO_KERNEL_H
