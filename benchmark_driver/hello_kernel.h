#ifndef NEON_SVE_BENCH_HELLO_KERNEL_H
#define NEON_SVE_BENCH_HELLO_KERNEL_H

#include "driver_common.h"

static void bm_helloNeon(benchmark::State &state) {
  for (auto _ : state) {
    neon_kernels::helloNeon();
  }
}

static void bm_helloSVE(benchmark::State &state) {
  for (auto _ : state) {
    sve_kernels::helloSVE();
  }
}

BENCHMARK(bm_helloNeon);
BENCHMARK(bm_helloSVE);

#endif // NEON_SVE_BENCH_HELLO_KERNEL_H
