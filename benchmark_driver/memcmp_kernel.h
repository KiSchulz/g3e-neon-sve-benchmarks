#ifndef NEON_SVE_BENCH_MEMCMP_KERNEL_H
#define NEON_SVE_BENCH_MEMCMP_KERNEL_H

#include "benchmark_common.h"

static void bm_memcmpNeon(benchmark::State &state) {
  for (auto _ : state) {
    neon::memcmp(nullptr, nullptr, 0);
  }
}

static void bm_memcmpSVE(benchmark::State &state) {
  for (auto _ : state) {
    sve::memcmp(nullptr, nullptr, 0);
  }
}

BENCHMARK(bm_memcmpNeon);
BENCHMARK(bm_memcmpSVE);

#endif // NEON_SVE_BENCH_MEMCMP_KERNEL_H
