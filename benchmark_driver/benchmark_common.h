#ifndef NEON_SVE_BENCH_BENCHMARK_COMMON_H
#define NEON_SVE_BENCH_BENCHMARK_COMMON_H

#include <benchmark/benchmark.h>
#include <kernels/neon/neon_kernels.h>
#include <kernels/reference/reference_kernels.h>
#include <kernels/sve/sve_kernels.h>

namespace ref = reference_kernels;
namespace neon = neon_kernels;
namespace sve = sve_kernels;

void addClockCounter(benchmark::State &state) {
  state.counters["clock"] = benchmark::Counter{state.counters["CYCLES"], benchmark::Counter::kIsRate};
}

void addByteCounters(benchmark::State &state, uint64_t bytes_per_iter) {
  state.counters["bytes"] = (double)bytes_per_iter;
  state.counters["bytes_p_sec"] =
      benchmark::Counter{(double)state.iterations() * (double)bytes_per_iter, benchmark::Counter::kIsRate};
}

#endif // NEON_SVE_BENCH_BENCHMARK_COMMON_H
