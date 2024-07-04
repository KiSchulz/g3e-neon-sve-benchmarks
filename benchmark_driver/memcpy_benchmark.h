#ifndef NEON_SVE_BENCH_MEMCPY_BENCHMARK_H
#define NEON_SVE_BENCH_MEMCPY_BENCHMARK_H

#include "benchmark_common.h"

#include "common/random_data_generator.h"

struct BM_memcpy_args {
  static constexpr int64_t min_len = 1;       // in byte
  static constexpr int64_t max_len = 1 << 28; // in byte
  static constexpr int range_multiplier = 4;
  static constexpr uint64_t buff_alignment = 4096;
};

template <class... Args> void BM_aligned_memcpy(benchmark::State &state, Args &&...args) {
  const auto args_tuple = std::make_tuple(std::move(args)...);
  const auto func = std::get<0>(args_tuple);

  static RandomDataGenerator gen;
  std::size_t count = state.range(0);
  auto *src = new (std::align_val_t(std::get<1>(args_tuple))) char[count];
  auto *dest = new (std::align_val_t(std::get<1>(args_tuple))) char[count];
  gen.initArrWithRand(src, count);

  for (auto _ : state) {
    auto result = func(dest, src, count);
    benchmark::DoNotOptimize(result);
  }

  addByteCounters(state, 2 * count);
  addClockCounter(state);

  delete[] src;
  delete[] dest;
}

BENCHMARK_CAPTURE(BM_aligned_memcpy, Ref, &ref::memcpy, BM_memcpy_args::buff_alignment)
->RangeMultiplier(BM_memcpy_args::range_multiplier)
    ->Range(BM_memcpy_args::min_len, BM_memcpy_args::max_len)->ThreadPerCpu();
BENCHMARK_CAPTURE(BM_aligned_memcpy, Neon, &neon::memcpy, BM_memcpy_args::buff_alignment)
->RangeMultiplier(BM_memcpy_args::range_multiplier)
    ->Range(BM_memcpy_args::min_len, BM_memcpy_args::max_len)->ThreadPerCpu();
BENCHMARK_CAPTURE(BM_aligned_memcpy, SVE, &sve::memcpy, BM_memcpy_args::buff_alignment)
->RangeMultiplier(BM_memcpy_args::range_multiplier)
    ->Range(BM_memcpy_args::min_len, BM_memcpy_args::max_len)->ThreadPerCpu();

#endif // NEON_SVE_BENCH_MEMCPY_BENCHMARK_H
