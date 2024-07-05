#ifndef NEON_SVE_BENCH_MEMSET_BENCHMARK_H
#define NEON_SVE_BENCH_MEMSET_BENCHMARK_H

#include "benchmark_common.h"

#include "common/random_data_generator.h"

struct BM_memset_args {
  static constexpr int64_t min_len = 1;       // in byte
  static constexpr int64_t max_len = 1 << 26; // in byte
  static constexpr int range_multiplier = 2;
  static constexpr uint64_t buff_alignment = 4096;
};

template <class... Args> void BM_aligned_memset(benchmark::State &state, Args &&...args) {
  const auto args_tuple = std::make_tuple(std::move(args)...);
  const auto func = std::get<0>(args_tuple);

  static RandomDataGenerator gen;
  std::size_t count = state.range(0);
  auto *arr = new (std::align_val_t(std::get<1>(args_tuple))) char[count];
  gen.initArrWithRand(arr, count);

  for (auto _ : state) {
    auto result = func(arr, gen.getRandomInt<int>(), count);
    benchmark::DoNotOptimize(result);
  }

  addByteCounters(state, count);
  addClockCounter(state);

  delete[] arr;
}

BENCHMARK_CAPTURE(BM_aligned_memset, Ref, &ref::memset, BM_memset_args::buff_alignment)
    ->RangeMultiplier(BM_memset_args::range_multiplier)
    ->Range(BM_memset_args::min_len, BM_memset_args::max_len)->ThreadPerCpu();
BENCHMARK_CAPTURE(BM_aligned_memset, Neon, &neon::memset, BM_memset_args::buff_alignment)
    ->RangeMultiplier(BM_memset_args::range_multiplier)
    ->Range(BM_memset_args::min_len, BM_memset_args::max_len)->ThreadPerCpu();
BENCHMARK_CAPTURE(BM_aligned_memset, SVE, &sve::memset, BM_memset_args::buff_alignment)
    ->RangeMultiplier(BM_memset_args::range_multiplier)
    ->Range(BM_memset_args::min_len, BM_memset_args::max_len)->ThreadPerCpu();

#endif // NEON_SVE_BENCH_MEMSET_BENCHMARK_H
