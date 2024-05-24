#ifndef NEON_SVE_BENCH_ACCUMULATE_BENCHMARK_H
#define NEON_SVE_BENCH_ACCUMULATE_BENCHMARK_H

#include "benchmark_common.h"

#include "common/random_data_generator.h"

struct BM_accumulate_args {
  static constexpr int64_t min_len = 1;       // in byte
  static constexpr int64_t max_len = 1 << 26; // in byte
  static constexpr int range_multiplier = 4;
  static constexpr uint64_t buff_alignment = 4096;
};

template <class... Args> void BM_aligned_accumulate(benchmark::State &state, Args &&...args) {
  const auto args_tuple = std::make_tuple(std::move(args)...);
  const auto func = std::get<0>(args_tuple);

  static RandomDataGenerator gen;
  std::size_t size = state.range(0);
  auto *arr = new (std::align_val_t(std::get<1>(args_tuple))) uint64_t[size];
  gen.initArrWithRand(arr, size * sizeof(uint64_t));

  for (auto _ : state) {
    auto result = func(arr, size);
    benchmark::DoNotOptimize(result);
  }

  delete [] arr;
}

BENCHMARK_CAPTURE(BM_aligned_accumulate, Ref, &ref::accumulate, BM_accumulate_args::buff_alignment)
->RangeMultiplier(BM_accumulate_args::range_multiplier)
    ->Range(BM_accumulate_args::min_len, BM_accumulate_args::max_len);
BENCHMARK_CAPTURE(BM_aligned_accumulate, Neon, &neon::accumulate, BM_accumulate_args::buff_alignment)
  ->RangeMultiplier(BM_accumulate_args::range_multiplier)
  ->Range(BM_accumulate_args::min_len, BM_accumulate_args::max_len);
BENCHMARK_CAPTURE(BM_aligned_accumulate, SVE, &sve::accumulate, BM_accumulate_args::buff_alignment)
->RangeMultiplier(BM_accumulate_args::range_multiplier)
    ->Range(BM_accumulate_args::min_len, BM_accumulate_args::max_len);

#endif // NEON_SVE_BENCH_ACCUMULATE_BENCHMARK_H
