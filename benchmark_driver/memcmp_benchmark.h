#ifndef NEON_SVE_BENCH_MEMCMP_BENCHMARK_H
#define NEON_SVE_BENCH_MEMCMP_BENCHMARK_H

#include "benchmark_common.h"

#include "common/random_data_generator.h"

struct BM_memcmp_args {
  static constexpr int64_t min_prefix_length = 1;        // in byte
  static constexpr int64_t max_prefix_length = 1 << 29; // in byte
  static constexpr int range_multiplier = 2;
};

template <class... Args> void BM_memcmp(benchmark::State &state, Args &&...args) {
  const auto args_tuple = std::make_tuple(std::move(args)...);
  const auto func = std::get<0>(args_tuple);

  static RandomDataGenerator gen;
  char *buff1 = new char[BM_memcmp_args::max_prefix_length];
  char *buff2 = new char[BM_memcmp_args::max_prefix_length];

  // generate common prefix for buff1 and buff2
  const std::size_t pref_len = state.range(0);
  gen.initArrWithRand(buff1, pref_len);
  std::copy(buff1, buff1 + pref_len, buff2);
  // generate random tail for buff1 and buff2
  // use min to avoid reinitializing the hole array everytime
  gen.initArrWithRand(buff1 + pref_len, std::min(BM_memcmp_args::max_prefix_length - pref_len, 2 * pref_len));
  gen.initArrWithRand(buff2 + pref_len, std::min(BM_memcmp_args::max_prefix_length - pref_len, 2 * pref_len));

  for (auto _ : state) {
    int result = func(buff1, buff2, BM_memcmp_args::max_prefix_length);
    benchmark::DoNotOptimize(result);
  }

  delete[] buff1;
  delete[] buff2;
}

BENCHMARK_CAPTURE(BM_memcmp, Ref, &ref::memcmp)
->RangeMultiplier(BM_memcmp_args::range_multiplier)
    ->Range(BM_memcmp_args::min_prefix_length, BM_memcmp_args::max_prefix_length);
BENCHMARK_CAPTURE(BM_memcmp, Neon, &neon::memcmp)
    ->RangeMultiplier(BM_memcmp_args::range_multiplier)
    ->Range(BM_memcmp_args::min_prefix_length, BM_memcmp_args::max_prefix_length);
BENCHMARK_CAPTURE(BM_memcmp, SVE, &sve::memcmp)
    ->RangeMultiplier(BM_memcmp_args::range_multiplier)
    ->Range(BM_memcmp_args::min_prefix_length, BM_memcmp_args::max_prefix_length);

#endif // NEON_SVE_BENCH_MEMCMP_BENCHMARK_H
