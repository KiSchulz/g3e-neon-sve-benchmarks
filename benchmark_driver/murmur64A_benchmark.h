#ifndef NEON_SVE_BENCH_MURMUR64A_BENCHMARK_H
#define NEON_SVE_BENCH_MURMUR64A_BENCHMARK_H

#include "benchmark_common.h"

#include "common/random_data_generator.h"

struct BM_murmur64A_args {
  static constexpr int64_t min_length = 1;       // in byte
  static constexpr int64_t max_length = 1 << 26; // in byte
  static constexpr int range_multiplier = 4;
  static constexpr uint64_t buff_alignment = 4096;
};

template <class... Args> void BM_murmur64A(benchmark::State &state, Args &&...args) {
  const auto args_tuple = std::make_tuple(std::move(args)...);
  const auto func = std::get<0>(args_tuple);
  const std::size_t len = state.range(0);

  static RandomDataGenerator gen;
  auto *key = new (std::align_val_t(std::get<1>(args_tuple))) uint8_t [len];
  gen.initArrWithRand(key, len);
  const auto seed = gen.getRandomInt<uint64_t>();

  for (auto _ : state) {
    uint64_t result = func(key, len, seed);
    benchmark::DoNotOptimize(result);
  }

  addByteCounters(state, len);
  addClockCounter(state);

  delete[] key;
}

BENCHMARK_CAPTURE(BM_murmur64A, Ref, &ref::murmur64A, BM_murmur64A_args::buff_alignment)
->RangeMultiplier(BM_murmur64A_args::range_multiplier)
    ->Range(BM_murmur64A_args::min_length, BM_murmur64A_args::max_length);
BENCHMARK_CAPTURE(BM_murmur64A, Neon, &neon::murmur64A, BM_murmur64A_args::buff_alignment)
->RangeMultiplier(BM_murmur64A_args::range_multiplier)
    ->Range(BM_murmur64A_args::min_length, BM_murmur64A_args::max_length);
BENCHMARK_CAPTURE(BM_murmur64A, SVE, &sve::murmur64A, BM_murmur64A_args::buff_alignment)
->RangeMultiplier(BM_murmur64A_args::range_multiplier)
    ->Range(BM_murmur64A_args::min_length, BM_murmur64A_args::max_length);

#endif // NEON_SVE_BENCH_MURMUR64A_BENCHMARK_H
