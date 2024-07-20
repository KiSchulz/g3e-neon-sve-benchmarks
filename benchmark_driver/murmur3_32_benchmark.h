#ifndef NEON_SVE_BENCH_MURMUR3_32_BENCHMARK_H
#define NEON_SVE_BENCH_MURMUR3_32_BENCHMARK_H

#include "benchmark_common.h"

#include "common/random_data_generator.h"

struct BM_murmur3_32_args {
  static constexpr int64_t min_length = 1;       // in byte
  static constexpr int64_t max_length = 1 << 26; // in byte
  static constexpr int range_multiplier = 4;
  static constexpr uint64_t buff_alignment = 4096;
};

template <class... Args> void BM_murmur3_32(benchmark::State &state, Args &&...args) {
  const auto args_tuple = std::make_tuple(std::move(args)...);
  const auto func = std::get<0>(args_tuple);
  const std::size_t len = state.range(0);

  static RandomDataGenerator gen;
  auto *key = new (std::align_val_t(std::get<1>(args_tuple))) uint8_t [len];
  gen.initArrWithRand(key, len);
  const auto seed = gen.getRandomInt<uint32_t>();

  for (auto _ : state) {
    uint32_t result = func(key, len, seed);
    benchmark::DoNotOptimize(result);
  }

  addByteCounters(state, len);
  addClockCounter(state);

  delete[] key;
}

BENCHMARK_CAPTURE(BM_murmur3_32, Ref, &ref::murmur3_32, BM_murmur3_32_args::buff_alignment)
->RangeMultiplier(BM_murmur3_32_args::range_multiplier)
    ->Range(BM_murmur3_32_args::min_length, BM_murmur3_32_args::max_length)->ThreadPerCpu();
BENCHMARK_CAPTURE(BM_murmur3_32, Neon, &neon::murmur3_32, BM_murmur3_32_args::buff_alignment)
->RangeMultiplier(BM_murmur3_32_args::range_multiplier)
    ->Range(BM_murmur3_32_args::min_length, BM_murmur3_32_args::max_length)->ThreadPerCpu();
BENCHMARK_CAPTURE(BM_murmur3_32, SVE, &sve::murmur3_32, BM_murmur3_32_args::buff_alignment)
->RangeMultiplier(BM_murmur3_32_args::range_multiplier)
    ->Range(BM_murmur3_32_args::min_length, BM_murmur3_32_args::max_length)->ThreadPerCpu();

#endif // NEON_SVE_BENCH_MURMUR3_32_BENCHMARK_H
