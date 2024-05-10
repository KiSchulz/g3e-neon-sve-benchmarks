#ifndef NEON_SVE_BENCH_NBODY_STEP_BENCHMARK_H
#define NEON_SVE_BENCH_NBODY_STEP_BENCHMARK_H

#include "benchmark_common.h"

#include "common/random_data_generator.h"

struct BM_nBody_step_args {
  static constexpr int64_t min_num_Bodies = 1;
  static constexpr int64_t max_num_Bodies = 8192;
  static constexpr int range_multiplier = 2;
  static constexpr uint64_t buff_alignment = 4096;
};

template <class... Args> void BM_nBody_step(benchmark::State &state, Args &&...args) {
  const auto args_tuple = std::make_tuple(std::move(args)...);
  const auto func = std::get<0>(args_tuple);

  const int64_t n = state.range(0);
  double *buff[7];
  for (auto &i : buff) {
    i = new (std::align_val_t(std::get<1>(args_tuple))) double[n];
    benchmark::DoNotOptimize(i);
  }

  RandomDataGenerator gen;
  gen.initNBodySystem(buff[0], buff[1], buff[2], buff[3], buff[4], buff[5], buff[6], n);

  for (auto _ : state) {
    func(buff[0], buff[1], buff[2], buff[3], buff[4], buff[5], buff[6], 0.01, n);
    benchmark::ClobberMemory();
  }

  for (auto &i : buff) {
    delete[] i;
  }
}

BENCHMARK_CAPTURE(BM_nBody_step, Ref, &ref::nBody_step, BM_nBody_step_args::buff_alignment)
    ->RangeMultiplier(BM_nBody_step_args::range_multiplier)
    ->Range(BM_nBody_step_args::min_num_Bodies, BM_nBody_step_args::max_num_Bodies);
BENCHMARK_CAPTURE(BM_nBody_step, Neon, &neon::nBody_step<false>, BM_nBody_step_args::buff_alignment)
    ->RangeMultiplier(BM_nBody_step_args::range_multiplier)
    ->Range(BM_nBody_step_args::min_num_Bodies, BM_nBody_step_args::max_num_Bodies);
BENCHMARK_CAPTURE(BM_nBody_step, SVE, &sve::nBody_step<false>, BM_nBody_step_args::buff_alignment)
    ->RangeMultiplier(BM_nBody_step_args::range_multiplier)
    ->Range(BM_nBody_step_args::min_num_Bodies, BM_nBody_step_args::max_num_Bodies);
BENCHMARK_CAPTURE(BM_nBody_step, Neon_fastMath, &neon::nBody_step<true>, BM_nBody_step_args::buff_alignment)
->RangeMultiplier(BM_nBody_step_args::range_multiplier)
    ->Range(BM_nBody_step_args::min_num_Bodies, BM_nBody_step_args::max_num_Bodies);
BENCHMARK_CAPTURE(BM_nBody_step, SVE_fastMath, &sve::nBody_step<true>, BM_nBody_step_args::buff_alignment)
->RangeMultiplier(BM_nBody_step_args::range_multiplier)
    ->Range(BM_nBody_step_args::min_num_Bodies, BM_nBody_step_args::max_num_Bodies);

#endif // NEON_SVE_BENCH_NBODY_STEP_BENCHMARK_H
