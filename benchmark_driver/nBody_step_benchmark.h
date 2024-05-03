#ifndef NEON_SVE_BENCH_NBODY_STEP_BENCHMARK_H
#define NEON_SVE_BENCH_NBODY_STEP_BENCHMARK_H

#include "benchmark_common.h"

#include "common/random_data_generator.h"

struct BM_nBody_step_args {
  static constexpr int64_t min_num_Bodies = 1;
  static constexpr int64_t max_num_Bodies = 1024;
};

template <class... Args> void BM_nBody_step(benchmark::State &state, Args &&...args) {
  const auto args_tuple = std::make_tuple(std::move(args)...);
  const auto func = std::get<0>(args_tuple);

  for (auto _ : state) {
    //func();
  }
}

BENCHMARK_CAPTURE(BM_nBody_step, Ref, &ref::nBody_step)
    ->Range(BM_nBody_step_args::min_num_Bodies, BM_nBody_step_args::max_num_Bodies);
BENCHMARK_CAPTURE(BM_nBody_step, Neon, &neon::nBody_step)
    ->Range(BM_nBody_step_args::min_num_Bodies, BM_nBody_step_args::max_num_Bodies);
BENCHMARK_CAPTURE(BM_nBody_step, SVE, &sve::nBody_step)
    ->Range(BM_nBody_step_args::min_num_Bodies, BM_nBody_step_args::max_num_Bodies);

#endif // NEON_SVE_BENCH_NBODY_STEP_BENCHMARK_H
