#ifndef NEON_SVE_BENCH_MAXOPS_BENCHMARK_H
#define NEON_SVE_BENCH_MAXOPS_BENCHMARK_H

#include "benchmark_common.h"

struct BM_maxOps_args {
  static constexpr int64_t num_ops = 1ul << 30;
};

template <class... Args> void BM_maxOps(benchmark::State &state, Args &&...args) {
  const auto args_tuple = std::make_tuple(std::move(args)...);
  const auto func = std::get<0>(args_tuple);
  const auto num_ops = std::get<1>(args_tuple);

  state.SetLabel("ops = " + std::to_string(num_ops) + " = 2^" + std::to_string((int)std::log2(num_ops)));

  for (auto _ : state) {
    float result = func(num_ops);
    benchmark::DoNotOptimize(result);
  }
}

BENCHMARK_CAPTURE(BM_maxOps, Neon<float32_t>, &neon::maxOps<float>, BM_maxOps_args::num_ops);
BENCHMARK_CAPTURE(BM_maxOps, Neon<float64_t>, &neon::maxOps<double>, BM_maxOps_args::num_ops);
BENCHMARK_CAPTURE(BM_maxOps, Neon<uint32_t>, &neon::maxOps<uint32_t>, BM_maxOps_args::num_ops);
BENCHMARK_CAPTURE(BM_maxOps, Neon<uint64_t>, &neon::maxOps<uint64_t>, BM_maxOps_args::num_ops);

BENCHMARK_CAPTURE(BM_maxOps, SVE<float32_t>, &sve::maxOps<float>, BM_maxOps_args::num_ops);
BENCHMARK_CAPTURE(BM_maxOps, SVE<float64_t>, &sve::maxOps<double>, BM_maxOps_args::num_ops);
BENCHMARK_CAPTURE(BM_maxOps, SVE<uint32_t>, &sve::maxOps<uint32_t>, BM_maxOps_args::num_ops);
BENCHMARK_CAPTURE(BM_maxOps, SVE<uint64_t>, &sve::maxOps<uint64_t>, BM_maxOps_args::num_ops);

#endif // NEON_SVE_BENCH_MAXOPS_BENCHMARK_H
