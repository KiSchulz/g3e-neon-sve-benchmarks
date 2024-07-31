#ifndef NEON_SVE_BENCH_VECTORLOADFACTOR_BENCHMARK_H
#define NEON_SVE_BENCH_VECTORLOADFACTOR_BENCHMARK_H

#include "benchmark_common.h"

struct BM_vectorLoadFactor_args {
  static constexpr int64_t num_inst = 1ul << 20;
};

template <class... Args> void BM_vectorLoadFactor(benchmark::State &state, Args &&...args) {
  const auto args_tuple = std::make_tuple(std::move(args)...);
  const auto func = std::get<0>(args_tuple);
  const auto num_inst = std::get<1>(args_tuple);

  state.SetLabel("inst = " + std::to_string(num_inst) + " = 2^" + std::to_string((int)std::log2(num_inst)));

  const std::size_t maxActiveElements = state.range(0);

  for (auto _ : state) {
    float result = func(num_inst, maxActiveElements);
    benchmark::DoNotOptimize(result);
  }

  state.counters["num_inst"] = (double)num_inst;
  state.counters["active_elements"] = (double)maxActiveElements;
}

BENCHMARK_CAPTURE(BM_vectorLoadFactor, SVE<float32_t>, &sve::vectorLoadFactor<float>,
                  BM_vectorLoadFactor_args::num_inst)
    ->DenseRange(1, 8);

BENCHMARK_CAPTURE(BM_vectorLoadFactor, SVE<uint32_t>, &sve::vectorLoadFactor<uint32_t>,
                  BM_vectorLoadFactor_args::num_inst)
->DenseRange(1, 8);

#endif // NEON_SVE_BENCH_VECTORLOADFACTOR_BENCHMARK_H
