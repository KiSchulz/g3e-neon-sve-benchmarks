#ifndef NEON_SVE_BENCH_INTERSECTP_BENCHMARK_H
#define NEON_SVE_BENCH_INTERSECTP_BENCHMARK_H

#include "benchmark_common.h"

#include "common/random_data_generator.h"
#include "common/types.h"

struct BM_intersectP_args {
  static constexpr int64_t min_numBoxes = 64;
  static constexpr int64_t max_numBoxes = 2 << 15;
  static constexpr int64_t ray_reuse_ratio = 32;
  static constexpr int range_multiplier = 2;
  static constexpr uint64_t buff_alignment = 4096;
};

template <class... Args> void BM_intersectP(benchmark::State &state, Args &&...args) {
  const auto args_tuple = std::make_tuple(std::move(args)...);
  const auto func = std::get<0>(args_tuple);
  const std::size_t width = std::get<1>(args_tuple);
  const std::size_t rayReuseRatio = std::get<2>(args_tuple);
  const std::size_t numBoxes = state.range(0);

  const std::size_t numRays = numBoxes / rayReuseRatio;

  auto *boxes = new (std::align_val_t(std::get<3>(args_tuple))) Bounds3f[numBoxes];
  auto *orig = new (std::align_val_t(std::get<3>(args_tuple))) Vec3f[numRays];
  auto *dir = new (std::align_val_t(std::get<3>(args_tuple))) Vec3f[numRays];
  float tMax = std::numeric_limits<float>::infinity();
  int *results = (int *)alloca(width * sizeof(int));

  Bounds3f baseBox{Vec3f{-1, -1, -1}, Vec3f{1, 1, 1}};
  RandomDataGenerator gen;
  gen.initAABBArr(baseBox, boxes, numBoxes);
  gen.initRayArr(baseBox, orig, dir, numRays);

  for (auto _ : state) {
    std::size_t boxOffset = 0;
    for (std::size_t i = 0; i < numRays; i++) {
      Vec3f invDir = dir[i].invertElements();
      int dirIsNeg[3];
      for (std::size_t k = 0; k < 3; k++) {
        dirIsNeg[k] = invDir[k] < 0 ? 1 : 0;
      }
      for (std::size_t j = 0; j < rayReuseRatio; j += width, boxOffset += width) {
        func(boxes + boxOffset, &orig[i], &tMax, &invDir, dirIsNeg, results);

        benchmark::DoNotOptimize(results);
        benchmark::ClobberMemory();
      }
    }
  }

  state.counters["num_intersects"] = (double)numBoxes;

  delete[] boxes;
  delete[] orig;
  delete[] dir;
}

BENCHMARK_CAPTURE(BM_intersectP, Ref, &ref::intersectP, ref::intersectPWidth(), BM_intersectP_args::ray_reuse_ratio,
                  BM_intersectP_args::buff_alignment)
    ->RangeMultiplier(BM_intersectP_args::range_multiplier)
    ->Range(BM_intersectP_args::min_numBoxes, BM_intersectP_args::max_numBoxes);
BENCHMARK_CAPTURE(BM_intersectP, Neon, &neon::intersectP, neon::intersectPWidth(), BM_intersectP_args::ray_reuse_ratio,
                  BM_intersectP_args::buff_alignment)
    ->RangeMultiplier(BM_intersectP_args::range_multiplier)
    ->Range(BM_intersectP_args::min_numBoxes, BM_intersectP_args::max_numBoxes);
BENCHMARK_CAPTURE(BM_intersectP, SVE, &sve::intersectP, sve::intersectPWidth(), BM_intersectP_args::ray_reuse_ratio,
                  BM_intersectP_args::buff_alignment)
    ->RangeMultiplier(BM_intersectP_args::range_multiplier)
    ->Range(BM_intersectP_args::min_numBoxes, BM_intersectP_args::max_numBoxes);

#endif // NEON_SVE_BENCH_INTERSECTP_BENCHMARK_H
