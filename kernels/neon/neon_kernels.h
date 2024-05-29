#ifndef NEON_SVE_BENCH_NEON_KERNELS_H
#define NEON_SVE_BENCH_NEON_KERNELS_H

#include <cstdint>

#include "common/types.h"

namespace neon_kernels {
uint64_t helloNeon();

int memcmp(const void *lhs, const void *rhs, std::size_t count);

uint64_t accumulate(const uint64_t *arr, std::size_t len);

template <class T> T maxOps(std::size_t n_ops);

template <bool fastMath = false>
void nBody_step(double *px, double *py, double *pz, double *vx, double *vy, double *vz, const double *m, double dt,
                std::size_t len);

std::size_t intersectPWidth();

void intersectP(const Bounds3f *vResult, const Vec3f *rayOrig, const float *rayTMax, const Vec3f *invRayDir,
                const int *dirIsNeg, int *result);

} // namespace neon_kernels

#endif // NEON_SVE_BENCH_NEON_KERNELS_H
