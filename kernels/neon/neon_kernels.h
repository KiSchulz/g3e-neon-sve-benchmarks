#ifndef NEON_SVE_BENCH_NEON_KERNELS_H
#define NEON_SVE_BENCH_NEON_KERNELS_H

#include <cstdint>

namespace neon_kernels {
uint64_t helloNeon();
int memcmp(const void *lhs, const void *rhs, std::size_t count);
template<bool fastMath = false>
void nBody_step(double *px, double *py, double *pz, double *vx, double *vy, double *vz, const double *m, double dt,
                std::size_t len);
} // namespace neon_kernels

#endif // NEON_SVE_BENCH_NEON_KERNELS_H
