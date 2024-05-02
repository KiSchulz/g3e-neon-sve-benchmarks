#ifndef NEON_SVE_BENCH_REFERENCE_KERNELS_H
#define NEON_SVE_BENCH_REFERENCE_KERNELS_H

#include <cstdint>

namespace reference_kernels {
uint64_t helloReference();
int memcmp(const void *lhs, const void *rhs, std::size_t count);
void nBody_step(double *px, double *py, double *pz, double *vx, double *vy, double *vz, const double *m, double dt,
                std::size_t len);
} // namespace reference_kernels

#endif // NEON_SVE_BENCH_REFERENCE_KERNELS_H
