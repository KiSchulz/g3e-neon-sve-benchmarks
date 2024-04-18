#ifndef NEON_SVE_BENCH_SVE_KERNELS_H
#define NEON_SVE_BENCH_SVE_KERNELS_H

#include <cstdint>

namespace sve_kernels {
uint64_t helloSVE();
int memcmp(const void* lhs, const void* rhs, std::size_t count);
} // namespace sve_kernels

#endif // NEON_SVE_BENCH_SVE_KERNELS_H
