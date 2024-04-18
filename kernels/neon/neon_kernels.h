#ifndef NEON_SVE_BENCH_NEON_KERNELS_H
#define NEON_SVE_BENCH_NEON_KERNELS_H

#include <cstdint>

namespace neon_kernels {
uint64_t helloNeon();
int memcmp(const void* lhs, const void* rhs, std::size_t count);
} // namespace neon_kernels

#endif // NEON_SVE_BENCH_NEON_KERNELS_H
