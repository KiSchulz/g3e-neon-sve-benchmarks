#ifndef NEON_SVE_BENCH_REFERENCE_KERNELS_H
#define NEON_SVE_BENCH_REFERENCE_KERNELS_H

#include <cstdint>

namespace reference_kernels {
uint64_t helloReference();
int memcmp(const void* lhs, const void* rhs, std::size_t count);
}

#endif // NEON_SVE_BENCH_REFERENCE_KERNELS_H
