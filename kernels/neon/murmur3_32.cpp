#include "neon_common.h"

std::size_t neon_kernels::murmur3_32Width() { return 4; }

void neon_kernels::murmur3_32(const uint8_t *key, const size_t *len, uint32_t seed, uint32_t *out) {
  //TODO
}