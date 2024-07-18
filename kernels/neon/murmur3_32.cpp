#include "neon_common.h"

__always_inline uint32_t murmur_32_scramble(uint32_t k) {
  k *= 0xcc9e2d51;
  k = (k << 15) | (k >> 17);
  k *= 0x1b873593;
  return k;
}

uint32_t neon_kernels::murmur3_32(const uint8_t *key, size_t len, uint32_t seed) {
  constexpr std::size_t n = 1;
  constexpr std::size_t numEl = reg_width / sizeof(uint32_t);
  const std::size_t len32 = len / sizeof(uint32_t);
  const std::size_t vLen = len32 - (len32 % (numEl * n));


  uint32_t h = seed;
  for (std::size_t i = 0; i < vLen; i += numEl * n) {
    uint32x4_t k = vld1q_u32(((const uint32_t *)key) + i);

    k = vmulq_n_u32(k, 0xcc9e2d51);
    k = vorrq_u32(vshlq_n_u32(k, 15), vshrq_n_u32(k, 17));
    k = vmulq_n_u32(k, 0x1b873593);

#pragma clang loop unroll_count(numEl)
    for (std::size_t j = 0; j < numEl; j++) {
      h ^= k[j];
      h = (h << 13) | (h >> 19);
      h = h * 5 + 0xe6546b64;
    }
  }

  uint32_t k;
  // 1. loop-tail
  for (std::size_t i = vLen; i < len32; i++) {
    k = *(((const uint32_t*)key) + i);
    h ^= murmur_32_scramble(k);

    h = (h << 13) | (h >> 19);
    h = h * 5 + 0xe6546b64;
  }

  key += len32 * sizeof(uint32_t);
  k = 0;
  // 2. loop-tail
  for (std::size_t i = len & 3; i; i--) {
    k <<= 8;
    k |= key[i - 1];
  }

  h ^= murmur_32_scramble(k);

  h ^= len;
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;
  return h;
}