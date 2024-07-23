#include "neon_common.h"

__always_inline uint32_t murmur_32_scramble(uint32_t k) {
  k *= 0xcc9e2d51;
  k = (k << 15) | (k >> 17);
  k *= 0x1b873593;
  return k;
}

__always_inline uint32_t mergeIntoHash(uint32_t h, const uint32_t *buff, uint64_t len) {
#pragma clang loop unroll_count(4)
  for (uint64_t j = 0; j < len; j++) {
    h ^= buff[j];
    h = (h << 13) | (h >> 19);
    h = h * 5 + 0xe6546b64;
  }
  return h;
}

uint32_t neon_kernels::murmur3_32(const uint8_t *key, size_t len, uint32_t seed) {
  constexpr std::size_t n = 4;
  constexpr std::size_t numEl = reg_width / sizeof(uint32_t);
  const std::size_t len32 = len / sizeof(uint32_t);
  const std::size_t vLen = len32 - (len32 % (numEl * n));

  const auto *data = (const uint32_t *)key;
  auto *buff = (uint32_t *)alloca(sizeof(uint32_t) * numEl * n);

  uint32_t h = seed;
  for (std::size_t i = 0; i < vLen; i += numEl * n) {
    uint32x4_t k0 = vld1q_u32(data + i + 0 * numEl);
    uint32x4_t k1 = vld1q_u32(data + i + 1 * numEl);
    uint32x4_t k2 = vld1q_u32(data + i + 2 * numEl);
    uint32x4_t k3 = vld1q_u32(data + i + 3 * numEl);

    k0 = vmulq_n_u32(k0, 0xcc9e2d51);
    k1 = vmulq_n_u32(k1, 0xcc9e2d51);
    k2 = vmulq_n_u32(k2, 0xcc9e2d51);
    k3 = vmulq_n_u32(k3, 0xcc9e2d51);
    k0 = vorrq_u32(vshlq_n_u32(k0, 15), vshrq_n_u32(k0, 17));
    k1 = vorrq_u32(vshlq_n_u32(k1, 15), vshrq_n_u32(k1, 17));
    k2 = vorrq_u32(vshlq_n_u32(k2, 15), vshrq_n_u32(k2, 17));
    k3 = vorrq_u32(vshlq_n_u32(k3, 15), vshrq_n_u32(k3, 17));
    k0 = vmulq_n_u32(k0, 0x1b873593);
    k1 = vmulq_n_u32(k1, 0x1b873593);
    k2 = vmulq_n_u32(k2, 0x1b873593);
    k3 = vmulq_n_u32(k3, 0x1b873593);

    vst1q_u32(buff + 0 * numEl, k0);
    vst1q_u32(buff + 1 * numEl, k1);
    vst1q_u32(buff + 2 * numEl, k2);
    vst1q_u32(buff + 3 * numEl, k3);

    h = mergeIntoHash(h, buff, numEl * n);
  }

  uint32_t k;
  // 1. loop-tail
  for (std::size_t i = vLen; i < len32; i++) {
    k = *(data + i);
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