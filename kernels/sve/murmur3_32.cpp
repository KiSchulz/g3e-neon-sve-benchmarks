#include "sve_common.h"

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

uint32_t sve_kernels::murmur3_32(const uint8_t *key, size_t len, uint32_t seed) {
  constexpr uint64_t n = 2;
  const uint64_t vl = svcntw();
  const uint64_t len32 = len / sizeof(uint32_t);
  const uint64_t vLen = len32 - (len32 % (vl * n));
  const svbool_t vTrue = svptrue_b32();

  const auto *data = (const uint32_t*)key;
  auto *buff = (uint32_t *)alloca(sizeof(uint32_t) * vl * n);

  uint32_t h = seed;
  for (uint64_t i = 0; i < vLen; i += vl * n) {
    svuint32_t k0 = svld1_u32(vTrue, data + i + 0 * vl);
    svuint32_t k1 = svld1_u32(vTrue, data + i + 1 * vl);

    k0 = svmul_n_u32_x(vTrue, k0, 0xcc9e2d51);
    k1 = svmul_n_u32_x(vTrue, k1, 0xcc9e2d51);
    k0 = svorr_x(vTrue, svlsl_n_u32_x(vTrue, k0, 15), svlsr_n_u32_x(vTrue, k0, 17));
    k1 = svorr_x(vTrue, svlsl_n_u32_x(vTrue, k1, 15), svlsr_n_u32_x(vTrue, k1, 17));
    k0 = svmul_n_u32_x(vTrue, k0, 0x1b873593);
    k1 = svmul_n_u32_x(vTrue, k1, 0x1b873593);

    svst1(vTrue, buff + 0 * vl, k0);
    svst1(vTrue, buff + 1 * vl, k1);

    h = mergeIntoHash(h, buff, vl * n);
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