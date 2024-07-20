#include "sve_common.h"

__always_inline uint32_t murmur_32_scramble(uint32_t k) {
  k *= 0xcc9e2d51;
  k = (k << 15) | (k >> 17);
  k *= 0x1b873593;
  return k;
}

uint32_t sve_kernels::murmur3_32(const uint8_t *key, size_t len, uint32_t seed) {
  constexpr uint64_t n = 1;
  const uint64_t vl = svcntw();
  const uint64_t len32 = len / sizeof(uint32_t);
  const uint64_t vLen = len32 - (len32 % (vl * n));
  svbool_t vTrue = svptrue_b32();

  uint32_t h = seed;
  for (uint64_t i = 0; i < vLen; i += vl * n) {
    svuint32_t k = svld1_u32(vTrue, ((const uint32_t*)key) + i);

    k = svmul_n_u32_x(vTrue, k, 0xcc9e2d51);
    k = svorr_x(vTrue, svlsl_n_u32_x(vTrue, k, 15), svlsr_n_u32_x(vTrue, k, 17));
    k = svmul_n_u32_x(vTrue, k, 0x1b873593);

    svbool_t pred = svpfalse();
#pragma clang loop unroll_count(4)
    for (uint64_t j = 0; j < vl; j++) {
      pred = svpnext_b32(vTrue, pred);
      uint32_t k_j = svlastb_u32(pred, k);

      h ^= k_j;
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