#include "neon_common.h"

const uint64_t m = 0xc6a4a7935bd1e995ul;

__always_inline uint64_t mergeIntoHash(uint64_t h, const uint64_t *buff, uint64_t len) {
#pragma clang loop unroll_count(2)
  for (uint64_t j = 0; j < len; j++) {
    h ^= buff[j];
    h *= m;
  }
  return h;
}

uint64_t neon_kernels::murmur64A(const uint8_t *key, size_t len, uint64_t seed) {
  constexpr std::size_t n = 2;
  constexpr std::size_t numEl = reg_width / sizeof(uint64_t);
  const std::size_t len64 = len / sizeof(uint64_t);
  const std::size_t vLen = len64 - (len64 % (numEl * n));

  const int r = 47;

  const auto *data = (const uint64_t *)key;
  auto *buff = (uint64_t *)alloca(sizeof(uint64_t) * numEl * n);

  uint64_t h = seed ^ (len * m);
#pragma clang loop unroll_count(4)
  for (std::size_t i = 0; i < vLen; i += numEl * n) {
    uint64x2_t k0 = vld1q_u64(data + i + 0 * numEl);
    uint64x2_t k1 = vld1q_u64(data + i + 1 * numEl);

    k0 *= m;
    k1 *= m;
    k0 = veorq_u64(k0, vshrq_n_u64(k0, r));
    k1 = veorq_u64(k1, vshrq_n_u64(k1, r));
    k0 *= m;
    k1 *= m;

    vst1q_u64(buff + 0 * numEl, k0);
    vst1q_u64(buff + 1 * numEl, k1);

    h = mergeIntoHash(h, buff, numEl * n);
  }

  // 1. loop-tail
  for (std::size_t i = vLen; i < len64; i++) {
    uint64_t k = *(data + i);

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }

  // 2. loop-tail
  const auto *data2 = (const unsigned char *)(data + len64);

  switch (len & 7) {
  case 7:
    h ^= uint64_t(data2[6]) << 48;
  case 6:
    h ^= uint64_t(data2[5]) << 40;
  case 5:
    h ^= uint64_t(data2[4]) << 32;
  case 4:
    h ^= uint64_t(data2[3]) << 24;
  case 3:
    h ^= uint64_t(data2[2]) << 16;
  case 2:
    h ^= uint64_t(data2[1]) << 8;
  case 1:
    h ^= uint64_t(data2[0]);
    h *= m;
  };

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
}