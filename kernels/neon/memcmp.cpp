#include "neon_common.h"

int neon_kernels::memcmp(const void *in_lhs, const void *in_rhs, std::size_t count) {
  const auto *lhs = (const uint64_t *)in_lhs;
  const auto *rhs = (const uint64_t *)in_rhs;
  constexpr uint64_t n = 2;
  constexpr uint64_t numEl = (reg_width / sizeof(*lhs));

  const auto *lhs_end = (const uint64_t *)((const uint8_t *)in_lhs + count - (count % (n * reg_width)));

  for (; lhs < lhs_end; lhs += numEl * n, rhs += numEl * n) {
    // load the data
    const uint64x2_t l0 = vld1q_u64(lhs + 0 * numEl);
    const uint64x2_t l1 = vld1q_u64(lhs + 1 * numEl);

    const uint64x2_t r0 = vld1q_u64(rhs + 0 * numEl);
    const uint64x2_t r1 = vld1q_u64(rhs + 1 * numEl);
    // perform xor
    uint64x2_t e0 = veorq_u64(l0, r0);
    uint64x2_t e1 = veorq_u64(l1, r1);
    e0 = veorq_u64(e0, e1);

    // find the max uint32_t value and check if it is not 0
    if (vmaxvq_u32(vreinterpretq_u32_u64(e0))) [[unlikely]] {
      break;
    }
  }

  // creating tail vars of the correct type
  const auto *tail_lhs = (const uint8_t *)lhs;
  const auto *tail_rhs = (const uint8_t *)rhs;
  const auto *tail_lhs_end = (const uint8_t *)in_lhs + count;
  // loop tail
  for (; tail_lhs < tail_lhs_end; tail_lhs++, tail_rhs++) {
    if (*tail_lhs != *tail_rhs) {
      return (int)*tail_lhs - (int)*tail_rhs;
    }
  }

  return 0;
}