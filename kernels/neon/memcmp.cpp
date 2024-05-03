#include "neon_common.h"

int neon_kernels::memcmp(const void *in_lhs, const void *in_rhs, std::size_t count) {
  const auto *lhs = (const uint64_t *)in_lhs;
  const auto *rhs = (const uint64_t *)in_rhs;

  const auto *lhs_end = (const uint64_t *)((const uint8_t *)in_lhs + count - (count % reg_width));

  constexpr uint64_t ptr_inc = reg_width / sizeof(*lhs);
  for (; lhs < lhs_end; lhs += ptr_inc, rhs += ptr_inc) {
    // load the data
    const uint64x2_t l = vld1q_u64(lhs);
    const uint64x2_t r = vld1q_u64(rhs);
    // perform xor
    const uint64x2_t eor = veorq_u64(l, r);

    // find the max uint32_t value and check if it is not 0
    if (vmaxvq_u32(vreinterpretq_u32_u64(eor))) {
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