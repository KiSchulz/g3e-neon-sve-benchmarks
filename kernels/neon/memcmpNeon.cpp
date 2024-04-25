#include "neon_common.h"

int neon_kernels::memcmp(const void *in_lhs, const void *in_rhs, std::size_t count) {
  constexpr int reg_width = 16;
  const auto *lhs = (const uint8_t *)in_lhs;
  const auto *rhs = (const uint8_t *)in_rhs;

  const auto *lhs_end = lhs + count;

  for (; lhs + reg_width < lhs_end; lhs += reg_width, rhs += reg_width) {
    // load the data
    const uint8x16_t l = vld1q_u8(lhs);
    const uint8x16_t r = vld1q_u8(rhs);
    // perform bitwise compare (1 iff l[i] == r[i])
    const uint8x16_t cmp_res = vceqq_u8(l, r);
    // count the number of bits set in each byte of the register
    const uint8x16_t cnt_res = vcntq_u8(cmp_res);
    // accumulate the all the bytes into one register
    const uint8_t num_set_bits = vaddvq_u8(cnt_res);
    // check if one bit was not set
    if (num_set_bits != 8 * reg_width) {
      break;
    }
  }

  // loop tail
  for (; lhs < lhs_end; lhs++, rhs++) {
    if (*lhs != *rhs) {
      return (int)*lhs - (int)*rhs;
    }
  }

  return 0;
}