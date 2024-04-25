#include "sve_common.h"

int sve_kernels::memcmp(const void *in_lhs, const void *in_rhs, std::size_t count) {
  const auto *lhs = (const uint8_t *)in_lhs;
  const auto *rhs = (const uint8_t *)in_rhs;

  const auto *lhs_end = lhs + count;

  while (lhs < lhs_end) {
    const svbool_t pred = svwhilelt_b8_u64((uint64_t)lhs, (uint64_t)lhs_end);
    const svuint8_t l = svld1_u8(pred, lhs);
    const svuint8_t r = svld1_u8(pred, rhs);
    const svbool_t cmp_res = svcmpne_u8(pred, l, r);

    if (svptest_any(pred, cmp_res)) {
      const svbool_t first_ne_mask = svbrka_b_z(pred, cmp_res);
      const int l_ne = svlastb_u8(first_ne_mask, l);
      const int r_ne = svlastb_u8(first_ne_mask, r);
      return l_ne - r_ne;
    }

    const uint64_t vl = svcntp_b8(pred, pred);
    lhs += vl;
    rhs += vl;
  }

  return 0;
}