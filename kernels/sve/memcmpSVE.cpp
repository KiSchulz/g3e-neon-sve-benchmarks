#include "sve_common.h"

int sve_kernels::memcmp(const void *in_lhs, const void *in_rhs, std::size_t count) {
  const uint64_t vl = svcntb();
  const auto *lhs = (const uint8_t *)in_lhs;
  const auto *rhs = (const uint8_t *)in_rhs;

  const auto *lhs_end = lhs + count;

  for (; lhs < lhs_end; lhs += vl, rhs += vl) {
    // initializing pred with a fitting length
    const svbool_t pred = svwhilelt_b8_u64((uint64_t)lhs, (uint64_t)lhs_end);
    // load data
    const svuint8_t l = svldff1_u8(pred, lhs);
    const svuint8_t r = svldff1_u8(pred, rhs);
    // perform not equal comparison
    const svbool_t cmp_res = svcmpne(pred, l, r);

    // test if any active element was not equal in l an r
    if (svptest_any(pred, cmp_res)) {
      // set all elements to active until the first that were not equal in l and r, including the not equal element
      const svbool_t first_ne_mask = svbrka_b_z(pred, cmp_res);
      // find the last active element in l and r (the not equal elements)
      const int l_ne = svlastb_u8(first_ne_mask, l);
      const int r_ne = svlastb_u8(first_ne_mask, r);
      return l_ne - r_ne;
    }
  }

  return 0;
}