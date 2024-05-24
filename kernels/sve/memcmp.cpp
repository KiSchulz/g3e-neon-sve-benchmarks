#include "sve_common.h"

int sve_kernels::memcmp(const void *in_lhs, const void *in_rhs, std::size_t count) {
  const uint64_t vl = svcntb();
  const auto *lhs = (const uint8_t *)in_lhs;
  const auto *rhs = (const uint8_t *)in_rhs;

  const auto *lhs_end = lhs + count;

  svbool_t pred = svwhilelt_b8_u64((uint64_t)lhs, (uint64_t)lhs_end);
  while (lhs < lhs_end) {
    // initializing pred with a fitting length
    // load data
    const svuint8_t l = svld1_u8(pred, lhs);
    const svuint8_t r = svld1_u8(pred, rhs);
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
    lhs += vl;
    rhs += vl;
    pred = svwhilelt_b8_u64((uint64_t)lhs, (uint64_t)lhs_end);
  }

  return 0;

  //__asm__("cmp x2, #0x1\n"
  //        "b.lt _rz\n"
  //        "add x8, x0, x2\n"
  //        "whilelo p0.b, x0, x8\n"
  //        "_l1:\n"
  //        "ldff1b {z1.b}, p0/z, [x0, xzr]\n"
  //        "ldff1b {z0.b}, p0/z, [x1, xzr]\n"
  //        "cmpne p1.b, p0/z, z1.b, z0.b\n"
  //        "b.ne _rne\n"
  //        "add x0, x0, #0x20\n"
  //        "add x1, x1, #0x20\n"
  //        "whilelo p0.b, x0, x8\n"
  //        "b.first _l1\n"
  //        "_rz:\n"
  //        "mov w0, wzr\n"
  //        "ret\n"
  //        "_rne:\n"
  //        "brka p0.b, p0/z, p1.b\n"
  //        "lastb w8, p0, z1.b\n"
  //        "lastb w9, p0, z0.b\n"
  //        "and w8, w8, #0xff\n"
  //        "sub w0, w8, w9, uxtb\n"
  //        "ret\n");
}