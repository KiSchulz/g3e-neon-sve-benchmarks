#include "sve_common.h"

int sve_kernels::memcmp(const void *in_lhs, const void *in_rhs, std::size_t count) {
  constexpr uint64_t n = 4;
  const uint64_t vl = svcntb();
  const uint64_t vLen = count - (count % (vl * n));
  const auto *lhs = (const uint8_t *)in_lhs;
  const auto *rhs = (const uint8_t *)in_rhs;

  svbool_t p0 = svptrue_b8();
  for (; lhs < (const uint8_t*)in_lhs + vLen; lhs += n * vl, rhs += n * vl) {
    // load data
    svuint8_t l0 = svld1_u8(p0, lhs + 0 * vl);
    svuint8_t l1 = svld1_u8(p0, lhs + 1 * vl);
    svuint8_t l2 = svld1_u8(p0, lhs + 2 * vl);
    svuint8_t ll = svld1_u8(p0, lhs + 3 * vl);

    svuint8_t r0 = svld1_u8(p0, rhs + 0 * vl);
    svuint8_t r1 = svld1_u8(p0, rhs + 1 * vl);
    svuint8_t r2 = svld1_u8(p0, rhs + 2 * vl);
    svuint8_t rl = svld1_u8(p0, rhs + 3 * vl);

    svuint8_t e0 = sveor_x(p0, l0, r0);
    svuint8_t e1 = sveor_x(p0, l1, r1);
    svuint8_t e2 = sveor_x(p0, l2, r2);
    svuint8_t el = sveor_x(p0, ll, rl);

    svuint8_t et0 = sveor_x(p0, e0, e1);
    svuint8_t et1 = sveor_x(p0, e2, el);
    et0 = sveor_x(p0, et0, et1);

    // test if any active element was not equal in l0 and r0
    if (svmaxv(p0, et0)) [[unlikely]] {
      break;
    }
  }

  for (; lhs < (const uint8_t*)in_lhs + count; lhs += vl, rhs += vl) {
    p0 = svwhilelt_b8_u64((uint64_t)lhs, (uint64_t)in_lhs + count);
    svuint8_t l = svld1(p0, lhs);
    svuint8_t r = svld1(p0, rhs);
    svuint8_t e = sveor_x(p0, l, r);
    if (svmaxv(p0, e)) {
      // set all elements to active until the first that were not equal in l0 and r0, including the not equal element
      const svbool_t firstNeMask = svbrka_b_z(svptrue_b8(), svcmpne(p0, l, r));
      // extract the last active element in l0 and r0 (the not equal elements)
      const int lNe = svlastb_u8(firstNeMask, l);
      const int rNe = svlastb_u8(firstNeMask, r);
      return lNe - rNe;
    }
  }

  return 0;
}