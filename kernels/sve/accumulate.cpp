#include "sve_common.h"

uint64_t sve_kernels::accumulate(const uint64_t *arr, std::size_t len) {
  constexpr uint64_t n = 4;
  const uint64_t vl = svcntd();
  const uint64_t vLen = len - (len % (vl * n));

  svuint64_t acc0 = svdup_u64(0);
  svuint64_t acc1 = svdup_u64(0);
  svuint64_t acc2 = svdup_u64(0);
  svuint64_t acc3 = svdup_u64(0);
  svbool_t pred0 = svptrue_b64();
  svbool_t pred1 = svptrue_b64();
  svbool_t pred2 = svptrue_b64();
  svbool_t pred3 = svptrue_b64();
  for (std::size_t i = 0; i < vLen; i += vl * n) {
    svuint64_t v0 = svld1(pred0, arr + i + 0 * vl);
    svuint64_t v1 = svld1(pred1, arr + i + 1 * vl);
    svuint64_t v2 = svld1(pred2, arr + i + 2 * vl);
    svuint64_t v3 = svld1(pred3, arr + i + 3 * vl);

    acc0 = svadd_u64_m(pred0, acc0, v0);
    acc1 = svadd_u64_m(pred1, acc1, v1);
    acc2 = svadd_u64_m(pred2, acc2, v2);
    acc3 = svadd_u64_m(pred3, acc3, v3);
  }

  for (std::size_t i = vLen; i < len; i += vl) {
    pred0 = svwhilelt_b64_u64(i, len);
    svuint64_t v0 = svld1(pred0, arr + i);
    acc0 = svadd_u64_m(pred0, acc0, v0);
  }

  acc0 = svadd_u64_x(svptrue_b64(), acc0, acc1);
  acc2 = svadd_u64_x(svptrue_b64(), acc2, acc3);

  acc0 = svadd_u64_x(svptrue_b64(), acc0, acc2);
  return svaddv_u64(svptrue_b64(), acc0);
}