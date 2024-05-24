#include "sve_common.h"

uint64_t sve_kernels::accumulate(const uint64_t *arr, std::size_t len) {
  const uint64_t vl = svcntd();

  svuint64_t acc0 = svdup_u64(0);
  svuint64_t acc1 = svdup_u64(0);
  for (std::size_t i = 0; i < len; i += vl * 2) {
    svbool_t pred0 = svwhilelt_b64_u64(i, len);
    svbool_t pred1 = svwhilelt_b64_u64(i + vl, len);
    svuint64_t v0 = svld1(pred0, arr + i);
    svuint64_t v1 = svld1(pred1, arr + i + vl);
    acc0 = svadd_u64_m(pred0, acc0, v0);
    acc1 = svadd_u64_m(pred1, acc1, v1);
  }
  acc0 = svadd_u64_x(svptrue_b64(), acc0, acc1);
  return svaddv_u64(svptrue_b64(), acc0);
}