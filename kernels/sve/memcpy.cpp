#include "sve_common.h"

void *sve_kernels::memcpy(void *dest, const void *src, std::size_t count) {
  constexpr uint64_t n = 4;
  const uint64_t vl = svcntb();
  const uint64_t vLen = count - (count % (vl * n));

  svbool_t p0 = svptrue_b8();
  for (std::size_t i = 0; i < vLen; i += vl * n) {
    svuint8_t v0 = svld1(p0, (const uint8_t *)src + i + 0 * vl);
    svuint8_t v1 = svld1(p0, (const uint8_t *)src + i + 1 * vl);
    svuint8_t v2 = svld1(p0, (const uint8_t *)src + i + 2 * vl);
    svuint8_t v3 = svld1(p0, (const uint8_t *)src + i + 3 * vl);

    svst1(p0, (uint8_t *)dest + i + 0 * vl, v0);
    svst1(p0, (uint8_t *)dest + i + 1 * vl, v1);
    svst1(p0, (uint8_t *)dest + i + 2 * vl, v2);
    svst1(p0, (uint8_t *)dest + i + 3 * vl, v3);
  }

  for (std::size_t i = vLen; i < count; i += vl) {
    p0 = svwhilelt_b8_u64(i, count);
    svuint8_t v0 = svld1(p0, (const uint8_t *)src + i);
    svst1(p0, (uint8_t *)dest + i, v0);
  }

  return dest;
}