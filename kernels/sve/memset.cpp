#include "sve_common.h"
#include <cstring>
void *sve_kernels::memset(void *dest, int ch, std::size_t count) {
  constexpr uint64_t n = 4;
  const uint64_t vl = svcntb();
  const uint64_t vLen = count - (count % (vl * n));
  const svuint8_t vCh = svdup_u8(static_cast<unsigned char>(ch));

  svbool_t p0 = svptrue_b8();
  for (std::size_t i = 0; i < vLen; i += vl * n) {
    svst1(p0, (unsigned char *)dest + i + 0 * vl, vCh);
    svst1(p0, (unsigned char *)dest + i + 1 * vl, vCh);
    svst1(p0, (unsigned char *)dest + i + 2 * vl, vCh);
    svst1(p0, (unsigned char *)dest + i + 3 * vl, vCh);
  }

  for (std::size_t i = vLen; i < count; i += vl) {
    p0 = svwhilelt_b8_u64(i, count);
    svst1(p0, (unsigned char *)dest + i, vCh);
  }

  return dest;
}