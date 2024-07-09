#include "sve_common.h"

template <uint32_t version> void *sve_kernels::memcpy(void *dest, const void *src, std::size_t count) {
  const uint64_t vl = svcntb();

  if constexpr (version == 0) {
    for (std::size_t i = 0; i < count; i += vl) {
      svbool_t p = svwhilelt_b8_u64(i, count);
      svst1(p, (uint8_t *)dest + i, svld1(p, (const uint8_t *)src + i));
    }
  } else if (version == 1) {
    const uint64_t vLen = count - (count % vl);
    svbool_t p = svptrue_b8();
    for (std::size_t i = 0; i < vLen; i += vl) {
      svst1(p, (uint8_t *)dest + i, svld1(p, (const uint8_t *)src + i));
    }

    for (std::size_t i = vLen; i < count; i += vl) {
      p = svwhilelt_b8_u64(i, count);
      svst1(p, (uint8_t *)dest + i, svld1(p, (const uint8_t *)src + i));
    }
  } else if (version == 2) {
    constexpr uint64_t n = 4;
    const uint64_t vLen = count - (count % (vl * n));

    svbool_t p = svptrue_b8();
    for (std::size_t i = 0; i < vLen; i += n * vl) {
      svst1(p, (uint8_t *)dest + i + 0 * vl, svld1(p, (const uint8_t *)src + i + 0 * vl));
      svst1(p, (uint8_t *)dest + i + 1 * vl, svld1(p, (const uint8_t *)src + i + 1 * vl));
      svst1(p, (uint8_t *)dest + i + 2 * vl, svld1(p, (const uint8_t *)src + i + 2 * vl));
      svst1(p, (uint8_t *)dest + i + 3 * vl, svld1(p, (const uint8_t *)src + i + 3 * vl));
    }

    for (std::size_t i = vLen; i < count; i += vl) {
      p = svwhilelt_b8_u64(i, count);
      svst1(p, (uint8_t *)dest + i, svld1(p, (const uint8_t *)src + i));
    }
  } else {
    constexpr uint64_t n = 4;
    const uint64_t vLen = count - (count % (vl * n));

    svbool_t p = svptrue_b8();
    for (std::size_t i = 0; i < vLen; i += vl * n) {
      svuint8_t v0 = svld1(p, (const uint8_t *)src + i + 0 * vl);
      svuint8_t v1 = svld1(p, (const uint8_t *)src + i + 1 * vl);
      svuint8_t v2 = svld1(p, (const uint8_t *)src + i + 2 * vl);
      svuint8_t v3 = svld1(p, (const uint8_t *)src + i + 3 * vl);

      svst1(p, (uint8_t *)dest + i + 0 * vl, v0);
      svst1(p, (uint8_t *)dest + i + 1 * vl, v1);
      svst1(p, (uint8_t *)dest + i + 2 * vl, v2);
      svst1(p, (uint8_t *)dest + i + 3 * vl, v3);
    }

    for (std::size_t i = vLen; i < count; i += vl) {
      p = svwhilelt_b8_u64(i, count);
      svuint8_t v0 = svld1(p, (const uint8_t *)src + i);
      svst1(p, (uint8_t *)dest + i, v0);
    }
  }

  return dest;
}

template void *sve_kernels::memcpy<0xffffffff>(void *dest, const void *src, std::size_t count);

template void *sve_kernels::memcpy<0>(void *dest, const void *src, std::size_t count);
template void *sve_kernels::memcpy<1>(void *dest, const void *src, std::size_t count);
template void *sve_kernels::memcpy<2>(void *dest, const void *src, std::size_t count);
template void *sve_kernels::memcpy<3>(void *dest, const void *src, std::size_t count);
