#include "neon_common.h"

void *neon_kernels::memcpy(void *dest, const void *src, std::size_t count) {
  constexpr std::size_t n = 4;
  constexpr std::size_t numEl = reg_width / sizeof(char);
  const std::size_t vLen = count - (count % (numEl * n));

  for (std::size_t i = 0; i < vLen; i += numEl * n) {
    uint8x16_t v0 = vld1q_u8((const uint8_t *)src + i + 0 * numEl);
    uint8x16_t v1 = vld1q_u8((const uint8_t *)src + i + 1 * numEl);
    uint8x16_t v2 = vld1q_u8((const uint8_t *)src + i + 2 * numEl);
    uint8x16_t v3 = vld1q_u8((const uint8_t *)src + i + 3 * numEl);

    vst1q_u8((uint8_t *)dest + i + 0 * numEl, v0);
    vst1q_u8((uint8_t *)dest + i + 1 * numEl, v1);
    vst1q_u8((uint8_t *)dest + i + 2 * numEl, v2);
    vst1q_u8((uint8_t *)dest + i + 3 * numEl, v3);
  }

  for (std::size_t i = vLen; i < count; i++) {
    ((char *)dest)[i] = ((char *)src)[i];
  }

  return dest;
}