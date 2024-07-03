#include "neon_common.h"

void *neon_kernels::memset(void *dest, int ch, std::size_t count) {
  constexpr std::size_t n = 8;
  constexpr std::size_t numEl = reg_width / sizeof(unsigned char);
  const std::size_t vLen = count - (count % (numEl * n));
  const uint8x16_t vCh = vdupq_n_u8(static_cast<unsigned char>(ch));

  for (std::size_t i = 0; i < vLen; i += numEl * n) {
    vst1q_u8((unsigned char *)dest + i + 0 * numEl, vCh);
    vst1q_u8((unsigned char *)dest + i + 1 * numEl, vCh);
    vst1q_u8((unsigned char *)dest + i + 2 * numEl, vCh);
    vst1q_u8((unsigned char *)dest + i + 3 * numEl, vCh);
    vst1q_u8((unsigned char *)dest + i + 4 * numEl, vCh);
    vst1q_u8((unsigned char *)dest + i + 5 * numEl, vCh);
    vst1q_u8((unsigned char *)dest + i + 6 * numEl, vCh);
    vst1q_u8((unsigned char *)dest + i + 7 * numEl, vCh);
  }

  for (std::size_t i = vLen; i < count; i++) {
    ((unsigned char *)dest)[i] = static_cast<unsigned char>(ch);
  }

  return dest;
}