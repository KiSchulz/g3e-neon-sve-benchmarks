#include "neon_common.h"

uint64_t neon_kernels::accumulate(const uint64_t *arr, std::size_t len) {
  constexpr std::size_t n = 8;
  constexpr std::size_t numEl = reg_width / sizeof(uint64_t);
  const std::size_t vLen = len - (len % (numEl * n));
  uint64x2_t vAcc0 = vdupq_n_u64(0);
  uint64x2_t vAcc1 = vdupq_n_u64(0);
  uint64x2_t vAcc2 = vdupq_n_u64(0);
  uint64x2_t vAcc3 = vdupq_n_u64(0);
  uint64x2_t vAcc4 = vdupq_n_u64(0);
  uint64x2_t vAcc5 = vdupq_n_u64(0);
  uint64x2_t vAcc6 = vdupq_n_u64(0);
  uint64x2_t vAcc7 = vdupq_n_u64(0);

  for (std::size_t i = 0; i < vLen; i += numEl * n) {
    uint64x2_t v0 = vld1q_u64(arr + i + 0 * numEl);
    uint64x2_t v1 = vld1q_u64(arr + i + 1 * numEl);
    uint64x2_t v2 = vld1q_u64(arr + i + 2 * numEl);
    uint64x2_t v3 = vld1q_u64(arr + i + 3 * numEl);
    uint64x2_t v4 = vld1q_u64(arr + i + 4 * numEl);
    uint64x2_t v5 = vld1q_u64(arr + i + 5 * numEl);
    uint64x2_t v6 = vld1q_u64(arr + i + 6 * numEl);
    uint64x2_t v7 = vld1q_u64(arr + i + 7 * numEl);

    vAcc0 = vaddq_u64(vAcc0, v0);
    vAcc1 = vaddq_u64(vAcc1, v1);
    vAcc2 = vaddq_u64(vAcc2, v2);
    vAcc3 = vaddq_u64(vAcc3, v3);
    vAcc4 = vaddq_u64(vAcc4, v4);
    vAcc5 = vaddq_u64(vAcc5, v5);
    vAcc6 = vaddq_u64(vAcc6, v6);
    vAcc7 = vaddq_u64(vAcc7, v7);
  }

  vAcc0 = vaddq_u64(vAcc0, vAcc1);
  vAcc2 = vaddq_u64(vAcc2, vAcc3);
  vAcc4 = vaddq_u64(vAcc4, vAcc5);
  vAcc6 = vaddq_u64(vAcc6, vAcc7);

  vAcc0 = vaddq_u64(vAcc0, vAcc2);
  vAcc4 = vaddq_u64(vAcc4, vAcc6);

  vAcc0 = vaddq_u64(vAcc0, vAcc4);
  uint64_t acc = vaddvq_u64(vAcc0);

  for (std::size_t i = vLen; i < len; i++) {
    acc += arr[i];
  }

  return acc;
}