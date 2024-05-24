#include "neon_common.h"

uint64_t neon_kernels::accumulate(const uint64_t *arr, std::size_t len) {
  constexpr std::size_t n = 4;
  const std::size_t vLen = len - (len % (2 * n));
  uint64x2_t vAcc0 = vdupq_n_u64(0);
  uint64x2_t vAcc1 = vdupq_n_u64(0);
  uint64x2_t vAcc2 = vdupq_n_u64(0);
  uint64x2_t vAcc3 = vdupq_n_u64(0);

  for (std::size_t i = 0; i < vLen; i += 2 * n) {
    uint64x2_t v0 = vld1q_u64(arr + i);
    uint64x2_t v1 = vld1q_u64(arr + i + 2);
    uint64x2_t v2 = vld1q_u64(arr + i + 2 + 2);
    uint64x2_t v3 = vld1q_u64(arr + i + 2 + 2 + 2);
    vAcc0 = vaddq_u64(vAcc0, v0);
    vAcc1 = vaddq_u64(vAcc1, v1);
    vAcc2 = vaddq_u64(vAcc2, v2);
    vAcc3 = vaddq_u64(vAcc3, v3);
  }

  vAcc0 = vaddq_u64(vAcc0, vAcc1);
  vAcc2 = vaddq_u64(vAcc2, vAcc3);
  vAcc0 = vaddq_u64(vAcc0, vAcc2);
  uint64_t acc = vaddvq_u64(vAcc0);

  for (std::size_t i = vLen; i < len; i++) {
    acc += arr[i];
  }

  return acc;
}