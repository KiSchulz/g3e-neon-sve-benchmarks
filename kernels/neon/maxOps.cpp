#include "neon_common.h"

template <> float32_t neon_kernels::maxOps(std::size_t n_ops) {
  constexpr std::size_t num_instruction_per_iteration = 8;
  constexpr std::size_t ops_per_instruction_per_lane = 2;
  constexpr std::size_t num_lanes = 4;
  const std::size_t ops_per_iter = num_instruction_per_iteration * ops_per_instruction_per_lane * num_lanes;

  float32x4_t acc0 = vdupq_n_f32(1);
  float32x4_t acc1 = vdupq_n_f32(2);
  float32x4_t acc2 = vdupq_n_f32(3);
  float32x4_t acc3 = vdupq_n_f32(5);
  float32x4_t acc4 = vdupq_n_f32(7);
  float32x4_t acc5 = vdupq_n_f32(11);
  float32x4_t acc6 = vdupq_n_f32(13);
  float32x4_t acc7 = vdupq_n_f32(17);

  const float32x4_t fac0 = vdupq_n_f32(2);
  const float32x4_t fac1 = vdupq_n_f32(3);
  const float32x4_t fac2 = vdupq_n_f32(5);
  const float32x4_t fac3 = vdupq_n_f32(7);

  for (std::size_t i = 0; i < n_ops; i += ops_per_iter) {
    acc0 = vfmaq_f32(acc0, fac0, fac1);
    acc1 = vfmaq_f32(acc1, fac0, fac2);
    acc2 = vfmaq_f32(acc2, fac0, fac3);
    acc3 = vfmaq_f32(acc3, fac1, fac0);
    acc4 = vfmaq_f32(acc4, fac1, fac2);
    acc5 = vfmaq_f32(acc5, fac1, fac3);
    acc6 = vfmaq_f32(acc6, fac2, fac0);
    acc7 = vfmaq_f32(acc7, fac2, fac1);
  }

  acc0 = vaddq_f32(acc0, acc1);
  acc0 = vaddq_f32(acc0, acc2);
  acc0 = vaddq_f32(acc0, acc3);
  acc0 = vaddq_f32(acc0, acc4);
  acc0 = vaddq_f32(acc0, acc5);
  acc0 = vaddq_f32(acc0, acc6);
  acc0 = vaddq_f32(acc0, acc7);

  return vaddvq_f32(acc0);
}

template <> float64_t neon_kernels::maxOps(std::size_t n_ops) {
  constexpr std::size_t num_instruction_per_iteration = 8;
  constexpr std::size_t ops_per_instruction_per_lane = 2;
  constexpr std::size_t num_lanes = 2;
  const std::size_t ops_per_iter = num_instruction_per_iteration * ops_per_instruction_per_lane * num_lanes;

  float64x2_t acc0 = vdupq_n_f64(1);
  float64x2_t acc1 = vdupq_n_f64(2);
  float64x2_t acc2 = vdupq_n_f64(3);
  float64x2_t acc3 = vdupq_n_f64(5);
  float64x2_t acc4 = vdupq_n_f64(7);
  float64x2_t acc5 = vdupq_n_f64(11);
  float64x2_t acc6 = vdupq_n_f64(13);
  float64x2_t acc7 = vdupq_n_f64(17);

  const float64x2_t fac0 = vdupq_n_f64(2);
  const float64x2_t fac1 = vdupq_n_f64(3);
  const float64x2_t fac2 = vdupq_n_f64(5);
  const float64x2_t fac3 = vdupq_n_f64(7);

  for (std::size_t i = 0; i < n_ops; i += ops_per_iter) {
    acc0 = vfmaq_f64(acc0, fac0, fac1);
    acc1 = vfmaq_f64(acc1, fac0, fac2);
    acc2 = vfmaq_f64(acc2, fac0, fac3);
    acc3 = vfmaq_f64(acc3, fac1, fac0);
    acc4 = vfmaq_f64(acc4, fac1, fac2);
    acc5 = vfmaq_f64(acc5, fac1, fac3);
    acc6 = vfmaq_f64(acc6, fac2, fac0);
    acc7 = vfmaq_f64(acc7, fac2, fac1);
  }

  acc0 = vaddq_f64(acc0, acc1);
  acc0 = vaddq_f64(acc0, acc2);
  acc0 = vaddq_f64(acc0, acc3);
  acc0 = vaddq_f64(acc0, acc4);
  acc0 = vaddq_f64(acc0, acc5);
  acc0 = vaddq_f64(acc0, acc6);
  acc0 = vaddq_f64(acc0, acc7);

  return vaddvq_f64(acc0);
}

template <> uint32_t neon_kernels::maxOps(std::size_t n_ops) {
  constexpr std::size_t num_instruction_per_iteration = 8;
  constexpr std::size_t ops_per_instruction_per_lane = 2;
  constexpr std::size_t num_lanes = 4;
  const std::size_t ops_per_iter = num_instruction_per_iteration * ops_per_instruction_per_lane * num_lanes;

  uint32x4_t acc0 = vdupq_n_u32(1);
  uint32x4_t acc1 = vdupq_n_u32(2);
  uint32x4_t acc2 = vdupq_n_u32(3);
  uint32x4_t acc3 = vdupq_n_u32(5);
  uint32x4_t acc4 = vdupq_n_u32(7);
  uint32x4_t acc5 = vdupq_n_u32(11);
  uint32x4_t acc6 = vdupq_n_u32(13);
  uint32x4_t acc7 = vdupq_n_u32(17);

  const uint32x4_t fac0 = vdupq_n_u32(2);
  const uint32x4_t fac1 = vdupq_n_u32(3);
  const uint32x4_t fac2 = vdupq_n_u32(5);
  const uint32x4_t fac3 = vdupq_n_u32(7);

  for (std::size_t i = 0; i < n_ops; i += ops_per_iter) {
    acc0 = vmlaq_u32(acc0, fac0, fac1);
    acc1 = vmlaq_u32(acc1, fac0, fac2);
    acc2 = vmlaq_u32(acc2, fac0, fac3);
    acc3 = vmlaq_u32(acc3, fac1, fac0);
    acc4 = vmlaq_u32(acc4, fac1, fac2);
    acc5 = vmlaq_u32(acc5, fac1, fac3);
    acc6 = vmlaq_u32(acc6, fac2, fac0);
    acc7 = vmlaq_u32(acc7, fac2, fac1);
  }

  acc0 = vaddq_u32(acc0, acc1);
  acc0 = vaddq_u32(acc0, acc2);
  acc0 = vaddq_u32(acc0, acc3);
  acc0 = vaddq_u32(acc0, acc4);
  acc0 = vaddq_u32(acc0, acc5);
  acc0 = vaddq_u32(acc0, acc6);
  acc0 = vaddq_u32(acc0, acc7);

  return vaddvq_u32(acc0);
}

template <> uint64_t neon_kernels::maxOps(std::size_t n_ops) {
  //TODO fix this to do the same as the sve version
  constexpr std::size_t num_instruction_per_iteration = 8;
  constexpr std::size_t ops_per_instruction_per_lane = 2;
  constexpr std::size_t num_lanes = 2;
  const std::size_t ops_per_iter = num_instruction_per_iteration * ops_per_instruction_per_lane * num_lanes;

  uint64x2_t acc0 = vdupq_n_u64(1);
  uint64x2_t acc1 = vdupq_n_u64(2);
  uint64x2_t acc2 = vdupq_n_u64(3);
  uint64x2_t acc3 = vdupq_n_u64(5);
  uint64x2_t acc4 = vdupq_n_u64(7);
  uint64x2_t acc5 = vdupq_n_u64(11);
  uint64x2_t acc6 = vdupq_n_u64(13);
  uint64x2_t acc7 = vdupq_n_u64(17);

  const uint64x2_t fac0 = vdupq_n_u64(2);
  const uint64x2_t fac1 = vdupq_n_u64(3);
  const uint64x2_t fac2 = vdupq_n_u64(5);
  const uint64x2_t fac3 = vdupq_n_u64(7);

  for (std::size_t i = 0; i < n_ops; i += ops_per_iter) {
    acc0 = vfmaq_f64(acc0, fac0, fac1);
    acc1 = vfmaq_f64(acc1, fac0, fac2);
    acc2 = vfmaq_f64(acc2, fac0, fac3);
    acc3 = vfmaq_f64(acc3, fac1, fac0);
    acc4 = vfmaq_f64(acc4, fac1, fac2);
    acc5 = vfmaq_f64(acc5, fac1, fac3);
    acc6 = vfmaq_f64(acc6, fac2, fac0);
    acc7 = vfmaq_f64(acc7, fac2, fac1);
  }

  acc0 = vaddq_u64(acc0, acc1);
  acc0 = vaddq_u64(acc0, acc2);
  acc0 = vaddq_u64(acc0, acc3);
  acc0 = vaddq_u64(acc0, acc4);
  acc0 = vaddq_u64(acc0, acc5);
  acc0 = vaddq_u64(acc0, acc6);
  acc0 = vaddq_u64(acc0, acc7);

  return vaddvq_u64(acc0);
}
