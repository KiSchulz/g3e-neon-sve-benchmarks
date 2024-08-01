#include "sve_common.h"

template <> float32_t sve_kernels::maxOps(std::size_t n_ops) {
  constexpr std::size_t num_instruction_per_iteration = 8;
  constexpr std::size_t ops_per_instruction_per_lane = 2;
  const std::size_t num_lanes = svcntw();
  const std::size_t ops_per_iter = num_instruction_per_iteration * ops_per_instruction_per_lane * num_lanes;

  const svbool_t vTrue = svptrue_b32();
  svfloat32_t acc0 = svdup_f32(1);
  svfloat32_t acc1 = svdup_f32(2);
  svfloat32_t acc2 = svdup_f32(3);
  svfloat32_t acc3 = svdup_f32(5);
  svfloat32_t acc4 = svdup_f32(7);
  svfloat32_t acc5 = svdup_f32(11);
  svfloat32_t acc6 = svdup_f32(13);
  svfloat32_t acc7 = svdup_f32(17);

  const svfloat32_t fac0 = svdup_f32(2);
  const svfloat32_t fac1 = svdup_f32(3);
  const svfloat32_t fac2 = svdup_f32(5);
  const svfloat32_t fac3 = svdup_f32(7);

  for (std::size_t i = 0; i < n_ops; i += ops_per_iter) {
    acc0 = svmad_f32_x(vTrue, acc0, fac0, fac1);
    acc1 = svmad_f32_x(vTrue, acc1, fac0, fac2);
    acc2 = svmad_f32_x(vTrue, acc2, fac0, fac3);
    acc3 = svmad_f32_x(vTrue, acc3, fac1, fac0);
    acc4 = svmad_f32_x(vTrue, acc4, fac1, fac2);
    acc5 = svmad_f32_x(vTrue, acc5, fac1, fac3);
    acc6 = svmad_f32_x(vTrue, acc6, fac2, fac0);
    acc7 = svmad_f32_x(vTrue, acc7, fac2, fac1);
  }

  acc0 = svadd_f32_x(vTrue, acc0, acc1);
  acc0 = svadd_f32_x(vTrue, acc0, acc2);
  acc0 = svadd_f32_x(vTrue, acc0, acc3);
  acc0 = svadd_f32_x(vTrue, acc0, acc4);
  acc0 = svadd_f32_x(vTrue, acc0, acc5);
  acc0 = svadd_f32_x(vTrue, acc0, acc6);
  acc0 = svadd_f32_x(vTrue, acc0, acc7);

  return svadda_f32(vTrue, 0, acc0);
}

template <> float64_t sve_kernels::maxOps(std::size_t n_ops) {
  constexpr std::size_t num_instruction_per_iteration = 8;
  constexpr std::size_t ops_per_instruction_per_lane = 2;
  const std::size_t num_lanes = svcntd();
  const std::size_t ops_per_iter = num_instruction_per_iteration * ops_per_instruction_per_lane * num_lanes;

  const svbool_t vTrue = svptrue_b32();
  svfloat64_t acc0 = svdup_f64(1);
  svfloat64_t acc1 = svdup_f64(2);
  svfloat64_t acc2 = svdup_f64(3);
  svfloat64_t acc3 = svdup_f64(5);
  svfloat64_t acc4 = svdup_f64(7);
  svfloat64_t acc5 = svdup_f64(11);
  svfloat64_t acc6 = svdup_f64(13);
  svfloat64_t acc7 = svdup_f64(17);

  const svfloat64_t fac0 = svdup_f64(2);
  const svfloat64_t fac1 = svdup_f64(3);
  const svfloat64_t fac2 = svdup_f64(5);
  const svfloat64_t fac3 = svdup_f64(7);

  for (std::size_t i = 0; i < n_ops; i += ops_per_iter) {
    acc0 = svmad_f64_x(vTrue, acc0, fac0, fac1);
    acc1 = svmad_f64_x(vTrue, acc1, fac0, fac2);
    acc2 = svmad_f64_x(vTrue, acc2, fac0, fac3);
    acc3 = svmad_f64_x(vTrue, acc3, fac1, fac0);
    acc4 = svmad_f64_x(vTrue, acc4, fac1, fac2);
    acc5 = svmad_f64_x(vTrue, acc5, fac1, fac3);
    acc6 = svmad_f64_x(vTrue, acc6, fac2, fac0);
    acc7 = svmad_f64_x(vTrue, acc7, fac2, fac1);
  }

  acc0 = svadd_f64_x(vTrue, acc0, acc1);
  acc0 = svadd_f64_x(vTrue, acc0, acc2);
  acc0 = svadd_f64_x(vTrue, acc0, acc3);
  acc0 = svadd_f64_x(vTrue, acc0, acc4);
  acc0 = svadd_f64_x(vTrue, acc0, acc5);
  acc0 = svadd_f64_x(vTrue, acc0, acc6);
  acc0 = svadd_f64_x(vTrue, acc0, acc7);

  return svadda_f64(vTrue, 0, acc0);
}

template <> uint32_t sve_kernels::maxOps(std::size_t n_ops) {
  constexpr std::size_t num_instruction_per_iteration = 8;
  constexpr std::size_t ops_per_instruction_per_lane = 2;
  const std::size_t num_lanes = svcntw();
  const std::size_t ops_per_iter = num_instruction_per_iteration * ops_per_instruction_per_lane * num_lanes;

  const svbool_t vTrue = svptrue_b32();
  svuint32_t acc0 = svdup_u32(1);
  svuint32_t acc1 = svdup_u32(3);
  svuint32_t acc2 = svdup_u32(5);
  svuint32_t acc3 = svdup_u32(7);
  svuint32_t acc4 = svdup_u32(11);
  svuint32_t acc5 = svdup_u32(13);
  svuint32_t acc6 = svdup_u32(17);
  svuint32_t acc7 = svdup_u32(19);

  const svuint32_t fac0 = svdup_u32(23);
  const svuint32_t fac1 = svdup_u32(29);
  const svuint32_t fac2 = svdup_u32(31);
  const svuint32_t fac3 = svdup_u32(37);
  const svuint32_t fac4 = svdup_u32(41);
  const svuint32_t fac5 = svdup_u32(43);
  const svuint32_t fac6 = svdup_u32(47);
  const svuint32_t fac7 = svdup_u32(53);

  for (std::size_t i = 0; i < n_ops; i += ops_per_iter) {
    acc0 = svmad_u32_x(vTrue, acc0, fac0, fac7);
    acc1 = svmad_u32_x(vTrue, acc1, fac1, fac6);
    acc2 = svmad_u32_x(vTrue, acc2, fac2, fac5);
    acc3 = svmad_u32_x(vTrue, acc3, fac3, fac4);
    acc4 = svmad_u32_x(vTrue, acc4, fac4, fac3);
    acc5 = svmad_u32_x(vTrue, acc5, fac5, fac2);
    acc6 = svmad_u32_x(vTrue, acc6, fac6, fac1);
    acc7 = svmad_u32_x(vTrue, acc7, fac7, fac0);
  }

  acc0 = svadd_u32_x(vTrue, acc0, acc1);
  acc0 = svadd_u32_x(vTrue, acc0, acc2);
  acc0 = svadd_u32_x(vTrue, acc0, acc3);
  acc0 = svadd_u32_x(vTrue, acc0, acc4);
  acc0 = svadd_u32_x(vTrue, acc0, acc5);
  acc0 = svadd_u32_x(vTrue, acc0, acc6);
  acc0 = svadd_u32_x(vTrue, acc0, acc7);

  return svaddv_u32(vTrue, acc0);
}

template <> uint64_t sve_kernels::maxOps(std::size_t n_ops) {
  constexpr std::size_t num_instruction_per_iteration = 8;
  constexpr std::size_t ops_per_instruction_per_lane = 1;
  const std::size_t num_lanes = svcntd();
  const std::size_t ops_per_iter = num_instruction_per_iteration * ops_per_instruction_per_lane * num_lanes;

  const svbool_t vTrue = svptrue_b32();
  svuint64_t acc0 = svdup_u64(0);
  svuint64_t acc1 = svdup_u64(0);
  svuint64_t acc2 = svdup_u64(0);
  svuint64_t acc3 = svdup_u64(0);
  svuint64_t acc4 = svdup_u64(0);
  svuint64_t acc5 = svdup_u64(0);
  svuint64_t acc6 = svdup_u64(0);
  svuint64_t acc7 = svdup_u64(0);

  const svuint64_t fac0 = svdup_u64(1);
  const svuint64_t fac1 = svdup_u64(2);
  const svuint64_t fac2 = svdup_u64(3);
  const svuint64_t fac3 = svdup_u64(5);
  const svuint64_t fac4 = svdup_u64(7);
  const svuint64_t fac5 = svdup_u64(11);
  const svuint64_t fac6 = svdup_u64(13);
  const svuint64_t fac7 = svdup_u64(17);

  for (std::size_t i = 0; i < n_ops; i += ops_per_iter) {
    //acc0 = svmad_u64_x(vTrue, acc0, fac0, fac7);
    //acc1 = svmad_u64_x(vTrue, acc1, fac1, fac6);
    //acc2 = svmad_u64_x(vTrue, acc2, fac2, fac5);
    //acc3 = svmad_u64_x(vTrue, acc3, fac3, fac4);
    //acc4 = svmad_u64_x(vTrue, acc4, fac4, fac3);
    //acc5 = svmad_u64_x(vTrue, acc5, fac5, fac2);
    //acc6 = svmad_u64_x(vTrue, acc6, fac6, fac1);
    //acc7 = svmad_u64_x(vTrue, acc7, fac7, fac0);

    acc0 = svadd_u64_x(vTrue, acc0, fac0);
    acc1 = svadd_u64_x(vTrue, acc1, fac1);
    acc2 = svadd_u64_x(vTrue, acc2, fac2);
    acc3 = svadd_u64_x(vTrue, acc3, fac3);
    acc4 = svadd_u64_x(vTrue, acc4, fac4);
    acc5 = svadd_u64_x(vTrue, acc5, fac5);
    acc6 = svadd_u64_x(vTrue, acc6, fac6);
    acc7 = svadd_u64_x(vTrue, acc7, fac7);
  }

  acc0 = svadd_u64_x(vTrue, acc0, acc1);
  acc0 = svadd_u64_x(vTrue, acc0, acc2);
  acc0 = svadd_u64_x(vTrue, acc0, acc3);
  acc0 = svadd_u64_x(vTrue, acc0, acc4);
  acc0 = svadd_u64_x(vTrue, acc0, acc5);
  acc0 = svadd_u64_x(vTrue, acc0, acc6);
  acc0 = svadd_u64_x(vTrue, acc0, acc7);

  return svaddv_u64(vTrue, acc0);
}
