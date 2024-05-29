#include "sve_common.h"

template<> float32_t sve_kernels::vectorLoadFactor(std::size_t n_inst, std::size_t maxActiveElements) {
  constexpr std::size_t num_instruction_per_iteration = 8;

  const std::size_t num_lanes = std::min(svcntw(), maxActiveElements);
  const svbool_t pred = svwhilelt_b32(0ul, num_lanes);
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

  for (std::size_t i = 0; i < n_inst; i += num_instruction_per_iteration) {
    acc0 = svmad_f32_x(pred, acc0, fac0, fac1);
    acc1 = svmad_f32_x(pred, acc1, fac0, fac2);
    acc2 = svmad_f32_x(pred, acc2, fac0, fac3);
    acc3 = svmad_f32_x(pred, acc3, fac1, fac0);
    acc4 = svmad_f32_x(pred, acc4, fac1, fac2);
    acc5 = svmad_f32_x(pred, acc5, fac1, fac3);
    acc6 = svmad_f32_x(pred, acc6, fac2, fac0);
    acc7 = svmad_f32_x(pred, acc7, fac2, fac1);
  }

  acc0 = svadd_f32_x(pred, acc0, acc1);
  acc0 = svadd_f32_x(pred, acc0, acc2);
  acc0 = svadd_f32_x(pred, acc0, acc3);
  acc0 = svadd_f32_x(pred, acc0, acc4);
  acc0 = svadd_f32_x(pred, acc0, acc5);
  acc0 = svadd_f32_x(pred, acc0, acc6);
  acc0 = svadd_f32_x(pred, acc0, acc7);

  return svadda_f32(pred, 0, acc0);
}