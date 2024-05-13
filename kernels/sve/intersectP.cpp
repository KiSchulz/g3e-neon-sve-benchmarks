#include "sve_common.h"

std::size_t sve_kernels::intersectPWidth() { return svcntw(); }

void sve_kernels::intersectP(const Bounds3f *b, const Vec3f *rayOrig, const float *rayTMax, const Vec3f *invRayDir,
                             const int *dirIsNeg, int *result) {
  constexpr int x = 0;
  constexpr int y = 1;
  constexpr int z = 2;
  const svuint32_t vIndices = svindex_u32(0, sizeof(Bounds3f) / sizeof(float32_t));
  constexpr float32_t widenFac = 1 + 2 * EPSILON_F;

  svbool_t vResult = svptrue_b32();
  auto computeT = [&](int axis, int bIdx) {
    auto *base = (float32_t *)((uint8_t *)b + sizeof(Vec3f) * bIdx + sizeof(float32_t) * axis);

    svfloat32_t res = svld1_gather_u32index_f32(vResult, base, vIndices);
    const svfloat32_t o = svdup_f32((*rayOrig)[axis]);
    const svfloat32_t id = svdup_f32((*invRayDir)[axis]);

    res = svsub_f32_x(vResult, res, o);
    res = svmul_f32_x(vResult, res, id);

    return res;
  };

  auto updateResult = [&](svfloat32_t tMin, svfloat32_t tMax, svfloat32_t tnMin, svfloat32_t tnMax) {
    const svbool_t cond1 = svcmple_f32(vResult, tMin, tnMax);
    const svbool_t cond2 = svcmple_f32(vResult, tnMin, tMax);
    return svand_b_z(cond1, cond2, vResult);
  };

  enum class UpdateType { min, max };
  auto updateT = [&]<UpdateType u>(svfloat32_t t, svfloat32_t tn) {
    svbool_t pred;
    if constexpr (u == UpdateType::min) {
      pred = svcmpgt_f32(vResult, tn, t);
    } else {
      pred = svcmplt_f32(vResult, tn, t);
    }
    return svsel_f32(pred, tn, t);
  };

  // float tMin = (bounds[dirIsNeg[0]][x] - rayOrig[x]) * invRayDir[x];
  // float tMax = (bounds[1 - dirIsNeg[0]][x] - rayOrig[x]) * invRayDir[x];
  svfloat32_t tMin = computeT(x, dirIsNeg[x]);
  svfloat32_t tMax = computeT(x, 1 - dirIsNeg[x]);
  // float tyMin = (bounds[dirIsNeg[1]][y] - rayOrig[y]) * invRayDir[y];
  // float tyMax = (bounds[1 - dirIsNeg[1]][y] - rayOrig[y]) * invRayDir[y];
  svfloat32_t tyMin = computeT(y, dirIsNeg[y]);
  svfloat32_t tyMax = computeT(y, 1 - dirIsNeg[y]);

  // tMax *= 1 + 2 * EPSILON_F;
  // tyMax *= 1 + 2 * EPSILON_F;
  tMax = svmul_n_f32_x(vResult, tMax, widenFac);
  tyMax = svmul_n_f32_x(vResult, tyMax, widenFac);

  // if (tMin > tyMax || tyMin > tMax) {
  //   *result = false;
  //   return;
  // }
  // if (tyMin > tMin)
  //   tMin = tyMin;
  // if (tyMax < tMax)
  //   tMax = tyMax;
  vResult = updateResult(tMin, tMax, tyMin, tyMax);
  tMin = updateT.operator()<UpdateType::min>(tMin, tyMin);
  tMax = updateT.operator()<UpdateType::max>(tMax, tyMax);

  // float tzMin = (bounds[dirIsNeg[2]][z] - rayOrig[z]) * invRayDir[z];
  // float tzMax = (bounds[1 - dirIsNeg[2]][z] - rayOrig[z]) * invRayDir[z];
  svfloat32_t tzMin = computeT(z, dirIsNeg[z]);
  svfloat32_t tzMax = computeT(z, 1 - dirIsNeg[z]);

  // tzMax *= 1 + 2 * EPSILON_F;
  tzMax = svmul_n_f32_x(vResult, tzMax, widenFac);

  // if (tMin > tzMax || tzMin > tMax) {
  //   *result = false;
  //   return;
  // }
  // if (tzMin > tMin)
  //   tMin = tzMin;
  // if (tzMax < tMax)
  //   tMax = tzMax;
  vResult = updateResult(tMin, tMax, tzMin, tzMax);
  tMin = updateT.operator()<UpdateType::min>(tMin, tzMin);
  tMax = updateT.operator()<UpdateType::max>(tMax, tzMax);

  //*result = (tMin < *rayTMax) && (tMax > 0);
  const svfloat32_t vTMax = svdup_f32_x(vResult, *rayTMax);
  const svbool_t cond1 = svcmplt_f32(vResult, tMin, vTMax);
  const svbool_t cond2 = svcmpgt_n_f32(vResult, tMax, 0);
  vResult = svand_b_z(cond1, cond2, vResult);

  // using some instruction to filter set inactive elements to zero
  svint32_t res = svdup_n_s32(0);
  res = svnot_s32_z(vResult, res);

  svst1_s32(svptrue_b32(), result, res);
}