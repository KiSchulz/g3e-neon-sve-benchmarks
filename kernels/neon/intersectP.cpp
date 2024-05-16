#include "neon_common.h"

std::size_t neon_kernels::intersectPWidth() { return 4; }

void neon_kernels::intersectP(const Bounds3f *b, const Vec3f *rayOrig, const float *rayTMax, const Vec3f *invRayDir,
                              const int *dirIsNeg, int *result) {
  constexpr int x = 0;
  constexpr int y = 1;
  constexpr int z = 2;
  const uint32x4_t vZeros = vdupq_n_u32(0);
  const float32_t widenFac = 1 + 2 * EPSILON_F;

  auto computeT = [&](int axis, int bIdx) {
    float32x4_t pMin_axis = {b[0][0][axis], b[1][0][axis], b[2][0][axis], b[3][0][axis]};
    float32x4_t pMax_axis = {b[0][1][axis], b[1][1][axis], b[2][1][axis], b[3][1][axis]};
    float32x4_t res = bIdx ? pMax_axis : pMin_axis;

    float32x4_t o = vdupq_n_f32((*rayOrig)[axis]);
    res = vsubq_f32(res, o);
    float32x4_t id = vdupq_n_f32((*invRayDir)[axis]);
    res = vmulq_f32(res, id);

    return res;
  };

  auto updateResult = [&](uint32x4_t vResult, float32x4_t tMin, float32x4_t tMax, float32x4_t tnMin, float32x4_t tnMax) {
    uint32x4_t cond1 = vcleq_f32(tMin, tnMax);
    uint32x4_t cond2 = vcleq_f32(tnMin, tMax);
    return vandq_u32(vResult, vandq_u32(cond1, cond2));
  };

  enum class UpdateType { min, max };
  auto updateT = [&]<UpdateType u>(float32x4_t t, float32x4_t tn) {
    uint32x4_t mask;
    if constexpr (u == UpdateType::min) {
      mask = vcgtq_f32(tn, t);
    } else {
      mask = vcltq_f32(tn, t);
    }
    return vbslq_f32(mask, tn, t);
  };

  uint32x4_t vResult = vmvnq_u32(vZeros);
  // float tMin = (bounds[dirIsNeg[0]][x] - rayOrig[x]) * invRayDir[x];
  // float tMax = (bounds[1 - dirIsNeg[0]][x] - rayOrig[x]) * invRayDir[x];
  float32x4_t tMin = computeT(x, dirIsNeg[x]);
  float32x4_t tMax = computeT(x, 1 - dirIsNeg[x]);

  // float tyMin = (bounds[dirIsNeg[1]][y] - rayOrig[y]) * invRayDir[y];
  // float tyMax = (bounds[1 - dirIsNeg[1]][y] - rayOrig[y]) * invRayDir[y];
  float32x4_t tyMin = computeT(y, dirIsNeg[y]);
  float32x4_t tyMax = computeT(y, 1 - dirIsNeg[y]);

  // tMax *= 1 + 2 * EPSILON_F;
  // tyMax *= 1 + 2 * EPSILON_F;
  tMax = vmulq_n_f32(tMax, widenFac);
  tyMax = vmulq_n_f32(tyMax, widenFac);

  // if (tMin > tyMax || tyMin > tMax) {
  //   *result = false;
  //   return;
  // }
  // if (tyMin > tMin)
  //   tMin = tyMin;
  // if (tyMax < tMax)
  //   tMax = tyMax;
  vResult = updateResult(vResult, tMin, tMax, tyMin, tyMax);
  tMin = updateT.operator()<UpdateType::min>(tMin, tyMin);
  tMax = updateT.operator()<UpdateType::max>(tMax, tyMax);

  // float tzMin = (bounds[dirIsNeg[2]][z] - rayOrig[z]) * invRayDir[z];
  // float tzMax = (bounds[1 - dirIsNeg[2]][z] - rayOrig[z]) * invRayDir[z];
  float32x4_t tzMin = computeT(z, dirIsNeg[z]);
  float32x4_t tzMax = computeT(z, 1 - dirIsNeg[z]);

  // tzMax *= 1 + 2 * EPSILON_F;
  tzMax = vmulq_n_f32(tzMax, widenFac);

  // if (tMin > tzMax || tzMin > tMax) {
  //   *result = false;
  //   return;
  // }
  // if (tzMin > tMin)
  //   tMin = tzMin;
  // if (tzMax < tMax)
  //   tMax = tzMax;
  vResult = updateResult(vResult, tMin, tMax, tzMin, tzMax);
  tMin = updateT.operator()<UpdateType::min>(tMin, tzMin);
  tMax = updateT.operator()<UpdateType::max>(tMax, tzMax);

  //*result = (tMin < *rayTMax) && (tMax > 0);
  float32x4_t vTMax = vdupq_n_f32(*rayTMax);
  uint32x4_t cond1 = vcltq_f32(tMin, vTMax);
  uint32x4_t cond2 = vcgtq_f32(tMax, vreinterpretq_f32_u32(vZeros));
  vResult = vandq_u32(vResult, vandq_u32(cond1, cond2));
  vst1q_u32((uint32_t *)result, vResult);
}