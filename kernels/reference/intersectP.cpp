#include "reference_common.h"

void reference_kernels::intersectP(const Bounds3f *b, const Vec3f *rO, const float *rayTMax, const Vec3f *iD,
                                   const int *dirIsNeg, bool *result) {
  constexpr int x = 0;
  constexpr int y = 1;
  constexpr int z = 2;
  const Bounds3f &bounds = *b;
  const Vec3f &rayOrig = *rO;
  const Vec3f &invRayDir = *iD;

  float tMin = (bounds[dirIsNeg[0]][x] - rayOrig[x]) * invRayDir[x];
  float tMax = (bounds[1 - dirIsNeg[0]][x] - rayOrig[x]) * invRayDir[x];
  float tyMin = (bounds[dirIsNeg[1]][y] - rayOrig[y]) * invRayDir[y];
  float tyMax = (bounds[1 - dirIsNeg[1]][y] - rayOrig[y]) * invRayDir[y];

  tMax *= 1 + 2 * EPSILON_F;
  tyMax *= 1 + 2 * EPSILON_F;

  if (tMin > tyMax || tyMin > tMax) {
    *result = false;
    return;
  }
  if (tyMin > tMin)
    tMin = tyMin;
  if (tyMax < tMax)
    tMax = tyMax;

  float tzMin = (bounds[dirIsNeg[2]][z] - rayOrig[z]) * invRayDir[z];
  float tzMax = (bounds[1 - dirIsNeg[2]][z] - rayOrig[z]) * invRayDir[z];

  tzMax *= 1 + 2 * EPSILON_F;

  if (tMin > tzMax || tzMin > tMax) {
    *result = false;
    return;
  }
  if (tzMin > tMin)
    tMin = tzMin;
  if (tzMax < tMax)
    tMax = tzMax;

  *result = (tMin < *rayTMax) && (tMax > 0);
}
