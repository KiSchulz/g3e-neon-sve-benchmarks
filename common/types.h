#ifndef NEON_SVE_BENCH_TYPES_H
#define NEON_SVE_BENCH_TYPES_H

#include <cmath>

struct Vec3f {
  float v[3];

  float operator[](std::size_t i) const { return v[i]; }
  float &operator[](std::size_t i) { return v[i]; }
  Vec3f operator-(const Vec3f &o) const { return {v[0] - o.v[0], v[1] - o.v[1], v[2] - o.v[2]}; }

  [[nodiscard]] float norm() const { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
  [[nodiscard]] Vec3f normalize() const { return {v[0] / norm(), v[1] / norm(), v[2] / norm()}; }
  [[nodiscard]] Vec3f invertElements() const { return {1 / v[0], 1 / v[1], 1 / v[2]}; }
};

// AABB
struct Bounds3f {
  Vec3f v[2];

  Vec3f operator[](std::size_t i) const { return v[i]; }
  Vec3f &operator[](std::size_t i) { return v[i]; }
};

#endif // NEON_SVE_BENCH_TYPES_H
