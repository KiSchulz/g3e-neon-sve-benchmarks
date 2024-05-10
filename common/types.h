#ifndef NEON_SVE_BENCH_TYPES_H
#define NEON_SVE_BENCH_TYPES_H

struct Vec3f {
  float v[3];

  float operator[](std::size_t i) const { return v[i]; }
  float &operator[](std::size_t i) { return v[i]; }

  [[nodiscard]] Vec3f invertElements() const { return {1 / v[0], 1 / v[1], 1 / v[2]}; }
};

// AABB
struct Bounds3f {
  Vec3f v[2];

  Vec3f operator[](std::size_t i) const { return v[i]; }
  Vec3f &operator[](std::size_t i) { return v[i]; }
};

#endif // NEON_SVE_BENCH_TYPES_H
