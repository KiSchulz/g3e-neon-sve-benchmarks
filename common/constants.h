#ifndef NEON_SVE_BENCH_CONSTANTS_H
#define NEON_SVE_BENCH_CONSTANTS_H

#include <limits>

namespace physics {
constexpr double G = 6.67430e-11; // gravitational constant
}

constexpr double EPSILON_D = std::numeric_limits<double>::epsilon();
constexpr float EPSILON_F = std::numeric_limits<float>::epsilon();
constexpr float INFINITY_F = std::numeric_limits<float>::infinity();

#endif // NEON_SVE_BENCH_CONSTANTS_H
