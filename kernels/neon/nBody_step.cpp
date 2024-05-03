#include "neon_common.h"

#include <cmath>

// TODO add fast math check
void neon_kernels::nBody_step(double *px, double *py, double *pz, double *vx, double *vy, double *vz, const double *m,
                              double dt, std::size_t len) {
  constexpr std::size_t num_lanes = reg_width / sizeof(*px);
  const bool odd_len = len % 2 != 0;
  // TODO try to improve the constant situation
  const float64x2_t vEpsilon = {EPSILON_D, EPSILON_D};
  const float64x2_t vG = {physics::G, physics::G};

  // TODO use pointers directly instead of indices if not optimized
  for (std::size_t i = 0; i < len; i++) {
    // TODO optimize this initialization of the values
    float64x2_t ax = {0, 0}, ay = {0, 0}, az = {0, 0};

    float64x2_t px_i = {px[i], px[i]};
    float64x2_t py_i = {py[i], py[i]};
    float64x2_t pz_i = {pz[i], pz[i]};
    for (std::size_t j = 0; j < (odd_len ? len - 1 : len); j += num_lanes) {
      const float64x2_t px_j = vld1q_f64(px + j);
      const float64x2_t py_j = vld1q_f64(py + j);
      const float64x2_t pz_j = vld1q_f64(pz + j);

      const float64x2_t dx = vsubq_f64(px_j, px_i);
      const float64x2_t dy = vsubq_f64(py_j, py_i);
      const float64x2_t dz = vsubq_f64(pz_j, pz_i);

      // TODO look at scalar multiply add for epsilon
      float64x2_t r2 = vfmaq_f64(vEpsilon, dx, dx);
      r2 = vfmaq_f64(r2, dy, dy);
      r2 = vfmaq_f64(r2, dz, dz);

      // TODO try to use sqrt estimation
      const float64x2_t r = vsqrtq_f64(r2);
      // float64x2_t r = vrsqrteq_f64(r2);
      //  r = vrecpeq_f64(r);

      const float64x2_t m_j = vld1q_f64(m + j);
      const float64x2_t acc = vmulq_f64(vG, m_j);
      const float64x2_t ar = vdivq_f64(acc, r);
      // const float64x2_t ar = vmulq_f64(acc, r);

      ax = vfmaq_f64(ax, dx, ar);
      ay = vfmaq_f64(ay, dy, ar);
      az = vfmaq_f64(az, dz, ar);
    }
    double sAx = vaddvq_f64(ax);
    double sAy = vaddvq_f64(ay);
    double sAz = vaddvq_f64(az);

    // loop tail
    if (odd_len) {
      const std::size_t last = len - 1;

      const double dx = px[last] - px[i];
      const double dy = py[last] - py[i];
      const double dz = pz[last] - pz[i];

      const double r2 = (dx * dx) + (dy * dy) + (dz * dz) + EPSILON_D;
      const double r = std::sqrt(r2);
      const double acc = physics::G * m[last] / r2;
      const double ar = acc / r;

      sAx += dx * ar;
      sAy += dy * ar;
      sAz += dz * ar;
    }

    vx[i] += sAx * dt;
    vy[i] += sAy * dt;
    vz[i] += sAz * dt;

    px[i] += vx[i] * dt;
    py[i] += vy[i] * dt;
    pz[i] += vz[i] * dt;
  }
}